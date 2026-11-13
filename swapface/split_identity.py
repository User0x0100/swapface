"""Inference-only identity conditioning split for the InSwapper and StyleGAN2 Coarse models.

All identity-to-modulation weights live exclusively in IdentityProjection. The
runtime receives only precomputed modulation values; training is unchanged.
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.networks import Coarse, Generator, IDInject, StyleGAN2Coarse
from models.stylegan2 import ModulatedConv2d, StyledConv, SynthesisStage, ToOutput


def _static_condition_slices(conditioning: Tensor, widths: list[int]) -> list[Tensor]:
    """Expand static channel offsets into ONNX Slice, not Sequence operators.

    torch.split(conditioning, widths, dim=1) exports as SplitToSequence on
    some PyTorch versions, which TensorRT cannot parse even for fixed shapes.
    """
    styles: list[Tensor] = []
    start = 0
    for width in widths:
        styles.append(conditioning[:, start:start + width])
        start += width
    return styles


class InSwapperProjection(nn.Module):
    def __init__(self, coarse: Coarse) -> None:
        super().__init__()
        self.linears = nn.ModuleList(
            deepcopy(projection)
            for layer in coarse.latent_space.layers
            for projection in (layer.adain0.linear, layer.adain1.linear)
        )
        self.width = sum(linear.out_features for linear in self.linears)

    def forward(self, identity: Tensor) -> Tensor:
        return torch.cat([linear(identity) for linear in self.linears], dim=1)


class PrecomputedAdaIN(nn.Module):
    def __init__(self, adain: nn.Module) -> None:
        super().__init__()
        self.norm = deepcopy(adain.norm)

    def forward(self, x: Tensor, gamma_beta: Tensor) -> Tensor:
        # Static channel slices are exported as ONNX Slice, not SplitToSequence.
        channels = self.norm.num_features
        gamma = gamma_beta[:, :channels]
        beta = gamma_beta[:, channels:channels * 2]
        return self.norm(x) * gamma[:, :, None, None] + beta[:, :, None, None]


class PrecomputedIDInject(nn.Module):
    def __init__(self, block: IDInject) -> None:
        super().__init__()
        self.conv0 = deepcopy(block.conv0)
        self.conv1 = deepcopy(block.conv1)
        self.adain0 = PrecomputedAdaIN(block.adain0)
        self.adain1 = PrecomputedAdaIN(block.adain1)
        self.act = deepcopy(block.act)

    def forward(self, x: Tensor, style0: Tensor, style1: Tensor) -> Tensor:
        residual = self.act(self.adain0(self.conv0(x), style0))
        residual = self.adain1(self.conv1(residual), style1)
        return x + residual


class PrecomputedInSwapper(nn.Module):
    def __init__(self, coarse: Coarse) -> None:
        super().__init__()
        self.img_resolution = coarse.img_resolution
        self.from_rgb = deepcopy(coarse.from_rgb)
        self.encoder = deepcopy(coarse.encoder)
        self.layers = nn.ModuleList(PrecomputedIDInject(block) for block in coarse.latent_space.layers)
        self.decoder = deepcopy(coarse.decoder)
        self.to_rgb = deepcopy(coarse.to_rgb)
        self.style_widths = [block.adain0.linear.out_features for block in coarse.latent_space.layers for _ in range(2)]
        self.condition_width = sum(self.style_widths)

    def forward(self, target: Tensor, conditioning: Tensor) -> Tensor:
        x = F.interpolate(target, size=self.img_resolution, mode="bilinear", align_corners=False)
        x = self.encoder(self.from_rgb(x))
        styles = _static_condition_slices(conditioning, self.style_widths)
        for i, block in enumerate(self.layers):
            x = block(x, styles[2 * i], styles[2 * i + 1])
        return self.to_rgb(self.decoder(x))


def _modulated_layers(generator: nn.Module) -> list[ModulatedConv2d]:
    layers = [generator.conv4.conv, generator.to_rgb4.conv]
    for stage in generator.stages:
        layers.extend((stage.conv0.conv, stage.conv1.conv, stage.to_rgb.conv))
    return layers


class StyleGAN2Projection(nn.Module):
    def __init__(self, coarse: StyleGAN2Coarse) -> None:
        super().__init__()
        generator = coarse.generator
        self.id_mapping = deepcopy(generator.id_mapping)
        self.style = deepcopy(generator.style)
        self.modulations = nn.ModuleList(deepcopy(layer.modulation) for layer in _modulated_layers(generator))
        self.width = sum(layer.in_ch for layer in _modulated_layers(generator))

    def forward(self, identity: Tensor) -> Tensor:
        w = self.style(self.id_mapping(identity))
        return torch.cat([projection(w) for projection in self.modulations], dim=1)


class PrecomputedModulatedConv2d(nn.Module):
    def __init__(self, layer: ModulatedConv2d) -> None:
        super().__init__()
        self.in_ch = layer.in_ch
        self.out_ch = layer.out_ch
        self.kernel_size = layer.kernel_size
        self.upsample = layer.upsample
        self.demodulate = layer.demodulate
        self.scale = layer.scale
        self.weight = deepcopy(layer.weight)
        self.blur = deepcopy(layer.blur) if layer.upsample else None
        # Deliberately omit layer.modulation and all of its weights.

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        weight = self.weight * self.scale * style.reshape(batch, 1, self.in_ch, 1, 1)
        if self.demodulate:
            demod = torch.rsqrt(weight.float().square().sum(dim=(2, 3, 4)) + 1e-8)
            weight = weight * demod.to(weight.dtype).reshape(batch, self.out_ch, 1, 1, 1)
        if self.upsample:
            weight = weight.reshape(batch, self.out_ch, self.in_ch, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * self.in_ch, self.out_ch, self.kernel_size, self.kernel_size)
            x = F.conv_transpose2d(x.reshape(1, batch * channels, height, width), weight, stride=2, groups=batch)
            x = x.reshape(batch, self.out_ch, x.shape[-2], x.shape[-1])
            assert self.blur is not None
            return self.blur(x)
        weight = weight.reshape(batch * self.out_ch, self.in_ch, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, batch * channels, height, width), weight, padding=self.kernel_size // 2, groups=batch)
        return x.reshape(batch, self.out_ch, height, width)


class PrecomputedStyledConv(nn.Module):
    def __init__(self, layer: StyledConv) -> None:
        super().__init__()
        self.conv = PrecomputedModulatedConv2d(layer.conv)
        self.inject_gain = deepcopy(layer.inject_gain)
        self.bias = deepcopy(layer.bias)

    def forward(self, x: Tensor, style: Tensor, target_feat: Tensor) -> Tensor:
        x = torch.cat((self.conv(x, style), self.inject_gain * target_feat), dim=1)
        return F.leaky_relu(x + self.bias, negative_slope=0.2) * (2.0 ** 0.5)


class PrecomputedToOutput(nn.Module):
    def __init__(self, layer: ToOutput) -> None:
        super().__init__()
        self.conv = PrecomputedModulatedConv2d(layer.conv)
        self.bias = deepcopy(layer.bias)
        self.upsample = deepcopy(layer.upsample)

    def forward(self, x: Tensor, style: Tensor, previous: Tensor | None) -> Tensor:
        out = self.conv(x, style) + self.bias
        if previous is not None:
            assert self.upsample is not None
            out = out + self.upsample(previous)
        return out


class PrecomputedSynthesisStage(nn.Module):
    def __init__(self, stage: SynthesisStage) -> None:
        super().__init__()
        self.conv0 = PrecomputedStyledConv(stage.conv0)
        self.conv1 = PrecomputedStyledConv(stage.conv1)
        self.to_rgb = PrecomputedToOutput(stage.to_rgb)

    def forward(self, x: Tensor, styles: tuple[Tensor, Tensor, Tensor], target_feat: Tensor, rgb: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv0(x, styles[0], target_feat)
        x = self.conv1(x, styles[1], target_feat)
        return x, self.to_rgb(x, styles[2], rgb)


class PrecomputedStyleGAN2(nn.Module):
    def __init__(self, coarse: StyleGAN2Coarse) -> None:
        super().__init__()
        generator = coarse.generator
        self.img_resolution = coarse.img_resolution
        self.encoder = deepcopy(generator.encoder)
        self.constant = deepcopy(generator.constant)
        self.conv4 = PrecomputedStyledConv(generator.conv4)
        self.to_rgb4 = PrecomputedToOutput(generator.to_rgb4)
        self.stages = nn.ModuleList(PrecomputedSynthesisStage(stage) for stage in generator.stages)
        self.style_widths = [layer.in_ch for layer in _modulated_layers(generator)]
        self.condition_width = sum(self.style_widths)

    def forward(self, target: Tensor, conditioning: Tensor) -> Tensor:
        target = F.interpolate(target, size=self.img_resolution, mode="bilinear", align_corners=False)
        features = self.encoder(target)
        styles = _static_condition_slices(conditioning, self.style_widths)
        x = self.constant.expand(target.shape[0], -1, -1, -1)
        x = self.conv4(x, styles[0], features[4])
        rgb = self.to_rgb4(x, styles[1], None)
        for i, stage in enumerate(self.stages):
            resolution = 2 ** (i + 3)
            x, rgb = stage(x, (styles[2 + 3 * i], styles[3 + 3 * i], styles[4 + 3 * i]), features[resolution], rgb)
        return rgb


class IdentityProjection(nn.Module):
    """Consumes the original normalized [B,512] embedding, outputs [B,width]."""

    def __init__(self, generator: Generator) -> None:
        super().__init__()
        if isinstance(generator.coarse, Coarse):
            self.mapping = InSwapperProjection(generator.coarse)
            self.coarse_type = "inswapper"
        elif isinstance(generator.coarse, StyleGAN2Coarse):
            self.mapping = StyleGAN2Projection(generator.coarse)
            self.coarse_type = "stylegan2"
        else:
            raise ValueError(f"unsupported Coarse type: {type(generator.coarse).__name__}")
        self.condition_width = self.mapping.width

    def forward(self, identity: Tensor) -> Tensor:
        return self.mapping(identity)


class SplitIdentityEncoder(nn.Module):
    """Server artifact: aligned identity image -> encoder -> all identity mappings."""

    def __init__(self, encoder: nn.Module, projection: IdentityProjection) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, faces: Tensor) -> Tensor:
        return self.projection(self.encoder(faces))


class SplitGenerator(nn.Module):
    """Client artifact: target image + precomputed condition -> generated image."""

    def __init__(self, generator: Generator) -> None:
        super().__init__()
        self.img_resolution = generator.img_resolution
        self.coarse_type = "stylegan2" if isinstance(generator.coarse, StyleGAN2Coarse) else "inswapper"
        if isinstance(generator.coarse, Coarse):
            self.coarse = PrecomputedInSwapper(generator.coarse)
        elif isinstance(generator.coarse, StyleGAN2Coarse):
            self.coarse = PrecomputedStyleGAN2(generator.coarse)
        else:
            raise ValueError(f"unsupported Coarse type: {type(generator.coarse).__name__}")
        self.hq = deepcopy(generator.hq)
        self.condition_width = self.coarse.condition_width

    def forward(self, target: Tensor, conditioning: Tensor) -> Tensor:
        return self.hq(target, self.coarse(target, conditioning))


def build_split_models(generator: Generator, encoder: nn.Module) -> tuple[SplitIdentityEncoder, SplitGenerator]:
    projection = IdentityProjection(generator)
    runtime = SplitGenerator(generator)
    if projection.condition_width != runtime.condition_width:
        raise ValueError("identity projector and runtime condition widths differ")
    return SplitIdentityEncoder(encoder, projection).eval(), runtime.eval()
