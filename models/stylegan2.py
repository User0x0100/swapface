"""StyleSwap-inspired StyleGAN2 face-swap generator.

The target-image encoder supplies spatial features at every synthesis resolution.
The source identity embedding controls the synthesis convolutions via StyleGAN2
weight modulation/demodulation. This module is independent of the current
Coarse/HQ training model; it does not change the active generator selection.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .discriminator.upfirdn2d import UpFIRDn2d, upfirdn2d
from .discriminator.upfirdn2d.upfirdn2d import setup_filter


class EqualLinear(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, bias_init: float = 0.0, lr_mul: float = 1.0, activate: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch) / lr_mul)
        self.bias = nn.Parameter(torch.full((out_ch,), bias_init))
        self.scale = lr_mul / math.sqrt(in_ch)
        self.lr_mul = lr_mul
        self.activate = activate

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)
        if self.activate:
            x = F.leaky_relu(x, negative_slope=0.2) * math.sqrt(2.0)
        return x


class PixelNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + 1e-8)


class FIRBlur(nn.Module):
    """Apply the repository's existing UpFirDn kernel with StyleSwap padding."""

    def __init__(self, kernel: Sequence[int], padding: tuple[int, int], gain: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("kernel", setup_filter(list(kernel)), persistent=False)
        self.padding = padding
        self.gain = gain

    def forward(self, x: Tensor) -> Tensor:
        p0, p1 = self.padding
        return upfirdn2d(x, self.kernel, 1, 1, 1, 1, p0, p1, p0, p1, False, self.gain)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, *, kernel_size: int = 3, upsample: bool = False, demodulate: bool = True, blur_kernel: Sequence[int] = (1, 3, 3, 1)) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.demodulate = demodulate
        self.scale = 1.0 / math.sqrt(in_ch * kernel_size * kernel_size)
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))
        self.modulation = EqualLinear(w_dim, in_ch, bias_init=1.0)

        if upsample:
            # StyleGAN2 transposed convolution produces (2 * H + 1);
            # FIR filtering with these pads returns precisely 2 * H.
            p = len(blur_kernel) - 2 - (kernel_size - 1)
            self.blur = FIRBlur(blur_kernel, ((p + 1) // 2 + 1, p // 2 + 1), gain=4.0)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        style = self.modulation(w).reshape(batch, 1, self.in_ch, 1, 1)
        weight = self.weight * self.scale * style
        if self.demodulate:
            demod = torch.rsqrt(weight.float().square().sum(dim=(2, 3, 4)) + 1e-8)
            weight = weight * demod.to(weight.dtype).reshape(batch, self.out_ch, 1, 1, 1)

        if self.upsample:
            weight = weight.reshape(batch, self.out_ch, self.in_ch, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * self.in_ch, self.out_ch, self.kernel_size, self.kernel_size)
            x = F.conv_transpose2d(x.reshape(1, batch * channels, height, width), weight, stride=2, groups=batch)
            x = x.reshape(batch, self.out_ch, x.shape[-2], x.shape[-1])
            return self.blur(x)

        weight = weight.reshape(batch * self.out_ch, self.in_ch, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, batch * channels, height, width), weight, padding=self.kernel_size // 2, groups=batch)
        return x.reshape(batch, self.out_ch, height, width)


class StyledConv(nn.Module):
    """StyleGAN2 convolution with additive target-attribute feature injection."""

    def __init__(self, in_ch: int, out_ch: int, w_dim: int, *, upsample: bool = False, blur_kernel: Sequence[int] = (1, 3, 3, 1)) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, w_dim, upsample=upsample, blur_kernel=blur_kernel)
        # Retain StyleSwap's zero-initialized injection gain but add features
        # instead of concatenating them, preserving rosinality channel widths.
        self.inject_gain = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x: Tensor, w: Tensor, target_feat: Tensor) -> Tensor:
        x = self.conv(x, w)
        return F.leaky_relu(x + self.inject_gain * target_feat + self.bias, negative_slope=0.2) * math.sqrt(2.0)


class ToOutput(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, *, upsample: bool, blur_kernel: Sequence[int]) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, w_dim, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.upsample = UpFIRDn2d(filt_size=len(blur_kernel)) if upsample else None
        # UpFIRDn2d uses the same default [1,3,3,1] binomial filter.
        if tuple(blur_kernel) != (1, 3, 3, 1):
            raise ValueError("ToOutput currently supports blur_kernel=(1,3,3,1) only")

    def forward(self, x: Tensor, w: Tensor, previous: Tensor | None) -> Tensor:
        out = self.conv(x, w)
        out = out + self.bias
        if previous is not None:
            assert self.upsample is not None
            out = out + self.upsample(previous)
        return out


class EqualConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, *, downsample: bool, blur_kernel: Sequence[int]) -> None:
        super().__init__()
        if downsample:
            p = len(blur_kernel) - 2 + kernel_size - 1
            self.blur = FIRBlur(blur_kernel, ((p + 1) // 2, p // 2))
        else:
            self.blur = None
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.scale = 1.0 / math.sqrt(in_ch * kernel_size * kernel_size)
        self.stride = 2 if downsample else 1
        self.padding = 0 if downsample else kernel_size // 2

    def forward(self, x: Tensor) -> Tensor:
        if self.blur is not None:
            x = self.blur(x)
        x = F.conv2d(x, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return F.leaky_relu(x, negative_slope=0.2) * math.sqrt(2.0)


class AttributeEncoder(nn.Module):
    def __init__(self, channels: dict[int, int], img_resolution: int, img_channels: int, blur_kernel: Sequence[int]) -> None:
        super().__init__()
        self.from_rgb = EqualConv2d(img_channels, channels[img_resolution], 1, downsample=False, blur_kernel=blur_kernel)
        self.downs = nn.ModuleList()
        resolution = img_resolution
        while resolution > 4:
            self.downs.append(EqualConv2d(channels[resolution], channels[resolution // 2], 3, downsample=True, blur_kernel=blur_kernel))
            resolution //= 2
        self.img_resolution = img_resolution

    def forward(self, target: Tensor) -> dict[int, Tensor]:
        resolution = self.img_resolution
        x = self.from_rgb(target)
        features = {resolution: x}
        for down in self.downs:
            x = down(x)
            resolution //= 2
            features[resolution] = x
        return features


class SynthesisStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, img_channels: int, *, blur_kernel: Sequence[int]) -> None:
        super().__init__()
        self.conv0 = StyledConv(in_ch, out_ch, w_dim, upsample=True, blur_kernel=blur_kernel)
        self.conv1 = StyledConv(out_ch, out_ch, w_dim, blur_kernel=blur_kernel)
        self.to_rgb = ToOutput(out_ch, img_channels, w_dim, upsample=True, blur_kernel=blur_kernel)

    def forward(self, x: Tensor, w: Tensor, target_feat: Tensor, rgb: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv0(x, w, target_feat)
        x = self.conv1(x, w, target_feat)
        rgb = self.to_rgb(x, w, rgb)
        return x, rgb


class Generator(nn.Module):
    """StyleSwap Generator with a project-style interface.

    Input: target image [B,C,R,R] and identity embedding [B,id_dim].
    Output: generated RGB image [B,C,R,R] without a mask branch.

    The identity embedding is mapped into the StyleGAN2 W space; the target
    encoder supplies per-resolution features by addition after each styled
    convolution. Synthesis channels follow rosinality/stylegan2-pytorch.
    """

    def __init__(
        self,
        img_resolution: int = 128,
        img_channels: int = 3,
        id_dim: int = 512,
        w_dim: int = 512,
        mapping_layers: int = 8,
        channel_multiplier: int = 1,
        blur_kernel: Sequence[int] = (1, 3, 3, 1),
    ) -> None:
        super().__init__()
        if img_resolution < 4 or img_resolution > 1024 or img_resolution & (img_resolution - 1):
            raise ValueError("img_resolution must be a power of two in [4,1024]")
        if img_channels <= 0 or id_dim <= 0 or w_dim <= 0 or mapping_layers < 1:
            raise ValueError("img_channels, id_dim, w_dim, mapping_layers must be positive")
        if channel_multiplier <= 0:
            raise ValueError("channel_multiplier must be positive")
        if tuple(blur_kernel) != (1, 3, 3, 1):
            raise ValueError("only the original [1,3,3,1] StyleGAN2 FIR kernel is supported")

        self.network_cfg = {
            "img_resolution": img_resolution,
            "img_channels": img_channels,
            "id_dim": id_dim,
            "w_dim": w_dim,
            "mapping_layers": mapping_layers,
            "channel_multiplier": channel_multiplier,
            "blur_kernel": tuple(blur_kernel),
        }
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.id_dim = id_dim

        channel_base = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.channels = channel_base

        self.encoder = AttributeEncoder(self.channels, img_resolution, img_channels, blur_kernel)
        self.id_mapping = EqualLinear(id_dim, w_dim, activate=True)
        self.style = nn.Sequential(
            PixelNorm(),
            *[EqualLinear(w_dim, w_dim, lr_mul=0.01, activate=True) for _ in range(mapping_layers)],
        )
        self.constant = nn.Parameter(torch.randn(1, self.channels[4], 4, 4))
        self.conv4 = StyledConv(self.channels[4], self.channels[4], w_dim, blur_kernel=blur_kernel)
        self.to_rgb4 = ToOutput(self.channels[4], img_channels, w_dim, upsample=False, blur_kernel=blur_kernel)

        self.stages = nn.ModuleList()
        for res in (2**i for i in range(3, img_resolution.bit_length())):
            self.stages.append(SynthesisStage(self.channels[res // 2], self.channels[res], w_dim, img_channels, blur_kernel=blur_kernel))

    def forward(self, target: Tensor, id_feat: Tensor) -> Tensor:
        expected = (self.img_channels, self.img_resolution, self.img_resolution)
        if target.ndim != 4 or target.shape[1:] != expected:
            raise ValueError(f"target must be [B,{self.img_channels},{self.img_resolution},{self.img_resolution}], got {tuple(target.shape)}")
        if id_feat.ndim != 2 or id_feat.shape != (target.shape[0], self.id_dim):
            raise ValueError(f"id_feat must be [B,{self.id_dim}], got {tuple(id_feat.shape)}")

        features = self.encoder(target)
        w = self.style(self.id_mapping(id_feat))
        x = self.constant.expand(target.shape[0], -1, -1, -1)
        x = self.conv4(x, w, features[4])
        rgb = self.to_rgb4(x, w, None)

        for res, stage in zip((2**i for i in range(3, self.img_resolution.bit_length())), self.stages, strict=True):
            x, rgb = stage(x, w, features[res], rgb)

        return rgb


__all__ = ["Generator"]


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    network_cfg = {
        "img_resolution": 128,
        "img_channels": 3,
        "id_dim": 512,
        "w_dim": 512,
        "mapping_layers": 8,
        "channel_multiplier": 1,
        "blur_kernel": (1, 3, 3, 1),
    }

    model = Generator(**network_cfg).to(device).eval()
    target = torch.randn(
        (batch_size, network_cfg["img_channels"], network_cfg["img_resolution"], network_cfg["img_resolution"]),
        device=device,
    )
    id_feat = torch.randn((batch_size, network_cfg["id_dim"]), device=device)

    summary(
        model,
        input_data=(target, id_feat),
        depth=3,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        row_settings=("var_names",),
    )

    with torch.no_grad():
        output = model(target, id_feat)
    print(f"output: {tuple(output.shape)}")
    print("NetWork_Info:")
    for k, v in network_cfg.items():
        print(f"  {k:30}: {v}")

    flops = FlopCountAnalysis(model, (target, id_feat))
    flops.unsupported_ops_warnings(False)
    macs = flops.total()
    print(f"\n模型总MACs（fvcore 可统计算子）: {macs / 1e9:.4f} GMACs")
    print(f"约合FLOPs（按 1 MAC = 2 FLOPs）: {2 * macs / 1e9:.4f} GFLOPs")
    unsupported = flops.unsupported_ops()
    if unsupported:
        print(f"未计入算子: {', '.join(sorted(unsupported))}")
