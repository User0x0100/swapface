"""StyleSwap-inspired StyleGAN2 face-swap generator.

The target-image encoder optionally supplies spatial features at selected synthesis resolutions.
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
    """Modulated convolution with optional Target feature injection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        w_dim: int,
        *,
        inject_mode: str = "concat",
        inject_target: bool = True,
        upsample: bool = False,
        blur_kernel: Sequence[int] = (1, 3, 3, 1),
    ) -> None:
        super().__init__()
        if inject_mode not in {"concat", "add"}:
            raise ValueError("inject_mode must be concat or add")
        self.inject_mode = inject_mode
        self.inject_target = inject_target
        self.output_channels = out_ch * 2 if inject_target and inject_mode == "concat" else out_ch
        self.conv = ModulatedConv2d(in_ch, out_ch, w_dim, upsample=upsample, blur_kernel=blur_kernel)
        if inject_target:
            # The original StyleSwap path uses zero-initialized learnable gain.
            self.inject_gain = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(1, self.output_channels, 1, 1))

    def forward(self, x: Tensor, w: Tensor, target_feat: Tensor | None = None) -> Tensor:
        x = self.conv(x, w)
        if self.inject_target:
            if target_feat is None:
                raise ValueError("target_feat is required for an enabled injection layer")
            injected = self.inject_gain * target_feat
            if self.inject_mode == "concat":
                x = torch.cat((x, injected), dim=1)
            else:
                x = x + injected
        return F.leaky_relu(x + self.bias, negative_slope=0.2) * math.sqrt(2.0)


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


def mask_probability(logits: Tensor) -> Tensor:
    """Keep Mask probabilities and their sigmoid gradients in FP32 under autocast."""
    return torch.sigmoid(logits.float())


class ToMask(nn.Module):
    """Accumulate Mask logits across synthesis resolutions, without intermediate sigmoid."""

    def __init__(self, in_ch: int, w_dim: int, *, upsample: bool, blur_kernel: Sequence[int]) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, 1, w_dim, kernel_size=1, demodulate=False)
        self.upsample = UpFIRDn2d(filt_size=len(blur_kernel)) if upsample else None

    def forward(self, x: Tensor, w: Tensor, previous_logits: Tensor | None) -> Tensor:
        logits = self.conv(x, w).float()
        if previous_logits is not None:
            assert self.upsample is not None
            logits = logits + self.upsample(previous_logits)
        return logits


def stylegan2_mask_layer_index(resolution: int) -> int:
    """Legacy StyleSwap synthesis index, anchored at 4x4 even when synthesis starts later.

    8->1, 16->3, 32->5, 64->7, 128->9, 256->11, 512->13.
    """
    return 2 * (resolution.bit_length() - 1) - 5


def mask_guided_target_features(
    target_feat: Tensor, mask: Tensor, *, epsilon: float, interpolation: str,
) -> Tensor:
    """Replace masked Target features by their spatially weighted channel mean.

    Corresponds to the original StyleSwap `conver_noise_withmask` operation;
    it does not blend the generated RGB image with the input image.
    """
    mask = F.interpolate(mask, size=target_feat.shape[-2:], mode=interpolation)
    weights = mask.float()
    target_fp32 = target_feat.float()
    mean = (target_fp32 * weights).sum(dim=(2, 3), keepdim=True) / weights.sum(dim=(2, 3), keepdim=True).clamp_min(epsilon)
    return torch.lerp(target_fp32, mean, weights).to(target_feat.dtype)


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
    def __init__(
        self, channels: dict[int, int], img_resolution: int, min_resolution: int,
        img_channels: int, blur_kernel: Sequence[int],
    ) -> None:
        super().__init__()
        self.from_rgb = EqualConv2d(img_channels, channels[img_resolution], 1, downsample=False, blur_kernel=blur_kernel)
        self.downs = nn.ModuleList()
        resolution = img_resolution
        while resolution > min_resolution:
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
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        w_dim: int,
        img_channels: int,
        *,
        inject_mode: str,
        inject_target: bool,
        blur_kernel: Sequence[int],
        mask_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.conv0 = StyledConv(
            in_ch, out_ch, w_dim, inject_mode=inject_mode, inject_target=inject_target,
            upsample=True, blur_kernel=blur_kernel,
        )
        self.conv1 = StyledConv(
            self.conv0.output_channels, out_ch, w_dim, inject_mode=inject_mode,
            inject_target=inject_target, blur_kernel=blur_kernel,
        )
        self.to_rgb = ToOutput(self.conv1.output_channels, img_channels, w_dim, upsample=True, blur_kernel=blur_kernel)
        self.to_mask = ToMask(self.conv1.output_channels, w_dim, upsample=True, blur_kernel=blur_kernel) if mask_enabled else None

    def forward(
        self, x: Tensor, w: Tensor, target_feat: Tensor | None, rgb: Tensor, mask_logits: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        x = self.conv0(x, w, target_feat)
        x = self.conv1(x, w, target_feat)
        rgb = self.to_rgb(x, w, rgb)
        if self.to_mask is not None:
            assert mask_logits is not None
            mask_logits = self.to_mask(x, w, mask_logits)
        return x, rgb, mask_logits


class Generator(nn.Module):
    """StyleSwap Generator with a project-style interface.

    Input: target image [B,C,R,R] and identity embedding [B,id_dim].
    Output: generated RGB image [B,C,R,R]; optional StyleSwap mask head.

    The identity embedding is mapped into the StyleGAN2 W space; the target
    encoder optionally supplies per-resolution Target features using concatenation
    or addition after the styled convolutions.
    """

    def __init__(
        self,
        img_resolution: int = 128,
        img_channels: int = 3,
        id_dim: int = 512,
        w_dim: int = 512,
        mapping_layers: int = 8,
        start_resolution: int = 4,
        channels: Sequence[int] = (128, 512),
        target_inject_mode: str = "concat",
        target_inject_layers: Sequence[int] | None = None,
        blur_kernel: Sequence[int] = (1, 3, 3, 1),
        mask_enabled: bool = False,
        mask_noise_blank_level: int = 9,
        mask_guidance_epsilon: float = 1e-4,
        mask_guidance_interpolation: str = "bilinear",
    ) -> None:
        super().__init__()
        if img_resolution < 4 or img_resolution > 1024 or img_resolution & (img_resolution - 1):
            raise ValueError("img_resolution must be a power of two in [4,1024]")
        if img_channels <= 0 or id_dim <= 0 or w_dim <= 0 or mapping_layers < 1:
            raise ValueError("img_channels, id_dim, w_dim, mapping_layers must be positive")
        if (type(start_resolution) is not int or start_resolution < 4 or
                start_resolution > img_resolution or start_resolution & (start_resolution - 1)):
            raise ValueError("start_resolution must be a power of two in [4, img_resolution]")
        if (not isinstance(channels, (tuple, list)) or len(channels) != 2 or
                any(type(ch) is not int or ch <= 0 for ch in channels) or channels[1] < channels[0]):
            raise ValueError("channels must be (base_ch, max_ch) positive integers with max_ch >= base_ch")
        base_ch, max_ch = channels
        if tuple(blur_kernel) != (1, 3, 3, 1):
            raise ValueError("only the original [1,3,3,1] StyleGAN2 FIR kernel is supported")
        if target_inject_mode not in {"concat", "add"}:
            raise ValueError("target_inject_mode must be concat or add")
        if type(mask_enabled) is not bool:
            raise TypeError("mask_enabled must be bool")
        if type(mask_noise_blank_level) is not int or mask_noise_blank_level < -1:
            raise ValueError("mask_noise_blank_level must be an integer >= -1")
        if not math.isfinite(mask_guidance_epsilon) or mask_guidance_epsilon <= 0:
            raise ValueError("mask_guidance_epsilon must be a finite positive value")
        if mask_guidance_interpolation not in {"nearest", "bilinear"}:
            raise ValueError("mask_guidance_interpolation must be nearest or bilinear")
        max_layer = img_resolution.bit_length() - start_resolution.bit_length()
        if target_inject_layers is None:
            target_inject_layers = tuple(range(max_layer + 1))
        if not isinstance(target_inject_layers, (list, tuple)) or any(
            type(index) is not int or index < 0 or index > max_layer for index in target_inject_layers
        ):
            raise ValueError(f"target_inject_layers must contain integer indices in [0, {max_layer}]")
        if len(set(target_inject_layers)) != len(target_inject_layers):
            raise ValueError("target_inject_layers must not contain duplicate indices")
        target_inject_layers = tuple(sorted(target_inject_layers))
        self.target_inject_layers = target_inject_layers
        self.mask_enabled = mask_enabled
        self.mask_noise_blank_level = mask_noise_blank_level
        self.mask_guidance_epsilon = mask_guidance_epsilon
        self.mask_guidance_interpolation = mask_guidance_interpolation

        self.network_cfg = {
            "img_resolution": img_resolution,
            "img_channels": img_channels,
            "id_dim": id_dim,
            "w_dim": w_dim,
            "mapping_layers": mapping_layers,
            "start_resolution": start_resolution,
            "channels": (base_ch, max_ch),
            "blur_kernel": tuple(blur_kernel),
            "target_inject_mode": target_inject_mode,
            "target_inject_layers": target_inject_layers,
            "mask_enabled": mask_enabled,
            "mask_noise_blank_level": mask_noise_blank_level,
            "mask_guidance_epsilon": mask_guidance_epsilon,
            "mask_guidance_interpolation": mask_guidance_interpolation,
        }
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.id_dim = id_dim
        self.start_resolution = start_resolution
        self.synthesis_resolutions = tuple(
            2**exponent
            for exponent in range(start_resolution.bit_length() - 1, img_resolution.bit_length())
        )
        # base_ch belongs to the output-resolution stage. Each halving of
        # resolution doubles the width, capped by max_ch.
        self.channels = {
            resolution: min(max_ch, base_ch * (img_resolution // resolution))
            for resolution in self.synthesis_resolutions
        }
        # Encoding stops at the lowest resolution that actually consumes a
        # Target feature. With no injection, Target is not encoded at all.
        injected_resolutions = [self.synthesis_resolutions[index] for index in self.target_inject_layers]
        self.encoder = (
            AttributeEncoder(
                self.channels, img_resolution, min(injected_resolutions), img_channels, blur_kernel,
            )
            if injected_resolutions else None
        )
        self.id_mapping = EqualLinear(id_dim, w_dim, activate=True)
        self.style = nn.Sequential(
            PixelNorm(),
            *[EqualLinear(w_dim, w_dim, lr_mul=0.01, activate=True) for _ in range(mapping_layers)],
        )
        self.constant = nn.Parameter(torch.randn(
            1, self.channels[start_resolution], start_resolution, start_resolution,
        ))
        enabled = set(self.target_inject_layers)
        self.conv_first = StyledConv(
            self.channels[start_resolution], self.channels[start_resolution], w_dim,
            inject_mode=target_inject_mode, inject_target=0 in enabled, blur_kernel=blur_kernel,
        )
        self.to_rgb_first = ToOutput(
            self.conv_first.output_channels, img_channels, w_dim,
            upsample=False, blur_kernel=blur_kernel,
        )
        self.to_mask_first = (
            ToMask(self.conv_first.output_channels, w_dim, upsample=False, blur_kernel=blur_kernel)
            if mask_enabled else None
        )

        self.stages = nn.ModuleList()
        previous_channels = self.conv_first.output_channels
        for layer_index, resolution in enumerate(self.synthesis_resolutions[1:], start=1):
            stage = SynthesisStage(
                previous_channels, self.channels[resolution], w_dim, img_channels,
                inject_mode=target_inject_mode, inject_target=layer_index in enabled,
                blur_kernel=blur_kernel, mask_enabled=mask_enabled,
            )
            self.stages.append(stage)
            previous_channels = stage.conv1.output_channels

    def forward(
        self, target: Tensor, id_feat: Tensor, return_mask: bool = False, return_mask_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if return_mask and return_mask_logits:
            raise ValueError("return_mask and return_mask_logits are mutually exclusive")
        if (return_mask or return_mask_logits) and not self.mask_enabled:
            raise ValueError("return_mask/return_mask_logits requires mask_enabled=True")
        expected = (self.img_channels, self.img_resolution, self.img_resolution)
        if target.ndim != 4 or target.shape[1:] != expected:
            raise ValueError(f"target must be [B,{self.img_channels},{self.img_resolution},{self.img_resolution}], got {tuple(target.shape)}")
        if id_feat.ndim != 2 or id_feat.shape != (target.shape[0], self.id_dim):
            raise ValueError(f"id_feat must be [B,{self.id_dim}], got {tuple(id_feat.shape)}")

        features = self.encoder(target) if self.encoder is not None else {}
        w = self.style(self.id_mapping(id_feat))
        x = self.constant.expand(target.shape[0], -1, -1, -1)
        x = self.conv_first(x, w, features.get(self.start_resolution))
        rgb = self.to_rgb_first(x, w, None)
        mask_logits = self.to_mask_first(x, w, None) if self.to_mask_first is not None else None

        for layer_index, (res, stage) in enumerate(zip(self.synthesis_resolutions[1:], self.stages, strict=True), start=1):
            target_feat = features.get(res)
            # Layer indices are absolute: 8x8=1, 16x16=3, ..., 256x256=11.
            if (
                mask_logits is not None and layer_index in self.target_inject_layers
                and target_feat is not None and stylegan2_mask_layer_index(res) > self.mask_noise_blank_level
            ):
                target_feat = mask_guided_target_features(
                    target_feat, mask_probability(mask_logits),
                    epsilon=self.mask_guidance_epsilon,
                    interpolation=self.mask_guidance_interpolation,
                )
            x, rgb, mask_logits = stage(x, w, target_feat, rgb, mask_logits)

        if return_mask_logits:
            assert mask_logits is not None
            return rgb, mask_logits
        if return_mask:
            assert mask_logits is not None
            return rgb, mask_probability(mask_logits)
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
        "start_resolution": 4,
        "channels": (128, 512),
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
