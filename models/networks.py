
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .stylegan2 import Generator as StyleGAN2Generator


class AdaIN(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False, eps=1e-8)
        # InSwapper-style affine modulation: one id_dim -> 2*C projection,
        # first half gamma, second half beta. The scale/shift initialization
        # preserves the project's previous near-identity AdaIN starting point.
        self.linear = nn.Linear(w_dim, channels * 2)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.linear.bias[:channels].fill_(1.0)
            self.linear.bias[channels:].zero_()

    def forward(self, x: Tensor, id_feat: Tensor) -> Tensor:
        gamma, beta = self.linear(id_feat).chunk(2, dim=1)
        return self.norm(x) * gamma[:, :, None, None] + beta[:, :, None, None]


class IDInject(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.act = nn.SiLU()

        self.adain0 = AdaIN(channels, w_dim)
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

        self.adain1 = AdaIN(channels, w_dim)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        residual = self.conv0(x)
        residual = self.adain0(residual, w)
        residual = self.act(residual)

        residual = self.conv1(residual)
        residual = self.adain1(residual, w)

        return x + residual


class FromRGB(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.SiLU(),
            nn.Conv2d(out_ch // 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )


class ToRGB(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_ch // 2, out_ch, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
        )


class UpSample(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )


class DownSample(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )


class WSpaceMap(nn.Sequential):
    """Two Linear-SiLU layers mapping identity into one shared W."""

    def __init__(self, id_dim: int) -> None:
        super().__init__(
            nn.Linear(id_dim, id_dim),
            nn.SiLU(),
            nn.Linear(id_dim, id_dim),
            nn.SiLU(),
        )


class LatentBlock(nn.Module):
    def __init__(self, channels: int, w_dim: int, num_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([IDInject(channels, w_dim) for _ in range(num_layers)])

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, w)
        return x


class InSwapper(nn.Module):
    """Full-resolution U-Net generator with identity-modulated bottleneck."""

    def __init__(
        self,
        img_resolution: int = 512,
        img_channels: int = 3,
        latent_resolution: int = 32,
        num_latent: int = 6,
        base_ch: int = 64,
        max_ch: int = 256,
        id_dim: int = 512,
        skip_mode: str = "cat",
        skip_layers: tuple[int, ...] | list[int] = (),
    ) -> None:
        super().__init__()
        if max_ch < base_ch or base_ch < 2 or max_ch < 2:
            raise ValueError("invalid base_ch/max_ch")
        if latent_resolution < 2 or img_resolution < latent_resolution or img_resolution % latent_resolution:
            raise ValueError("latent_resolution must be >= 2 and divide img_resolution")
        scale = img_resolution // latent_resolution
        if scale < 2 or scale & (scale - 1):
            raise ValueError("img_resolution / latent_resolution must be a power of 2 >= 2")
        depth = scale.bit_length() - 1
        if skip_mode not in ("cat", "add"):
            raise ValueError("skip_mode must be cat or add")
        if not isinstance(skip_layers, (list, tuple)) or any(type(i) is not int or i < 0 or i >= depth for i in skip_layers) or len(set(skip_layers)) != len(skip_layers):
            raise ValueError(f"skip_layers must be unique integer indices in [0,{depth-1}]")
        self.img_resolution = img_resolution
        self.skip_mode = skip_mode
        self.skip_layers = tuple(sorted(skip_layers))
        enabled = set(self.skip_layers)
        channels = [min(max_ch, base_ch * 2**i) for i in range(depth + 1)]
        self.from_rgb = FromRGB(img_channels, base_ch)
        self.encoder = nn.ModuleList(DownSample(channels[i], channels[i + 1]) for i in range(depth))
        self.w_space_map = WSpaceMap(id_dim)
        self.latent_space = LatentBlock(channels[-1], id_dim, num_latent)
        self.decoder = nn.ModuleList()
        current_ch = channels[-1]
        for index in range(depth):
            out_ch = channels[-index - 2]
            self.decoder.append(UpSample(current_ch, out_ch))
            current_ch = out_ch * (2 if index in enabled and skip_mode == "cat" else 1)
        self.to_rgb = ToRGB(current_ch, img_channels)

    def forward(self, x: Tensor, id_feat: Tensor, return_resize_in: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        resized = x if x.shape[-2:] == (self.img_resolution, self.img_resolution) else F.interpolate(x, size=self.img_resolution, mode="bilinear", align_corners=False)
        feature = self.from_rgb(resized)
        # Retain only encoder features consumed by enabled Decoder skip layers.
        skips: dict[int, Tensor] = {}
        last_skip = len(self.decoder) - 1
        if last_skip in self.skip_layers:
            skips[last_skip] = feature
        for encoder_index, block in enumerate(self.encoder):
            feature = block(feature)
            skip_index = len(self.encoder) - encoder_index - 2
            if skip_index in self.skip_layers:
                skips[skip_index] = feature
        feature = self.latent_space(feature, self.w_space_map(id_feat))
        for index, block in enumerate(self.decoder):
            feature = block(feature)
            if index in self.skip_layers:
                target_feature = skips.pop(index)
                feature = torch.cat((feature, target_feature), dim=1) if self.skip_mode == "cat" else feature + target_feature
        result = self.to_rgb(feature)
        return (result, resized) if return_resize_in else result


class StyleGAN2Swap(nn.Module):
    """Single-stage StyleSwap synthesis with optional mask head."""

    def __init__(
        self,
        img_resolution: int = 128,
        img_channels: int = 3,
        id_dim: int = 512,
        w_dim: int = 512,
        mapping_layers: int = 8,
        start_resolution: int = 4,
        channels: tuple[int, int] | list[int] = (128, 512),
        target_inject_mode: str = "concat",
        target_inject_layers: tuple[int, ...] | list[int] | None = None,
        mask_enabled: bool = False,
        mask_noise_blank_level: int = 9,
        mask_guidance_epsilon: float = 1e-4,
        mask_guidance_interpolation: str = "bilinear",
    ) -> None:
        super().__init__()
        self.img_resolution = img_resolution
        self.generator = StyleGAN2Generator(
            img_resolution=img_resolution,
            img_channels=img_channels,
            id_dim=id_dim,
            w_dim=w_dim,
            mapping_layers=mapping_layers,
            start_resolution=start_resolution,
            channels=channels,
            target_inject_mode=target_inject_mode,
            target_inject_layers=target_inject_layers,
            mask_enabled=mask_enabled,
            mask_noise_blank_level=mask_noise_blank_level,
            mask_guidance_epsilon=mask_guidance_epsilon,
            mask_guidance_interpolation=mask_guidance_interpolation,
        )

    def forward(
        self, x: Tensor, id_feat: Tensor, return_resize_in: bool = False, return_mask: bool = False,
        return_mask_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        resize_in = x if x.shape[-2:] == (self.img_resolution, self.img_resolution) else F.interpolate(x, size=self.img_resolution, mode="bilinear", align_corners=False)
        # Preserve StyleGAN2's unbounded RGB output; the loss functions
        # supervise its range without saturating the output gradient.
        result = self.generator(
            resize_in, id_feat, return_mask=return_mask, return_mask_logits=return_mask_logits,
        )
        if return_mask or return_mask_logits:
            output, mask_or_logits = result
            return (output, resize_in, mask_or_logits) if return_resize_in else (output, mask_or_logits)
        if return_resize_in:
            return result, resize_in
        return result


class Generator(nn.Module):
    """Single-stage generator, selecting InSwapper U-Net or StyleSwap synthesis."""

    def __init__(
        self,
        img_resolution: int = 512,
        id_dim: int = 512,
        type: str = "inswapper",
        latent_resolution: int = 32,
        num_latent: int = 6,
        base_ch: int = 64,
        max_ch: int = 256,
        skip_mode: str = "cat",
        skip_layers: tuple[int, ...] | list[int] = (),
        w_dim: int = 512,
        mapping_layers: int = 8,
        start_resolution: int = 4,
        channels: tuple[int, int] | list[int] = (128, 512),
        target_inject_mode: str = "concat",
        target_inject_layers: tuple[int, ...] | list[int] | None = None,
        mask_enabled: bool = False,
        mask_noise_blank_level: int = 9,
        mask_guidance_epsilon: float = 1e-4,
        mask_guidance_interpolation: str = "bilinear",
    ) -> None:
        super().__init__()
        img_channels = 3
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.id_dim = id_dim
        self.type = type
        common = {"type": type, "img_resolution": img_resolution, "id_dim": id_dim}
        if type == "inswapper":
            self.core = InSwapper(img_resolution=img_resolution, img_channels=img_channels,
                                  latent_resolution=latent_resolution, num_latent=num_latent,
                                  base_ch=base_ch, max_ch=max_ch, id_dim=id_dim,
                                  skip_mode=skip_mode, skip_layers=skip_layers)
            self.network_cfg = common | {
                "latent_resolution": latent_resolution, "num_latent": num_latent,
                "base_ch": base_ch, "max_ch": max_ch,
                "skip_mode": skip_mode, "skip_layers": list(self.core.skip_layers),
            }
        elif type == "stylegan2":
            self.core = StyleGAN2Swap(
                img_resolution=img_resolution, img_channels=img_channels,
                id_dim=id_dim, w_dim=w_dim, mapping_layers=mapping_layers,
                start_resolution=start_resolution, channels=channels,
                target_inject_mode=target_inject_mode, target_inject_layers=target_inject_layers,
                mask_enabled=mask_enabled, mask_noise_blank_level=mask_noise_blank_level,
                mask_guidance_epsilon=mask_guidance_epsilon,
                mask_guidance_interpolation=mask_guidance_interpolation,
            )
            self.network_cfg = common | {
                "w_dim": w_dim, "mapping_layers": mapping_layers, "start_resolution": start_resolution,
                "channels": list(channels), "target_inject_mode": target_inject_mode,
                "target_inject_layers": list(self.core.generator.target_inject_layers),
                "mask_enabled": mask_enabled, "mask_noise_blank_level": mask_noise_blank_level,
                "mask_guidance_epsilon": mask_guidance_epsilon,
                "mask_guidance_interpolation": mask_guidance_interpolation,
            }
        else:
            raise ValueError(f"unknown Generator type: {type!r}")

    def forward(self, target: Tensor, id_feat: Tensor, return_mask: bool = False,
                return_mask_logits: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        if target.ndim != 4 or target.shape[1:] != (self.img_channels, self.img_resolution, self.img_resolution):
            raise ValueError("target shape does not match Generator")
        if id_feat.shape != (target.shape[0], self.id_dim):
            raise ValueError("identity feature shape does not match Generator")
        if return_mask or return_mask_logits:
            if not isinstance(self.core, StyleGAN2Swap):
                raise ValueError("Mask requires StyleGAN2")
            return self.core(target, id_feat, return_mask=return_mask, return_mask_logits=return_mask_logits)
        return self.core(target, id_feat)


if __name__ == "__main__":
    import argparse
    import tomllib
    from pathlib import Path

    from fvcore.nn import FlopCountAnalysis
    from torchinfo import summary

    from swapface.config import _normalize_generator

    parser = argparse.ArgumentParser(description="统计单阶段 Generator 的参数量与推理计算量")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "experiments/train.toml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    with args.config.open("rb") as file:
        network_cfg = _normalize_generator({"generator": tomllib.load(file)["generator"]})
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    model = Generator(**network_cfg).to(device).eval()
    target = torch.randn(args.batch_size, 3, network_cfg["img_resolution"], network_cfg["img_resolution"], device=device)
    identity = torch.randn(args.batch_size, network_cfg["id_dim"], device=device)

    summary(
        model,
        input_data=(target, identity),
        depth=3,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        row_settings=("var_names",),
    )
    with torch.inference_mode():
        output = model(target, identity)
    print(f"output: {tuple(output.shape)}")
    print("NetWork_Info:")
    for key, value in network_cfg.items():
        print(f"  {key:30}: {value}")

    flops = FlopCountAnalysis(model, (target, identity))
    flops.unsupported_ops_warnings(False)
    macs = flops.total()
    print(f"\n模型总MACs（fvcore 可统计算子）: {macs / 1e9:.4f} GMACs")
    print(f"约合FLOPs（按 1 MAC = 2 FLOPs）: {2 * macs / 1e9:.4f} GFLOPs")
    unsupported = flops.unsupported_ops()
    if unsupported:
        print(f"未计入算子: {', '.join(sorted(unsupported))}")
