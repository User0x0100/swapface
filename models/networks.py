from __future__ import annotations

import torch
from torch import Tensor, nn


def _num_downsamples(resolution: int, bottleneck_resolution: int, *, name: str) -> int:
    if resolution <= 0 or bottleneck_resolution <= 0:
        raise ValueError(f"{name} resolutions must be positive")
    if resolution < bottleneck_resolution or resolution % bottleneck_resolution != 0:
        raise ValueError(f"{name}: resolution={resolution} must be an integer multiple of bottleneck_resolution={bottleneck_resolution}")

    ratio = resolution // bottleneck_resolution
    if ratio & (ratio - 1):
        raise ValueError(f"{name}: resolution/bottleneck_resolution must be a power of two, got {ratio}")
    return ratio.bit_length() - 1


def _doubling_channels(base_ch: int, max_ch: int, num_levels: int) -> tuple[int, ...]:
    if base_ch <= 0 or max_ch <= 0:
        raise ValueError("base_ch and max_ch must be positive")
    if max_ch < base_ch:
        raise ValueError(f"max_ch={max_ch} must be >= base_ch={base_ch}")
    if num_levels <= 0:
        raise ValueError("num_levels must be positive")
    return tuple(min(max_ch, base_ch * (2**level)) for level in range(num_levels))


def _hq_channels(base_ch: int, max_ch: int, num_levels: int, hold_level: int | None) -> tuple[int, ...]:
    if hold_level is None:
        return _doubling_channels(base_ch, max_ch, num_levels)
    if hold_level <= 0 or hold_level >= num_levels:
        raise ValueError(f"hq_channel_hold_level must be in [1, {num_levels - 1}], got {hold_level}")
    if base_ch <= 0 or max_ch < base_ch:
        raise ValueError(f"invalid HQ channels: base_ch={base_ch}, max_ch={max_ch}")

    # One level keeps the previous channel count; later levels resume doubling.
    # Default: base=8, levels=6, hold=2 -> [8, 16, 16, 32, 64, 128].
    channels: list[int] = []
    for level in range(num_levels):
        exponent = level - (1 if level >= hold_level else 0)
        channels.append(min(max_ch, base_ch * (2**exponent)))
    return tuple(channels)


class CenteredRMSNorm2d(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x = x - x.mean(dim=(2, 3), keepdim=True)
        rms = torch.sqrt((x * x).mean(dim=(2, 3), keepdim=True) + self.eps)
        return x / rms


class ConvAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, activation: nn.Module | None = None) -> None:
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=True)]
        if activation is not None:
            layers.append(activation)
        super().__init__(*layers)


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__(ConvAct(in_ch, mid_ch, activation=nn.ReLU()), ConvAct(mid_ch, out_ch, activation=nn.ReLU()))


class UpConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, *, align_corners: bool, activation: nn.Module) -> None:
        super().__init__(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=align_corners), ConvAct(in_ch, out_ch, activation=activation))


class StyleResidualBlock(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.channels = channels
        self.pad = nn.ReflectionPad2d(1)
        self.conv0 = nn.Conv2d(channels, channels, 3, bias=True)
        self.norm0 = CenteredRMSNorm2d(eps)
        self.conv1 = nn.Conv2d(channels, channels, 3, bias=True)
        self.norm1 = CenteredRMSNorm2d(eps)
        self.act = nn.ReLU()

    @staticmethod
    def modulate(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        return x * gamma[:, :, None, None] + beta[:, :, None, None]

    def forward(
        self,
        x: Tensor,
        gamma0: Tensor,
        beta0: Tensor,
        gamma1: Tensor,
        beta1: Tensor,
    ) -> Tensor:
        residual = x
        x = self.conv0(self.pad(x))
        x = self.act(self.modulate(self.norm0(x), gamma0, beta0))
        x = self.conv1(self.pad(x))
        x = self.modulate(self.norm1(x), gamma1, beta1)
        return residual + x


class StyleBankProjector(nn.Module):
    sites_per_block = 2
    affine_components = 2  # gamma + beta

    def __init__(self, id_dim: int, channels: int, num_blocks: int) -> None:
        super().__init__()
        if id_dim <= 0 or channels <= 0 or num_blocks <= 0:
            raise ValueError("id_dim, channels and num_blocks must be positive")

        self.id_dim = id_dim
        self.channels = channels
        self.num_blocks = num_blocks
        self.output_dim = num_blocks * self.sites_per_block * self.affine_components * channels
        self.proj = nn.Linear(id_dim, self.output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            bias = self.proj.bias.view(self.num_blocks, self.sites_per_block, self.affine_components, self.channels)
            bias[:, :, 0].fill_(1.0)  # gamma
            bias[:, :, 1].zero_()  # beta

    def forward(self, style: Tensor) -> tuple[Tensor, ...]:
        # One Gemm + one Split in ONNX. Flat ordering is
        # [gamma0, beta0, gamma1, beta1, ...], each [B,C].
        return self.proj(style).split(self.channels, dim=1)


class CoarseSwapCore(nn.Module):
    def __init__(
        self,
        *,
        img_resolution: int = 512,
        img_channels: int = 3,
        id_dim: int = 512,
        resolution: int = 128,
        bottleneck_resolution: int = 32,
        base_ch: int = 32,
        max_ch: int = 256,
        num_style_blocks: int = 6,
        norm_eps: float = 1e-8,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if resolution > img_resolution:
            raise ValueError(f"coarse resolution={resolution} must be <= img_resolution={img_resolution}")
        if num_style_blocks <= 0:
            raise ValueError("num_style_blocks must be positive")

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.id_dim = id_dim
        self.resolution = resolution
        self.bottleneck_resolution = bottleneck_resolution
        self.num_down = _num_downsamples(resolution, bottleneck_resolution, name="coarse")

        # One 7x7 stem + one same-resolution 3x3 stage + one stage per downsample.
        self.feature_channels = _doubling_channels(base_ch, max_ch, self.num_down + 2)
        self.style_channels = self.feature_channels[-1]
        self.num_style_blocks = num_style_blocks

        self.resize_in = nn.Upsample(size=(resolution, resolution), mode="bilinear", align_corners=False)

        encoder: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            ConvAct(img_channels, self.feature_channels[0], 7, padding=0, activation=nn.LeakyReLU(leaky_relu_slope)),
            ConvAct(self.feature_channels[0], self.feature_channels[1], activation=nn.LeakyReLU(leaky_relu_slope)),
        ]
        for level in range(self.num_down):
            encoder.append(ConvAct(self.feature_channels[level + 1], self.feature_channels[level + 2], stride=2, activation=nn.LeakyReLU(leaky_relu_slope)))
        self.encoder = nn.Sequential(*encoder)

        self.style_projector = StyleBankProjector(id_dim=id_dim, channels=self.style_channels, num_blocks=num_style_blocks)
        self.style_blocks = nn.ModuleList([StyleResidualBlock(self.style_channels, norm_eps) for _ in range(num_style_blocks)])

        decoder: list[nn.Module] = []
        current_ch = self.feature_channels[-1]
        decoder_channels = tuple(reversed(self.feature_channels[:-1]))
        for level, out_ch in enumerate(decoder_channels):
            if level < self.num_down:
                decoder.append(UpConv(current_ch, out_ch, align_corners=False, activation=nn.LeakyReLU(leaky_relu_slope)))
            else:
                decoder.append(ConvAct(current_ch, out_ch, activation=nn.LeakyReLU(leaky_relu_slope)))
            current_ch = out_ch
        self.decoder = nn.Sequential(*decoder)

        self.to_rgb = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(current_ch, img_channels, 7, bias=True),
            nn.Tanh(),
        )

    def forward(self, target: Tensor, style: Tensor) -> Tensor:
        expected_target = (self.img_channels, self.img_resolution, self.img_resolution)
        if target.ndim != 4 or target.shape[1:] != expected_target:
            raise ValueError(f"target must be [B,{self.img_channels},{self.img_resolution},{self.img_resolution}], got {tuple(target.shape)}")
        if style.ndim != 2 or style.shape != (target.shape[0], self.id_dim):
            raise ValueError(f"input_2 must be [B,{self.id_dim}], got {tuple(style.shape)}")

        x = self.encoder(self.resize_in(target))
        affine = self.style_projector(style)
        for block_index, block in enumerate(self.style_blocks):
            offset = block_index * 4
            x = block(
                x,
                affine[offset],
                affine[offset + 1],
                affine[offset + 2],
                affine[offset + 3],
            )

        return self.to_rgb(self.decoder(x))


class HQRefiner(nn.Module):
    def __init__(
        self,
        *,
        img_resolution: int = 512,
        img_channels: int = 3,
        bottleneck_resolution: int = 16,
        base_ch: int = 8,
        max_ch: int = 128,
        channel_hold_level: int | None = 2,
    ) -> None:
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.bottleneck_resolution = bottleneck_resolution
        self.num_down = _num_downsamples(img_resolution, bottleneck_resolution, name="hq")
        if self.num_down < 1:
            raise ValueError("HQ refiner requires at least one downsampling stage")

        self.feature_channels = _hq_channels(
            base_ch,
            max_ch,
            self.num_down + 1,
            channel_hold_level,
        )

        self.coarse_resize = nn.Upsample(
            size=(img_resolution, img_resolution),
            mode="bilinear",
            align_corners=False,
        )
        self.pool = nn.MaxPool2d(2, 2)

        encoder: list[nn.Module] = []
        in_ch = img_channels * 2
        for out_ch in self.feature_channels[:-1]:
            encoder.append(DoubleConv(in_ch, out_ch, out_ch))
            in_ch = out_ch
        self.encoder = nn.ModuleList(encoder)

        self.bottleneck = DoubleConv(
            self.feature_channels[-2],
            self.feature_channels[-1],
            self.feature_channels[-1],
        )

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        decoder: list[nn.Module] = []
        decoder_ch = self.feature_channels[-1]
        for skip_ch in reversed(self.feature_channels[:-1]):
            concat_ch = skip_ch + decoder_ch
            mid_ch = concat_ch // 2
            decoder.append(DoubleConv(concat_ch, mid_ch, skip_ch))
            decoder_ch = skip_ch
        self.decoder = nn.ModuleList(decoder)

        self.to_rgb = nn.Conv2d(decoder_ch, img_channels, 1, bias=True)

    def forward(self, target: Tensor, coarse: Tensor) -> Tensor:
        x = torch.cat((self.coarse_resize(coarse), target), dim=1)

        skips: list[Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        for block, skip in zip(self.decoder, reversed(skips), strict=True):
            x = block(torch.cat((skip, self.up(x)), dim=1))

        return torch.tanh(self.to_rgb(x))


class Generator(nn.Module):
    def __init__(
        self,
        img_resolution: int = 512,
        img_channels: int = 3,
        id_dim: int = 512,
        coarse_resolution: int = 128,
        coarse_bottleneck_resolution: int = 32,
        coarse_base_ch: int = 32,
        coarse_max_ch: int = 256,
        num_style_blocks: int = 6,
        hq_bottleneck_resolution: int = 16,
        hq_base_ch: int = 8,
        hq_max_ch: int = 128,
        hq_channel_hold_level: int | None = 2,
        norm_eps: float = 1e-8,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.network_cfg = {
            "img_resolution": img_resolution,
            "img_channels": img_channels,
            "id_dim": id_dim,
            "coarse_resolution": coarse_resolution,
            "coarse_bottleneck_resolution": coarse_bottleneck_resolution,
            "coarse_base_ch": coarse_base_ch,
            "coarse_max_ch": coarse_max_ch,
            "num_style_blocks": num_style_blocks,
            "hq_bottleneck_resolution": hq_bottleneck_resolution,
            "hq_base_ch": hq_base_ch,
            "hq_max_ch": hq_max_ch,
            "hq_channel_hold_level": hq_channel_hold_level,
            "norm_eps": norm_eps,
            "leaky_relu_slope": leaky_relu_slope,
        }

        self.coarse = CoarseSwapCore(
            img_resolution=img_resolution,
            img_channels=img_channels,
            id_dim=id_dim,
            resolution=coarse_resolution,
            bottleneck_resolution=coarse_bottleneck_resolution,
            base_ch=coarse_base_ch,
            max_ch=coarse_max_ch,
            num_style_blocks=num_style_blocks,
            norm_eps=norm_eps,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.hq = HQRefiner(
            img_resolution=img_resolution,
            img_channels=img_channels,
            bottleneck_resolution=hq_bottleneck_resolution,
            base_ch=hq_base_ch,
            max_ch=hq_max_ch,
            channel_hold_level=hq_channel_hold_level,
        )

    def forward(self, target: Tensor, input_2: Tensor) -> Tensor:
        coarse = self.coarse(target, input_2)
        return self.hq(target, coarse)


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    network_cfg = {
        "img_resolution": 512,
        "img_channels": 3,
        "id_dim": 512,
        "coarse_resolution": 128,
        "coarse_bottleneck_resolution": 32,
        "coarse_base_ch": 32,
        "coarse_max_ch": 256,
        "num_style_blocks": 6,
        "hq_bottleneck_resolution": 16,
        "hq_base_ch": 8,
        "hq_max_ch": 128,
        "hq_channel_hold_level": 2,
        "norm_eps": 1e-8,
        "leaky_relu_slope": 0.2,
    }

    model = Generator(**network_cfg).to(device)
    model.eval()

    target = torch.randn(
        batch_size,
        network_cfg["img_channels"],
        network_cfg["img_resolution"],
        network_cfg["img_resolution"],
        device=device,
    )
    identity = torch.randn(batch_size, network_cfg["id_dim"], device=device)

    summary(
        model,
        input_data=(target, identity),
        depth=3,
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ),
        row_settings=("var_names",),
    )

    print("\nNetwork_Info:")
    for key, value in network_cfg.items():
        print(f"  {key:35}: {value}")

    print("\nDerived_Architecture:")
    print(f"  {'coarse_num_down':35}: {model.coarse.num_down}")
    print(f"  {'coarse_feature_channels':35}: {model.coarse.feature_channels}")
    print(f"  {'coarse_style_channels':35}: {model.coarse.style_channels}")
    print(f"  {'style_bank_dim':35}: {model.coarse.style_projector.output_dim}")
    print(f"  {'hq_num_down':35}: {model.hq.num_down}")
    print(f"  {'hq_feature_channels':35}: {model.hq.feature_channels}")

    flops = FlopCountAnalysis(model, (target, identity))
    print(f"\nModel FLOPs: {flops.total() / 1e9:.4f} GFLOPs")
