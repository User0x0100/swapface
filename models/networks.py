import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.gamma_fc = nn.Linear(w_dim, channels)
        self.beta_fc = nn.Linear(w_dim, channels)

        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.ones_(self.gamma_fc.bias)

        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        gamma = self.gamma_fc(w)[:, :, None, None]
        beta = self.beta_fc(w)[:, :, None, None]

        return self.norm(x) * gamma + beta


class IDInject(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.act = nn.SiLU()

        self.adain0 = AdaIN(channels, w_dim)
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.adain1 = AdaIN(channels, w_dim)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        residual = self.conv0(x)
        residual = self.adain0(residual, w)
        residual = self.act(residual)

        residual = self.conv1(residual)
        residual = self.adain1(residual, w)
        residual = self.act(residual)

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
            nn.Tanh(),
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


class WSpaceMap(nn.Module):
    def __init__(self, id_dim: int, num: int, num_share_layers: int = 4, num_w_p_layers: int = 2) -> None:
        super().__init__()

        self.shared_delta = nn.Sequential()
        for _ in range(num_share_layers - 1):
            self.shared_delta.append(nn.Linear(id_dim, id_dim))
            self.shared_delta.append(nn.SiLU())
        self.shared_delta.append(nn.Linear(id_dim, id_dim))

        self.private_delta = nn.ModuleList()
        for _ in range(num):
            layers = nn.Sequential()
            for _ in range(num_w_p_layers):
                layers.append(nn.SiLU())
                layers.append(nn.Linear(id_dim, id_dim))
            self.private_delta.append(layers)

        self.num = num
        self.id_dim = id_dim

        shared_output = self.shared_delta[-1]
        assert isinstance(shared_output, nn.Linear)
        nn.init.zeros_(shared_output.weight)
        nn.init.zeros_(shared_output.bias)

        for head in self.private_delta:
            assert isinstance(head, nn.Sequential)
            private_output = head[-1]
            assert isinstance(private_output, nn.Linear)
            nn.init.zeros_(private_output.weight)
            nn.init.zeros_(private_output.bias)

    def forward(self, id_feat: Tensor) -> tuple[Tensor, ...]:
        w = id_feat + self.shared_delta(id_feat)
        outputs: list[Tensor] = []
        for head in self.private_delta:
            outputs.append(w + head(w))
        return tuple(outputs)


class LatentBlock(nn.Module):
    def __init__(self, channels: int, w_dim: int, num_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([IDInject(channels, w_dim) for _ in range(num_layers)])

    def forward(self, x: Tensor, w_space: tuple[Tensor, ...]) -> Tensor:

        assert len(w_space) == len(self.layers), f"w_p_all length {len(w_space)} != layers length {len(self.layers)}"

        for layer_index, layer in enumerate(self.layers):
            x = layer(x, w_space[layer_index])

        return x


class Coarse(nn.Module):
    def __init__(
        self,
        img_resolution: int = 512,
        img_channels: int = 3,
        latent_resolution: int = 32,
        num_latent: int = 6,
        base_ch: int = 64,
        max_ch: int = 512,
        id_dim: int = 512,
    ) -> None:
        super().__init__()

        assert max_ch >= base_ch, f"max_ch={max_ch} must be >= base_ch={base_ch}"
        if latent_resolution <= 0:
            raise ValueError(f"latent_resolution must be greater than 0, got {latent_resolution}")

        if latent_resolution > img_resolution:
            raise ValueError(f"latent_resolution={latent_resolution} must be <= img_resolution={img_resolution}")

        if img_resolution % latent_resolution != 0:
            raise ValueError(f"img_resolution={img_resolution} must be divisible by latent_resolution={latent_resolution}")

        scale = img_resolution // latent_resolution
        if scale & (scale - 1):
            raise ValueError(f"img_resolution / latent_resolution must be a power of 2, got {img_resolution} / {latent_resolution} = {scale}")

        self.img_resolution = img_resolution

        num_depth = int(math.log2(scale))

        self.w_space_map = WSpaceMap(id_dim, num_latent)
        features = [min(max_ch, base_ch * (2**i)) for i in range(num_depth + 1)]

        self.from_rgb = FromRGB(img_channels, base_ch)
        self.encoder = nn.Sequential(*[DownSample(features[i], features[i + 1]) for i in range(num_depth)])
        self.latent_space = LatentBlock(features[-1], id_dim, num_latent)
        self.decoder = nn.Sequential(*[UpSample(features[-(i + 1)], features[-(i + 2)]) for i in range(num_depth)])
        self.to_rgb = ToRGB(base_ch, img_channels)

    def forward(self, x: Tensor, id_feat: Tensor) -> Tensor:
        w_space = self.w_space_map(id_feat)

        resize_in = F.interpolate(x, size=self.img_resolution, mode="bilinear", align_corners=False)

        feat = self.from_rgb(resize_in)
        encoder_feat = self.encoder(feat)
        latent_feat = self.latent_space(encoder_feat, w_space)
        decoder_feat = self.decoder(latent_feat)

        return self.to_rgb(decoder_feat)


class HQRefiner(nn.Module):
    def __init__(
        self,
        img_resolution: int = 512,
        img_channels: int = 3,
        coarse_resolution: int = 128,
        bottleneck_resolution: int = 16,
        base_ch: int = 8,
        max_ch: int = 128,
    ) -> None:
        super().__init__()

        if coarse_resolution <= 0 or coarse_resolution > img_resolution:
            raise ValueError(f"coarse_resolution must be in (0, {img_resolution}], got {coarse_resolution}")
        if img_resolution % bottleneck_resolution != 0:
            raise ValueError(f"img_resolution={img_resolution} must be divisible by bottleneck_resolution={bottleneck_resolution}")

        scale = img_resolution // bottleneck_resolution
        if scale <= 1 or scale & (scale - 1):
            raise ValueError(f"img_resolution / bottleneck_resolution must be a power of 2 greater than 1, got {img_resolution} / {bottleneck_resolution} = {scale}")
        if base_ch <= 0 or max_ch < base_ch:
            raise ValueError(f"invalid HQ channels: base_ch={base_ch}, max_ch={max_ch}")

        num_down = int(math.log2(scale))
        num_levels = num_down + 1

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.coarse_resolution = coarse_resolution
        self.bottleneck_resolution = bottleneck_resolution

        self.feature_channels = tuple(min(max_ch, base_ch * (2**level)) for level in range(num_levels))

        self.pool = nn.MaxPool2d(2, 2)

        encoder: list[nn.Module] = []
        in_ch = img_channels * 2
        for out_ch in self.feature_channels[:-1]:
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.SiLU(),
                )
            )
            in_ch = out_ch
        self.encoder = nn.ModuleList(encoder)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.feature_channels[-2], self.feature_channels[-1], kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.feature_channels[-1], self.feature_channels[-1], kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        decoder: list[nn.Module] = []
        decoder_ch = self.feature_channels[-1]
        for skip_ch in reversed(self.feature_channels[:-1]):
            concat_ch = skip_ch + decoder_ch
            mid_ch = concat_ch // 2
            decoder.append(
                nn.Sequential(
                    nn.Conv2d(concat_ch, mid_ch, kernel_size=3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(mid_ch, skip_ch, kernel_size=3, stride=1, padding=1),
                    nn.SiLU(),
                )
            )
            decoder_ch = skip_ch
        self.decoder = nn.ModuleList(decoder)
        self.to_rgb = nn.Conv2d(decoder_ch, img_channels, kernel_size=1, bias=True)

    def forward(self, target: Tensor, coarse: Tensor) -> Tensor:
        expected_target = (self.img_channels, self.img_resolution, self.img_resolution)
        expected_coarse = (self.img_channels, self.coarse_resolution, self.coarse_resolution)
        if target.ndim != 4 or target.shape[1:] != expected_target:
            raise ValueError(f"target must be [B,{self.img_channels},{self.img_resolution},{self.img_resolution}], got {tuple(target.shape)}")
        if coarse.ndim != 4 or coarse.shape[0] != target.shape[0] or coarse.shape[1:] != expected_coarse:
            raise ValueError(f"coarse must be [B,{self.img_channels},{self.coarse_resolution},{self.coarse_resolution}], got {tuple(coarse.shape)}")

        resize_coarse = F.interpolate(coarse, size=self.img_resolution, mode="bilinear", align_corners=False)
        x = torch.cat((resize_coarse, target), dim=1)

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
        coarse_latent_resolution: int = 32,
        coarse_num_latent: int = 8,
        coarse_base_ch: int = 64,
        coarse_max_ch: int = 512,
        hq_bottleneck_resolution: int = 16,
        hq_base_ch: int = 8,
        hq_max_ch: int = 128,
    ) -> None:
        super().__init__()

        self.network_cfg = {
            "img_resolution": img_resolution,
            "img_channels": img_channels,
            "id_dim": id_dim,
            "coarse_resolution": coarse_resolution,
            "coarse_latent_resolution": coarse_latent_resolution,
            "coarse_num_latent": coarse_num_latent,
            "coarse_base_ch": coarse_base_ch,
            "coarse_max_ch": coarse_max_ch,
            "hq_bottleneck_resolution": hq_bottleneck_resolution,
            "hq_base_ch": hq_base_ch,
            "hq_max_ch": hq_max_ch,
        }
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.id_dim = id_dim
        self.coarse_resolution = coarse_resolution

        self.coarse = Coarse(
            img_resolution=coarse_resolution,
            img_channels=img_channels,
            latent_resolution=coarse_latent_resolution,
            num_latent=coarse_num_latent,
            base_ch=coarse_base_ch,
            max_ch=coarse_max_ch,
            id_dim=id_dim,
        )
        self.hq = HQRefiner(
            img_resolution=img_resolution,
            img_channels=img_channels,
            coarse_resolution=coarse_resolution,
            bottleneck_resolution=hq_bottleneck_resolution,
            base_ch=hq_base_ch,
            max_ch=hq_max_ch,
        )

    def forward(self, target: Tensor, id_feat: Tensor, return_coarse: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        expected_target = (self.img_channels, self.img_resolution, self.img_resolution)
        if target.ndim != 4 or target.shape[1:] != expected_target:
            raise ValueError(f"target must be [B,{self.img_channels},{self.img_resolution},{self.img_resolution}], got {tuple(target.shape)}")
        if id_feat.ndim != 2 or id_feat.shape != (target.shape[0], self.id_dim):
            raise ValueError(f"id_feat must be [B,{self.id_dim}], got {tuple(id_feat.shape)}")

        coarse = self.coarse(target, id_feat)
        output = self.hq(target, coarse)
        if return_coarse:
            return output, coarse
        return output


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    network_cfg = {
        "img_resolution": 256,
        "img_channels": 3,
        "id_dim": 512,
        "coarse_resolution": 128,
        "coarse_latent_resolution": 32,
        "coarse_num_latent": 8,
        "coarse_base_ch": 64,
        "coarse_max_ch": 512,
        "hq_bottleneck_resolution": 16,
        "hq_base_ch": 8,
        "hq_max_ch": 128,
    }

    model = Generator(**network_cfg).to(device).eval()
    target = torch.randn((batch_size, 3, network_cfg["img_resolution"], network_cfg["img_resolution"]), device=device)
    id_feat = torch.randn((batch_size, network_cfg["id_dim"]), device=device)

    summary(
        model,
        input_data=(target, id_feat),
        depth=3,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        row_settings=("var_names",),
    )

    output, coarse = model(target, id_feat, return_coarse=True)
    print(f"output: {tuple(output.shape)}")
    print(f"coarse: {tuple(coarse.shape)}")
    print("NetWork_Info:")
    for k, v in network_cfg.items():
        print(f"  {k:30}: {v}")

    flops = FlopCountAnalysis(model, (target, id_feat))
    print(f"\n模型总FLOPs: {flops.total() / 1e9:.4f} GFLOPs")
