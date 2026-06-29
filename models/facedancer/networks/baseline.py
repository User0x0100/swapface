import torch
from torch import Tensor
import torch.nn as nn


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
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=7, stride=1, padding=3),
            nn.SiLU(),
            nn.Conv2d(out_ch // 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )


class ToRGB(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_ch // 2, out_ch, kernel_size=7, stride=1, padding=3),
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

        nn.init.zeros_(self.shared_delta[-1].weight)
        nn.init.zeros_(self.shared_delta[-1].bias)

        for head in self.private_delta:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, id_feat: Tensor) -> tuple[Tensor, ...]:
        w = id_feat + self.shared_delta(id_feat)

        return tuple(w + head(w) for head in self.private_delta)


class LatentBlock(nn.Module):
    def __init__(self, channels: int, w_dim: int, num_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([IDInject(channels, w_dim) for _ in range(num_layers)])

    def forward(self, x: Tensor, w_space: tuple[Tensor, ...]) -> Tensor:

        assert len(w_space) == len(self.layers), f"w_p_all length {len(w_space)} != layers length {len(self.layers)}"

        for layer, w_p in zip(self.layers, w_space):
            x = layer(x, w_p)

        return x


class Generator(nn.Module):
    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        num_depth: int = 3,
        num_latent: int = 6,
        base_ch: int = 256,
        max_ch: int = 1024,
        id_dim: int = 512,
        skip: bool = True,
    ) -> None:
        super().__init__()

        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        assert max_ch >= base_ch, f"max_ch={max_ch} must be >= base_ch={base_ch}"

        self.w_space_map = WSpaceMap(id_dim, num_latent)

        features = [min(max_ch, base_ch * (2**i)) for i in range(num_depth + 1)]

        self.from_rgb = FromRGB(img_channels, base_ch)
        self.encoder = nn.Sequential(*[DownSample(features[i], features[i + 1]) for i in range(num_depth)])
        self.latent_space = LatentBlock(features[-1], id_dim, num_latent)
        self.decoder = nn.Sequential(*[UpSample(features[-(i + 1)], features[-(i + 2)]) for i in range(num_depth)])
        self.to_rgb = ToRGB(base_ch * 2 if skip else base_ch, img_channels)

    def forward(self, x: Tensor, id_feat: Tensor) -> Tensor:

        w_space = self.w_space_map(id_feat)

        skip = self.from_rgb(x)
        feat = self.encoder(skip)
        feat = self.latent_space(feat, w_space)
        feat = self.decoder(feat)

        if self.network_cfg["skip"]:
            feat = torch.cat([skip, feat], dim=1)

        feat = self.to_rgb(feat)

        return feat


if __name__ == "__main__":
    import torch
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    network_cfg = {
        "img_resolution": 512,
        "img_channels": 3,
        "num_depth": 5,
        "num_latent": 6,
        "base_ch": 16,
        "max_ch": 2048,
        "id_dim": 512,
        "skip": True,
    }

    model = Generator(**network_cfg).to(device)
    model.eval()

    x_target = torch.randn((batch_size, network_cfg["img_channels"], network_cfg["img_resolution"], network_cfg["img_resolution"]), device=device)
    id_feat = torch.randn((batch_size, network_cfg["id_dim"]), device=device)
    summary(model, input_data=(x_target, id_feat), depth=2, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), row_settings=("var_names",))

    print("NetWork_Info:")
    for k, v in network_cfg.items():
        print(f"  {k:25}: {v}")

    flops = FlopCountAnalysis(model, (x_target, id_feat))
    print(f"\n模型总FLOPs: {flops.total() / 1e9:.4f} GFLOPs")
