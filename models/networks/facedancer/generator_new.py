import torch
from torch import Tensor
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False)

        self.fc_gamma = nn.Linear(w_dim, channels)
        self.fc_beta = nn.Linear(w_dim, channels)

        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        x = self.norm(x)

        gamma = self.fc_gamma(w).view(w.size(0), -1, 1, 1)
        beta = self.fc_beta(w).view(w.size(0), -1, 1, 1)

        return x * gamma + beta


class AdainRB(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.adain0 = AdaIN(channels, w_dim)
        self.act0 = nn.SiLU()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.adain1 = AdaIN(channels, w_dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        residual = self.adain0(x, w)
        residual = self.act0(residual)
        residual = self.conv0(residual)

        residual = self.adain1(residual, w)
        residual = self.act1(residual)
        residual = self.conv1(residual)

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


class DownSample(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )


class UpSample(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.inject = AdainRB(in_ch, w_dim)
        self.act = nn.SiLU()
        self.up = UpSample(in_ch, out_ch)

    def forward(self, x, w) -> Tensor:
        x = self.conv(x)
        x = self.inject(x, w)
        x = self.act(x)
        x = self.up(x)
        return x


class IDMapping(nn.Module):
    def __init__(self, id_dim: int = 512, num_layers: int = 4) -> None:
        super().__init__()

        self.delta = nn.Sequential()
        for _ in range(num_layers - 1):
            self.delta.append(nn.Linear(id_dim, id_dim))
            self.delta.append(nn.SiLU())
        self.delta.append(nn.Linear(id_dim, id_dim))

        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, id_feat: Tensor) -> Tensor:
        return id_feat + self.delta(id_feat)


class Generator(nn.Module):
    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        num_depth: int = 2,
        base_ch: int = 64,
        max_ch: int = 512,
        id_dim: int = 512,
    ) -> None:
        super().__init__()

        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}

        self.mapping = IDMapping(id_dim)

        self.from_rgb = FromRGB(img_channels, base_ch)

        features = [min(max_ch, base_ch * (2**i)) for i in range(num_depth + 1)]

        self.encoder = nn.Sequential(*[DownSample(features[i], features[i + 1]) for i in range(num_depth)])

        self.decoder = nn.ModuleList([DecoderBlock(features[-(i + 1)], features[-(i + 2)], id_dim) for i in range(num_depth)])

        self.to_rgb = ToRGB(base_ch, img_channels)

    def forward(self, x: Tensor, id_feat: Tensor) -> Tensor:

        w = self.mapping(id_feat)

        x = self.from_rgb(x)
        x = self.encoder(x)

        for decode_layer in self.decoder:
            x = decode_layer(x, w)

        x = self.to_rgb(x)

        return x


if __name__ == "__main__":
    import torch
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    network_cfg = {
        "img_resolution": 128,
        "img_channels": 3,
        "num_depth": 3,
        "base_ch": 64,
        "max_ch": 512,
        "id_dim": 512,
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
