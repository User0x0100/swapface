import torch
from torch import Tensor
import torch.nn as nn


class Linear2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        # [B, C, H, W] -> [B, H, W, C]
        x = x.movedim(1, -1)
        x = self.linear(x)
        # [B, H, W, C] -> [B, C, H, W]
        return x.movedim(-1, 1).contiguous()


class AdaINMLPRB(nn.Module):
    def __init__(self, channels: int, w_dim: int, hidden_ratio: float = 1.0) -> None:
        super().__init__()

        hidden_ch = max(16, int(channels * hidden_ratio))

        self.adain0 = AdaIN(channels, w_dim)
        self.act0 = nn.SiLU()
        self.fc0 = Linear2d(channels, hidden_ch)

        self.adain1 = AdaIN(hidden_ch, w_dim)
        self.act1 = nn.SiLU()
        self.fc1 = Linear2d(hidden_ch, channels)

        # 让 block 初始接近恒等映射，训练更稳
        nn.init.zeros_(self.fc1.linear.weight)
        nn.init.zeros_(self.fc1.linear.bias)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        y = self.adain0(x, w)
        y = self.act0(y)
        y = self.fc0(y)

        y = self.adain1(y, w)
        y = self.act1(y)
        y = self.fc1(y)

        return x + y


class AdaIN(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.affine = nn.Linear(w_dim, channels * 2)

        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        nn.init.ones_(self.affine.bias[:channels])

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        gamma, beta = self.affine(w)[:, :, None, None].chunk(2, dim=1)
        return self.norm(x) * gamma + beta


class AdainRB(nn.Module):
    def __init__(self, channels: int, w_dim: int, hidden_ratio: float = 0.5) -> None:
        super().__init__()

        hidden_ch = int(channels * hidden_ratio)

        self.adain0 = AdaIN(channels, w_dim)
        self.act0 = nn.SiLU()
        self.conv0 = nn.Conv2d(channels, hidden_ch, kernel_size=1, stride=1, padding=0)

        self.adain1 = AdaIN(hidden_ch, w_dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(hidden_ch, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        residual = self.adain0(x, w)
        residual = self.act0(residual)
        residual = self.conv0(residual)

        residual = self.adain1(residual, w)
        residual = self.act1(residual)
        residual = self.conv1(residual)

        return x + residual


# class LowRankConv3x3(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int, group_size: int = 32) -> None:
#         super().__init__()

#         assert out_ch % group_size == 0, f"out_ch={out_ch} must be divisible by group_size={group_size}"

#         groups = out_ch // group_size

#         self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

#         self.spatial = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups)

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.proj(x)
#         x = self.spatial(x)
#         return x


# class AdainRB(nn.Module):
#     def __init__(self, channels: int, w_dim: int, hidden_ratio: float = 0.5, group_size0: int = 32, group_size1: int = 32) -> None:
#         super().__init__()

#         hidden_ch = int(channels * hidden_ratio)

#         self.adain0 = AdaIN(channels, w_dim)
#         self.act0 = nn.SiLU()
#         self.conv0 = LowRankConv3x3(channels, hidden_ch, group_size=group_size0)

#         self.adain1 = AdaIN(hidden_ch, w_dim)
#         self.act1 = nn.SiLU()
#         self.conv1 = LowRankConv3x3(hidden_ch, channels, group_size=group_size1)

#     def forward(self, x: Tensor, w: Tensor) -> Tensor:
#         residual = self.adain0(x, w)
#         residual = self.act0(residual)
#         residual = self.conv0(residual)

#         residual = self.adain1(residual, w)
#         residual = self.act1(residual)
#         residual = self.conv1(residual)

#         return x + residual


class FromRGB(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=3),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
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


class WPMappings(nn.Module):
    def __init__(self, id_dim: int = 512, num: int = 4, num_share_layers: int = 4, num_w_p_layers: int = 2) -> None:
        super().__init__()

        self.shared_delta = nn.Sequential()
        for _ in range(num_share_layers - 1):
            self.shared_delta.append(nn.Linear(id_dim, id_dim))
            self.shared_delta.append(nn.SiLU())
        self.shared_delta.append(nn.Linear(id_dim, id_dim))

        self.private_delta = nn.Sequential(nn.SiLU(), nn.Linear(id_dim, id_dim * num))
        for _ in range(num_w_p_layers - 1):
            self.private_delta.append(nn.SiLU())
            self.private_delta.append(nn.Linear(id_dim * num, id_dim * num))

        self.num = num
        self.id_dim = id_dim

        nn.init.zeros_(self.shared_delta[-1].weight)
        nn.init.zeros_(self.shared_delta[-1].bias)

        nn.init.zeros_(self.private_delta[-1].weight)
        nn.init.zeros_(self.private_delta[-1].bias)

    def forward(self, id_feat: Tensor) -> tuple[Tensor]:
        w = id_feat + self.shared_delta(id_feat)

        delta = self.private_delta(w)
        delta = delta.view(id_feat.size(0), self.num, self.id_dim)

        w_p_all = w[:, None, :] + delta

        return w_p_all.unbind(dim=1)


class Generator(nn.Module):
    def __init__(
        self,
        img_resolution: int = 128,
        img_channels: int = 3,
        num_depth: int = 2,
        num_bottleneck: int = 6,
        base_ch: int = 64,
        max_ch: int = 512,
        id_dim: int = 512,
    ) -> None:
        super().__init__()

        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}

        self.w_p_mapping = WPMappings(id_dim, num_bottleneck)

        self.from_rgb = FromRGB(img_channels, base_ch)

        features = [min(max_ch, base_ch * (2**i)) for i in range(num_depth + 1)]
        features[-1] = max(features[-1], max_ch)

        self.encoder = nn.Sequential(*[DownSample(features[i], features[i + 1]) for i in range(num_depth)])

        self.bottleneck = nn.ModuleList([AdaINMLPRB(features[-1], id_dim) for _ in range(num_bottleneck)])

        self.decoder = nn.Sequential(*[UpSample(features[-(i + 1)], features[-(i + 2)]) for i in range(num_depth)])

        self.to_rgb = ToRGB(base_ch, img_channels)

    def forward(self, x: Tensor, id_feat: Tensor) -> Tensor:

        w_p_all = self.w_p_mapping(id_feat)

        x = self.from_rgb(x)
        x = self.encoder(x)

        for blk, w_p in zip(self.bottleneck, w_p_all):
            x = blk(x, w_p)

        x = self.decoder(x)
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
        "num_depth": 2,
        "num_bottleneck": 6,
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
