import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


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


class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, style_dim: int, kernel: int = 3, demod: bool = True) -> None:
        super().__init__()

        assert kernel % 2 == 1, "kernel must be odd to preserve spatial size"

        self.scale = 1.0 / math.sqrt(in_ch * kernel * kernel)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.demod = demod

        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel))
        self.style = nn.Linear(style_dim, in_ch)

        nn.init.zeros_(self.style.weight)
        nn.init.ones_(self.style.bias)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        B, C, H, W = x.shape

        assert C == self.in_ch, f"expected input channels {self.in_ch}, got {C}"
        assert w.shape[0] == B, f"batch mismatch: x batch {B}, style batch {w.shape[0]}"

        style = self.style(w).view(B, 1, C, 1, 1)

        weight = self.weight * self.scale
        weight = weight * style

        if self.demod:
            d = torch.rsqrt(weight.float().pow(2).sum((2, 3, 4)) + 1e-8)
            weight = weight * d.to(weight.dtype).view(B, self.out_ch, 1, 1, 1)

        x = x.reshape(1, B * C, H, W)
        weight = weight.reshape(B * self.out_ch, self.in_ch, self.kernel, self.kernel)

        return F.conv2d(x, weight, padding=self.kernel // 2, groups=B).reshape(B, self.out_ch, H, W)


# class IDInject(nn.Module):
#     def __init__(self, channels: int, w_dim: int) -> None:
#         super().__init__()

#         self.adain0 = AdaIN(channels, w_dim)
#         self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

#         self.adain1 = AdaIN(channels, w_dim)
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, x: Tensor, w: Tensor) -> Tensor:

#         residual = self.adain0(x, w)
#         residual = F.silu(residual)
#         residual = self.conv0(residual)

#         residual = self.adain1(residual, w)
#         residual = F.silu(residual)
#         residual = self.conv1(residual)

#         return x + residual


class IDInject(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.modconv0 = ModulatedConv2d(channels, channels, w_dim)
        self.bias0 = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.modconv1 = ModulatedConv2d(channels, channels, w_dim)
        self.bias1 = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        residual = self.modconv0(x, w) + self.bias0
        residual = F.silu(residual)

        residual = self.modconv1(residual, w) + self.bias1
        residual = F.silu(residual)

        return x + residual


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


class BottleneckLayer(nn.Module):
    def __init__(self, channels: int, w_dim: int, num_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([IDInject(channels, w_dim) for _ in range(num_layers)])

    def forward(self, x: Tensor, w_p_all: tuple[Tensor, ...]) -> Tensor:

        assert len(w_p_all) == len(self.layers), f"w_p_all length {len(w_p_all)} != layers length {len(self.layers)}"

        for layer, w_p in zip(self.layers, w_p_all):
            x = layer(x, w_p)

        return x


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
        self.num_bottleneck = num_bottleneck
        self.w_p_mapping = WPMappings(id_dim, num_bottleneck + num_depth)

        self.from_rgb = FromRGB(img_channels, base_ch)

        features = [min(max_ch, base_ch * (2**i)) for i in range(num_depth + 1)]

        self.encoder = nn.ModuleList([DownSample(features[i], features[i + 1]) for i in range(num_depth)])

        self.bottleneck = BottleneckLayer(features[-1], id_dim, num_bottleneck)

        self.decoder_inject = nn.ModuleList([IDInject(features[-(i + 1)], id_dim) for i in range(num_depth)])
        self.decoder = nn.ModuleList([UpSample(features[-(i + 1)], features[-(i + 2)]) for i in range(num_depth)])

        self.to_rgb = ToRGB(base_ch, img_channels)

    def forward(self, x: Tensor, id_feat: Tensor) -> Tensor:

        w_p_all = self.w_p_mapping(id_feat)

        x = self.from_rgb(x)

        encode_feats = []
        for encode_layer in self.encoder:
            x = encode_layer(x)
            encode_feats.append(x)

        # x = self.encoder(x)

        x = self.bottleneck(x, w_p_all[0 : self.num_bottleneck])

        for decode_layer, encode_inject, w_p, encode_feat in zip(self.decoder, self.decoder_inject, w_p_all[self.num_bottleneck :], reversed(encode_feats)):
            x = x + encode_inject(encode_feat, w_p)
            x = decode_layer(x)

        # x = self.decoder(x)
        x = self.to_rgb(x)

        return x


if __name__ == "__main__":
    import torch
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    network_cfg = {
        "img_resolution": 256,
        "img_channels": 3,
        "num_depth": 4,
        "num_bottleneck": 6,
        "base_ch": 32,
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
