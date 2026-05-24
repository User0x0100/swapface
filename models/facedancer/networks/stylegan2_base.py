import math
import torch
from torch import nn as nn, Tensor
import torch.nn.functional as F


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


class StyledConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, style_dim: int, kernel: int = 3, demod=True):
        super().__init__()

        self.conv = ModulatedConv2d(in_ch, out_ch, style_dim, kernel, demod)
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.activate = nn.SiLU()

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        out = self.conv(input, style) + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_ch: int, w_dim: int, upsample: bool = True):
        super().__init__()

        self.upsample = upsample
        self.conv = ModulatedConv2d(in_ch, 3, w_dim, demod=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input: Tensor, style: Tensor, skip: Tensor | None = None) -> Tensor:
        out = self.conv(input, style)
        out = out + self.bias

        if self.upsample and skip is not None:
            out = out + F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)

        return out


class IDInjec(nn.Module):
    def __init__(self, channels: int, style_dim: int, kernel: int = 3, demod=True):
        super().__init__()

        self.style_conv0 = StyledConv(channels, channels, style_dim, kernel, demod)
        self.style_conv1 = StyledConv(channels, channels, style_dim, kernel, demod)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        residual = self.style_conv0(x, style)
        residual = self.style_conv1(residual, style)

        return x + residual


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


class InputStem(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=3),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )


class FinalRGBHead(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_ch // 2, out_ch, kernel_size=7, stride=1, padding=3),
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


class Generator(nn.Module):
    def __init__(
        self,
        img_resolution: int = 128,
        img_channels: int = 3,
        num_depth: int = 2,
        base_ch: int = 64,
        max_ch: int = 512,
        id_dim: int = 512,
    ) -> None:
        super().__init__()

        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}

        features = [min(max_ch, base_ch * (2**i)) for i in range(num_depth + 1)]
        # features[-1] = max(features[-1], max_ch)

        self.w_p_mapping = WPMappings(id_dim, num_depth)

        self.input_stem = InputStem(img_channels, base_ch)

        self.encoder = nn.Sequential(*[DownSample(features[i], features[i + 1]) for i in range(num_depth)])

        self.idinject_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for i in range(num_depth):
            in_ch, out_ch = features[-(i + 1)], features[-(i + 2)]
            self.idinject_layers.append(IDInjec(in_ch, id_dim))
            self.upsample_layers.append(UpSample(in_ch, out_ch))

        self.final_rgb = FinalRGBHead(base_ch, img_channels)

    def forward(self, x: Tensor, id_feat: Tensor) -> Tensor:

        w_p_all = self.w_p_mapping(id_feat)

        x = self.input_stem(x)
        x = self.encoder(x)

        for idinject, upsample, w_p in zip(self.idinject_layers, self.upsample_layers, w_p_all):
            x = idinject(x, w_p)
            x = upsample(x)

        rgb = self.final_rgb(x)

        rgb = torch.tanh(rgb)

        return rgb


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
