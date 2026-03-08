import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class EqualizedWeight(nn.Module):
    def __init__(self, shape: list[int]) -> None:
        super().__init__()

        self.c = 1.0 / math.sqrt(math.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualizedConv2d(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__()

        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class Smooth(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: Tensor) -> Tensor:

        B, C, H, W = x.shape
        x = x.view(-1, 1, H, W)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(B, C, H, W)


class DownSample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.smooth = Smooth()

    def forward(self, x: Tensor) -> Tensor:
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode="bilinear", align_corners=False)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        self.residual = nn.Sequential(DownSample(), EqualizedConv2d(in_ch, out_ch, kernel_size=1))

        self.block = nn.Sequential(
            EqualizedConv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.down_sample = DownSample()
        self.scale = 1.0 / math.sqrt(2)

    def forward(self, x: Tensor) -> Tensor:

        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4) -> Tensor:
        super().__init__()

        self.group_size = group_size

    def forward(self, x: Tensor) -> Tensor:

        grouped = x.view(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1)
        B, _, H, W = x.shape
        std = std.expand(B, -1, H, W)
        return torch.cat([x, std], dim=1)


class EqualizedLinear(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bias: float = 0.0) -> None:
        super().__init__()

        self.weight = EqualizedWeight([out_ch, in_ch])
        self.bias = nn.Parameter(torch.ones(out_ch) * bias)

    def forward(self, x: Tensor) -> None:

        return F.linear(x, self.weight(), bias=self.bias)


class Stylegan2DiscriminatorLite(nn.Module):
    def __init__(self, input_res: int, base_ch: int = 64, max_ch: int = 512, group_size: int = 5) -> None:
        super().__init__()

        self.network_cfg = {
            "input_res": input_res,
            "base_ch": base_ch,
            "max_ch": max_ch,
            "group_size": group_size,
        }

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, base_ch, 1),
            nn.LeakyReLU(0.2),
        )

        features = [min(max_ch, base_ch * (2**i)) for i in range(int(math.log2(input_res)) - 1)]

        n_blocks = len(features) - 1

        self.blocks = nn.ModuleList([DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)])

        self.std_dev = MiniBatchStdDev(group_size)

        final_features = features[-1] + 1

        self.conv = EqualizedConv2d(final_features, final_features, 3)

        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x: Tensor, return_feats: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:

        feats = [] if return_feats else None

        x = self.from_rgb(x)

        for block in self.blocks:
            x = block(x)
            if return_feats:
                feats.append(x)

        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.final(x)

        return (x, feats) if feats is not None else x
