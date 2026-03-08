from math import ceil
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class BlurPool(nn.Module):
    def __init__(self, channels: int, filt_size: int = 4, stride: int = 2, pad_off: int = 0) -> None:
        super().__init__()

        self.filt_size = filt_size
        self.pad_off = pad_off
        pad = (filt_size - 1) / 2
        self.pad_sizes = [int(pad), int(ceil(pad)), int(pad), int(ceil(pad))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.channels = channels

        a = {
            1: [1.0],
            2: [1.0, 1.0],
            3: [1.0, 2.0, 1.0],
            4: [1.0, 3.0, 3.0, 1.0],
            5: [1.0, 4.0, 6.0, 4.0, 1.0],
            6: [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
            7: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        }

        a = torch.tensor(a[filt_size], dtype=torch.float)
        filt = a[:, None] * a[None, :]
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = nn.ReflectionPad2d(self.pad_sizes)

    def forward(self, x: Tensor) -> Tensor:
        if self.filt_size == 1:
            if self.pad_off == 0:
                return x[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(x)[:, :, :: self.stride, :: self.stride]
        else:
            return F.conv2d(self.pad(x), self.filt, stride=self.stride, groups=x.shape[1])


class LearnableLP(nn.Module):
    """可学习低通滤波器，深度可分离，保持空间尺寸"""

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()

        self.dw_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )

        nn.init.constant_(self.dw_conv.weight, 1.0 / kernel_size**2)

    def forward(self, x: Tensor) -> Tensor:
        return self.dw_conv(x)


class SoftNorm(nn.Module):
    def __init__(self, channels: int, init_scale: float = 0.1):
        super().__init__()

        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), init_scale))

    def forward(self, x: Tensor) -> Tensor:
        std = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        return (x / std) * self.scale


class SpatialGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        nn.init.constant_(self.proj[-2].bias, 2.0)

    def forward(self, x_target: Tensor, x_source: Tensor) -> Tensor:
        diff = (x_target - x_source).abs()
        return self.proj(diff)


class FreqResidualBlend(nn.Module):
    def __init__(self, channels: int, w_dim: int, blur_kernel_size: int = 5, init_scale: float = 0.1):
        super().__init__()

        self.blur = LearnableLP(channels, blur_kernel_size)

        self.id_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=4),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

        self.soft_norm = SoftNorm(channels, init_scale)
        self.spatial_gate = SpatialGate(channels)
        self.w_gate = nn.Sequential(
            nn.Linear(w_dim, channels),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.w_gate[0].weight)
        nn.init.constant_(self.w_gate[0].bias, 1.0)

        self.last_attn_mask = None

    def forward(self, x_target: Tensor, x_source: Tensor, w: Tensor):

        low_target = self.blur(x_target)
        low_source = self.blur(x_source)
        high_source = x_source - low_source

        delta = self.id_encoder(high_source) + high_source

        delta = self.soft_norm(delta)

        s_gate = self.spatial_gate(x_target, x_source)  # (B, 1, H, W)
        self.last_attn_mask = s_gate.detach()

        w_gate = self.w_gate(w).view(-1, delta.shape[1], 1, 1)  # (B, C, 1, 1)

        return low_target + s_gate * w_gate * delta
