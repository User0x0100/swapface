from enum import Enum, auto
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class BlurPool(nn.Module):
    def __init__(self, ch: int, stride: int = 2, filter_size: int = 4):
        super().__init__()
        self.ch = ch
        self.stride = stride

        kernels = {
            1: [1],
            2: [1, 1],
            3: [1, 2, 1],
            4: [1, 3, 3, 1],
            5: [1, 4, 6, 4, 1],
        }
        kernel = torch.tensor(kernels[filter_size], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None].repeat(ch, 1, 1, 1)

        self.register_buffer("kernel", kernel)
        self.pad = (filter_size - 1) // 2

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        return F.conv2d(x, self.kernel, stride=self.stride, groups=self.ch)


class ResBlockMode(Enum):
    UPSAMPLE = auto()
    DOWNSAMPLE = auto()


class ResBlock(nn.Module):
    def __init__(self, mode: ResBlockMode, in_ch: int, out_ch: int, resample: bool):
        super().__init__()

        self.residual = nn.Sequential(nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1))

        if resample:
            match mode:
                case ResBlockMode.UPSAMPLE:
                    self.residual.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
                    self.shortcut.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

                case ResBlockMode.DOWNSAMPLE:
                    self.residual.append(nn.AvgPool2d(kernel_size=2))
                    self.shortcut.append(nn.AvgPool2d(kernel_size=2))
