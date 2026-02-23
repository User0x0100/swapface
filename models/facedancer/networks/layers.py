from enum import Enum
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


class RBResampleMode(Enum):
    NONE = "none"
    UPSAMPLE = "upsample"
    DOWNSAMPLE = "downsample"


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, resample_mode: RBResampleMode):
        super().__init__()

        self.residual = nn.Sequential(nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1))

        match resample_mode:
            case RBResampleMode.UPSAMPLE:
                self.residual.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
                self.shortcut.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

            case RBResampleMode.DOWNSAMPLE:
                self.residual.append(nn.AvgPool2d(kernel_size=2))
                self.shortcut.append(nn.AvgPool2d(kernel_size=2))

            case RBResampleMode.NONE:
                pass
