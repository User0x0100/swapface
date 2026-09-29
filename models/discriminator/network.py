import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from .upfirdn2d import DownFIRDn2d


class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size: int = 4, num_channels: int = 1) -> None:
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        if channels % self.num_channels:
            raise ValueError(f"channels={channels} 必须能被 num_channels={self.num_channels} 整除")

        group = min(self.group_size, batch)
        while batch % group:
            group -= 1

        channels_per_group = channels // self.num_channels
        y = x.reshape(group, -1, self.num_channels, channels_per_group, height, width)
        y = y - y.mean(dim=0)
        y = (y.square().mean(dim=0) + 1e-8).sqrt()
        y = y.mean(dim=(2, 3, 4))
        y = y.reshape(-1, self.num_channels, 1, 1)
        y = y.repeat(group, 1, height, width)
        return torch.cat((x, y), dim=1)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        self.shortcut_filter = DownFIRDn2d()
        self.shortcut_conv = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))

        self.conv0 = spectral_norm(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1))
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.residual_filter = DownFIRDn2d()

        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut_conv(self.shortcut_filter(x))

        residual = F.leaky_relu(self.conv0(x), negative_slope=0.2)
        residual = F.leaky_relu(self.conv1(residual), negative_slope=0.2)
        residual = self.residual_filter(residual)

        return (shortcut + residual) * self.scale


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = F.leaky_relu(self.conv(x), negative_slope=0.2)
        return (x + skip) * self.scale


class Discriminator(nn.Module):
    """共享深层 Encoder 的 Global + Dense 双头判别器。

    Encoder 使用 SpectralNorm residual blocks 与 FIR 抗混叠下采样，到
    bottleneck_resolution 后分叉：
    - Global head 继续下采样到 4x4，输出每张图一个 logit；
    - Dense head 通过 U-Net skip 恢复到输入分辨率，输出逐位置 realism logits。
    """

    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        base_ch: int = 64,
        max_ch: int = 512,
        bottleneck_resolution: int = 8,
        minibatch_std_group_size: int = 4,
    ) -> None:
        super().__init__()

        if img_resolution < 16 or img_resolution & (img_resolution - 1):
            raise ValueError(f"img_resolution 必须是 >=16 的 2 的整数次幂，实际为 {img_resolution}")
        if bottleneck_resolution != 8:
            raise ValueError(f"当前 Global head 要求 bottleneck_resolution=8，实际为 {bottleneck_resolution}")
        if bottleneck_resolution >= img_resolution or img_resolution % bottleneck_resolution:
            raise ValueError(
                f"bottleneck_resolution={bottleneck_resolution} 必须小于且整除 img_resolution={img_resolution}"
            )
        if base_ch <= 0 or max_ch < base_ch:
            raise ValueError(f"无效通道配置：base_ch={base_ch}, max_ch={max_ch}")
        if minibatch_std_group_size <= 0:
            raise ValueError("minibatch_std_group_size 必须为正数")

        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}

        num_down = int(math.log2(img_resolution // bottleneck_resolution))
        channels = [min(max_ch, base_ch * (2**level)) for level in range(num_down + 1)]

        self.from_rgb = spectral_norm(nn.Conv2d(img_channels, channels[0], kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(
            [DownBlock(channels[index], channels[index + 1]) for index in range(num_down)]
        )
        self.feature_count = len(channels)

        self.global_down = DownBlock(channels[-1], channels[-1])
        self.global_std = MinibatchStdLayer(minibatch_std_group_size)
        global_ch = channels[-1] + 1
        self.global_conv = spectral_norm(
            nn.Conv2d(global_ch, global_ch, kernel_size=3, padding=1)
        )
        self.global_fc0 = spectral_norm(nn.Linear(4 * 4 * global_ch, global_ch))
        self.global_fc1 = spectral_norm(nn.Linear(global_ch, 1))

        self.up_blocks = nn.ModuleList(
            [UpBlock(channels[index], channels[index - 1]) for index in range(num_down, 0, -1)]
        )
        self.dense_conv0 = spectral_norm(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False)
        )
        self.dense_conv1 = spectral_norm(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False)
        )
        self.dense_out = spectral_norm(nn.Conv2d(channels[0], 1, kernel_size=3, padding=1))

    def _encode(self, x: Tensor, max_layer: int | None = None) -> list[Tensor]:
        if max_layer is not None and not 0 <= max_layer < self.feature_count:
            raise ValueError(f"max_layer 必须在 [0, {self.feature_count - 1}]，实际为 {max_layer}")

        feats = [F.leaky_relu(self.from_rgb(x), negative_slope=0.2)]
        if max_layer == 0:
            return feats

        for layer_index, block in enumerate(self.down_blocks, start=1):
            feats.append(block(feats[-1]))
            if max_layer == layer_index:
                break
        return feats

    def get_feats(self, x: Tensor, max_layer: int | None = None) -> list[Tensor]:
        return self._encode(x, max_layer)

    def _global_head(self, bottleneck: Tensor, split_minibatch_std: bool = False) -> Tensor:
        x = self.global_down(bottleneck)
        if split_minibatch_std:
            if x.shape[0] % 2:
                raise ValueError(f"split_minibatch_std 要求偶数 batch，实际为 {x.shape[0]}")
            first, second = x.chunk(2, dim=0)
            x = torch.cat((self.global_std(first), self.global_std(second)), dim=0)
        else:
            x = self.global_std(x)
        x = F.leaky_relu(self.global_conv(x), negative_slope=0.2)
        x = x.flatten(1)
        x = F.leaky_relu(self.global_fc0(x), negative_slope=0.2)
        return self.global_fc1(x)

    def _dense_head(self, feats: list[Tensor]) -> Tensor:
        x = feats[-1]
        for block, skip in zip(self.up_blocks, reversed(feats[:-1]), strict=True):
            x = block(x, skip)
        x = F.leaky_relu(self.dense_conv0(x), negative_slope=0.2)
        x = F.leaky_relu(self.dense_conv1(x), negative_slope=0.2)
        return self.dense_out(x)

    def forward(
        self,
        x: Tensor,
        return_feats: bool = False,
        split_minibatch_std: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[tuple[Tensor, Tensor], list[Tensor]]:
        expected = (
            self.network_cfg["img_channels"],
            self.network_cfg["img_resolution"],
            self.network_cfg["img_resolution"],
        )
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"x 必须为 [B,{expected[0]},{expected[1]},{expected[2]}]，实际为 {tuple(x.shape)}"
            )

        feats = self._encode(x)
        scores = (self._global_head(feats[-1], split_minibatch_std), self._dense_head(feats))
        return (scores, feats) if return_feats else scores
