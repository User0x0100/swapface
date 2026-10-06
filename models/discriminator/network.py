import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .upfirdn2d import DownFIRDn2d

INSTANCE_NORM_RESIDUAL_MIN_RESOLUTION = 4


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


class FIRDownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.shortcut_filter = DownFIRDn2d()
        self.shortcut_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.conv0 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.residual_filter = DownFIRDn2d()
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut_conv(self.shortcut_filter(x))

        residual = F.leaky_relu(self.conv0(x), negative_slope=0.2)
        residual = F.leaky_relu(self.conv1(residual), negative_slope=0.2)
        residual = self.residual_filter(residual)
        return (shortcut + residual) * self.scale


class FIRMinibatchStdDiscriminator(nn.Module):
    """FIR 抗混叠下采样 + MinibatchStd 的全局判别器。"""

    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        base_ch: int = 64,
        max_ch: int = 512,
        minibatch_std_group_size: int = 4,
    ) -> None:
        super().__init__()

        if img_resolution < 16 or img_resolution & (img_resolution - 1):
            raise ValueError(f"img_resolution 必须是 >=16 的 2 的整数次幂，实际为 {img_resolution}")
        if base_ch <= 0 or max_ch < base_ch:
            raise ValueError(f"无效通道配置：base_ch={base_ch}, max_ch={max_ch}")
        if minibatch_std_group_size <= 0:
            raise ValueError("minibatch_std_group_size 必须为正数")

        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}

        num_down = int(math.log2(img_resolution)) - 2
        channels = [min(max_ch, base_ch * (2**level)) for level in range(num_down + 1)]

        self.from_rgb = nn.Conv2d(img_channels, channels[0], kernel_size=1)
        self.down_blocks = nn.ModuleList(
            [FIRDownBlock(channels[index], channels[index + 1]) for index in range(num_down)]
        )
        self.minibatch_std = MinibatchStdLayer(minibatch_std_group_size)
        final_ch = channels[-1] + 1
        self.final_conv = nn.Conv2d(final_ch, final_ch, kernel_size=3, padding=1)
        self.final_fc0 = nn.Linear(4 * 4 * final_ch, final_ch)
        self.final_fc1 = nn.Linear(final_ch, 1)

    def _encode(self, x: Tensor, return_feats: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        x = F.leaky_relu(self.from_rgb(x), negative_slope=0.2)
        feats = [] if return_feats else None
        for block in self.down_blocks:
            x = block(x)
            if feats is not None:
                feats.append(x)
        return (x, feats) if feats is not None else x

    def _score(self, x: Tensor, split_minibatch_std: bool = False) -> Tensor:
        if split_minibatch_std:
            if x.shape[0] % 2:
                raise ValueError(f"split_minibatch_std 要求偶数 batch，实际为 {x.shape[0]}")
            fake, real = x.chunk(2, dim=0)
            x = torch.cat((self.minibatch_std(fake), self.minibatch_std(real)), dim=0)
        else:
            x = self.minibatch_std(x)

        x = F.leaky_relu(self.final_conv(x), negative_slope=0.2)
        x = x.flatten(1)
        x = F.leaky_relu(self.final_fc0(x), negative_slope=0.2)
        return self.final_fc1(x)

    def forward(
        self,
        x: Tensor,
        split_minibatch_std: bool = False,
        return_feats: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        expected = (
            self.network_cfg["img_channels"],
            self.network_cfg["img_resolution"],
            self.network_cfg["img_resolution"],
        )
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"x 必须为 [B,{expected[0]},{expected[1]},{expected[2]}]，实际为 {tuple(x.shape)}"
            )

        encoded = self._encode(x, return_feats=return_feats)
        if return_feats:
            assert isinstance(encoded, tuple)
            feature, feats = encoded
            return self._score(feature, split_minibatch_std), feats
        assert isinstance(encoded, Tensor)
        return self._score(encoded, split_minibatch_std)


class InstanceNormResidualDownBlock(nn.Module):
    """2026-06 旧全局 D 的 InstanceNorm + AvgPool residual down block。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.AvgPool2d(2),
        )
        self.residual = nn.Sequential(
            nn.InstanceNorm2d(in_ch, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.AvgPool2d(2),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class InstanceNormResidualDiscriminator(nn.Module):
    """2026-06 旧全局 D：InstanceNorm residual encoder + AvgPool，下采样固定到 4x4。"""

    MIN_RESOLUTION = INSTANCE_NORM_RESIDUAL_MIN_RESOLUTION

    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        base_ch: int = 64,
        max_ch: int = 512,
    ) -> None:
        super().__init__()

        if img_resolution < 16 or img_resolution & (img_resolution - 1):
            raise ValueError(f"img_resolution 必须是 >=16 的 2 的整数次幂，实际为 {img_resolution}")
        if img_resolution < self.MIN_RESOLUTION or img_resolution % self.MIN_RESOLUTION:
            raise ValueError(f"img_resolution={img_resolution} 必须能整除固定最小分辨率 {self.MIN_RESOLUTION}")
        ratio = img_resolution // self.MIN_RESOLUTION
        if ratio & (ratio - 1):
            raise ValueError(f"img_resolution/{self.MIN_RESOLUTION} 必须为 2 的整数次幂")
        num_down = int(math.log2(ratio))
        if num_down <= 0:
            raise ValueError(f"img_resolution 必须大于固定最小分辨率 {self.MIN_RESOLUTION}")
        if base_ch <= 0 or max_ch < base_ch:
            raise ValueError(f"无效通道配置：base_ch={base_ch}, max_ch={max_ch}")

        self.network_cfg = {
            "img_resolution": img_resolution,
            "img_channels": img_channels,
            "base_ch": base_ch,
            "max_ch": max_ch,
        }
        self.from_rgb = nn.Conv2d(img_channels, base_ch, kernel_size=3, padding=1)
        channels = [min(max_ch, base_ch * (2**level)) for level in range(num_down + 1)]
        self.down_blocks = nn.ModuleList(
            [InstanceNormResidualDownBlock(channels[index], channels[index + 1]) for index in range(num_down)]
        )

        final_ch = channels[-1]
        final_resolution = self.MIN_RESOLUTION
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, kernel_size=final_resolution),
            nn.LeakyReLU(0.2),
            nn.Conv2d(final_ch, 1, kernel_size=1),
            nn.Flatten(),
        )

    def forward(
        self,
        x: Tensor,
        split_minibatch_std: bool = False,
        return_feats: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        del split_minibatch_std
        expected = (
            self.network_cfg["img_channels"],
            self.network_cfg["img_resolution"],
            self.network_cfg["img_resolution"],
        )
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"x 必须为 [B,{expected[0]},{expected[1]},{expected[2]}]，实际为 {tuple(x.shape)}"
            )

        x = self.from_rgb(x)
        feats = [] if return_feats else None
        for block in self.down_blocks:
            x = block(x)
            if feats is not None:
                feats.append(x)
        score = self.final_conv(x)
        return (score, feats) if feats is not None else score
