"""U-Net discriminator for 128x128 CoarseSwapCore.

Based on the shared Encoder / Global Head / Spatial Decoder structure
of lucidrains/unet-stylegan2; the adversarial training policy is separate.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=2)
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = F.leaky_relu(self.conv0(x), 0.2)
        skip = F.leaky_relu(self.conv1(skip), 0.2)
        return (self.down(skip) + self.shortcut(x)) * self.scale, skip


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        y = F.leaky_relu(self.conv0(torch.cat((up, skip), dim=1)), 0.2)
        y = F.leaky_relu(self.conv1(y), 0.2)
        return (y + self.shortcut(up)) * self.scale


class UNetDiscriminator(nn.Module):
    """Shared encoder, image-level logit and full-resolution spatial logits.

    WFM uses four encoder-downsample features. Input is always [B,3,128,128].
    When return_spatial=False, the decoder is not run and the logit
    output is a single [B,1] tensor (with or without return_feats).
    """

    def __init__(
        self,
        img_resolution: int = 128,
        img_channels: int = 3,
        base_ch: int = 32,
        max_ch: int = 256,
    ) -> None:
        super().__init__()
        if img_resolution != 128:
            raise ValueError(f"UNetDiscriminator 仅支持 128x128，收到 {img_resolution}")
        if img_channels <= 0 or base_ch <= 0 or max_ch < base_ch:
            raise ValueError("UNetDiscriminator 要求通道数为正且 max_ch >= base_ch")
        self.network_cfg = {
            "img_resolution": img_resolution,
            "img_channels": img_channels,
            "base_ch": base_ch,
            "max_ch": max_ch,
        }
        channels = [min(base_ch * 2 ** level, max_ch) for level in range(5)]
        self.from_rgb = nn.Conv2d(img_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            _DownBlock(channels[i], channels[i + 1]) for i in range(4)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1),
        )
        self.global_head = nn.Linear(channels[-1], 1)
        self.up_blocks = nn.ModuleList(
            _UpBlock(channels[i + 1], channels[i + 1], channels[i])
            for i in reversed(range(4))
        )
        self.spatial_head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[0], 1, 1),
        )

    def forward(
        self,
        x: Tensor,
        split_minibatch_std: bool = False,
        return_feats: bool = False,
        return_spatial: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor | tuple[Tensor, Tensor], list[Tensor]]:
        del split_minibatch_std  # API compatibility; no batch-coupled normalization
        cfg = self.network_cfg
        expected = (cfg["img_channels"], cfg["img_resolution"], cfg["img_resolution"])
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"x 必须为 [B,{expected[0]},{expected[1]},{expected[2]}]，实际为 {tuple(x.shape)}")

        x = F.leaky_relu(self.from_rgb(x), 0.2)
        residuals: list[Tensor] = []
        features: list[Tensor] = []
        for block in self.down_blocks:
            x, skip = block(x)
            residuals.append(skip)
            if return_feats:
                features.append(x)
        x = (x + self.bottleneck(x)) * (1.0 / math.sqrt(2.0))
        global_logits = self.global_head(F.leaky_relu(x, 0.2).mean(dim=(2, 3)))
        if not return_spatial:
            return (global_logits, features) if return_feats else global_logits

        for block, skip in zip(self.up_blocks, reversed(residuals), strict=True):
            x = block(x, skip)
        spatial_logits = self.spatial_head(x)
        result = (global_logits, spatial_logits)
        return (result, features) if return_feats else result
