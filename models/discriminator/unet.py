"""BasicSR / Real-ESRGAN 的 UNetDiscriminatorSN（适配项目 WFM 接口）。

原始实现：https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/discriminator_arch.py
保留 conv0..conv9、三层下采样、双线性上采样、加法跳连与 SN 层分布。
"""

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm


class UNetDiscriminatorSN(nn.Module):
    """全分辨率像素判别器；可选返回四层 encoder 特征用于 WFM。"""

    def __init__(self, num_in_ch: int, num_feat: int = 64, skip_connection: bool = True) -> None:
        super().__init__()
        if num_in_ch <= 0 or num_feat <= 0:
            raise ValueError("UNetDiscriminatorSN 要求输入及基础通道数为正")
        self.skip_connection = skip_connection
        self.num_in_ch = num_in_ch
        # 与 BasicSR 完全一致：首层和输出层不使用 spectral_norm。
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(
        self,
        x: Tensor,
        split_minibatch_std: bool = False,
        return_feats: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        # 项目判别器统一接口；本网络没有 minibatch 统计层。
        del split_minibatch_std
        if x.ndim != 4 or x.shape[1] != self.num_in_ch or x.shape[2] % 8 or x.shape[3] % 8:
            raise ValueError(f"UNetDiscriminatorSN 输入须为 [B,{self.num_in_ch},H,W] 且 H/W 能被 8 整除，收到 {tuple(x.shape)}")

        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        x4 = F.leaky_relu(self.conv4(F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)), 0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2
        x5 = F.leaky_relu(self.conv5(F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)), 0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1
        x6 = F.leaky_relu(self.conv6(F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)), 0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0
        out = F.leaky_relu(self.conv7(x6), 0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), 0.2, inplace=True)
        logits = self.conv9(out)
        return (logits, [x0, x1, x2, x3]) if return_feats else logits
