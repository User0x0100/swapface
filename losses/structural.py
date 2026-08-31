import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .functional import EPS, Reduction, _reduce_loss


class DSSIMLoss(nn.Module):
    """基于局部 SSIM 的结构差异损失。

    使用固定高斯窗口 对 RGB 三通道分别计算局部统计量，再对通道取平均。
    DSSIM 定义为 ``(1 - SSIM) / 2``。"""

    CHANNELS = 3

    def __init__(
        self,
        weight: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
        reduction: Reduction = "none",
        range_norm: bool = True,
    ) -> None:
        """初始化 DSSIM 损失。

        参数:
            weight: 最终损失权重。
            window_size: 高斯窗口尺寸，必须为正奇数。
            sigma: 高斯标准差，必须为正数。
            reduction: ``none`` 返回 ``(N, 1, H, W)``；``mean``/``sum`` 返回标量。
            range_norm: 是否先将输入从 ``[-1, 1]`` 映射到 ``[0, 1]``。"""
        super().__init__()

        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be a positive odd integer, got {window_size}"
            )
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.range_norm = range_norm
        self.weight = weight
        self.window_size = window_size
        self.reduction = reduction

        window = self._create_window(window_size, sigma).expand(
            self.CHANNELS, 1, window_size, window_size
        )
        self.register_buffer("window", window, persistent=False)

    @staticmethod
    def _gaussian_1d(window_size: int, sigma: float) -> Tensor:
        """生成归一化的一维高斯核。"""
        coordinates = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gaussian = torch.exp(-(coordinates.square()) / (2 * sigma**2))
        return gaussian / gaussian.sum()

    @classmethod
    def _create_window(cls, window_size: int, sigma: float) -> Tensor:
        """由一维高斯核 构造二维 SSIM 窗口。"""
        gaussian_1d = cls._gaussian_1d(window_size, sigma)
        return torch.outer(gaussian_1d, gaussian_1d)[None, None]

    def _ssim(self, prediction: Tensor, target: Tensor) -> Tensor:
        """计算 RGB 输入的局部 SSIM 图，并对颜色通道取平均。"""
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes must match: {tuple(prediction.shape)} != {tuple(target.shape)}"
            )
        if prediction.ndim != 4 or prediction.shape[1] != self.CHANNELS:
            raise ValueError(
                f"DSSIM expects NCHW RGB tensors, got shape={tuple(prediction.shape)}"
            )

        window = self.get_buffer("window")
        padding = self.window_size // 2

        mu_prediction = F.conv2d(
            prediction, window, padding=padding, groups=self.CHANNELS
        )
        mu_target = F.conv2d(target, window, padding=padding, groups=self.CHANNELS)

        mu_prediction2 = mu_prediction.square()
        mu_target2 = mu_target.square()
        mu_cross = mu_prediction * mu_target

        sigma_prediction2 = (
            F.conv2d(prediction.square(), window, padding=padding, groups=self.CHANNELS)
            - mu_prediction2
        ).clamp_min(0.0)
        sigma_target2 = (
            F.conv2d(target.square(), window, padding=padding, groups=self.CHANNELS)
            - mu_target2
        ).clamp_min(0.0)
        sigma_cross = (
            F.conv2d(prediction * target, window, padding=padding, groups=self.CHANNELS)
            - mu_cross
        )

        c1 = 0.01**2
        c2 = 0.03**2
        ssim_map = ((2 * mu_cross + c1) * (2 * sigma_cross + c2)) / (
            (mu_prediction2 + mu_target2 + c1)
            * (sigma_prediction2 + sigma_target2 + c2)
            + EPS
        )
        return ssim_map.mean(dim=1, keepdim=True)

    @torch.compile(
        fullgraph=True,
        dynamic=False,
        options={"epilogue_fusion": True, "max_autotune": True},
    )
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """计算预测图像与目标图像的 DSSIM。

        参数:
            prediction: ``(N, 3, H, W)`` RGB 张量。
            target: 与 prediction 同形的 RGB 张量。

        返回:
            按 ``reduction`` 处理后的 DSSIM。"""
        if self.range_norm:
            prediction = prediction.add(1.0).mul(0.5)
            target = target.add(1.0).mul(0.5)

        dssim = (1.0 - self._ssim(prediction, target)) * (0.5 * self.weight)
        return _reduce_loss(dssim, self.reduction)
