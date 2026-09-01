from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, autocast, nn

from .functional import Reduction, _reduce_loss


class DiscriminatorAdversarialLoss(nn.Module):
    """计算判别器对抗损失。

    支持 hinge、Wasserstein、最小二乘和 BCE-with-logits 四种形式。
    ``reduction="none"`` 时保留判别器得分张量的形状；``mean``/``sum`` 返回标量。
    """

    SUPPORTED_TYPES = frozenset({"hinge", "wgan", "ls", "bce"})

    def __init__(
        self,
        loss_type: Literal["hinge", "wgan", "ls", "bce"] = "hinge",
        weight: float = 1.0,
        reduction: Reduction = "none",
    ) -> None:
        """初始化判别器对抗损失。

        参数:
            loss_type: 对抗损失类型，支持 ``hinge``、``wgan``、``ls``、``bce``。
            weight: 最终损失的标量权重。
            reduction: 输出聚合方式，支持 ``none``、``mean``、``sum``。

        异常:
            ValueError: ``loss_type`` 不受支持。
        """
        super().__init__()
        if loss_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported loss_type: {loss_type!r}. Supported types: {sorted(self.SUPPORTED_TYPES)}")
        self.loss_type = loss_type
        self.reduction = reduction
        self.weight = weight

    @staticmethod
    def _hinge(fake_score: Tensor, real_score: Tensor) -> Tensor:
        """计算逐元素 hinge 判别器损失。"""
        return torch.relu(1.0 - real_score) + torch.relu(1.0 + fake_score)

    @staticmethod
    def _wgan(fake_score: Tensor, real_score: Tensor) -> Tensor:
        """计算逐元素 Wasserstein 判别器损失。"""
        return fake_score - real_score

    @staticmethod
    def _ls(fake_score: Tensor, real_score: Tensor) -> Tensor:
        """计算逐元素最小二乘判别器损失。"""
        return (real_score - 1).square() + fake_score.square()

    @staticmethod
    def _bce(fake_score: Tensor, real_score: Tensor) -> Tensor:
        """使用 FP32 计算逐元素 BCE-with-logits 判别器损失。"""
        with autocast(device_type="cuda", enabled=False):
            fake_score = fake_score.float()
            real_score = real_score.float()
            return F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score), reduction="none") + F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score), reduction="none")

    def forward(self, fake_score: Tensor, real_score: Tensor) -> Tensor:
        """计算生成样本与真实样本得分对应的判别器损失。

        参数:
            fake_score: 判别器对生成样本输出的原始得分。
            real_score: 判别器对真实样本输出的原始得分。

        返回:
            按 ``reduction`` 聚合并乘以 ``weight`` 的损失张量。
        """
        match self.loss_type:
            case "hinge":
                loss = self._hinge(fake_score, real_score)
            case "wgan":
                loss = self._wgan(fake_score, real_score)
            case "ls":
                loss = self._ls(fake_score, real_score)
            case "bce":
                loss = self._bce(fake_score, real_score)
            case _:
                raise RuntimeError(f"Unexpected loss_type: {self.loss_type!r}")
        return _reduce_loss(loss * self.weight, self.reduction)


class GeneratorAdversarialLoss(nn.Module):
    """计算生成器对抗损失。

    hinge 与 Wasserstein 的生成器目标均为 ``-D(fake)``，另外支持最小二乘和
    BCE-with-logits。
    """

    SUPPORTED_TYPES = frozenset({"hinge", "wgan", "ls", "bce"})

    def __init__(
        self,
        loss_type: Literal["hinge", "wgan", "ls", "bce"] = "hinge",
        weight: float = 1.0,
        reduction: Reduction = "none",
    ) -> None:
        """初始化生成器对抗损失。

        参数:
            loss_type: 对抗损失类型，支持 ``hinge``、``wgan``、``ls``、``bce``。
            weight: 最终损失的标量权重。
            reduction: 输出聚合方式，支持 ``none``、``mean``、``sum``。

        异常:
            ValueError: ``loss_type`` 不受支持。
        """
        super().__init__()
        if loss_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported loss_type: {loss_type!r}. Supported types: {sorted(self.SUPPORTED_TYPES)}")
        self.loss_type = loss_type
        self.reduction = reduction
        self.weight = weight

    @staticmethod
    def _hinge_or_wgan(predicted_score: Tensor) -> Tensor:
        """计算 hinge/Wasserstein 的逐元素生成器目标 ``-D(fake)``。"""
        return -predicted_score

    @staticmethod
    def _ls(predicted_score: Tensor) -> Tensor:
        """计算逐元素最小二乘生成器损失。"""
        return (predicted_score - 1).square()

    @staticmethod
    def _bce(predicted_score: Tensor) -> Tensor:
        """使用 FP32 计算逐元素 BCE-with-logits 生成器损失。"""
        with autocast(device_type="cuda", enabled=False):
            predicted_score = predicted_score.float()
            return F.binary_cross_entropy_with_logits(predicted_score, torch.ones_like(predicted_score), reduction="none")

    def forward(self, predicted_score: Tensor) -> Tensor:
        """计算生成样本得分对应的生成器损失。

        参数:
            predicted_score: 判别器对生成样本输出的原始得分。

        返回:
            按 ``reduction`` 聚合并乘以 ``weight`` 的损失张量。
        """
        match self.loss_type:
            case "hinge" | "wgan":
                loss = self._hinge_or_wgan(predicted_score)
            case "ls":
                loss = self._ls(predicted_score)
            case "bce":
                loss = self._bce(predicted_score)
            case _:
                raise RuntimeError(f"Unexpected loss_type: {self.loss_type!r}")
        return _reduce_loss(loss * self.weight, self.reduction)
