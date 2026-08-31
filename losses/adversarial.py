from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, autocast, nn

from .functional import Reduction, _reduce_loss


class DiscriminatorAdversarialLoss(nn.Module):
    SUPPORTED_TYPES = frozenset({"hinge", "wgan", "ls", "bce"})

    def __init__(
        self,
        loss_type: Literal["hinge", "wgan", "ls", "bce"] = "hinge",
        weight: float = 1.0,
        reduction: Reduction = "none",
    ) -> None:
        super().__init__()
        if loss_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported loss_type: {loss_type!r}. Supported types: {sorted(self.SUPPORTED_TYPES)}"
            )
        self.loss_type = loss_type
        self.reduction = reduction
        self.weight = weight

    @staticmethod
    def _hinge(fake_score: Tensor, real_score: Tensor) -> Tensor:
        return torch.relu(1.0 - real_score) + torch.relu(1.0 + fake_score)

    @staticmethod
    def _wgan(fake_score: Tensor, real_score: Tensor) -> Tensor:
        return fake_score - real_score

    @staticmethod
    def _ls(fake_score: Tensor, real_score: Tensor) -> Tensor:
        return (real_score - 1).square() + fake_score.square()

    @staticmethod
    def _bce(fake_score: Tensor, real_score: Tensor) -> Tensor:
        with autocast(device_type="cuda", enabled=False):
            fake_score = fake_score.float()
            real_score = real_score.float()
            return F.binary_cross_entropy_with_logits(
                real_score, torch.ones_like(real_score), reduction="none"
            ) + F.binary_cross_entropy_with_logits(
                fake_score, torch.zeros_like(fake_score), reduction="none"
            )

    def forward(self, fake_score: Tensor, real_score: Tensor) -> Tensor:
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
    SUPPORTED_TYPES = frozenset({"hinge", "wgan", "ls", "bce"})

    def __init__(
        self,
        loss_type: Literal["hinge", "wgan", "ls", "bce"] = "hinge",
        weight: float = 1.0,
        reduction: Reduction = "none",
    ) -> None:
        super().__init__()
        if loss_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported loss_type: {loss_type!r}. Supported types: {sorted(self.SUPPORTED_TYPES)}"
            )
        self.loss_type = loss_type
        self.reduction = reduction
        self.weight = weight

    @staticmethod
    def _hinge_or_wgan(predicted_score: Tensor) -> Tensor:
        return -predicted_score

    @staticmethod
    def _ls(predicted_score: Tensor) -> Tensor:
        return (predicted_score - 1).square()

    @staticmethod
    def _bce(predicted_score: Tensor) -> Tensor:
        with autocast(device_type="cuda", enabled=False):
            predicted_score = predicted_score.float()
            return F.binary_cross_entropy_with_logits(
                predicted_score, torch.ones_like(predicted_score), reduction="none"
            )

    def forward(self, predicted_score: Tensor) -> Tensor:
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
