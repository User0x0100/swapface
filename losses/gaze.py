"""基于冻结 L2CS-Net 的 gaze consistency loss。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from misc.models import ImageInputRange
from misc.models.l2cs import L2CSNet


class GazeLoss(nn.Module):
    """约束生成人脸保持参考人脸的 gaze 方向。

    参考分支在 ``no_grad`` 下提取 soft target；生成分支保留输入梯度，但 L2CS-Net
    参数始终冻结。损失由 3D gaze direction cosine loss 和小权重的 logits 分布
    KL divergence 组成，并可使用参考预测的归一化熵自动降低低置信样本权重。
    """

    def __init__(
        self,
        weight: float = 1.0,
        distribution_weight: float = 0.1,
        confidence_weighted: bool = True,
        input_range: ImageInputRange = ImageInputRange.MINUS_ONE_TO_ONE,
    ) -> None:
        super().__init__()
        if weight < 0.0:
            raise ValueError(f"weight 必须非负，实际为 {weight}")
        if distribution_weight < 0.0:
            raise ValueError(f"distribution_weight 必须非负，实际为 {distribution_weight}")

        self.weight = weight
        self.distribution_weight = distribution_weight
        self.confidence_weighted = confidence_weighted
        self.gaze_model = L2CSNet(input_range=input_range).eval().requires_grad_(False)
        bins = torch.arange(L2CSNet.NUM_BINS, dtype=torch.float32).mul(L2CSNet.BIN_WIDTH_DEGREES).add(L2CSNet.ANGLE_OFFSET_DEGREES)
        self.register_buffer("angle_bins_radians", bins.mul(math.pi / 180.0), persistent=False)
        self.eval()

    @staticmethod
    def _gaze_vector(angle0: Tensor, angle1: Tensor) -> Tensor:
        """复现 L2CS-Net 上游 ``gazeto3d`` 的二维角度到单位 gaze vector 映射。"""
        cos1 = torch.cos(angle1)
        return torch.stack(
            (
                -cos1 * torch.sin(angle0),
                -torch.sin(angle1),
                -cos1 * torch.cos(angle0),
            ),
            dim=1,
        )

    @staticmethod
    def _normalized_confidence(prob0: Tensor, prob1: Tensor) -> Tensor:
        max_entropy = math.log(prob0.shape[1])
        entropy0 = -(prob0 * prob0.clamp_min(1e-8).log()).sum(dim=1)
        entropy1 = -(prob1 * prob1.clamp_min(1e-8).log()).sum(dim=1)
        return (1.0 - (entropy0 + entropy1) * (0.5 / max_entropy)).clamp(0.0, 1.0)

    def forward(self, generated: Tensor, reference: Tensor) -> Tensor:
        """计算 ``generated`` 相对 ``reference`` 的 gaze consistency loss。"""
        if generated.shape != reference.shape:
            raise ValueError(f"generated/reference shape 必须一致：{tuple(generated.shape)} != {tuple(reference.shape)}")

        with torch.no_grad():
            target_logits0, target_logits1 = self.gaze_model(reference)
            target_prob0 = F.softmax(target_logits0.float(), dim=1)
            target_prob1 = F.softmax(target_logits1.float(), dim=1)

        predicted_logits0, predicted_logits1 = self.gaze_model(generated)
        predicted_logits0 = predicted_logits0.float()
        predicted_logits1 = predicted_logits1.float()
        predicted_prob0 = F.softmax(predicted_logits0, dim=1)
        predicted_prob1 = F.softmax(predicted_logits1, dim=1)

        angle_bins = self.get_buffer("angle_bins_radians")
        target_angle0 = (target_prob0 * angle_bins).sum(dim=1)
        target_angle1 = (target_prob1 * angle_bins).sum(dim=1)
        predicted_angle0 = (predicted_prob0 * angle_bins).sum(dim=1)
        predicted_angle1 = (predicted_prob1 * angle_bins).sum(dim=1)

        target_vector = self._gaze_vector(target_angle0, target_angle1)
        predicted_vector = self._gaze_vector(predicted_angle0, predicted_angle1)
        direction_loss = 1.0 - F.cosine_similarity(predicted_vector, target_vector, dim=1)

        distribution_loss = F.kl_div(F.log_softmax(predicted_logits0, dim=1), target_prob0, reduction="none").sum(dim=1)
        distribution_loss += F.kl_div(F.log_softmax(predicted_logits1, dim=1), target_prob1, reduction="none").sum(dim=1)

        loss = direction_loss + distribution_loss * self.distribution_weight
        if self.confidence_weighted:
            loss = loss * self._normalized_confidence(target_prob0, target_prob1)
        return loss.mean() * self.weight
