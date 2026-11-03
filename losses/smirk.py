"""使用冻结 SMIRK encoder 监督目标姿态、表情、下颌和眼睑状态。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from misc.models.smirk import SMIRKEncoder


class SMIRKMotionLoss(nn.Module):
    COMPONENTS = ("pose", "expression", "jaw", "eyelid")

    def __init__(
        self,
        *,
        pose_weight: float,
        expression_weight: float,
        jaw_weight: float,
        eyelid_weight: float,
        encoder: SMIRKEncoder | None = None,
    ) -> None:
        super().__init__()
        weights = {"pose": pose_weight, "expression": expression_weight, "jaw": jaw_weight, "eyelid": eyelid_weight}
        if any(not math.isfinite(x) or x < 0 for x in weights.values()):
            raise ValueError(f"SMIRK 损失权重必须为有限非负数：{weights}")
        self.weights = weights
        self.encoder = encoder if encoder is not None else SMIRKEncoder()
        self.eval()

    def train(self, mode: bool = True) -> SMIRKMotionLoss:
        # Never switch pretrained teacher into training mode (BN/Dropout).
        super().train(False)
        return self

    @staticmethod
    def _pose_matrix(axis_angle: Tensor) -> Tensor:
        """Stable Rodrigues transform for FLAME axis-angle; differentiable at zero."""
        a = axis_angle.float()
        theta2 = (a * a).sum(dim=-1, keepdim=True)
        theta = theta2.clamp_min(1e-8).sqrt()
        # Small-angle Taylor branches avoid cancellation and 0/0.
        sin_over_theta = torch.where(
            theta2 < 1e-4,
            1 - theta2 / 6 + theta2.square() / 120,
            torch.sin(theta) / theta,
        )
        one_minus_cos_over_theta2 = torch.where(
            theta2 < 1e-4,
            0.5 - theta2 / 24 + theta2.square() / 720,
            (1 - torch.cos(theta)) / theta2.clamp_min(1e-8),
        )
        x, y, z = a.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*a.shape[:-1], 3, 3)
        identity = torch.eye(3, device=a.device, dtype=a.dtype).expand_as(skew)
        return identity + sin_over_theta[..., None] * skew + one_minus_cos_over_theta2[..., None] * (skew @ skew)

    @staticmethod
    def _preprocess(images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"SMIRK 需要 NCHW RGB 图像，实际 {tuple(images.shape)}")
        # Official SMIRK demo: RGB [0, 1] without ImageNet normalization,
        # cropped face 224x224. Input here is FFHQ/ArcFace restored [-1,1].
        images = (images.float() + 1.0) * 0.5
        return F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def encode_reference(self, reference: Tensor) -> dict[str, Tensor]:
        """Compute a detached target once per batch, reusable across stages/retries.

        SMIRK has no calibrated face confidence. Finite-output filtering only
        protects training from invalid teacher results; it is not a face-quality
        estimator and does not reject plausible but inaccurate extreme poses.
        """
        with torch.autocast(device_type=reference.device.type, enabled=False):
            raw = self.encoder(self._preprocess(reference))
        params = {key: raw[key].detach().float() for key in ("pose_params", "expression_params", "jaw_params", "eyelid_params")}
        valid = torch.stack([torch.isfinite(v).flatten(1).all(dim=1) for v in params.values()]).all(dim=0)
        cleaned = {key: torch.where(valid[:, None], value, torch.zeros_like(value)) for key, value in params.items()}
        cleaned["valid"] = valid
        return cleaned

    @staticmethod
    def _weighted_loss(per_sample: Tensor, target_valid: Tensor, sample_weights: Tensor | None) -> Tensor:
        """Skip invalid targets and downweight synthetic occlusion without renormalizing it away."""
        valid = target_valid.detach().to(device=per_sample.device)
        weights = valid.to(per_sample.dtype)
        if sample_weights is not None:
            if sample_weights.shape != valid.shape:
                raise ValueError(f"SMIRK sample_weights shape 不匹配：{tuple(sample_weights.shape)} != {tuple(valid.shape)}")
            # Avoid a GPU -> CPU sync here: invalid weights propagate into
            # g_loss and are caught by the existing finite-loss guard.
            weights = weights * sample_weights.detach().to(device=per_sample.device, dtype=per_sample.dtype).clamp(0, 1)
        # Invalid target rows were replaced by zeros; explicitly zero their loss.
        # Dividing by valid count preserves the per-valid-sample scale, while
        # an occluded sample contributes proportionally less gradient.
        return (per_sample * weights).sum() / weights.new_tensor(1).maximum(valid.sum().to(weights.dtype))

    def forward_components(
        self,
        generated: Tensor,
        reference: Tensor,
        *,
        reference_features: dict[str, Tensor] | None = None,
        sample_weights: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if generated.shape != reference.shape:
            raise ValueError(f"generated/reference shape 不匹配：{tuple(generated.shape)} != {tuple(reference.shape)}")
        with torch.autocast(device_type=generated.device.type, enabled=False):
            target = self.encode_reference(reference) if reference_features is None else reference_features
            if target["valid"].shape != (generated.shape[0],):
                raise ValueError("SMIRK reference_features 的 batch 不匹配")
            if any(v.device != generated.device for v in target.values()):
                raise ValueError("SMIRK reference_features 的 device 不匹配")
            predicted = self.encoder(self._preprocess(generated))
            parameters = {
                "expression": "expression_params",
                "jaw": "jaw_params",
                "eyelid": "eyelid_params",
            }
            result = {
                name: self._weighted_loss(
                    F.smooth_l1_loss(predicted[key].float(), target[key], reduction="none").flatten(1).mean(dim=1),
                    target["valid"],
                    sample_weights,
                )
                * self.weights[name]
                for name, key in parameters.items()
                if self.weights[name] > 0
            }
            if self.weights["pose"] > 0:
                per_sample_pose = (
                    F
                    .smooth_l1_loss(
                        self._pose_matrix(predicted["pose_params"]),
                        self._pose_matrix(target["pose_params"]),
                        reduction="none",
                    )
                    .flatten(1)
                    .mean(dim=1)
                )
                result["pose"] = self._weighted_loss(per_sample_pose, target["valid"], sample_weights) * self.weights["pose"]
            return result

    def forward(self, generated: Tensor, reference: Tensor) -> Tensor:
        components = self.forward_components(generated, reference)
        if not components:
            return generated.sum() * 0
        return torch.stack(tuple(components.values())).sum()
