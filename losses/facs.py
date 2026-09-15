"""基于冻结 OpenGraphAU / MEFARG 的 FACS Action Unit 一致性损失。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from misc.models import ImageInputRange
from misc.models.openface_au import MAIN_AU_CODES, OpenFaceAU

# 27 个 global AU 按主要动作区域分组；14 个 bilateral AU 组成 7 对左右差异约束。
_BROW_AUS = (1, 2, 4)
_EYE_AUS = (5, 6, 7)
_NOSE_AUS = (9, 38, 39)
_MOUTH_AUS = (10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 23, 24, 25, 27, 32)
_LOWER_FACE_AUS = (17, 19, 26)
_NUM_ASYMMETRY_PAIRS = OpenFaceAU.NUM_SUB_AUS // 2
_NUM_FACS_SIGNALS = OpenFaceAU.NUM_MAIN_AUS + _NUM_ASYMMETRY_PAIRS


def _main_indices(codes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(MAIN_AU_CODES.index(code) for code in codes)


class FACSConsistencyLoss(nn.Module):
    """约束 generated 保持 reference 的连续 FACS AU 激活状态。

    reference 分支完全在 ``no_grad`` 下执行；generated 分支保留到图像的梯度，
    AU teacher 参数始终冻结。主 AU 使用绝对激活差异；bilateral AU 仅比较左右差值，
    避免对同一动作强度重复计权。默认归约是 27 个主 AU + 7 个左右差异，共 34 个
    语义信号的平均绝对误差。
    """

    def __init__(
        self,
        weight: float = 1.0,
        brow_weight: float = 1.0,
        eye_weight: float = 1.0,
        nose_weight: float = 1.0,
        mouth_weight: float = 1.0,
        lower_face_weight: float = 1.0,
        asymmetry_weight: float = 1.0,
        input_range: ImageInputRange = ImageInputRange.MINUS_ONE_TO_ONE,
    ) -> None:
        super().__init__()
        weights = {
            "weight": weight,
            "brow_weight": brow_weight,
            "eye_weight": eye_weight,
            "nose_weight": nose_weight,
            "mouth_weight": mouth_weight,
            "lower_face_weight": lower_face_weight,
            "asymmetry_weight": asymmetry_weight,
        }
        invalid = {name: value for name, value in weights.items() if not math.isfinite(value) or value < 0.0}
        if invalid:
            raise ValueError(f"FACS loss 权重必须为有限非负数：{invalid}")

        self.weight = weight
        self.component_weights = {
            "brow": brow_weight,
            "eye": eye_weight,
            "nose": nose_weight,
            "mouth": mouth_weight,
            "lower_face": lower_face_weight,
            "asymmetry": asymmetry_weight,
        }
        self.au_model = OpenFaceAU(input_range=input_range).eval().requires_grad_(False)

        groups = {
            "brow": _main_indices(_BROW_AUS),
            "eye": _main_indices(_EYE_AUS),
            "nose": _main_indices(_NOSE_AUS),
            "mouth": _main_indices(_MOUTH_AUS),
            "lower_face": _main_indices(_LOWER_FACE_AUS),
        }
        for name, indices in groups.items():
            self.register_buffer(f"{name}_indices", torch.tensor(indices, dtype=torch.long), persistent=False)
        self.eval()

    def _components(self, generated_scores: Tensor, reference_scores: Tensor) -> dict[str, Tensor]:
        generated_scores = generated_scores.float()
        reference_scores = reference_scores.float()
        main_error = (generated_scores[:, : OpenFaceAU.NUM_MAIN_AUS] - reference_scores[:, : OpenFaceAU.NUM_MAIN_AUS]).abs()

        components: dict[str, Tensor] = {}
        for name in ("brow", "eye", "nose", "mouth", "lower_face"):
            component_weight = self.component_weights[name]
            if component_weight <= 0.0 or self.weight <= 0.0:
                components[name] = main_error.new_zeros(())
                continue
            indices = self.get_buffer(f"{name}_indices")
            components[name] = main_error.index_select(1, indices).sum(dim=1).mean() * (self.weight * component_weight / _NUM_FACS_SIGNALS)

        asymmetry_weight = self.component_weights["asymmetry"]
        if asymmetry_weight <= 0.0 or self.weight <= 0.0:
            components["asymmetry"] = main_error.new_zeros(())
        else:
            generated_sub = generated_scores[:, OpenFaceAU.NUM_MAIN_AUS :].reshape(generated_scores.shape[0], _NUM_ASYMMETRY_PAIRS, 2)
            reference_sub = reference_scores[:, OpenFaceAU.NUM_MAIN_AUS :].reshape(reference_scores.shape[0], _NUM_ASYMMETRY_PAIRS, 2)
            generated_delta = generated_sub[..., 0] - generated_sub[..., 1]
            reference_delta = reference_sub[..., 0] - reference_sub[..., 1]
            # L-R 位于 [-1, 1]，两样本差值最大为 2；乘 0.5 后与主 AU 的 [0, 1] 误差尺度一致。
            asymmetry_error = (generated_delta - reference_delta).abs().mul(0.5)
            components["asymmetry"] = asymmetry_error.sum(dim=1).mean() * (self.weight * asymmetry_weight / _NUM_FACS_SIGNALS)
        return components

    def forward_components(self, generated: Tensor, reference: Tensor) -> dict[str, Tensor]:
        """返回 brow/eye/nose/mouth/lower_face/asymmetry 六个已加权标量损失。"""
        if generated.shape != reference.shape:
            raise ValueError(f"generated/reference shape 必须一致：{tuple(generated.shape)} != {tuple(reference.shape)}")

        with torch.no_grad():
            reference_scores = self.au_model(reference)
        generated_scores = self.au_model(generated)
        return self._components(generated_scores, reference_scores)

    def forward(self, generated: Tensor, reference: Tensor) -> Tensor:
        components = self.forward_components(generated, reference)
        return torch.stack(tuple(components.values())).sum()
