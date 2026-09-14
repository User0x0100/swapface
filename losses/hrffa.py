"""基于 HRFFA 关键点的面部几何一致性损失。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from misc.models import ImageInputRange
from misc.models.hrffa import HRFFALandmarkModel, HRFFAModelVariant, HRFFAScheme, HRFFAVisibility

from .functional import EPS

# WFLW98 topology（0-based）。
_CONTOUR = tuple(range(33))

# 眼睑开合：两个眼角定义宽度，三对上下眼睑点定义高度。
_RIGHT_EYE_CORNERS = (60, 64)
_RIGHT_EYE_PAIRS = ((61, 67), (62, 66), (63, 65))
_LEFT_EYE_CORNERS = (68, 72)
_LEFT_EYE_PAIRS = ((69, 75), (70, 74), (71, 73))

# 内唇开合。
_MOUTH_CORNERS = (88, 92)
_MOUTH_PAIRS = ((89, 95), (90, 94), (91, 93))

# 姿态只使用相对刚性的中央结构：四个眼角、鼻梁、鼻基底。
# 不使用嘴角/下巴，避免 smile、frown、jaw-open 污染 head-pose 监督。
_POSE_INDICES = (60, 64, 68, 72, 51, 52, 53, 54, 55, 57, 59)
_MIN_POSE_VISIBLE_POINTS = 6
_MIN_OPENNESS_PAIRS = 2
_MIN_OPENNESS_WIDTH = 0.02
_MIN_POSE_AXIS_NORM = 1e-3

# 通用 3D face template，仅用于从 2D landmarks 解出旋转；绝对单位没有意义。
# 坐标约定：x 向右、y 向上、z 向前。眼角与鼻部比例为 generic head 近似值，
# 并统一缩放到 O(1)，以改善弱透视最小二乘的条件数。
_POSE_TEMPLATE = torch.tensor(
    [
        (-225.0, 170.0, -135.0),
        (-75.0, 170.0, -135.0),
        (75.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (0.0, 150.0, -125.0),
        (0.0, 105.0, -95.0),
        (0.0, 60.0, -55.0),
        (0.0, 15.0, -20.0),
        (-70.0, -25.0, -50.0),
        (0.0, -35.0, 0.0),
        (70.0, -25.0, -50.0),
    ],
    dtype=torch.float32,
).div_(330.0)


def _confidence_mean(values: Tensor, weights: Tensor) -> Tensor:
    """按固定元素数归约置信加权损失，使低 coverage 真正降低监督强度。"""
    return (values * weights).mean()


def _masked_openness(
    points: Tensor,
    visibility: Tensor,
    corners: tuple[int, int],
    pairs: tuple[tuple[int, int], ...],
) -> tuple[Tensor, Tensor]:
    """计算仅由 target 可见点参与的局部开合度及其置信权重。

    宽度端点必须同时可见；上下点对仅在两个点都可见时参与高度均值，并要求至少
    ``_MIN_OPENNESS_PAIRS`` 对有效。这样大姿态下 HRFFA 对遮挡点的推断不会被当作
    眨眼/嘴部动作的监督信号。
    """
    visible = visibility == HRFFAVisibility.VISIBLE.value
    corner_valid = visible[:, corners[0]] & visible[:, corners[1]]
    pair_valid = torch.stack([visible[:, upper] & visible[:, lower] for upper, lower in pairs], dim=1)
    pair_count = pair_valid.sum(dim=1)

    # 关键点归一化到 [0, 1] 尺度后，正常眼宽约 0.1、嘴宽约 0.2。
    # generated 在训练早期可能发生角点塌缩；若直接除以极小 width，会产生 1/width
    # 级梯度爆炸。下限 0.02 远低于正常几何尺度，仅稳定退化样本。
    width = torch.linalg.vector_norm(points[:, corners[0]] - points[:, corners[1]], dim=1).clamp_min(_MIN_OPENNESS_WIDTH)
    heights = torch.stack([torch.linalg.vector_norm(points[:, upper] - points[:, lower], dim=1) for upper, lower in pairs], dim=1)
    height = (heights * pair_valid.to(dtype=heights.dtype)).sum(dim=1) / pair_count.clamp_min(1).to(dtype=heights.dtype)

    valid = corner_valid & (pair_count >= _MIN_OPENNESS_PAIRS)
    openness = torch.where(valid, height / width, torch.zeros_like(width))
    confidence = valid.to(dtype=points.dtype) * (pair_count.to(dtype=points.dtype) / len(pairs))
    return openness, confidence


def _minimum_projected_width_penalty(
    generated: Tensor,
    reference: Tensor,
    visibility: Tensor,
    corners: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """防止 generated 局部宽度塌缩，但不匹配 target 的具体宽度。

    target 仅提供局部横轴方向；generated 在该方向上的投影宽度达到
    ``_MIN_OPENNESS_WIDTH`` 后惩罚严格为 0。这样完全塌缩时仍有确定的展开梯度，
    又不会把 target 的眼宽/嘴宽身份几何复制到 generated。
    """
    visible = visibility == HRFFAVisibility.VISIBLE.value
    corner_visible = visible[:, corners[0]] & visible[:, corners[1]]

    reference_delta = reference[:, corners[1]] - reference[:, corners[0]]
    reference_width = torch.linalg.vector_norm(reference_delta, dim=1)
    reference_valid = corner_visible & torch.isfinite(reference_width) & (reference_width >= _MIN_OPENNESS_WIDTH)

    fallback = reference_delta.new_tensor((1.0, 0.0)).expand_as(reference_delta)
    safe_reference_delta = torch.where(reference_valid[:, None], reference_delta, fallback)
    reference_direction = F.normalize(safe_reference_delta, dim=1)

    generated_delta = generated[:, corners[1]] - generated[:, corners[0]]
    projected_width = (generated_delta * reference_direction).sum(dim=1)
    penalty = F.relu(_MIN_OPENNESS_WIDTH - projected_width) / _MIN_OPENNESS_WIDTH
    penalty = torch.where(reference_valid, penalty, torch.zeros_like(penalty))
    return penalty, reference_valid


class HRFFAFacialGeometryLoss(nn.Module):
    """保持 target 的头部姿态、眼睑状态、嘴部开合和面部外轮廓。

    HRFFA ``vitt-256`` 参数被冻结；reference 分支完全在 ``no_grad`` 下执行，
    generated 分支保留到输入的梯度。动态五官只比较尺度无关的局部几何描述量，
    避免直接匹配 target 的眼型/嘴型；外轮廓则有意匹配 target，因为换脸结果应
    保留 target 的脸型。
    """

    def __init__(
        self,
        pose_weight: float = 1.0,
        eye_weight: float = 1.0,
        mouth_weight: float = 1.0,
        contour_weight: float = 1.0,
        contour_shape_weight: float = 0.5,
        occluded_geometry_weight: float = 0.25,
        input_range: ImageInputRange = ImageInputRange.MINUS_ONE_TO_ONE,
    ) -> None:
        super().__init__()
        weights = {
            "pose_weight": pose_weight,
            "eye_weight": eye_weight,
            "mouth_weight": mouth_weight,
            "contour_weight": contour_weight,
            "contour_shape_weight": contour_shape_weight,
            "occluded_geometry_weight": occluded_geometry_weight,
        }
        non_finite = {name: value for name, value in weights.items() if not math.isfinite(value)}
        if non_finite:
            raise ValueError(f"HRFFA loss 权重必须为有限数值：{non_finite}")
        negative = {name: value for name, value in weights.items() if value < 0.0}
        if negative:
            raise ValueError(f"HRFFA loss 权重必须非负：{negative}")
        if occluded_geometry_weight > 1.0:
            raise ValueError(f"occluded_geometry_weight 必须位于 [0, 1]，实际为 {occluded_geometry_weight}")
        if not isinstance(input_range, ImageInputRange):
            raise TypeError(f"input_range 必须为 ImageInputRange，实际为 {type(input_range).__name__}")

        self.pose_weight = pose_weight
        self.eye_weight = eye_weight
        self.mouth_weight = mouth_weight
        self.contour_weight = contour_weight
        self.contour_shape_weight = contour_shape_weight
        self.occluded_geometry_weight = occluded_geometry_weight

        self.hrffa = (
            HRFFALandmarkModel(
                scheme=HRFFAScheme.WFLW98,
                input_range=input_range,
                variant=HRFFAModelVariant.VITT_256,
            )
            .eval()
            .requires_grad_(False)
        )
        self.register_buffer("pose_template", _POSE_TEMPLATE.clone(), persistent=False)
        self.eval()

    def _geometry_visibility_weights(self, visibility: Tensor) -> Tensor:
        """几何拟合权重：可见=1，遮挡=较小权重，画面外=0。"""
        weights = torch.zeros_like(visibility, dtype=torch.float32)
        weights = torch.where(visibility == HRFFAVisibility.VISIBLE.value, torch.ones_like(weights), weights)
        if self.occluded_geometry_weight > 0.0:
            weights = torch.where(
                visibility == HRFFAVisibility.OCCLUDED.value,
                torch.full_like(weights, self.occluded_geometry_weight),
                weights,
            )
        return weights

    def _fit_rotation(self, points: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
        """用加权弱透视最小二乘拟合 3×3 旋转矩阵和逐样本有效标记。

        姿态只使用眼角和鼻部刚性点。除最少可见点数外，还检查加权 canonical
        几何的秩；退化样本仍执行数值安全的正则化求解，但会被 ``valid`` 完全屏蔽，
        不产生 pose 监督。
        """
        observed = points[:, _POSE_INDICES]
        # 图像坐标 y 向下，转换为与 canonical template 一致的 y 向上坐标。
        observed = torch.stack((observed[..., 0], -observed[..., 1]), dim=-1)
        point_weights = weights[:, _POSE_INDICES].to(dtype=observed.dtype)

        template = self.get_buffer("pose_template").to(dtype=observed.dtype)
        template = template.unsqueeze(0).expand(observed.shape[0], -1, -1)
        denom = point_weights.sum(dim=1, keepdim=True).clamp_min(EPS)
        template_mean = (template * point_weights.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        observed_mean = (observed * point_weights.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)

        sqrt_weight = point_weights.clamp_min(0.0).sqrt().unsqueeze(-1)
        x = (template - template_mean) * sqrt_weight
        y = (observed - observed_mean) * sqrt_weight

        xt = x.transpose(1, 2)
        normal = xt @ x
        rhs = xt @ y
        eye = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0)
        projection = torch.linalg.solve(normal + eye * 1e-4, rhs)  # (B, 3, 2)

        # generated landmarks 可能在训练早期塌缩。若直接对接近零的 projection 做
        # F.normalize，1 / eps 会产生极大的反向梯度。先以无梯度布尔条件判断轴是否
        # 有足够长度；无效样本用固定正交轴替代，使其后续 rotation 对 generated
        # landmarks 的梯度严格为 0，再由 valid 完全屏蔽对应 pose loss。
        axis_x = projection[:, :, 0]
        axis_x_norm = torch.linalg.vector_norm(axis_x, dim=1)
        axis_x_valid = torch.isfinite(axis_x_norm) & (axis_x_norm > _MIN_POSE_AXIS_NORM)
        fallback_x = axis_x.new_tensor((1.0, 0.0, 0.0)).expand_as(axis_x)
        safe_axis_x = torch.where(axis_x_valid[:, None], axis_x, fallback_x)
        row_x = F.normalize(safe_axis_x, dim=1)

        axis_y = projection[:, :, 1]
        row_y_raw = axis_y - (axis_y * row_x).sum(dim=1, keepdim=True) * row_x
        axis_y_norm = torch.linalg.vector_norm(row_y_raw, dim=1)
        axis_y_valid = torch.isfinite(axis_y_norm) & (axis_y_norm > _MIN_POSE_AXIS_NORM)
        fallback_y = axis_y.new_tensor((0.0, 1.0, 0.0)).expand_as(axis_y)
        safe_axis_y = torch.where(axis_y_valid[:, None], row_y_raw, fallback_y)
        row_y = F.normalize(safe_axis_y, dim=1)
        row_z = torch.cross(row_x, row_y, dim=1)
        rotation = torch.stack((row_x, row_y, row_z), dim=1)

        eigenvalues = torch.linalg.eigvalsh(normal.float())
        full_rank = eigenvalues[:, 0] > eigenvalues[:, -1].clamp_min(EPS) * 1e-4
        finite = torch.isfinite(projection).all(dim=(1, 2)) & torch.isfinite(rotation).all(dim=(1, 2))
        return rotation, full_rank & finite & axis_x_valid & axis_y_valid

    def _pose_loss(self, generated: Tensor, reference: Tensor, visibility: Tensor) -> Tensor:
        # Pose 只使用 target 明确可见的刚性点，不使用遮挡点的推断坐标。
        pose_weights = (visibility == HRFFAVisibility.VISIBLE.value).to(dtype=generated.dtype)
        generated_rotation, generated_valid = self._fit_rotation(generated, pose_weights)
        reference_rotation, reference_valid = self._fit_rotation(reference, pose_weights)
        visible_count = pose_weights[:, _POSE_INDICES].sum(dim=1)
        valid = generated_valid & reference_valid & (visible_count >= _MIN_POSE_VISIBLE_POINTS)

        relative = generated_rotation @ reference_rotation.transpose(1, 2)
        trace = relative.diagonal(dim1=1, dim2=2).sum(dim=1)
        loss = ((3.0 - trace) * 0.5).clamp(0.0, 2.0)  # 1 - cos(theta)
        loss = torch.where(valid, loss, torch.zeros_like(loss))
        return _confidence_mean(loss, valid.to(dtype=loss.dtype))

    def _eye_loss(self, generated: Tensor, reference: Tensor, visibility: Tensor) -> Tensor:
        generated_right, right_confidence = _masked_openness(generated, visibility, _RIGHT_EYE_CORNERS, _RIGHT_EYE_PAIRS)
        reference_right, _ = _masked_openness(reference, visibility, _RIGHT_EYE_CORNERS, _RIGHT_EYE_PAIRS)
        right_width_penalty, right_width_valid = _minimum_projected_width_penalty(generated, reference, visibility, _RIGHT_EYE_CORNERS)
        right_confidence = right_confidence * right_width_valid.to(dtype=right_confidence.dtype)

        generated_left, left_confidence = _masked_openness(generated, visibility, _LEFT_EYE_CORNERS, _LEFT_EYE_PAIRS)
        reference_left, _ = _masked_openness(reference, visibility, _LEFT_EYE_CORNERS, _LEFT_EYE_PAIRS)
        left_width_penalty, left_width_valid = _minimum_projected_width_penalty(generated, reference, visibility, _LEFT_EYE_CORNERS)
        left_confidence = left_confidence * left_width_valid.to(dtype=left_confidence.dtype)

        losses = torch.cat(((generated_right - reference_right).abs(), (generated_left - reference_left).abs()))
        width_penalties = torch.cat((right_width_penalty, left_width_penalty))
        confidences = torch.cat((right_confidence, left_confidence))
        return _confidence_mean(losses + width_penalties, confidences)

    def _mouth_loss(self, generated: Tensor, reference: Tensor, visibility: Tensor) -> Tensor:
        generated_open, confidence = _masked_openness(generated, visibility, _MOUTH_CORNERS, _MOUTH_PAIRS)
        reference_open, _ = _masked_openness(reference, visibility, _MOUTH_CORNERS, _MOUTH_PAIRS)
        width_penalty, width_valid = _minimum_projected_width_penalty(generated, reference, visibility, _MOUTH_CORNERS)
        confidence = confidence * width_valid.to(dtype=confidence.dtype)
        return _confidence_mean((generated_open - reference_open).abs() + width_penalty, confidence)

    def _contour_loss(self, generated: Tensor, reference: Tensor, visibility: Tensor) -> Tensor:
        generated_contour = generated[:, _CONTOUR]
        reference_contour = reference[:, _CONTOUR]
        weights = self._geometry_visibility_weights(visibility)[:, _CONTOUR]

        point_error = torch.linalg.vector_norm(generated_contour - reference_contour, dim=2)
        point_loss = _confidence_mean(point_error, weights)
        if self.contour_shape_weight <= 0.0:
            return point_loss

        generated_segments = generated_contour[:, 1:] - generated_contour[:, :-1]
        reference_segments = reference_contour[:, 1:] - reference_contour[:, :-1]
        segment_error = torch.linalg.vector_norm(generated_segments - reference_segments, dim=2)
        segment_weights = torch.minimum(weights[:, 1:], weights[:, :-1])
        shape_loss = _confidence_mean(segment_error, segment_weights)
        return point_loss + shape_loss * self.contour_shape_weight

    def _landmark_components(self, generated: Tensor, reference: Tensor, visibility: Tensor) -> dict[str, Tensor]:
        """在归一化 WFLW98 坐标上以 FP32 计算各项加权几何约束。"""
        # Trainer 的 Generator step 可能处于 BF16 autocast。HRFFA 网络本身可以由
        # autocast 加速，但 linalg.solve/eigvalsh 等几何求解固定使用 FP32，避免
        # autocast 将矩阵乘降为 BF16 后形成混合 dtype 或数值不稳定。
        with torch.autocast(device_type=generated.device.type, enabled=False):
            generated = generated.float()
            reference = reference.float()
            zero = generated.new_zeros(())
            return {
                "pose": self._pose_loss(generated, reference, visibility) * self.pose_weight if self.pose_weight > 0.0 else zero,
                "eye": self._eye_loss(generated, reference, visibility) * self.eye_weight if self.eye_weight > 0.0 else zero,
                "mouth": self._mouth_loss(generated, reference, visibility) * self.mouth_weight if self.mouth_weight > 0.0 else zero,
                "contour": self._contour_loss(generated, reference, visibility) * self.contour_weight if self.contour_weight > 0.0 else zero,
            }

    def _extract_landmarks(self, generated: Tensor, reference: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if generated.shape != reference.shape:
            raise ValueError(f"generated/reference shape 必须一致：{tuple(generated.shape)} != {tuple(reference.shape)}")
        if generated.ndim != 4 or generated.shape[1] != 3 or generated.shape[2] != generated.shape[3]:
            raise ValueError(f"HRFFA loss 需要 NCHW 正方形 RGB 输入，实际 shape={tuple(generated.shape)}")

        with torch.no_grad():
            reference_points, reference_visibility = self.hrffa(reference, return_visibility=True)
        generated_points = self.hrffa(generated)

        # HRFFA wrapper 返回 caller 输入尺寸下的像素坐标；除以边长后所有几何损失
        # 与训练分辨率无关。坐标允许落在 [0, 1] 外，不做 clamp。
        scale = float(generated.shape[-1])
        return generated_points.float() / scale, reference_points.float() / scale, reference_visibility

    def forward_components(self, generated: Tensor, reference: Tensor) -> dict[str, Tensor]:
        """一次 HRFFA 前向后返回 pose/eye/mouth/contour 四个已加权标量损失。"""
        generated_points, reference_points, reference_visibility = self._extract_landmarks(generated, reference)
        return self._landmark_components(generated_points, reference_points, reference_visibility)

    def forward(self, generated: Tensor, reference: Tensor) -> Tensor:
        """计算 generated 相对 target/reference 的面部几何一致性总损失。"""
        components = self.forward_components(generated, reference)
        return torch.stack(tuple(components.values())).sum()
