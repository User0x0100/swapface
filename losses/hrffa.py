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
_MIN_OPENNESS_WIDTH = 0.02
_MIN_POSE_AXIS_NORM = 1e-3
_POSE_CONDITION_RATIO = 1e-4

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
    visible_probability: Tensor,
    corners: tuple[int, int],
    pairs: tuple[tuple[int, int], ...],
) -> tuple[Tensor, Tensor]:
    """按 target 的连续可见概率计算局部开合度和监督置信度。

    动态五官只使用 ``P(VISIBLE)``，不把 OCCLUDED 点的推断坐标当作眼睑/嘴部动作
    真值。点对权重使用两端置信度的较小值，使 visibility 在分类边界附近连续衰减，
    而不是因 argmax 翻转而突然丢失整项监督。
    """
    corner_confidence = torch.minimum(visible_probability[:, corners[0]], visible_probability[:, corners[1]])
    pair_confidence = torch.stack(
        [torch.minimum(visible_probability[:, upper], visible_probability[:, lower]) for upper, lower in pairs],
        dim=1,
    )
    pair_weight_sum = pair_confidence.sum(dim=1)

    # 关键点归一化到 [0, 1] 尺度后，正常眼宽约 0.1、嘴宽约 0.2。
    # generated 在训练早期可能发生角点塌缩；若直接除以极小 width，会产生 1/width
    # 级梯度爆炸。下限 0.02 远低于正常几何尺度，仅稳定退化样本。
    width = torch.linalg.vector_norm(points[:, corners[0]] - points[:, corners[1]], dim=1).clamp_min(_MIN_OPENNESS_WIDTH)
    heights = torch.stack([torch.linalg.vector_norm(points[:, upper] - points[:, lower], dim=1) for upper, lower in pairs], dim=1)
    height = (heights * pair_confidence).sum(dim=1) / pair_weight_sum.clamp_min(EPS)

    openness = torch.where(pair_weight_sum > 0.0, height / width, torch.zeros_like(width))

    # 单对关键点不足以稳定表示整体开合度，但也不应像硬阈值那样直接归零。
    # 对 3 对拓扑，完全可靠的 1/2/3 对分别得到约 0.1 / 0.67 / 1.0 的监督强度；
    # 中间概率连续插值，使大姿态下监督平滑衰减。
    pair_coverage = pair_confidence.mean(dim=1)
    support_factor = 0.3 + 0.7 * (pair_weight_sum - 1.0).clamp(0.0, 1.0)
    confidence = corner_confidence * pair_coverage * support_factor
    return openness, confidence


def _minimum_projected_width_penalty(
    generated: Tensor,
    reference: Tensor,
    corners: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """防止 generated 局部宽度塌缩，但不匹配 target 的具体宽度。

    target 仅提供局部横轴方向；generated 在该方向上的投影宽度达到
    ``_MIN_OPENNESS_WIDTH`` 后惩罚严格为 0。这样完全塌缩时仍有确定的展开梯度，
    又不会把 target 的眼宽/嘴宽身份几何复制到 generated。
    """
    reference_delta = reference[:, corners[1]] - reference[:, corners[0]]
    reference_width = torch.linalg.vector_norm(reference_delta, dim=1)
    reference_valid = torch.isfinite(reference_width) & (reference_width >= _MIN_OPENNESS_WIDTH)

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

    def _geometry_visibility_weights(self, visibility_probabilities: Tensor) -> Tensor:
        """连续几何置信度：P(VISIBLE) + α·P(OCCLUDED)，OUTSIDE 不直接贡献。"""
        visible = visibility_probabilities[..., HRFFAVisibility.VISIBLE.value]
        occluded = visibility_probabilities[..., HRFFAVisibility.OCCLUDED.value]
        return visible + occluded * self.occluded_geometry_weight

    def _fit_rotation(self, points: Tensor, weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """用加权弱透视最小二乘拟合 3×3 旋转矩阵和逐样本有效标记。

        姿态只使用眼角和鼻部刚性点。加权 canonical 几何的条件数作为连续置信度
        返回；只有 projection 非有限或旋转轴近零这类数值退化情况才硬屏蔽。
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
        condition_ratio = eigenvalues[:, 0].clamp_min(0.0) / eigenvalues[:, -1].clamp_min(EPS)
        # 旧实现以 ratio > 1e-4 做硬 full-rank 判定，导致 visibility 轻微变化即可
        # 让 pose loss 从 0 突然恢复。保留同一尺度作为“满置信”阈值，但改成连续权重。
        condition_confidence = (condition_ratio / _POSE_CONDITION_RATIO).clamp(0.0, 1.0)
        finite = torch.isfinite(projection).all(dim=(1, 2)) & torch.isfinite(rotation).all(dim=(1, 2))
        valid = finite & axis_x_valid & axis_y_valid
        return rotation, valid, condition_confidence.to(dtype=points.dtype)

    def _pose_loss(self, generated: Tensor, reference: Tensor, visibility_probabilities: Tensor) -> Tensor:
        # 大姿态下远侧刚性点常被标记为 OCCLUDED。使用 reference 的连续几何置信度
        # 做弱透视拟合，并按平均 coverage 衰减整样本监督，避免少于固定可见点数时骤然归零。
        pose_weights = self._geometry_visibility_weights(visibility_probabilities).to(dtype=generated.dtype)
        pose_point_weights = pose_weights[:, _POSE_INDICES]
        pose_confidence = pose_point_weights.mean(dim=1)

        # 拟合只保留各点之间的相对置信度，避免整体 coverage 下降时固定 ridge 正则
        # 改变 rotation 本身；整体置信度只在最终 loss 上衰减监督强度。
        fit_weights = pose_weights / pose_confidence.clamp_min(EPS).unsqueeze(1)
        generated_rotation, generated_valid, generated_condition = self._fit_rotation(generated, fit_weights)
        reference_rotation, reference_valid, reference_condition = self._fit_rotation(reference, fit_weights)
        valid = generated_valid & reference_valid & (pose_confidence > 0.0)
        condition_confidence = torch.minimum(generated_condition, reference_condition)

        # Softmax 会给几乎确定 OUTSIDE 的点留下极小非零概率，这些点不能靠“补齐秩”
        # 获得与真正可见点相同的 pose 权限。effective sample size 衡量实际由多少个独立
        # pose 点支撑拟合，并以连续曲线衰减欠约束样本；6 个有效点视为完整支持。
        squared_weight_sum = pose_point_weights.square().sum(dim=1)
        effective_points = pose_point_weights.sum(dim=1).square() / squared_weight_sum.clamp_min(
            torch.finfo(squared_weight_sum.dtype).tiny
        )
        pose_support = (effective_points / 6.0).clamp(max=1.0).square()

        relative = generated_rotation @ reference_rotation.transpose(1, 2)
        trace = relative.diagonal(dim1=1, dim2=2).sum(dim=1)
        loss = ((3.0 - trace) * 0.5).clamp(0.0, 2.0)  # 1 - cos(theta)
        loss = torch.where(valid, loss, torch.zeros_like(loss))
        confidence = pose_confidence * pose_support * condition_confidence * valid.to(dtype=loss.dtype)
        return _confidence_mean(loss, confidence)

    def _eye_loss(self, generated: Tensor, reference: Tensor, visibility_probabilities: Tensor) -> Tensor:
        visible_probability = visibility_probabilities[..., HRFFAVisibility.VISIBLE.value]
        generated_right, right_confidence = _masked_openness(generated, visible_probability, _RIGHT_EYE_CORNERS, _RIGHT_EYE_PAIRS)
        reference_right, _ = _masked_openness(reference, visible_probability, _RIGHT_EYE_CORNERS, _RIGHT_EYE_PAIRS)
        right_width_penalty, right_width_valid = _minimum_projected_width_penalty(generated, reference, _RIGHT_EYE_CORNERS)
        right_confidence = right_confidence * right_width_valid.to(dtype=right_confidence.dtype)

        generated_left, left_confidence = _masked_openness(generated, visible_probability, _LEFT_EYE_CORNERS, _LEFT_EYE_PAIRS)
        reference_left, _ = _masked_openness(reference, visible_probability, _LEFT_EYE_CORNERS, _LEFT_EYE_PAIRS)
        left_width_penalty, left_width_valid = _minimum_projected_width_penalty(generated, reference, _LEFT_EYE_CORNERS)
        left_confidence = left_confidence * left_width_valid.to(dtype=left_confidence.dtype)

        losses = torch.cat(((generated_right - reference_right).abs(), (generated_left - reference_left).abs()))
        width_penalties = torch.cat((right_width_penalty, left_width_penalty))
        confidences = torch.cat((right_confidence, left_confidence))
        return _confidence_mean(losses + width_penalties, confidences)

    def _mouth_loss(self, generated: Tensor, reference: Tensor, visibility_probabilities: Tensor) -> Tensor:
        visible_probability = visibility_probabilities[..., HRFFAVisibility.VISIBLE.value]
        generated_open, confidence = _masked_openness(generated, visible_probability, _MOUTH_CORNERS, _MOUTH_PAIRS)
        reference_open, _ = _masked_openness(reference, visible_probability, _MOUTH_CORNERS, _MOUTH_PAIRS)
        width_penalty, width_valid = _minimum_projected_width_penalty(generated, reference, _MOUTH_CORNERS)
        confidence = confidence * width_valid.to(dtype=confidence.dtype)
        return _confidence_mean((generated_open - reference_open).abs() + width_penalty, confidence)

    def _contour_loss(self, generated: Tensor, reference: Tensor, visibility_probabilities: Tensor) -> Tensor:
        generated_contour = generated[:, _CONTOUR]
        reference_contour = reference[:, _CONTOUR]
        weights = self._geometry_visibility_weights(visibility_probabilities)[:, _CONTOUR]

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

    def _landmark_components(self, generated: Tensor, reference: Tensor, visibility_probabilities: Tensor) -> dict[str, Tensor]:
        """在归一化 WFLW98 坐标上以 FP32 计算各项加权几何约束。"""
        # Trainer 的 Generator step 可能处于 BF16 autocast。HRFFA 网络本身可以由
        # autocast 加速，但 linalg.solve/eigvalsh 等几何求解固定使用 FP32，避免
        # autocast 将矩阵乘降为 BF16 后形成混合 dtype 或数值不稳定。
        with torch.autocast(device_type=generated.device.type, enabled=False):
            generated = generated.float()
            reference = reference.float()
            zero = generated.new_zeros(())
            return {
                "pose": self._pose_loss(generated, reference, visibility_probabilities) * self.pose_weight if self.pose_weight > 0.0 else zero,
                "eye": self._eye_loss(generated, reference, visibility_probabilities) * self.eye_weight if self.eye_weight > 0.0 else zero,
                "mouth": self._mouth_loss(generated, reference, visibility_probabilities) * self.mouth_weight if self.mouth_weight > 0.0 else zero,
                "contour": self._contour_loss(generated, reference, visibility_probabilities) * self.contour_weight if self.contour_weight > 0.0 else zero,
            }

    def _extract_landmarks(self, generated: Tensor, reference: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if generated.shape != reference.shape:
            raise ValueError(f"generated/reference shape 必须一致：{tuple(generated.shape)} != {tuple(reference.shape)}")
        if generated.ndim != 4 or generated.shape[1] != 3 or generated.shape[2] != generated.shape[3]:
            raise ValueError(f"HRFFA loss 需要 NCHW 正方形 RGB 输入，实际 shape={tuple(generated.shape)}")

        with torch.no_grad():
            reference_points, reference_visibility_logits = self.hrffa(reference)
            reference_visibility_probabilities = reference_visibility_logits.float().softmax(dim=-1)
        generated_points, _ = self.hrffa(generated)

        # visibility 只来自 reference 分支并已 detach；Generator 不能通过改变 fake 的
        # visibility 预测来降低自己的监督权重。坐标允许落在 [0, 1] 外，不做 clamp。
        scale = float(generated.shape[-1])
        return generated_points.float() / scale, reference_points.float() / scale, reference_visibility_probabilities

    def forward_components(self, generated: Tensor, reference: Tensor) -> dict[str, Tensor]:
        """一次 HRFFA 前向后返回 pose/eye/mouth/contour 四个已加权标量损失。"""
        generated_points, reference_points, reference_visibility_probabilities = self._extract_landmarks(generated, reference)
        return self._landmark_components(generated_points, reference_points, reference_visibility_probabilities)

    def forward(self, generated: Tensor, reference: Tensor) -> Tensor:
        """计算 generated 相对 target/reference 的面部几何一致性总损失。"""
        components = self.forward_components(generated, reference)
        return torch.stack(tuple(components.values())).sum()
