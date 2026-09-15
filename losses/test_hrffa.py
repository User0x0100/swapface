"""HRFFA 面部几何损失的纯几何回归检查。"""

from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import nn

from misc.models.hrffa import HRFFAVisibility

from .hrffa import _POSE_INDICES, _POSE_TEMPLATE, _RIGHT_EYE_CORNERS, _RIGHT_EYE_PAIRS, HRFFAFacialGeometryLoss, _confidence_mean, _masked_openness


def _make_loss_stub() -> HRFFAFacialGeometryLoss:
    """构造不加载 HRFFA 权重、只测试几何层的实例。"""
    loss = HRFFAFacialGeometryLoss.__new__(HRFFAFacialGeometryLoss)
    nn.Module.__init__(loss)  # noqa: PLC2801 - 绕过重模型初始化，仅构造几何单元测试实例。
    loss.pose_weight = 1.0
    loss.eye_weight = 1.0
    loss.mouth_weight = 1.0
    loss.contour_weight = 1.0
    loss.contour_shape_weight = 0.5
    loss.occluded_geometry_weight = 0.25
    loss.register_buffer("pose_template", _POSE_TEMPLATE.clone(), persistent=False)
    return loss


def _rotation(yaw: float, pitch: float, roll: float) -> torch.Tensor:
    yaw, pitch, roll = map(math.radians, (yaw, pitch, roll))
    ry = torch.tensor(
        (
            (math.cos(yaw), 0.0, math.sin(yaw)),
            (0.0, 1.0, 0.0),
            (-math.sin(yaw), 0.0, math.cos(yaw)),
        ),
        dtype=torch.float32,
    )
    rx = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(pitch), -math.sin(pitch)),
            (0.0, math.sin(pitch), math.cos(pitch)),
        ),
        dtype=torch.float32,
    )
    rz = torch.tensor(
        (
            (math.cos(roll), -math.sin(roll), 0.0),
            (math.sin(roll), math.cos(roll), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=torch.float32,
    )
    return rz @ ry @ rx


def _project_pose(rotation: torch.Tensor) -> torch.Tensor:
    points = torch.zeros(1, 98, 2, dtype=torch.float32)
    projected = _POSE_TEMPLATE @ rotation[:2].T
    points[0, list(_POSE_INDICES), 0] = projected[:, 0]
    points[0, list(_POSE_INDICES), 1] = -projected[:, 1]
    return points


def _fill_eye(points: torch.Tensor, start: int) -> None:
    points[0, start] = torch.tensor((0.0, 0.0))
    points[0, start + 4] = torch.tensor((1.0, 0.0))
    for upper, lower, x in ((start + 1, start + 7, 0.2), (start + 2, start + 6, 0.5), (start + 3, start + 5, 0.8)):
        points[0, upper] = torch.tensor((x, -0.1))
        points[0, lower] = torch.tensor((x, 0.1))


def _visibility_probabilities(state: HRFFAVisibility = HRFFAVisibility.VISIBLE) -> torch.Tensor:
    probabilities = torch.zeros(1, 98, 3, dtype=torch.float32)
    probabilities[..., state.value] = 1.0
    return probabilities


def _set_visibility_state(probabilities: torch.Tensor, index: int, state: HRFFAVisibility) -> None:
    probabilities[:, index] = 0.0
    probabilities[:, index, state.value] = 1.0


def test_occluded_eye_point_is_excluded() -> None:
    loss = _make_loss_stub()
    reference = torch.zeros(1, 98, 2)
    _fill_eye(reference, 60)
    _fill_eye(reference, 68)
    generated = reference.clone()
    generated[0, 61] = torch.tensor((0.2, -5.0))

    visibility = _visibility_probabilities()
    _set_visibility_state(visibility, 61, HRFFAVisibility.OCCLUDED)
    actual = loss._eye_loss(generated, reference, visibility)
    assert torch.allclose(actual, torch.zeros_like(actual)), actual


def test_occluded_mouth_point_is_excluded() -> None:
    loss = _make_loss_stub()
    reference = torch.zeros(1, 98, 2)
    reference[0, 88] = torch.tensor((0.0, 0.0))
    reference[0, 92] = torch.tensor((1.0, 0.0))
    for upper, lower, x in ((89, 95, 0.2), (90, 94, 0.5), (91, 93, 0.8)):
        reference[0, upper] = torch.tensor((x, -0.1))
        reference[0, lower] = torch.tensor((x, 0.1))
    generated = reference.clone()
    generated[0, 89] = torch.tensor((0.2, -5.0))

    visibility = _visibility_probabilities()
    _set_visibility_state(visibility, 89, HRFFAVisibility.OCCLUDED)
    actual = loss._mouth_loss(generated, reference, visibility)
    assert torch.allclose(actual, torch.zeros_like(actual)), actual


def test_outside_pose_is_masked() -> None:
    loss = _make_loss_stub()
    points = _project_pose(torch.eye(3))
    visibility = _visibility_probabilities(HRFFAVisibility.OUTSIDE_IMAGE)
    actual = loss._pose_loss(points, points, visibility)
    assert torch.allclose(actual, torch.zeros_like(actual)), actual


def test_pose_retains_supervision_with_occluded_points() -> None:
    loss = _make_loss_stub()
    reference = _project_pose(_rotation(65.0, 10.0, 5.0))
    generated = _project_pose(_rotation(50.0, 10.0, 5.0))
    visibility = _visibility_probabilities(HRFFAVisibility.OUTSIDE_IMAGE)
    for index in _POSE_INDICES[:4]:
        _set_visibility_state(visibility, index, HRFFAVisibility.VISIBLE)
    for index in _POSE_INDICES[4:]:
        _set_visibility_state(visibility, index, HRFFAVisibility.OCCLUDED)

    actual = loss._pose_loss(generated, reference, visibility)
    assert actual > 0.0, actual


def test_pose_fit_is_invariant_to_uniform_visibility_scale() -> None:
    loss = _make_loss_stub()
    reference = _project_pose(_rotation(65.0, 10.0, 5.0))
    generated = _project_pose(_rotation(50.0, 10.0, 5.0))

    normalized_losses = []
    for scale in (1.0, 0.1, 0.01, 1e-5):
        visibility = torch.zeros(1, 98, 3)
        visibility[..., HRFFAVisibility.OUTSIDE_IMAGE.value] = 1.0 - scale
        visibility[..., HRFFAVisibility.VISIBLE.value] = scale
        actual = loss._pose_loss(generated, reference, visibility)
        normalized_losses.append(actual / scale)

    for normalized_loss in normalized_losses[1:]:
        torch.testing.assert_close(normalized_loss, normalized_losses[0], atol=1e-6, rtol=1e-5)


def test_sparse_pose_support_suppresses_residual_probabilities() -> None:
    loss = _make_loss_stub()
    reference = _project_pose(_rotation(65.0, 10.0, 5.0))
    generated = _project_pose(_rotation(50.0, 10.0, 5.0))

    for strong_count in (1, 2):
        visibility = torch.zeros(1, 98, 3)
        visibility[..., HRFFAVisibility.OUTSIDE_IMAGE.value] = 0.999
        visibility[..., HRFFAVisibility.OCCLUDED.value] = 0.0005
        visibility[..., HRFFAVisibility.VISIBLE.value] = 0.0005
        for index in _POSE_INDICES[:strong_count]:
            visibility[:, index] = 0.0
            visibility[:, index, HRFFAVisibility.VISIBLE.value] = 1.0

        corrupted = generated.clone()
        for index in _POSE_INDICES[strong_count:]:
            corrupted[:, index] += torch.tensor((1.5, -1.5))

        actual = loss._pose_loss(corrupted, reference, visibility)
        point_weights = loss._geometry_visibility_weights(visibility)[:, _POSE_INDICES]
        pose_confidence = point_weights.mean(dim=1)
        squared_weight_sum = point_weights.square().sum(dim=1)
        effective_points = point_weights.sum(dim=1).square() / squared_weight_sum.clamp_min(
            torch.finfo(squared_weight_sum.dtype).tiny
        )
        pose_support = (effective_points / 6.0).clamp(max=1.0).square()
        maximum = 2.0 * pose_confidence * pose_support

        assert 0.0 < actual < maximum + 1e-7, (strong_count, actual, maximum)



def test_pose_conditioning_is_continuous_near_old_rank_threshold() -> None:
    loss = _make_loss_stub()
    reference = _project_pose(_rotation(65.0, 10.0, 5.0))
    generated = _project_pose(_rotation(50.0, 10.0, 5.0))

    losses = []
    for probability in (0.05, 0.06, 0.07, 0.08, 0.09, 0.10):
        visibility = _visibility_probabilities(HRFFAVisibility.OUTSIDE_IMAGE)
        for index in _POSE_INDICES[:6]:
            _set_visibility_state(visibility, index, HRFFAVisibility.VISIBLE)
        seventh = _POSE_INDICES[6]
        visibility[:, seventh, HRFFAVisibility.OUTSIDE_IMAGE.value] = 1.0 - probability
        visibility[:, seventh, HRFFAVisibility.VISIBLE.value] = probability

        actual = loss._pose_loss(generated, reference, visibility)
        assert actual > 0.0, (probability, actual)
        losses.append(actual)

    # 旧实现会在约 0.08~0.10 附近从严格 0 突然跳到约 2e-2。
    # 连续 conditioning 后相邻小幅 visibility 变化不应产生这种开关式跳变。
    jumps = torch.stack([torch.abs(right - left) for left, right in pairwise(losses)])
    assert jumps.max() < 0.01, jumps

def test_collapsed_generated_pose_is_masked_without_gradient_spike() -> None:
    loss = _make_loss_stub()
    reference = _project_pose(_rotation(30.0, -15.0, 20.0))
    generated = reference.clone()
    generated[0, list(_POSE_INDICES)] = 0.0
    generated.requires_grad_(True)
    visibility = _visibility_probabilities()

    actual = loss._pose_loss(generated, reference, visibility)
    actual.backward()

    assert torch.allclose(actual, torch.zeros_like(actual)), actual
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert torch.count_nonzero(generated.grad) == 0, generated.grad.abs().max()


def test_collapsed_openness_width_has_bounded_recovery_gradient() -> None:
    loss = _make_loss_stub()
    visibility = _visibility_probabilities()

    eye_reference = torch.zeros(1, 98, 2)
    _fill_eye(eye_reference, 60)
    eye_generated = eye_reference.clone()
    eye_generated[0, 64] = eye_generated[0, 60]
    eye_generated.requires_grad_(True)
    eye_loss = loss._eye_loss(eye_generated, eye_reference, visibility)
    eye_loss.backward()
    assert eye_loss > 0
    assert eye_generated.grad is not None and torch.isfinite(eye_generated.grad).all()
    assert 0.0 < eye_generated.grad[0, 60, 0] < 100.0
    assert -100.0 < eye_generated.grad[0, 64, 0] < 0.0
    assert eye_generated.grad.abs().max() < 100.0

    mouth_reference = torch.zeros(1, 98, 2)
    mouth_reference[0, 88] = torch.tensor((0.0, 0.0))
    mouth_reference[0, 92] = torch.tensor((1.0, 0.0))
    for upper, lower, x in ((89, 95, 0.2), (90, 94, 0.5), (91, 93, 0.8)):
        mouth_reference[0, upper] = torch.tensor((x, -0.1))
        mouth_reference[0, lower] = torch.tensor((x, 0.1))
    mouth_generated = mouth_reference.clone()
    mouth_generated[0, 92] = mouth_generated[0, 88]
    mouth_generated.requires_grad_(True)
    mouth_loss = loss._mouth_loss(mouth_generated, mouth_reference, visibility)
    mouth_loss.backward()
    assert mouth_loss > 0
    assert mouth_generated.grad is not None and torch.isfinite(mouth_generated.grad).all()
    assert 0.0 < mouth_generated.grad[0, 88, 0] < 100.0
    assert -100.0 < mouth_generated.grad[0, 92, 0] < 0.0
    assert mouth_generated.grad.abs().max() < 100.0


def test_openness_confidence_decays_with_pair_support() -> None:
    points = torch.zeros(1, 98, 2)
    _fill_eye(points, 60)

    expected = (0.1, 2.0 / 3.0, 1.0)
    for pair_count, expected_confidence in enumerate(expected, start=1):
        visible_probability = torch.zeros(1, 98)
        visible_probability[:, 60] = 1.0
        visible_probability[:, 64] = 1.0
        for upper, lower in ((61, 67), (62, 66), (63, 65))[:pair_count]:
            visible_probability[:, upper] = 1.0
            visible_probability[:, lower] = 1.0

        _, confidence = _masked_openness(points, visible_probability, _RIGHT_EYE_CORNERS, _RIGHT_EYE_PAIRS)
        torch.testing.assert_close(confidence, torch.tensor([expected_confidence]), atol=1e-6, rtol=0.0)


def test_openness_width_floor_does_not_match_target_width() -> None:
    loss = _make_loss_stub()
    visibility = _visibility_probabilities()

    reference = torch.zeros(1, 98, 2)
    reference[0, 60] = torch.tensor((0.0, 0.0))
    reference[0, 64] = torch.tensor((0.1, 0.0))
    for upper, lower, x in ((61, 67, 0.02), (62, 66, 0.05), (63, 65, 0.08)):
        reference[0, upper] = torch.tensor((x, -0.01))
        reference[0, lower] = torch.tensor((x, 0.01))

    generated = reference.clone()
    generated[0, 60:68] *= 0.3
    actual = loss._eye_loss(generated, reference, visibility)
    assert torch.allclose(actual, torch.zeros_like(actual), atol=1e-7), actual


def test_pose_fit_survives_large_angles() -> None:
    loss = _make_loss_stub()
    weights = torch.zeros(1, 98)
    weights[0, list(_POSE_INDICES)] = 1.0

    for angles in ((0.0, 0.0, 0.0), (75.0, 0.0, 0.0), (-75.0, 0.0, 0.0), (0.0, 45.0, 0.0), (0.0, -45.0, 0.0), (60.0, 30.0, 25.0), (-60.0, -30.0, -90.0)):
        expected = _rotation(*angles)
        points = _project_pose(expected)
        actual, valid, condition_confidence = loss._fit_rotation(points, weights)
        assert bool(valid.item()), angles
        assert 0.0 < condition_confidence.item() <= 1.0, (angles, condition_confidence)
        relative = actual[0] @ expected.T
        pose_error = (3.0 - torch.trace(relative)) * 0.5
        assert pose_error < 1e-4, (angles, pose_error)


def test_pose_ignores_mouth_and_jaw_motion() -> None:
    loss = _make_loss_stub()
    reference = _project_pose(_rotation(35.0, -15.0, 10.0))
    generated = reference.clone()
    generated[0, 76] += torch.tensor((0.0, -0.25))
    generated[0, 82] += torch.tensor((0.0, -0.25))
    generated[0, 16] += torch.tensor((0.0, 0.35))

    visibility = _visibility_probabilities()
    actual = loss._pose_loss(generated, reference, visibility)
    assert torch.allclose(actual, torch.zeros_like(actual), atol=1e-6), actual


def test_confidence_reduces_absolute_loss_strength() -> None:
    values = torch.ones(4)
    assert torch.allclose(_confidence_mean(values, torch.ones(4)), torch.tensor(1.0))
    assert torch.allclose(_confidence_mean(values, torch.full((4,), 0.5)), torch.tensor(0.5))
    assert torch.allclose(_confidence_mean(values, torch.tensor((1.0, 0.0, 0.0, 0.0))), torch.tensor(0.25))


def test_soft_visibility_weights_are_continuous() -> None:
    loss = _make_loss_stub()
    lower = torch.zeros(1, 98, 3)
    upper = torch.zeros(1, 98, 3)
    lower[..., HRFFAVisibility.VISIBLE.value] = 0.49
    lower[..., HRFFAVisibility.OCCLUDED.value] = 0.51
    upper[..., HRFFAVisibility.VISIBLE.value] = 0.51
    upper[..., HRFFAVisibility.OCCLUDED.value] = 0.49

    lower_weight = loss._geometry_visibility_weights(lower)
    upper_weight = loss._geometry_visibility_weights(upper)
    assert torch.allclose(lower_weight, torch.full_like(lower_weight, 0.49 + 0.51 * 0.25))
    assert torch.allclose(upper_weight, torch.full_like(upper_weight, 0.51 + 0.49 * 0.25))
    assert (upper_weight - lower_weight).abs().max() < 0.02


def test_zero_weight_skips_component_computation() -> None:
    loss = _make_loss_stub()
    loss.pose_weight = 0.0
    loss.eye_weight = 0.0
    loss.mouth_weight = 0.0
    loss.contour_weight = 0.0

    def fail(*_args, **_kwargs):
        raise AssertionError("weight=0 的几何分量不应执行")

    loss._pose_loss = fail
    loss._eye_loss = fail
    loss._mouth_loss = fail
    loss._contour_loss = fail

    points = torch.zeros(2, 98, 2)
    visibility = _visibility_probabilities().expand(2, -1, -1).clone()
    components = loss._landmark_components(points, points, visibility)
    assert all(torch.equal(component, torch.zeros_like(component)) for component in components.values())


def test_geometry_is_fp32_under_bf16_autocast() -> None:
    if not torch.cuda.is_available():
        return

    loss = _make_loss_stub().cuda()
    reference = _project_pose(_rotation(30.0, -15.0, 20.0)).cuda()
    generated = _project_pose(_rotation(35.0, -10.0, 15.0)).cuda().requires_grad_(True)
    visibility = _visibility_probabilities().cuda()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        components = loss._landmark_components(generated, reference, visibility)
        total = torch.stack(tuple(components.values())).sum()

    assert all(component.dtype == torch.float32 for component in components.values())
    assert torch.isfinite(total)
    total.backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()


def main() -> None:
    test_occluded_eye_point_is_excluded()
    test_occluded_mouth_point_is_excluded()
    test_outside_pose_is_masked()
    test_pose_retains_supervision_with_occluded_points()
    test_pose_fit_is_invariant_to_uniform_visibility_scale()
    test_sparse_pose_support_suppresses_residual_probabilities()
    test_pose_conditioning_is_continuous_near_old_rank_threshold()
    test_collapsed_generated_pose_is_masked_without_gradient_spike()
    test_collapsed_openness_width_has_bounded_recovery_gradient()
    test_openness_confidence_decays_with_pair_support()
    test_openness_width_floor_does_not_match_target_width()
    test_pose_fit_survives_large_angles()
    test_pose_ignores_mouth_and_jaw_motion()
    test_confidence_reduces_absolute_loss_strength()
    test_soft_visibility_weights_are_continuous()
    test_zero_weight_skips_component_computation()
    test_geometry_is_fp32_under_bf16_autocast()
    print("PASS: HRFFA soft visibility, large-pose coverage, pose/openness degeneracy, BF16 geometry and ablation skipping")


if __name__ == "__main__":
    main()
