"""HRFFA 面部几何损失的纯几何回归检查。"""

from __future__ import annotations

import math

import torch
from torch import nn

from misc.models.hrffa import HRFFAVisibility

from .hrffa import _POSE_INDICES, _POSE_TEMPLATE, HRFFAFacialGeometryLoss, _confidence_mean


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


def test_occluded_eye_point_is_excluded() -> None:
    loss = _make_loss_stub()
    reference = torch.zeros(1, 98, 2)
    _fill_eye(reference, 60)
    _fill_eye(reference, 68)
    generated = reference.clone()
    generated[0, 61] = torch.tensor((0.2, -5.0))

    visibility = torch.full((1, 98), HRFFAVisibility.VISIBLE.value, dtype=torch.long)
    visibility[0, 61] = HRFFAVisibility.OCCLUDED.value
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

    visibility = torch.full((1, 98), HRFFAVisibility.VISIBLE.value, dtype=torch.long)
    visibility[0, 89] = HRFFAVisibility.OCCLUDED.value
    actual = loss._mouth_loss(generated, reference, visibility)
    assert torch.allclose(actual, torch.zeros_like(actual)), actual


def test_degenerate_pose_is_masked() -> None:
    loss = _make_loss_stub()
    points = _project_pose(torch.eye(3))
    for visible_count in range(6):
        visibility = torch.full((1, 98), HRFFAVisibility.OUTSIDE_IMAGE.value, dtype=torch.long)
        visibility[0, list(_POSE_INDICES[:visible_count])] = HRFFAVisibility.VISIBLE.value
        actual = loss._pose_loss(points, points, visibility)
        assert torch.allclose(actual, torch.zeros_like(actual)), (visible_count, actual)


def test_collapsed_generated_pose_is_masked_without_gradient_spike() -> None:
    loss = _make_loss_stub()
    reference = _project_pose(_rotation(30.0, -15.0, 20.0))
    generated = reference.clone()
    generated[0, list(_POSE_INDICES)] = 0.0
    generated.requires_grad_(True)
    visibility = torch.full((1, 98), HRFFAVisibility.VISIBLE.value, dtype=torch.long)

    actual = loss._pose_loss(generated, reference, visibility)
    actual.backward()

    assert torch.allclose(actual, torch.zeros_like(actual)), actual
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert torch.count_nonzero(generated.grad) == 0, generated.grad.abs().max()


def test_collapsed_openness_width_has_bounded_recovery_gradient() -> None:
    loss = _make_loss_stub()
    visibility = torch.full((1, 98), HRFFAVisibility.VISIBLE.value, dtype=torch.long)

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


def test_openness_width_floor_does_not_match_target_width() -> None:
    loss = _make_loss_stub()
    visibility = torch.full((1, 98), HRFFAVisibility.VISIBLE.value, dtype=torch.long)

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
        actual, valid = loss._fit_rotation(points, weights)
        assert bool(valid.item()), angles
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

    visibility = torch.full((1, 98), HRFFAVisibility.VISIBLE.value, dtype=torch.long)
    actual = loss._pose_loss(generated, reference, visibility)
    assert torch.allclose(actual, torch.zeros_like(actual), atol=1e-6), actual


def test_confidence_reduces_absolute_loss_strength() -> None:
    values = torch.ones(4)
    assert torch.allclose(_confidence_mean(values, torch.ones(4)), torch.tensor(1.0))
    assert torch.allclose(_confidence_mean(values, torch.full((4,), 0.5)), torch.tensor(0.5))
    assert torch.allclose(_confidence_mean(values, torch.tensor((1.0, 0.0, 0.0, 0.0))), torch.tensor(0.25))


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
    visibility = torch.full((2, 98), HRFFAVisibility.VISIBLE.value, dtype=torch.long)
    components = loss._landmark_components(points, points, visibility)
    assert all(torch.equal(component, torch.zeros_like(component)) for component in components.values())


def test_geometry_is_fp32_under_bf16_autocast() -> None:
    if not torch.cuda.is_available():
        return

    loss = _make_loss_stub().cuda()
    reference = _project_pose(_rotation(30.0, -15.0, 20.0)).cuda()
    generated = _project_pose(_rotation(35.0, -10.0, 15.0)).cuda().requires_grad_(True)
    visibility = torch.full((1, 98), HRFFAVisibility.VISIBLE.value, dtype=torch.long, device="cuda")

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
    test_degenerate_pose_is_masked()
    test_collapsed_generated_pose_is_masked_without_gradient_spike()
    test_collapsed_openness_width_has_bounded_recovery_gradient()
    test_openness_width_floor_does_not_match_target_width()
    test_pose_fit_survives_large_angles()
    test_pose_ignores_mouth_and_jaw_motion()
    test_confidence_reduces_absolute_loss_strength()
    test_zero_weight_skips_component_computation()
    test_geometry_is_fp32_under_bf16_autocast()
    print("PASS: HRFFA visibility, pose/openness degeneracy, BF16 geometry, confidence reduction and ablation skipping")


if __name__ == "__main__":
    main()
