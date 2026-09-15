"""FACS AU 分组与归约逻辑的轻量回归检查。"""

from __future__ import annotations

import torch
from torch import nn

from misc.models.openface_au import MAIN_AU_CODES, SUB_AU_CODES, OpenFaceAU

from .facs import (
    _BROW_AUS,
    _EYE_AUS,
    _LOWER_FACE_AUS,
    _MOUTH_AUS,
    _NOSE_AUS,
    _NUM_ASYMMETRY_PAIRS,
    _NUM_FACS_SIGNALS,
    FACSConsistencyLoss,
    _main_indices,
)


def _make_loss_stub(**weights: float) -> FACSConsistencyLoss:
    loss = FACSConsistencyLoss.__new__(FACSConsistencyLoss)
    nn.Module.__init__(loss)  # noqa: PLC2801 - 绕过预训练模型初始化，只测试 AU 归约逻辑。
    loss.weight = weights.pop("weight", 1.0)
    loss.component_weights = {
        "brow": weights.pop("brow", 1.0),
        "eye": weights.pop("eye", 1.0),
        "nose": weights.pop("nose", 1.0),
        "mouth": weights.pop("mouth", 1.0),
        "lower_face": weights.pop("lower_face", 1.0),
        "asymmetry": weights.pop("asymmetry", 1.0),
    }
    assert not weights
    groups = {
        "brow": _main_indices(_BROW_AUS),
        "eye": _main_indices(_EYE_AUS),
        "nose": _main_indices(_NOSE_AUS),
        "mouth": _main_indices(_MOUTH_AUS),
        "lower_face": _main_indices(_LOWER_FACE_AUS),
    }
    for name, indices in groups.items():
        loss.register_buffer(f"{name}_indices", torch.tensor(indices, dtype=torch.long), persistent=False)
    return loss


def test_main_groups_cover_all_27_aus_exactly_once() -> None:
    groups = (_BROW_AUS, _EYE_AUS, _NOSE_AUS, _MOUTH_AUS, _LOWER_FACE_AUS)
    codes = tuple(code for group in groups for code in group)
    assert len(codes) == OpenFaceAU.NUM_MAIN_AUS
    assert set(codes) == set(MAIN_AU_CODES)
    assert len(set(codes)) == len(codes)


def test_sub_aus_are_seven_ordered_left_right_pairs() -> None:
    assert len(SUB_AU_CODES) == OpenFaceAU.NUM_SUB_AUS == _NUM_ASYMMETRY_PAIRS * 2
    for left, right in zip(SUB_AU_CODES[0::2], SUB_AU_CODES[1::2], strict=True):
        assert left.startswith("L") and right.startswith("R")
        assert left[1:] == right[1:]


def test_default_components_equal_mean_of_34_semantic_signals() -> None:
    loss = _make_loss_stub()
    generated = torch.linspace(0.0, 1.0, steps=2 * OpenFaceAU.NUM_AUS).reshape(2, OpenFaceAU.NUM_AUS)
    reference = torch.flip(generated, dims=(1,))

    components = loss._components(generated, reference)
    actual = torch.stack(tuple(components.values())).sum()

    main_error = (generated[:, : OpenFaceAU.NUM_MAIN_AUS] - reference[:, : OpenFaceAU.NUM_MAIN_AUS]).abs()
    generated_sub = generated[:, OpenFaceAU.NUM_MAIN_AUS :].reshape(2, _NUM_ASYMMETRY_PAIRS, 2)
    reference_sub = reference[:, OpenFaceAU.NUM_MAIN_AUS :].reshape(2, _NUM_ASYMMETRY_PAIRS, 2)
    asymmetry_error = ((generated_sub[..., 0] - generated_sub[..., 1]) - (reference_sub[..., 0] - reference_sub[..., 1])).abs().mul(0.5)
    expected = torch.cat((main_error, asymmetry_error), dim=1).sum(dim=1).mean() / _NUM_FACS_SIGNALS
    assert torch.allclose(actual, expected), (actual, expected)


def test_equal_bilateral_shift_has_zero_asymmetry_loss() -> None:
    loss = _make_loss_stub()
    reference = torch.zeros(1, OpenFaceAU.NUM_AUS)
    generated = reference.clone()
    reference[:, OpenFaceAU.NUM_MAIN_AUS :] = 0.2
    generated[:, OpenFaceAU.NUM_MAIN_AUS :] = 0.8

    components = loss._components(generated, reference)
    assert components["asymmetry"].item() == 0.0


def test_changed_left_right_difference_is_penalized() -> None:
    loss = _make_loss_stub()
    reference = torch.zeros(1, OpenFaceAU.NUM_AUS)
    generated = reference.clone()
    generated[:, OpenFaceAU.NUM_MAIN_AUS] = 0.5

    components = loss._components(generated, reference)
    expected = torch.tensor(0.25 / _NUM_FACS_SIGNALS)
    assert torch.allclose(components["asymmetry"], expected)


def test_component_weight_only_scales_its_region() -> None:
    generated = torch.ones(1, OpenFaceAU.NUM_AUS)
    reference = torch.zeros_like(generated)
    # 构造 bilateral 非对称，否则全 1/全 0 的左右差值都为 0。
    generated[:, OpenFaceAU.NUM_MAIN_AUS :: 2] = 0.5

    baseline = _make_loss_stub()._components(generated, reference)
    weighted = _make_loss_stub(mouth=2.5, asymmetry=1.75)._components(generated, reference)
    for name in baseline:
        factor = 2.5 if name == "mouth" else 1.75 if name == "asymmetry" else 1.0
        assert torch.allclose(weighted[name], baseline[name] * factor), name


def test_zero_component_weight_returns_zero() -> None:
    loss = _make_loss_stub(eye=0.0, asymmetry=0.0)
    generated = torch.ones(1, OpenFaceAU.NUM_AUS)
    reference = torch.zeros_like(generated)
    components = loss._components(generated, reference)
    assert components["eye"].item() == 0.0
    assert components["asymmetry"].item() == 0.0
