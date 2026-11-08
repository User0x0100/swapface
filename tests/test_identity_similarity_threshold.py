"""Per-sample smooth identity margin, legacy behavior and training config regression."""

import copy
import math
import tomllib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from losses.identity import MIN_IDENTITY_TRANSITION_WIDTH, IdentityLoss
from swapface.config import resolve_train_config
from swapface.train import Trainer


@pytest.fixture
def identity_criterion(monkeypatch):
    # Avoid any model download; this test concerns embeddings -> scalar only.
    monkeypatch.setattr("losses.identity.IDEncoder", lambda provider: nn.Identity())

    def make(*, weight=10.0, min_similarity=0.7, width=0.03, reduction="none"):
        return IdentityLoss(
            weight=weight,
            min_similarity=min_similarity,
            transition_width=width,
            reduction=reduction,
        )

    return make


def _vectors(similarities: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor(similarities, dtype=torch.float32)
    generated = torch.stack((values, (1.0 - values.square()).sqrt()), dim=1).requires_grad_(True)
    reference = torch.tensor([[1.0, 0.0]], dtype=torch.float32).expand(len(similarities), -1)
    return generated, reference


def test_per_sample_threshold_exact_values_and_gradients(identity_criterion) -> None:
    generated, reference = _vectors([0.9, 0.69, 0.4])
    criterion = identity_criterion()
    values = criterion(generated, reference)
    assert values.shape == (3,)
    expected = torch.tensor([0.0, 10.0 * (0.01**2) / (2.0 * 0.03), 10.0 * (0.3 - 0.015)])
    torch.testing.assert_close(values.detach(), expected, atol=3e-6, rtol=1e-5)
    values.mean().backward()
    assert torch.count_nonzero(generated.grad[0]) == 0
    assert generated.grad[1].abs().sum() > 0
    assert generated.grad[2].abs().sum() > 0


def test_transition_is_smooth_and_keeps_full_gradient_outside_band(identity_criterion) -> None:
    similarities = [0.7001, 0.7, 0.6999, 0.685, 0.6701, 0.65]
    generated, reference = _vectors(similarities)
    smooth = identity_criterion()
    legacy = identity_criterion(min_similarity=None)
    expected = [0.0, 0.0, 0.0001 / 0.03, 0.5, 0.0299 / 0.03, 1.0]

    for index, ratio in enumerate(expected):
        gradient_smooth = torch.autograd.grad(
            smooth(generated[index : index + 1], reference[index : index + 1]).sum(),
            generated,
            retain_graph=True,
        )[0][index]
        gradient_legacy = torch.autograd.grad(
            legacy(generated[index : index + 1], reference[index : index + 1]).sum(),
            generated,
            retain_graph=True,
        )[0][index]
        assert gradient_legacy.norm() > 0
        measured = (gradient_smooth.norm() / gradient_legacy.norm()).item()
        assert measured == pytest.approx(ratio, abs=2e-3)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_reductions_and_gradient_isolation(identity_criterion, reduction: str) -> None:
    generated, reference = _vectors([0.9, 0.4])
    criterion = identity_criterion(weight=3.0, reduction=reduction)
    result = criterion(generated, reference)
    assert result.shape == ((2,) if reduction == "none" else ())
    if reduction == "mean":
        result.backward()
    else:
        result.sum().backward()
    assert generated.grad[0].abs().sum() == 0
    assert generated.grad[1].abs().sum() > 0

    # Other losses must keep their independent gradients on satisfied samples.
    generated2, reference2 = _vectors([0.9, 0.4])
    (criterion(generated2, reference2).sum() + generated2.square().sum()).backward()
    torch.testing.assert_close(generated2.grad[0], 2.0 * generated2.detach()[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_threshold_computes_fp32_under_low_precision(identity_criterion, dtype) -> None:
    generated, reference = _vectors([0.9, 0.69, 0.4])
    generated = generated.detach().to(dtype).requires_grad_(True)
    reference = reference.to(dtype)
    result = identity_criterion()(generated, reference)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert torch.count_nonzero(generated.grad[0]) == 0
    assert torch.isfinite(generated.grad).all()
    assert generated.grad[1].abs().sum() > 0


def test_legacy_formula_unchanged(identity_criterion) -> None:
    generated, reference = _vectors([0.95, 0.3])
    for reduction in ("none", "mean", "sum"):
        criterion = identity_criterion(min_similarity=None, reduction=reduction, weight=7.5)
        expected = (1.0 - F.cosine_similarity(generated, reference, dim=1)) * 7.5
        if reduction == "mean":
            expected = expected.mean()
        if reduction == "sum":
            expected = expected.sum()
        torch.testing.assert_close(criterion(generated, reference), expected, rtol=0, atol=0)


def test_identity_threshold_validation_and_legacy_config(identity_criterion) -> None:
    for wrong in (-1.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="min_similarity"):
            identity_criterion(min_similarity=wrong)
    for wrong in (0.0, -0.01, 1e-50, MIN_IDENTITY_TRANSITION_WIDTH / 2, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="transition_width"):
            identity_criterion(width=wrong)

    source = Path(__file__).resolve().parents[1] / "experiments" / "train.toml"
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    # An original run without the opt-in fields must remain canonical as before.
    legacy = copy.deepcopy(raw)
    for stage in ("coarse", "hq"):
        legacy["loss"][stage]["identity"].pop("min_similarity", None)
        legacy["loss"][stage]["identity"].pop("transition_width", None)
    normalized = resolve_train_config(legacy)
    for stage in ("coarse", "hq"):
        assert "min_similarity" not in normalized["loss"][stage]["identity"]
        assert "transition_width" not in normalized["loss"][stage]["identity"]

    new = copy.deepcopy(legacy)
    new["loss"]["coarse"]["identity"]["min_similarity"] = 0.7
    new["loss"]["hq"]["identity"].update({"min_similarity": 0.8, "transition_width": 0.04})
    normalized = resolve_train_config(new)
    assert normalized["loss"]["coarse"]["identity"]["transition_width"] == pytest.approx(0.03)
    assert normalized["loss"]["hq"]["identity"]["min_similarity"] == pytest.approx(0.8)
    assert normalized["loss"]["hq"]["identity"]["transition_width"] == pytest.approx(0.04)
    assert resolve_train_config(normalized) == normalized

    for key, value in (
        ("min_similarity", -2.0),
        ("min_similarity", 1.1),
        ("min_similarity", True),
        ("min_similarity", math.nan),
        ("transition_width", -1.0),
        ("transition_width", 0.0),
        ("transition_width", 1e-50),
        ("transition_width", MIN_IDENTITY_TRANSITION_WIDTH / 2),
        ("transition_width", math.inf),
    ):
        invalid = copy.deepcopy(new)
        invalid["loss"]["coarse"]["identity"][key] = value
        with pytest.raises((ValueError, TypeError)):
            resolve_train_config(invalid)

    missing_threshold = copy.deepcopy(legacy)
    missing_threshold["loss"]["coarse"]["identity"]["transition_width"] = 0.03
    with pytest.raises(ValueError, match="min_similarity"):
        resolve_train_config(missing_threshold)


def test_trainer_logs_threshold_coverage_without_grad(identity_criterion) -> None:
    criterion = identity_criterion()
    generated, reference = _vectors([0.9, 0.69, 0.4])
    trainer = Trainer.__new__(Trainer)
    trainer._completed_step = 9
    trainer.log_interval = 10
    metrics = {}
    trainer.log = lambda key, value: metrics.update({key: value.detach()})
    trainer._log_identity_threshold_metrics("coarse", criterion, generated, reference)

    assert metrics["coarse_id_similarity_mean"].item() == pytest.approx(sum([0.9, 0.69, 0.4]) / 3, abs=1e-6)
    assert metrics["coarse_id_satisfied_fraction"].item() == pytest.approx(1 / 3)
    assert metrics["coarse_id_gradient_scale_mean"].item() == pytest.approx((0 + 1 / 3 + 1) / 3, abs=1e-5)
    assert not any(value.requires_grad for value in metrics.values())
    metrics.clear()
    trainer._log_identity_threshold_metrics("hq", identity_criterion(min_similarity=None), generated, reference)
    assert metrics == {}


def test_threshold_identity_loss_fullgraph_compile(identity_criterion) -> None:
    criterion = identity_criterion(reduction="mean")
    compiled = torch.compile(criterion, backend="eager", fullgraph=True, dynamic=False)
    generated, reference = _vectors([0.9, 0.69, 0.4])
    loss = compiled(generated, reference)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(generated.grad[0]) == 0
    assert generated.grad[1].abs().sum() > 0
    assert generated.grad[2].abs().sum() > 0


def test_distinct_stage_thresholds_independent(identity_criterion) -> None:
    generated, reference = _vectors([0.65])
    coarse = identity_criterion(min_similarity=0.6, reduction="mean")
    hq = identity_criterion(min_similarity=0.7, reduction="mean")
    assert coarse(generated, reference).item() == pytest.approx(0, abs=1e-8)
    assert hq(generated, reference).item() > 0


def test_minimum_transition_width_has_finite_backward(identity_criterion) -> None:
    criterion = identity_criterion(width=MIN_IDENTITY_TRANSITION_WIDTH, reduction="mean")
    generated, reference = _vectors([0.9, 0.6])
    loss = criterion(generated, reference)
    grad = torch.autograd.grad(loss, generated)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad[0]) == 0
    assert grad[1].abs().sum() > 0


def test_threshold_logging_skips_all_similarity_ops_between_intervals(identity_criterion, monkeypatch) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer._completed_step = 0
    trainer.log_interval = 10
    events = []
    trainer.log = lambda key, value: events.append(key)
    criterion = identity_criterion()

    def reject(*args, **kwargs):
        raise AssertionError("cosine similarity must not be computed on an unlogged step")

    monkeypatch.setattr("swapface.train.NF.cosine_similarity", reject)
    generated, reference = _vectors([0.9, 0.4])
    trainer._log_identity_threshold_metrics("coarse", criterion, generated, reference)
    trainer._log_identity_threshold_metrics("hq", criterion, generated, reference)
    assert not events

    # On the interval boundary, the guard must no longer skip computation.
    trainer._completed_step = 9
    with pytest.raises(AssertionError, match="must not be computed"):
        trainer._log_identity_threshold_metrics("coarse", criterion, generated, reference)
