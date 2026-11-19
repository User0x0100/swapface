"""Single-stage configuration, run, checkpoint, optimizer, and numeric regression tests.

Run: python -m swapface.test_experiment
"""

import copy
import json
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from torch import nn
from torch.amp import GradScaler

from swapface.config import DEFAULT_LOSS_CONFIG, load_train_config, resolve_train_config
from swapface.experiment import config_sha256
from swapface.train import (
    TRAINING_SEMANTICS_VERSION,
    Trainer,
    _canonical_generator_config,
    _format_sample_gradient_rows,
    _load_branch_checkpoint,
    _reduce_reconstruction_loss,
    _require_training_config,
    _scaled_backward_step,
)


def _config() -> dict:
    return tomllib.loads((Path(__file__).parents[1] / "experiments/train.toml").read_text())


def test_canonical_config_and_stage_removal() -> None:
    config = _config()
    normalized = resolve_train_config(config)
    assert resolve_train_config(normalized) == normalized
    assert json.loads(json.dumps(normalized)) == normalized
    assert normalized["generator"]["type"] == "inswapper"
    assert normalized["train"]["precision"] == "bf16"
    assert "stage" not in normalized["train"]
    assert set(normalized["optimizer"]) == {"generator", "discriminator"}
    assert set(normalized["scheduler"]) == {"generator", "discriminator"}
    assert "gan" in normalized["loss"] and "identity" in normalized["loss"]
    assert "mask" not in normalized["loss"]
    assert "type" in normalized["discriminator"]
    assert _canonical_generator_config(normalized["generator"]) == normalized["generator"]
    assert load_train_config(Path(__file__).parents[1] / "experiments/train.toml") == normalized
    assert len(config_sha256(normalized)) == 64
    assert DEFAULT_LOSS_CONFIG["identity"]["exclude_parsing_labels"] == []


@pytest.mark.parametrize("section, field, value", [
    ("train", "stage", "joint"),
    ("generator", "coarse_resolution", 128),
    ("generator", "hq_bottleneck_resolution", 16),
    ("optimizer", "hq_discriminator", {"lr": 1e-4}),
    ("scheduler", "coarse_discriminator", {}),
    ("loss", "hq", {}),
    ("discriminator", "coarse", {}),
])
def test_retired_stage_schema_is_rejected(section, field, value) -> None:
    raw = _config()
    raw[section][field] = value
    with pytest.raises((ValueError, TypeError)):
        resolve_train_config(raw)


def test_generator_resolution_discriminator_match_and_wfm() -> None:
    raw = _config()
    raw["generator"]["img_resolution"] = 128
    raw["generator"]["skip_layers"] = [0, 1]
    with pytest.raises(ValueError, match=r"discriminator\.img_resolution"):
        resolve_train_config(raw)
    raw["discriminator"]["img_resolution"] = 128
    normalized = resolve_train_config(raw)
    assert normalized["discriminator"]["img_resolution"] == 128
    raw["discriminator"] = {"type": "UNET", "img_resolution": 128, "base_ch": 64}
    normalized = resolve_train_config(raw)
    assert normalized["loss"]["r1"]["enable"] is False
    assert normalized["discriminator"]["type"] == "UNET"


def test_checkpoint_training_semantics() -> None:
    correct = {"training_config": {"semantics_version": TRAINING_SEMANTICS_VERSION, "precision": "bf16"}}
    assert _require_training_config(correct) == correct["training_config"]
    for invalid in (
        {"semantics_version": TRAINING_SEMANTICS_VERSION, "precision": "bf16", "stage": "coarse"},
        {"semantics_version": TRAINING_SEMANTICS_VERSION - 1, "precision": "bf16"},
        {"semantics_version": TRAINING_SEMANTICS_VERSION, "precision": "invalid"},
    ):
        with pytest.raises((ValueError, TypeError)):
            _require_training_config({"training_config": invalid})


def test_branch_accepts_different_metadata_and_defers_compatibility_to_weights(tmp_path) -> None:
    config = resolve_train_config(_config())
    ckpt = {
        "version": 1,
        "step": 3,
        "face_alignment": "arcface",
        "run": {"id": "run-id", "config_sha256": "digest"},
        "identity_encoders": {"generator": "SOME_OTHER_PROVIDER"},
        "training_state": {"net_g": {}},
        "net_g": {"network_cfg": {"type": "other"}, "state_dict": {}},
        "net_d": {"network_cfg": {"type": "other"}, "state_dict": {}},
    }
    path = tmp_path / "arbitrary_model_name.pth"
    torch.save(ckpt, path)
    loaded, parent = _load_branch_checkpoint(path, reset_discriminator=False)
    assert loaded["step"] == 3
    assert parent == {
        "run_id": "run-id", "checkpoint": path.name, "step": 3,
        "config_sha256": "digest", "branch_mode": "inherit",
        "generator": "inherit", "discriminator": "auto",
    }
    modified = copy.deepcopy(config)
    modified["generator"]["skip_layers"] = []
    modified["discriminator"]["base_ch"] = 32
    loaded, parent = _load_branch_checkpoint(path, reset_discriminator=True)
    assert parent["discriminator"] == "reset"
    assert loaded["net_g"]["network_cfg"] != modified["generator"]
    assert loaded["net_d"]["network_cfg"] != modified["discriminator"]



def test_reconstruction_scope() -> None:
    per_sample = torch.tensor([2., 4., 6.], requires_grad=True)
    mask = torch.tensor([True, False, True])
    torch.testing.assert_close(_reduce_reconstruction_loss(per_sample, mask, "same"), torch.tensor(4.))
    torch.testing.assert_close(_reduce_reconstruction_loss(per_sample, mask, "all"), torch.tensor(4.))
    with pytest.raises(ValueError, match="reconstruction_scope"):
        _reduce_reconstruction_loss(per_sample, mask, "invalid")
    _reduce_reconstruction_loss(per_sample, mask, "same").backward()
    torch.testing.assert_close(per_sample.grad, torch.tensor([.5, 0., .5]))


def test_gradient_map_presentation_is_finite() -> None:
    rows = [("GAN", torch.rand(2, 1, 8, 8)), ("ID", torch.rand(2, 1, 8, 8))]
    gradients = [torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)]
    for scale in ("shared", "independent"):
        output = _format_sample_gradient_rows(rows, gradients, heatmap_scale=scale)
        assert len(output) == 2
        for label, image in output:
            assert "RMS" in label and "DIR" in label and image.shape == (2, 3, 8, 8)
            assert torch.isfinite(image).all()


def test_optimizer_update_and_step_boundary() -> None:
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler("cpu", enabled=False)
    loss = model(torch.randn(2, 3)).square().mean()
    assert _scaled_backward_step(loss, optimizer, scaler, name="Generator", named_parameters=model.named_parameters())
    trainer = Trainer.__new__(Trainer)
    trainer._completed_step = 0
    trainer._step_in_progress = False
    assert not trainer.can_save_checkpoint
    trainer._completed_step = 1
    assert trainer.can_save_checkpoint
    trainer._step_in_progress = True
    assert not trainer.can_save_checkpoint


def main() -> None:
    import inspect
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    with TemporaryDirectory() as temp:
        for test in tests:
            if hasattr(test, "pytestmark") or getattr(test, "__wrapped__", None):
                continue
            params = inspect.signature(test).parameters
            if params:
                if list(params) == ["tmp_path"]: test(Path(temp))
                else: continue
            else:
                test()
    print("PASS: single-stage training configuration, branch, checkpoint, losses, optimizer")


if __name__ == "__main__":
    main()
