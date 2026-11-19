"""Single-stage run creation, frozen config, latest, resume and branch integration."""

import copy
import json
from pathlib import Path

import pytest

from swapface.config import load_train_config, resolve_train_config
from swapface.experiment import (
    RESOLVED_CONFIG_FORMAT_VERSION,
    RUN_CONFIG_FILE,
    RunLock,
    RunPaths,
    config_sha256,
    create_run,
    load_metadata,
    load_resolved_config,
    resolve_branch_target,
    resolve_resume_target,
    update_metadata,
    write_latest,
)
from swapface.train import _load_run_config


@pytest.fixture
def prepared_run(tmp_path: Path) -> tuple[RunPaths, dict]:
    source = tmp_path / "train.toml"
    original = Path(__file__).parents[1] / "experiments/train.toml"
    source.write_bytes(original.read_bytes())
    canonical = load_train_config(source)
    run = create_run(tmp_path / "runs", source, canonical, name="single-stage")
    return run, canonical


def test_run_freezes_source_and_canonical_config(prepared_run):
    paths, canonical = prepared_run
    assert paths.config.name == RUN_CONFIG_FILE
    assert paths.config.read_text() == (Path(__file__).parents[1] / "experiments/train.toml").read_text()
    assert load_resolved_config(paths) == canonical
    assert _load_run_config(paths) == canonical
    assert config_sha256(canonical) == load_metadata(paths)["config_sha256"]
    assert json.loads(paths.resolved_config.read_text())["format_version"] == RESOLVED_CONFIG_FORMAT_VERSION
    assert load_metadata(paths)["status"] == "created"
    assert {"checkpoints", "samples", "tensorboard"} <= {p.name for p in paths.root.iterdir()}
    update_metadata(paths, status="running")
    assert load_metadata(paths)["status"] == "running"
    update_metadata(paths, status="interrupted")
    assert load_metadata(paths)["status"] == "interrupted"


def test_run_config_hash_and_canonical_validation(prepared_run):
    paths, canonical = prepared_run
    changed = copy.deepcopy(canonical)
    changed["generator"]["skip_layers"] = []
    paths.resolved_config.write_text(json.dumps({"format_version": RESOLVED_CONFIG_FORMAT_VERSION, "config": changed}))
    with pytest.raises(ValueError, match="resolved config 已被修改"):
        load_resolved_config(paths)

    paths.resolved_config.write_text(json.dumps({"format_version": RESOLVED_CONFIG_FORMAT_VERSION, "config": canonical}))
    assert _load_run_config(paths) == canonical
    unc = copy.deepcopy(canonical)
    unc["generator"].pop("skip_layers")
    paths.resolved_config.write_text(json.dumps({"format_version": RESOLVED_CONFIG_FORMAT_VERSION, "config": unc}))
    metadata = load_metadata(paths)
    metadata["config_sha256"] = config_sha256(unc)
    paths.metadata.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="canonical"):
        _load_run_config(paths)


def test_run_ids_are_unique_and_parent_is_recorded(prepared_run):
    paths, canonical = prepared_run
    second = create_run(paths.root.parent, paths.config, canonical, name="single-stage", parent={"run_id": paths.root.name})
    assert second.root != paths.root
    assert second.root.name.startswith(paths.root.name)
    assert load_metadata(second)["parent"]["run_id"] == paths.root.name
    assert load_resolved_config(second) == canonical


def test_latest_resume_rejects_historical_checkpoint(prepared_run):
    paths, _ = prepared_run
    first = paths.checkpoints / "step_000000010.pth"
    second = paths.checkpoints / "step_000000020.pth"
    first.write_bytes(b"stub-checkpoint-one")
    second.write_bytes(b"stub-checkpoint-two")
    write_latest(paths, second)
    assert resolve_resume_target(paths.root) == (paths, second)
    assert resolve_resume_target(second) == (paths, second)
    assert resolve_branch_target(paths.root) == second
    assert resolve_branch_target(first) == first
    with pytest.raises(ValueError, match="latest"):
        resolve_resume_target(first)
    with pytest.raises(ValueError, match="resume checkpoint"):
        resolve_resume_target(paths.config)
    write_latest(paths, first)
    assert resolve_resume_target(paths.root) == (paths, first)
    assert resolve_branch_target(paths.root) == first


def test_missing_latest_checkpoint_fails_before_resume(prepared_run):
    paths, _ = prepared_run
    with pytest.raises(FileNotFoundError):
        resolve_resume_target(paths.root)
    missing = paths.checkpoints / "step_000000042.pth"
    write_latest(paths, missing)
    with pytest.raises(FileNotFoundError, match="latest checkpoint"):
        resolve_resume_target(paths.root)
    with pytest.raises(FileNotFoundError):
        resolve_branch_target(paths.root)


def test_run_lock_blocks_second_writer(prepared_run):
    paths, _ = prepared_run
    with RunLock(paths), pytest.raises(RuntimeError, match="占用"), RunLock(paths):
        pass
    with RunLock(paths):
        assert paths.root.joinpath(".run.lock").is_file()


def test_no_implicit_channel_field_and_generator_identity_dimension_stays_configurable():
    cfg = load_train_config(Path(__file__).parents[1] / "experiments/train.toml")
    assert "img_channels" not in cfg["generator"]
    assert "img_channels" not in cfg["discriminator"]
    cfg["generator"]["id_dim"] = 256  # Deliberately NOT restricted by config normalization.
    normalized = resolve_train_config(cfg)
    assert normalized["generator"]["id_dim"] == 256
    assert normalized == resolve_train_config(normalized)
    from models.networks import Generator

    with pytest.raises(TypeError, match="img_channels"):
        Generator(img_channels=4)
    for part in ("generator", "discriminator"):
        bad = copy.deepcopy(cfg)
        bad[part]["img_channels"] = 3
        with pytest.raises(ValueError, match="img_channels"):
            resolve_train_config(bad)
