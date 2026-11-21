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


def test_resume_compile_options_persist_in_both_configs_and_keep_hash_history(prepared_run):
    from swapface.experiment import update_run_train_options

    paths, canonical = prepared_run
    original_hash = config_sha256(canonical)
    changed = copy.deepcopy(canonical)
    changed['train'].update(compile_mode='reduce-overhead', compile_module=False)
    update_run_train_options(paths, changed, {'compile_mode': 'reduce-overhead', 'compile_module': False})

    assert load_resolved_config(paths) == changed
    assert load_train_config(paths.config) == changed
    assert load_metadata(paths)['prior_config_sha256'] == [original_hash]
    assert load_metadata(paths)['config_sha256'] == config_sha256(changed)
    assert 'compile_mode = "reduce-overhead"' in paths.config.read_text()
    assert 'compile_module = false' in paths.config.read_text()

    newer = copy.deepcopy(changed)
    newer['train']['compile_mode'] = 'default'
    update_run_train_options(paths, newer, {'compile_mode': 'default'})
    assert load_train_config(paths.config) == newer
    assert load_metadata(paths)['prior_config_sha256'] == [original_hash, config_sha256(changed)]
    assert load_resolved_config(paths) == newer


def test_resume_compile_override_cli_persists_only_after_trainer_load(prepared_run, monkeypatch):
    import sys
    import swapface.train as training

    paths, original = prepared_run
    checkpoint = paths.checkpoints / 'step_000000010.pth'
    checkpoint.write_bytes(b'test-checkpoint')
    write_latest(paths, checkpoint)
    original_hash = config_sha256(original)
    calls = []

    class FakeTrainer:
        def __init__(self, cfg, **kwargs):
            assert kwargs['checkpoint_path'] == checkpoint
            assert original_hash in kwargs['allowed_resume_config_hashes']
            assert kwargs['run_id'] == paths.root.name
            calls.append((cfg['train'].copy(), kwargs['resolved_config_sha256']))

        def train(self):
            pass

        def close_logs(self):
            pass

    monkeypatch.setattr(training, '_configure_training_runtime', lambda: None)
    monkeypatch.setattr(training, 'Trainer', FakeTrainer)
    monkeypatch.setattr(sys, 'argv', ['swapface.train', '--resume', str(paths.root),
                                    '--compile-mode', 'default', '--compile-module', 'false'])
    training.main()
    assert calls[-1][0]['compile_mode'] == 'default'
    assert calls[-1][0]['compile_module'] is False
    assert calls[-1][1] == load_metadata(paths)['config_sha256']
    assert load_train_config(paths.config) == load_resolved_config(paths)
    assert original_hash in load_metadata(paths)['prior_config_sha256']

    # A later resume without overrides must still accept a checkpoint with the original hash.
    monkeypatch.setattr(sys, 'argv', ['swapface.train', '--resume', str(paths.root)])
    training.main()
    assert calls[-1][0]['compile_mode'] == 'default'
    assert calls[-1][0]['compile_module'] is False
    assert len(load_metadata(paths)['prior_config_sha256']) == 1

    class FailedTrainer:
        def __init__(self, *_args, **_kwargs):
            raise ValueError('invalid checkpoint')

    monkeypatch.setattr(training, 'Trainer', FailedTrainer)
    before = (paths.config.read_bytes(), paths.resolved_config.read_bytes(), load_metadata(paths)['config_sha256'])
    monkeypatch.setattr(sys, 'argv', ['swapface.train', '--resume', str(paths.root), '--compile-mode', 'max-autotune'])
    with pytest.raises(ValueError, match='invalid checkpoint'):
        training.main()
    after = (paths.config.read_bytes(), paths.resolved_config.read_bytes(), load_metadata(paths)['config_sha256'])
    assert before == after


@pytest.mark.parametrize('args', [
    ['--compile-module', 'true'],
    ['--compile-mode', 'default'],
    ['--branch-from', 'anything.pth', '--config', 'anything.toml', '--compile-module', 'false'],
])
def test_compile_overrides_rejected_outside_resume(monkeypatch, capsys, args):
    import sys
    import swapface.train as training

    monkeypatch.setattr(training, '_configure_training_runtime', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['swapface.train', *args])
    with pytest.raises(SystemExit) as exc:
        training.main()
    assert exc.value.code == 2
    assert '--compile-mode/--compile-module only apply to resume' in capsys.readouterr().err
