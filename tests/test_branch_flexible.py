"""Branch accepts any loadable Generator weights; Resume remains strict."""

import copy

import pytest
import torch
from torch import nn

from models.discriminator import build_discriminator
from models.networks import Generator
from swapface.config import resolve_train_config
from swapface.experiment import resolve_branch_target
from swapface.train import Trainer, _branch_generator_state, _load_branch_checkpoint


def _config(*, alignment: str = "ffhq", generator_provider: str = "BLENDFACE",
            teacher: str = "BLENDFACE", d_base: int = 4, d_group: int = 2,
            latent_blocks: int = 1, skip_mode: str = "cat") -> dict:
    return resolve_train_config({
        "train": {"batch_size": 2, "device": "cuda", "precision": "fp32", "compile_module": False},
        "identity": {"provider": generator_provider},
        "generator": {"type": "inswapper", "img_resolution": 32,
                      "latent_resolution": 8, "base_ch": 4, "max_ch": 8,
                      "num_latent": latent_blocks, "skip_mode": skip_mode},
        "discriminator": {"img_resolution": 32, "base_ch": d_base, "max_ch": max(d_base, 8),
                          "minibatch_std_group_size": d_group},
        "loss": {"identity": {"provider": teacher}, "l1": {"enable": False},
                 "vgg": {"enable": False}, "gaze": {"enable": False},
                 "smirk": {"enable": False}, "wfm": {"enable": False}},
        "data": {"alignment": alignment, "src": [{"path": "unused_src"}],
                 "dst": [{"path": "unused_dst"}], "loader": {"py_num_workers": 0}},
    })


def _checkpoint(config: dict) -> dict:
    g = Generator(**config["generator"])
    d = build_discriminator(config["discriminator"])
    return {
        "version": 11,  # A branch does not interpret the historical semantics version.
        "step": 17,
        "face_alignment": "ffhq",
        "run": {"id": "source-run", "config_sha256": "old-hash"},
        "identity_encoders": {"generator": "BLENDFACE", "identity_loss": "BLENDFACE"},
        "training_config": {"semantics_version": -1, "precision": "different"},
        "net_g": {"network_cfg": {"irrelevant": True}, "state_dict": g.state_dict()},
        "net_d": {"network_cfg": {"irrelevant": True}, "state_dict": d.state_dict()},
        "training_state": {"net_g": g.state_dict()},
    }


class FakeEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, images):
        return images.new_zeros(images.shape[0], 512)


class FakeIdentityLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.min_similarity = None

    def extract_identity_embeddings(self, images):
        return images.new_zeros(images.shape[0], 512)


class FakeDataset:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def no_pretrained_weights(monkeypatch):
    monkeypatch.setattr("swapface.train.IDEncoder", FakeEncoder)
    monkeypatch.setattr("swapface.train.IdentityLoss", FakeIdentityLoss)
    monkeypatch.setattr("swapface.train.TrainingDataLoader", FakeDataset)


def _construct_branch(tmp_path, config, checkpoint, *, reset=False, step=None):
    path = tmp_path / "custom_pretrained_model.pth"
    torch.save(checkpoint, path)
    assert resolve_branch_target(path) == path
    preloaded, parent = _load_branch_checkpoint(path, reset_discriminator=reset)
    trainer = Trainer(config, run_dir=tmp_path / "branch", checkpoint_path=path,
                      checkpoint_mode="branch", preloaded_checkpoint=preloaded,
                      reset_discriminator=reset, start_step=step)
    return trainer, parent


def test_branch_loads_training_weights_first_then_ema_and_raw_state(tmp_path):
    ckpt = _checkpoint(_config())
    g = ckpt["training_state"]["net_g"]
    first_key = next(iter(g))
    # Train weights take precedence over EMA weights when both exist.
    ckpt["net_g"]["state_dict"] = {name: value.clone() for name, value in g.items()}
    ckpt["net_g"]["state_dict"][first_key].add_(1.0)
    assert _branch_generator_state(ckpt)[first_key].equal(g[first_key])
    infer_only = {"net_g": ckpt["net_g"]}
    assert _branch_generator_state(infer_only)[first_key].equal(ckpt["net_g"]["state_dict"][first_key])
    assert _branch_generator_state(g)[first_key].equal(g[first_key])
    for fixture, expected_step in ((infer_only, 0), (g, 0)):
        path = tmp_path / "nonstandard_name.pth"
        torch.save(fixture, path)
        loaded, parent = _load_branch_checkpoint(path, reset_discriminator=False)
        assert parent["step"] == expected_step
        assert parent["run_id"] is None
        assert _branch_generator_state(loaded)[first_key].equal(_branch_generator_state(fixture)[first_key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_branch_allows_changed_metadata_and_loadable_architecture(no_pretrained_weights, tmp_path):
    old = _config()
    checkpoint = _checkpoint(old)
    newer = _config(alignment="arcface", generator_provider="MS1MV3_ARCFACE_R50_FP16",
                    teacher="WEBFACE600K_ARCFACE_R50", d_group=1, skip_mode="add")
    trainer, parent = _construct_branch(tmp_path, newer, checkpoint)
    try:
        assert trainer.completed_step == 17
        assert trainer.face_alignment == "arcface"
        assert trainer.generator_id_encoder_provider.name == "MS1MV3_ARCFACE_R50_FP16"
        assert trainer.branch_discriminator_state == "inherit"
        assert parent["discriminator"] == "auto"
        for key, tensor in checkpoint["training_state"]["net_g"].items():
            torch.testing.assert_close(trainer.net_g.state_dict()[key].cpu(), tensor)
        for key, tensor in checkpoint["net_d"]["state_dict"].items():
            torch.testing.assert_close(trainer.net_d.state_dict()[key].cpu(), tensor)
    finally:
        trainer.log_writer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
@pytest.mark.parametrize("new_d_base,force_reset,expected", [(4, True, "reset"), (8, False, "reset")])
def test_branch_discriminator_reset_when_requested_or_incompatible(
    no_pretrained_weights, tmp_path, new_d_base, force_reset, expected,
):
    source = _checkpoint(_config())
    config = _config(d_base=new_d_base)
    trainer, _ = _construct_branch(tmp_path, config, source, reset=force_reset)
    try:
        assert trainer.branch_discriminator_state == expected
        for name, tensor in source["training_state"]["net_g"].items():
            torch.testing.assert_close(trainer.net_g.state_dict()[name].cpu(), tensor)
        key = "from_rgb.weight"
        if source["net_d"]["state_dict"][key].shape == trainer.net_d.state_dict()[key].shape:
            assert not torch.equal(source["net_d"]["state_dict"][key], trainer.net_d.state_dict()[key].cpu())
    finally:
        trainer.log_writer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_branch_accepts_inference_only_and_raw_checkpoint(no_pretrained_weights, tmp_path):
    source = _checkpoint(_config())
    raw = copy.deepcopy(source["training_state"]["net_g"])
    for value in ({"net_g": source["net_g"]}, raw):
        trainer, parent = _construct_branch(tmp_path, _config(), value, step=9)
        try:
            assert trainer.completed_step == 9
            assert trainer.branch_discriminator_state == "reset"
            assert parent["step"] == 0
        finally:
            trainer.log_writer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_branch_rejects_generator_weights_that_cannot_be_loaded(no_pretrained_weights, tmp_path):
    source = _checkpoint(_config())
    changed = _config(latent_blocks=2)
    with pytest.raises(ValueError, match="Branch Generator weights cannot be loaded"):
        _construct_branch(tmp_path, changed, source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_resume_still_rejects_incompatible_checkpoint_metadata(no_pretrained_weights, tmp_path):
    source = _checkpoint(_config())
    path = tmp_path / "resume.pth"
    torch.save(source, path)
    with pytest.raises(ValueError, match="checkpoint version mismatch"):
        Trainer(_config(), run_dir=tmp_path / "resume", checkpoint_path=path,
                checkpoint_mode="resume", run_id="source-run")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_incompatible_discriminator_never_keeps_partially_loaded_weights(no_pretrained_weights, tmp_path):
    source = _checkpoint(_config())
    # A previous partial load can copy from_rgb before failing at the final layer.
    # The fallback must reconstruct the entire Discriminator, not keep that copy.
    weights = source["net_d"]["state_dict"]
    weights["from_rgb.weight"] = torch.full_like(weights["from_rgb.weight"], 99.0)
    weights["final_fc1.weight"] = torch.empty(1, 1)  # deliberate shape mismatch
    trainer, _ = _construct_branch(tmp_path, _config(), source)
    try:
        assert trainer.branch_discriminator_state == "reset"
        assert not torch.allclose(trainer.net_d.state_dict()["from_rgb.weight"], torch.full_like(trainer.net_d.state_dict()["from_rgb.weight"], 99.0))
    finally:
        trainer.log_writer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
@pytest.mark.parametrize("raw_source", [False, True])
def test_branch_cli_persists_actual_discriminator_origin(no_pretrained_weights, tmp_path, monkeypatch, raw_source):
    """The initial 'auto' metadata must be resolved before the new run starts."""
    import json
    import sys

    from swapface import train as training_module

    source = _checkpoint(_config())
    checkpoint = source["training_state"]["net_g"] if raw_source else source
    checkpoint_path = tmp_path / "any_name.pth"
    torch.save(checkpoint, checkpoint_path)
    config_path = tmp_path / "new.toml"
    config_path.write_text(
        '[train]\nbatch_size = 2\nprecision = "fp32"\ncompile_module = false\n'
        '[identity]\nprovider = "BLENDFACE"\n'
        '[generator]\ntype = "inswapper"\nimg_resolution = 32\n'
        'latent_resolution = 8\nbase_ch = 4\nmax_ch = 8\nnum_latent = 1\n'
        '[discriminator]\nimg_resolution = 32\nbase_ch = 4\nmax_ch = 8\nminibatch_std_group_size = 2\n'
        '[loss.identity]\nprovider = "BLENDFACE"\n'
        '[loss.l1]\nenable = false\n[loss.vgg]\nenable = false\n'
        '[loss.gaze]\nenable = false\n[loss.smirk]\nenable = false\n'
        '[loss.wfm]\nenable = false\n'
        '[[data.src]]\npath = "unused_src"\n'
        '[[data.dst]]\npath = "unused_dst"\n'
        '[data.loader]\npy_num_workers = 0\n',
        encoding="utf-8",
    )
    runs_root = tmp_path / "runs"
    # The CLI test must not change the process-wide CUDA/cuDNN settings
    # or random seed for unrelated Pytest cases in the same interpreter.
    monkeypatch.setattr(training_module, "_configure_training_runtime", lambda: None)
    monkeypatch.setattr(training_module.Trainer, "train", lambda self: None)
    monkeypatch.setattr(
        sys, "argv", ["swapface.train", "--branch-from", str(checkpoint_path),
                      "--config", str(config_path), "--runs-root", str(runs_root)],
    )
    training_module.main()
    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1
    metadata = json.loads((run_dirs[0] / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["parent"]["discriminator"] == ("reset" if raw_source else "inherit")
