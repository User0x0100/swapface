"""Compilation mode configuration and independent identity encoder execution."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from swapface.config import resolve_train_config
from swapface.train import Trainer, _compile_training_callable


def test_compile_mode_is_forwarded_without_project_validation(monkeypatch):
    cfg = resolve_train_config({
        "train": {"precision": "fp32", "compile_mode": "custom-pytorch-mode"},
        "generator": {"img_resolution": 32, "latent_resolution": 8},
        "discriminator": {"img_resolution": 32},
        "data": {"src": [{"path": "unused"}], "dst": [{"path": "unused"}]},
    })
    assert cfg["train"]["compile_mode"] == "custom-pytorch-mode"

    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: calls.append((fn, kwargs)) or fn)
    fn = lambda x: x
    assert _compile_training_callable(fn, cfg["train"]["compile_mode"]) is fn
    assert calls == [(fn, {"fullgraph": True, "dynamic": False, "mode": "custom-pytorch-mode"})]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
@pytest.mark.parametrize("same_provider", [True, False])
@pytest.mark.parametrize("compile_module", [True, False])
def test_trainer_keeps_independent_identity_encoders(tmp_path, monkeypatch, same_provider, compile_module):
    created = []
    compiled = []

    class FakeEncoder(nn.Module):
        def __init__(self, provider):
            super().__init__()
            self.provider = provider
            self.weight = nn.Parameter(torch.ones(()))
            created.append(self)

        def forward(self, faces):
            rgb = faces.mean(dim=(2, 3)) * self.weight
            return F.normalize(F.pad(rgb, (0, 509)), dim=1)

    class FakeDataset:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("swapface.train.IDEncoder", FakeEncoder)
    monkeypatch.setattr("losses.identity.IDEncoder", FakeEncoder)
    monkeypatch.setattr("swapface.train.TrainingDataLoader", FakeDataset)
    monkeypatch.setattr("swapface.train.initialize_upfirdn2d", lambda: None)
    monkeypatch.setattr("swapface.train._compile_training_callable", lambda fn, mode: compiled.append((fn, mode)) or fn)

    cfg = resolve_train_config({
        "train": {"batch_size": 2, "device": "cuda", "precision": "fp32", "compile_module": compile_module, "compile_mode": "custom-pytorch-mode"},
        "identity": {"provider": "BLENDFACE"},
        "generator": {"img_resolution": 32, "latent_resolution": 8, "base_ch": 4, "max_ch": 8, "num_latent": 1},
        "discriminator": {"img_resolution": 32, "base_ch": 4, "max_ch": 8, "minibatch_std_group_size": 2},
        "loss": {
            "identity": {"provider": "BLENDFACE" if same_provider else "MS1MV3_ARCFACE_R50_FP16"},
            "l1": {"enable": False}, "vgg": {"enable": False}, "gaze": {"enable": False},
            "smirk": {"enable": False}, "wfm": {"enable": False},
        },
        "data": {"src": [{"path": "unused_src"}], "dst": [{"path": "unused_dst"}], "loader": {"py_num_workers": 0}},
    })

    trainer = Trainer(cfg, run_dir=tmp_path / "training")
    try:
        assert trainer.id_loss.id_encoder is not trainer.generator_id_encoder
        assert trainer.identity_embeddings_forward is not trainer.generator_id_encoder_forward
        assert len(created) == 2
        assert len(compiled) == (4 if compile_module else 0)
        assert all(mode == "custom-pytorch-mode" for _, mode in compiled)
        images = torch.randn(2, 3, 112, 112, device="cuda", requires_grad=True)
        trainer.identity_embeddings_forward(images)[:, 0].sum().backward()
        assert images.grad is not None and images.grad.abs().sum() > 0
        assert all(p.grad is None for p in trainer.id_loss.id_encoder.parameters())
    finally:
        trainer.log_writer.close()
