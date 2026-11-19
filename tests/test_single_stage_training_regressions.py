"""Single-stage training regression tests."""


import pytest
import torch
from torch import nn

from losses import DiscriminatorAdversarialLoss, WeakFeatureMatchingLoss
from models.discriminator import build_discriminator
from swapface.train import Trainer, _scaled_backward_step, _validate_checkpoint_identity_providers


class SimulatedScaler:
    """Deterministic stand-in for GradScaler's skipped-step semantics."""

    def __init__(self, *, enabled: bool = True, overflows: int = 0):
        self.enabled = enabled
        self.overflows = overflows
        self.scale_value = 8.0
        self.attempts = 0

    def get_scale(self) -> float:
        return self.scale_value

    def is_enabled(self) -> bool:
        return self.enabled

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.attempts += 1
        if self.attempts > self.overflows:
            optimizer.step()

    def update(self) -> None:
        if self.attempts <= self.overflows:
            self.scale_value *= 0.5


@pytest.mark.parametrize("overflows, expected_success", [(0, True), (1, False)])
def test_fp16_scaled_optimizer_skips_overflow(overflows: int, expected_success: bool) -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    scaler = SimulatedScaler(overflows=overflows)
    success = _scaled_backward_step(parameter.square(), optimizer, scaler)
    assert success is expected_success
    torch.testing.assert_close(parameter.detach(), torch.tensor(0.8 if expected_success else 1.0))


class FiniteLossWithNaNGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return torch.full_like(grad_output, torch.nan)


def test_bf16_fp32_step_does_not_inspect_gradients() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    success = _scaled_backward_step(
        FiniteLossWithNaNGradient.apply(parameter), optimizer, SimulatedScaler(enabled=False),
    )
    assert success
    assert torch.isnan(parameter.detach())


def test_discriminator_fp16_overflow_retries_only_one_optimizer() -> None:
    trainer = Trainer.__new__(Trainer)
    parameter = nn.Parameter(torch.tensor(1.0))
    unrelated = nn.Parameter(torch.tensor(2.0))
    trainer.optim_d = torch.optim.SGD([parameter], lr=0.1)
    unrelated_optimizer = torch.optim.SGD([unrelated], lr=0.1)
    trainer.scaler_d = SimulatedScaler(overflows=1)
    trainer._discriminator_loss = lambda _fake, _real: parameter.square()
    trainer._update_discriminator(torch.empty(0), torch.empty(0))
    assert trainer.scaler_d.attempts == 2
    assert trainer.scaler_d.get_scale() == pytest.approx(4.0)
    torch.testing.assert_close(parameter.detach(), torch.tensor(0.8))
    torch.testing.assert_close(unrelated.detach(), torch.tensor(2.0))
    assert not unrelated_optimizer.state


def test_lazy_r1_second_order_gradients_and_logging() -> None:
    trainer = Trainer.__new__(Trainer)
    model = build_discriminator({
        "type": "FIR_MINIBATCH_STD", "img_resolution": 32,
        "base_ch": 8, "max_ch": 32, "minibatch_std_group_size": 2,
    })
    trainer.net_d = model
    trainer.train_d = model
    trainer.enable_r1_loss = True
    trainer.r1_reg_step = 16
    trainer.r1_gamma = 10.0
    trainer.amp_dtype = None
    trainer.amp_enabled = False
    trainer.d_loss = DiscriminatorAdversarialLoss(weight=1.0, reduction="mean")
    logs: dict[str, float] = {}
    trainer.log = lambda key, value, **kwargs: logs.__setitem__(key, float(value.detach()))
    fake = torch.randn(4, 3, 32, 32)
    real = torch.randn(4, 3, 32, 32)

    trainer._completed_step = 0
    loss = trainer._discriminator_loss(fake, real)
    loss.backward()
    assert {"d_loss", "r1_loss_raw", "r1_loss"} <= logs.keys()
    assert logs["r1_loss"] == pytest.approx(logs["r1_loss_raw"] * 16, rel=1e-5, abs=1e-6)
    assert torch.isfinite(loss)
    assert all(param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters())

    model.zero_grad(set_to_none=True)
    logs.clear()
    trainer._completed_step = 1
    loss = trainer._discriminator_loss(fake, real)
    loss.backward()
    assert "d_loss" in logs and "r1_loss_raw" not in logs and "r1_loss" not in logs
    assert torch.isfinite(loss)


@pytest.mark.parametrize("kind", ["FIR_MINIBATCH_STD", "INSTANCE_NORM_RESIDUAL"])
def test_wfm_teacher_is_detached_but_fake_gradient_is_preserved(kind: str) -> None:
    cfg = {"type": kind, "img_resolution": 32, "base_ch": 8, "max_ch": 32}
    if kind == "FIR_MINIBATCH_STD":
        cfg["minibatch_std_group_size"] = 2
    model = build_discriminator(cfg)
    fake = torch.randn(2, 3, 32, 32, requires_grad=True)
    real = torch.randn(2, 3, 32, 32, requires_grad=True)
    _, generated_features = model(fake, return_feats=True)
    with torch.no_grad():
        _, reference_features = model(real, return_feats=True)
    loss = WeakFeatureMatchingLoss({0: 0.5, 1: 1.0})(generated_features, reference_features)
    loss.backward()
    assert torch.isfinite(loss) and torch.count_nonzero(fake.grad) > 0
    assert real.grad is None


def test_branch_can_change_loss_teacher_without_changing_generator_encoder() -> None:
    ckpt = {"identity_encoders": {"generator": "BLENDFACE", "identity_loss": "BLENDFACE"}}
    kwargs = {"generator_provider": "BLENDFACE", "identity_loss_provider": "MS1MV3_ARCFACE_R50_FP16"}
    _validate_checkpoint_identity_providers(ckpt, checkpoint_mode="branch", **kwargs)
    with pytest.raises(ValueError, match="identity loss provider"):
        _validate_checkpoint_identity_providers(ckpt, checkpoint_mode="resume", **kwargs)
    with pytest.raises(ValueError, match="generator identity provider"):
        _validate_checkpoint_identity_providers(ckpt, generator_provider="OTHER",
                                                identity_loss_provider="BLENDFACE", checkpoint_mode="resume")
    _validate_checkpoint_identity_providers(ckpt, generator_provider="OTHER",
                                            identity_loss_provider="BLENDFACE", checkpoint_mode="branch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_full_trainer_branch_with_changed_loss_teacher(tmp_path, monkeypatch) -> None:
    """Exercise the complete Trainer checkpoint path, not only its validation helper."""
    from models.discriminator import build_discriminator
    from models.networks import Generator
    from swapface.config import resolve_train_config
    from swapface.contracts import CHECKPOINT_VERSION
    from swapface.train import TRAINING_SEMANTICS_VERSION, _load_branch_checkpoint

    class FakeEncoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, images):
            return images.new_zeros((images.shape[0], 512))

    class FakeIdentityLoss(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.min_similarity = None

        def extract_identity_embeddings(self, images):
            return images.new_zeros((images.shape[0], 512))

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            pass

    cfg = resolve_train_config({
        "train": {"batch_size": 2, "device": "cuda", "precision": "fp32", "compile_module": False},
        "identity": {"provider": "BLENDFACE"},
        "generator": {"type": "inswapper", "img_resolution": 32, "latent_resolution": 8,
                      "base_ch": 4, "max_ch": 8, "num_latent": 1},
        "discriminator": {"img_resolution": 32, "base_ch": 4, "max_ch": 8,
                          "minibatch_std_group_size": 2},
        "loss": {"identity": {"provider": "MS1MV3_ARCFACE_R50_FP16"},
                 "l1": {"enable": False}, "vgg": {"enable": False},
                 "gaze": {"enable": False}, "smirk": {"enable": False},
                 "wfm": {"enable": False}},
        "data": {"src": [{"path": "dummy_src"}], "dst": [{"path": "dummy_dst"}],
                 "loader": {"py_num_workers": 0}},
    })
    generator = Generator(**cfg["generator"])
    discriminator = build_discriminator(cfg["discriminator"])
    checkpoint = {
        "version": CHECKPOINT_VERSION, "step": 3, "face_alignment": "ffhq",
        "run": {"id": "parent", "config_sha256": "parent-hash"},
        "training_config": {"semantics_version": TRAINING_SEMANTICS_VERSION, "precision": "fp32"},
        "identity_encoders": {"generator": "BLENDFACE", "identity_loss": "BLENDFACE"},
        "net_g": {"network_cfg": generator.network_cfg, "state_dict": generator.state_dict()},
        "net_d": {"network_cfg": discriminator.network_cfg, "state_dict": discriminator.state_dict()},
        "training_state": {"net_g": generator.state_dict()},
    }
    path = tmp_path / "step_000000003.pth"
    torch.save(checkpoint, path)
    loaded, _ = _load_branch_checkpoint(path, reset_discriminator=False)
    monkeypatch.setattr("swapface.train.IDEncoder", FakeEncoder)
    monkeypatch.setattr("swapface.train.IdentityLoss", FakeIdentityLoss)
    monkeypatch.setattr("swapface.train.TrainingDataLoader", FakeDataset)

    trainer = Trainer(cfg, run_dir=tmp_path / "branch", checkpoint_path=path,
                      checkpoint_mode="branch", preloaded_checkpoint=loaded, run_id="branch")
    try:
        assert trainer.completed_step == 3
        assert trainer.generator_id_encoder_provider.name == "BLENDFACE"
        assert trainer.identity_loss_provider.name == "MS1MV3_ARCFACE_R50_FP16"
        for name, value in generator.state_dict().items():
            torch.testing.assert_close(trainer.net_g.state_dict()[name].cpu(), value)
    finally:
        trainer.log_writer.close()

    with pytest.raises(ValueError, match="identity loss provider"):
        Trainer(cfg, run_dir=tmp_path / "resume", checkpoint_path=path,
                checkpoint_mode="resume", preloaded_checkpoint=loaded, run_id="parent")
