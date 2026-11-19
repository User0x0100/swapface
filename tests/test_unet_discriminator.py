"""Single-stage UNetDiscriminatorSN configuration and adversarial training."""

import tomllib
from pathlib import Path

import pytest
import torch

from losses import DiscriminatorAdversarialLoss, GeneratorAdversarialLoss
from models.discriminator import UNetDiscriminatorSN, build_discriminator
from swapface.config import resolve_train_config
from swapface.train import Trainer


def raw_config():
    return tomllib.loads((Path(__file__).parents[1] / "experiments/train.toml").read_text())


def test_single_unet_discriminator_and_r1_disabled():
    raw = raw_config()
    raw["discriminator"] = {"type": "UNET", "img_resolution": 512, "base_ch": 8}
    config = resolve_train_config(raw)
    assert config["discriminator"] == {"type": "UNET", "img_resolution": 512, "base_ch": 8, "skip_connection": True}
    assert not config["loss"]["r1"]["enable"]
    assert isinstance(build_discriminator(config["discriminator"]), UNetDiscriminatorSN)
    raw["discriminator"]["skip_connection"] = False
    assert not resolve_train_config(raw)["discriminator"]["skip_connection"]


@pytest.mark.parametrize("resolution", [32, 64])
def test_unet_forward_backward(resolution: int):
    torch.manual_seed(10)
    model = build_discriminator({"type": "UNET", "img_resolution": resolution,
                                 "base_ch": 8, "skip_connection": True})
    model.eval()  # spectral norm remains enabled, output retains spatial shape
    fake = torch.randn(2, 3, resolution, resolution, requires_grad=True)
    logits = model(fake)
    assert logits.shape == (2, 1, resolution, resolution)
    GeneratorAdversarialLoss(weight=1., reduction="mean")(logits).backward()
    assert torch.isfinite(fake.grad).all()


def test_unet_paired_fake_real_and_loss():
    model = build_discriminator({"type": "UNET", "img_resolution": 32,
                                 "img_channels": 3, "base_ch": 8, "skip_connection": False})
    fake, real = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    scores = model(torch.cat((fake, real), 0), True)
    f, r = scores.chunk(2)
    loss = DiscriminatorAdversarialLoss(weight=1., reduction="mean")(f, r)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_r1_skipped_for_unet_even_if_runtime_flag_true():
    trainer = Trainer.__new__(Trainer)
    trainer.net_d = build_discriminator({"type": "UNET", "img_resolution": 32,
                                        "base_ch": 8, "skip_connection": True})
    trainer.train_d = trainer.net_d
    trainer.enable_r1_loss = True
    trainer.r1_reg_step = 1
    trainer.r1_gamma = 10.
    trainer._completed_step = 0
    trainer.amp_enabled = False
    trainer.amp_dtype = None
    trainer.d_loss = DiscriminatorAdversarialLoss(weight=1., reduction="mean")
    logged = []
    trainer.log = lambda name, value, **kwargs: logged.append(name)
    output = trainer._discriminator_loss(torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32))
    assert torch.isfinite(output)
    assert "d_loss" in logged and "r1_loss" not in logged


@pytest.mark.parametrize("discriminator_type, invalid_field, value", [
    ("UNET", "max_ch", 128),
    ("UNET", "minibatch_std_group_size", 0),
    ("INSTANCE_NORM_RESIDUAL", "minibatch_std_group_size", 0),
    ("INSTANCE_NORM_RESIDUAL", "skip_connection", False),
    ("FIR_MINIBATCH_STD", "skip_connection", True),
    ("FIR_MINIBATCH_STD", "skip_connection", False),
])
def test_type_specific_discriminator_fields_are_rejected(discriminator_type, invalid_field, value):
    raw = raw_config()
    raw["discriminator"] = {"type": discriminator_type, invalid_field: value}
    with pytest.raises(ValueError, match=invalid_field):
        resolve_train_config(raw)


@pytest.mark.parametrize("discriminator_type", ["UNET", "INSTANCE_NORM_RESIDUAL", "FIR_MINIBATCH_STD"])
def test_discriminator_canonical_schema_is_strict_and_idempotent(discriminator_type):
    raw = raw_config()
    raw["discriminator"] = {"type": discriminator_type}
    normalized = resolve_train_config(raw)
    fields = set(normalized["discriminator"])
    assert "img_channels" not in fields
    if discriminator_type == "UNET":
        assert "skip_connection" in fields and "max_ch" not in fields and "minibatch_std_group_size" not in fields
    elif discriminator_type == "INSTANCE_NORM_RESIDUAL":
        assert "max_ch" in fields and "minibatch_std_group_size" not in fields and "skip_connection" not in fields
    else:
        assert "max_ch" in fields and "minibatch_std_group_size" in fields and "skip_connection" not in fields
    assert normalized == resolve_train_config(normalized)
    assert build_discriminator(normalized["discriminator"]).network_cfg == normalized["discriminator"]
