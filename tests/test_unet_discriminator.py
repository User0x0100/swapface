"""U-Net discriminator integration tests (CPU-safe; no dataset needed)."""

import copy
import tomllib
import unittest
from pathlib import Path

import torch

from losses import DiscriminatorAdversarialLoss, GeneratorAdversarialLoss, WeakFeatureMatchingLoss
from models.discriminator import DiscriminatorType, UNetDiscriminator, build_discriminator
from swapface.config import resolve_train_config
from swapface.train import Trainer


def _raw_config() -> dict:
    return tomllib.loads((Path(__file__).parents[1] / "experiments/train.toml").read_text())


def _check_unet_config() -> None:
    raw = _raw_config()
    original = resolve_train_config(raw)
    assert original["discriminator"]["coarse"]["type"] == "FIR_MINIBATCH_STD"
    assert "spatial_weight" not in original["loss"]["coarse"]["gan"]

    raw["discriminator"]["coarse"]["type"] = DiscriminatorType.UNET.name
    raw["discriminator"]["coarse"].pop("base_ch", None)
    raw["discriminator"]["coarse"].pop("max_ch", None)
    resolved = resolve_train_config(raw)
    assert resolved["discriminator"]["coarse"] == {
        "type": "UNET", "img_resolution": 128, "img_channels": 3, "base_ch": 32, "max_ch": 256,
    }
    assert resolved["loss"]["coarse"]["gan"]["spatial_weight"] == 0.25

    customized = copy.deepcopy(raw)
    customized["loss"]["coarse"]["gan"]["spatial_weight"] = 0.7
    assert resolve_train_config(customized)["loss"]["coarse"]["gan"]["spatial_weight"] == 0.7

    invalid = copy.deepcopy(customized)
    invalid["discriminator"]["coarse"]["type"] = "FIR_MINIBATCH_STD"
    with unittest.TestCase().assertRaisesRegex(ValueError, "spatial_weight"):
        resolve_train_config(invalid)

    invalid = copy.deepcopy(raw)
    invalid["discriminator"]["hq"]["type"] = "UNET"
    with unittest.TestCase().assertRaisesRegex(ValueError, "仅用于 Coarse"):
        resolve_train_config(invalid)

    invalid = copy.deepcopy(raw)
    invalid["generator"]["coarse_resolution"] = 64
    invalid["discriminator"]["coarse"]["img_resolution"] = 64
    with unittest.TestCase().assertRaisesRegex(ValueError, "128"):
        resolve_train_config(invalid)


def _check_unet_shapes() -> None:
    cfg = {"type": "UNET", "img_resolution": 128, "img_channels": 3, "base_ch": 4, "max_ch": 16}
    model = build_discriminator(cfg)
    assert isinstance(model, UNetDiscriminator) and model.network_cfg == cfg
    fake = torch.randn(1, 3, 128, 128, requires_grad=True)
    real = torch.randn(1, 3, 128, 128)
    (global_logits, spatial_logits), fake_features = model(fake, return_feats=True)
    assert global_logits.shape == (1, 1)
    assert spatial_logits.shape == (1, 1, 128, 128)
    assert [feature.shape[-2:] for feature in fake_features] == [(64, 64), (32, 32), (16, 16), (8, 8)]
    with torch.no_grad():
        _, real_features = model(real, return_feats=True)
    wfm = WeakFeatureMatchingLoss({0: 0.5, 1: 1.0, 2: 1.0, 3: 1.0})
    global_loss, spatial_loss = Trainer._unet_generator_gan_components(
        (global_logits, spatial_logits), GeneratorAdversarialLoss(weight=1.0, reduction="mean"), 0.25,
    )
    loss = global_loss + spatial_loss + wfm(fake_features, real_features)
    loss.backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
    assert model.global_head.weight.grad is not None
    assert model.spatial_head[-1].weight.grad is not None


def _check_unet_r1(step: int, expected_branch: int) -> None:
    model = UNetDiscriminator(base_ch=4, max_ch=16)
    trainer = Trainer.__new__(Trainer)
    trainer._completed_step = step
    trainer.coarse_spatial_gan_weight = 0.25
    trainer.amp_enabled = False
    trainer.amp_dtype = None
    trainer.d_loss = DiscriminatorAdversarialLoss(weight=1.0, reduction="mean")
    logs = {}
    trainer.log = lambda key, value, **_kwargs: logs.__setitem__(key, float(value.detach()))

    fake = torch.randn(2, 3, 128, 128)
    real = torch.randn(2, 3, 128, 128)
    loss = trainer._discriminator_stage_loss(
        "coarse", fake, real, model, model, r1_enabled=True, r1_interval=16, r1_gamma=10.0,
    )
    assert logs["coarse_r1_branch_spatial"] == expected_branch
    assert abs(logs["coarse_d_loss"] - (
        logs["coarse_d_global_loss"] + 0.25 * logs["coarse_d_spatial_loss"]
    )) < 1e-5
    assert abs(logs["coarse_r1_loss"] - 16 * logs["coarse_r1_loss_raw"]) < 1e-4
    loss.backward()
    assert model.from_rgb.weight.grad is not None
    if expected_branch:
        assert model.spatial_head[-1].weight.grad is not None
    else:
        assert model.global_head.weight.grad is not None
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()

    # Retry/resume on the same completed_step chooses the same branch.
    assert trainer._discriminator_stage_loss(
        "coarse", fake, real, model, model, r1_enabled=True, r1_interval=16, r1_gamma=10.0,
    ).isfinite()
    trainer._completed_step = step + 1
    logs.clear()
    regular = trainer._discriminator_stage_loss(
        "coarse", fake, real, model, model, r1_enabled=True, r1_interval=16, r1_gamma=10.0,
    )
    assert regular.isfinite() and "coarse_r1_loss_raw" not in logs
    regular.backward()


def _check_unet_paired() -> None:
    model = UNetDiscriminator(base_ch=4, max_ch=16).eval()
    fake = torch.randn(2, 3, 128, 128)
    real = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        separately_fake = model(fake)
        separately_real = model(real)
        paired = model(torch.cat((fake, real)), True)
    for reference_fake, reference_real, joint in zip(separately_fake, separately_real, paired, strict=True):
        a, b = joint.chunk(2, dim=0)
        torch.testing.assert_close(a, reference_fake)
        torch.testing.assert_close(b, reference_real)


def _check_unet_global_only_skips_decoder() -> None:
    model = UNetDiscriminator(base_ch=4, max_ch=16).eval()
    x = torch.randn(2, 3, 128, 128)
    called = []
    hook = model.up_blocks[0].register_forward_hook(lambda *_args: called.append("up"))
    try:
        with torch.no_grad():
            global_only = model(x, False, False, False)
            (global_with_features, features) = model(x, False, True, False)
            assert called == []
            assert global_only.shape == (2, 1)
            assert len(features) == 4
            torch.testing.assert_close(global_only, global_with_features)
            global_full, spatial_full = model(x)
            torch.testing.assert_close(global_only, global_full)
            assert spatial_full.shape == (2, 1, 128, 128)
            assert called == ["up"]
    finally:
        hook.remove()


def _stub_generator_trainer(model: UNetDiscriminator, weight: float):
    trainer = Trainer.__new__(Trainer)
    trainer._completed_step = 1000
    trainer.net_d_coarse = model
    trainer.train_d_coarse = model
    trainer.coarse_spatial_gan_weight = weight
    trainer.coarse_gan_loss = GeneratorAdversarialLoss(weight=1.0, reduction="mean")
    trainer.enable_coarse_wfm_loss = False
    trainer.enable_coarse_gaze_loss = False
    trainer.enable_coarse_hrffa_loss = False
    trainer.enable_coarse_facs_loss = False
    trainer.enable_coarse_vgg_loss = False
    trainer.enable_coarse_l1_loss = False
    trainer.coarse_reconstruction_scope = "all"
    trainer.coarse_identity_embeddings_forward = lambda x: x.mean(dim=(2, 3))
    trainer.coarse_id_loss = lambda a, b: (a - b).square().mean() * 0
    trainer.prepare_identity_encoder_faces = lambda x, theta_restore: x
    logs = {}
    trainer.log = lambda key, value, **_kwargs: logs.__setitem__(key, float(value.detach()))
    return trainer, logs


def _check_unet_generator_components_and_skipping() -> None:
    model = UNetDiscriminator(base_ch=4, max_ch=16)
    decoder_calls = []
    hook = model.spatial_head.register_forward_hook(lambda *_args: decoder_calls.append(1))
    try:
        for spatial_weight, expected_calls in ((0.25, 1), (0.0, 0)):
            for wfm_enabled in (False, True):
                decoder_calls.clear()
                trainer, logs = _stub_generator_trainer(model, spatial_weight)
                trainer.enable_coarse_wfm_loss = wfm_enabled
                if wfm_enabled:
                    trainer.coarse_wfm_loss = WeakFeatureMatchingLoss({0: 0.5, 1: 1.0})
                fake = torch.randn(2, 3, 128, 128, requires_grad=True)
                total = trainer._coarse_generator_stage_loss(
                    fake, fake.detach(), None, None, None,
                    torch.zeros(2, 3), None,
                )
                # WFM real forward is encoder-only, even when Spatial GAN is enabled.
                assert len(decoder_calls) == expected_calls
                assert "coarse_gan_loss" in logs
                assert "coarse_g_global_loss" in logs
                assert "coarse_g_spatial_loss" in logs
                assert ("coarse_wfm_loss" in logs) == wfm_enabled
                assert abs(logs["coarse_gan_loss"] - logs["coarse_g_global_loss"] - logs["coarse_g_spatial_loss"]) < 1e-6
                if spatial_weight == 0.0:
                    assert logs["coarse_g_spatial_loss"] == 0.0
                total.backward()
                assert fake.grad is not None and torch.isfinite(fake.grad).all()
    finally:
        hook.remove()


def _check_unet_global_only_discriminator_skips_decoder() -> None:
    model = UNetDiscriminator(base_ch=4, max_ch=16)
    trainer, logs = _stub_generator_trainer(model, 0.0)
    trainer.d_loss = DiscriminatorAdversarialLoss(weight=1.0, reduction="mean")
    trainer.amp_enabled = False
    trainer.amp_dtype = None
    count = []
    hook = model.up_blocks[0].register_forward_hook(lambda *_args: count.append(1))
    try:
        for step in (0, 1, 48):
            trainer._completed_step = step
            count.clear()
            loss = trainer._discriminator_stage_loss(
                "coarse", torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128),
                model, model, r1_enabled=True, r1_interval=16, r1_gamma=10.0,
            )
            assert len(count) == 0
            assert logs["coarse_d_spatial_loss"] == 0.0
            if step % 16 == 0:
                assert logs["coarse_r1_branch_spatial"] == 0.0
            loss.backward()
            model.zero_grad(set_to_none=True)
    finally:
        hook.remove()


def _check_unet_sample_gradient_rms() -> None:
    model = UNetDiscriminator(base_ch=4, max_ch=16).eval()

    class Writer:
        def __init__(self):
            self.values = {}

        def add_scalar(self, tag, value, step):
            self.values[tag] = (float(value), step)

    for spatial_weight in (0.25, 0.0):
        trainer, _ = _stub_generator_trainer(model, spatial_weight)
        trainer.log_writer = Writer()
        generated = torch.randn(2, 3, 128, 128)
        # Model code running in actual sample generation is wrapped by no_grad.
        with torch.no_grad():
            rows = trainer._sample_stage_grad_maps(
                "coarse", generated, generated, None, None, None,
                torch.zeros(2, 3), None, (128, 128),
            )
        assert rows[0][0].startswith("COARSE GAN GLOBAL GRAD")
        assert any(row[0].startswith("COARSE ID GRAD") for row in rows)
        assert len(rows) == (3 if spatial_weight else 2)
        assert ("Grad/coarse_g_global_rms" in trainer.log_writer.values)
        assert trainer.log_writer.values["Grad/coarse_g_global_rms"][1] == 1000
        if spatial_weight:
            assert rows[1][0].startswith("COARSE GAN SPATIAL GRAD")
            assert trainer.log_writer.values["Grad/coarse_g_spatial_rms"][0] > 0
        else:
            assert trainer.log_writer.values["Grad/coarse_g_spatial_rms"] == (0.0, 1000)

        x = generated.detach().requires_grad_(True)
        logits = model(x, False, False, spatial_weight > 0.0)
        loss_fn = trainer.coarse_gan_loss
        global_loss = loss_fn(logits[0] if isinstance(logits, tuple) else logits)
        (global_grad,) = torch.autograd.grad(global_loss, x, retain_graph=spatial_weight > 0)
        expected_global_rms = float(global_grad.square().mean().sqrt())
        actual_global_rms = trainer.log_writer.values["Grad/coarse_g_global_rms"][0]
        assert abs(expected_global_rms - actual_global_rms) < 1e-9
        if spatial_weight:
            (spatial_grad,) = torch.autograd.grad(spatial_weight * loss_fn(logits[1]), x)
            expected_spatial_rms = float(spatial_grad.square().mean().sqrt())
            actual_spatial_rms = trainer.log_writer.values["Grad/coarse_g_spatial_rms"][0]
            assert abs(expected_spatial_rms - actual_spatial_rms) < 1e-9


class TestUNetDiscriminator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._threads)

    def test_config(self) -> None:
        _check_unet_config()

    def test_shapes_wfm(self) -> None:
        _check_unet_shapes()

    def test_r1_global(self) -> None:
        _check_unet_r1(0, 0)

    def test_r1_spatial(self) -> None:
        _check_unet_r1(48, 1)

    def test_paired_inputs(self) -> None:
        _check_unet_paired()

    def test_global_only_skips_decoder(self) -> None:
        _check_unet_global_only_skips_decoder()

    def test_generator_components_and_skipping(self) -> None:
        _check_unet_generator_components_and_skipping()

    def test_d_r1_and_standard_global_only(self) -> None:
        _check_unet_global_only_discriminator_skips_decoder()

    def test_sample_gradient_rms(self) -> None:
        _check_unet_sample_gradient_rms()



if __name__ == "__main__":
    unittest.main()
