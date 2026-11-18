"""Optional StyleSwap mask branch, feature guidance, BCE supervision and split identity."""

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.networks import Generator as FullGenerator
from models.stylegan2 import Generator, ToMask, mask_guided_target_features, stylegan2_mask_layer_index
from swapface.config import _normalize_generator, resolve_train_config
from swapface.split_identity import PrecomputedStyleGAN2, StyleGAN2Projection
from swapface.train import Trainer


def _stylegan2(**kwargs) -> Generator:
    cfg = {"img_resolution": 16, "id_dim": 32, "w_dim": 16, "mapping_layers": 1, "channels": (8, 16), "mask_enabled": True}
    cfg.update(kwargs)
    return Generator(**cfg)


def _full(**kwargs) -> FullGenerator:
    cfg = {
        "img_resolution": 16,
        "coarse_resolution": 8,
        "id_dim": 32,
        "coarse_type": "stylegan2",
        "coarse_start_resolution": 4,
        "coarse_channels": [8, 16],
        "coarse_w_dim": 16,
        "coarse_mapping_layers": 1,
        "hq_bottleneck_resolution": 4,
        "hq_base_ch": 2,
        "hq_max_ch": 8,
        "coarse_mask_enabled": True,
        "coarse_mask_noise_blank_level": -1,
    }
    cfg.update(kwargs)
    return FullGenerator(**cfg)


def test_mask_disabled_keeps_original_rgb_only_model() -> None:
    generator = _stylegan2(mask_enabled=False)
    assert generator.to_mask_first is None
    assert all(stage.to_mask is None for stage in generator.stages)
    assert not any("to_mask" in key for key in generator.state_dict())
    x = torch.randn(1, 3, 16, 16)
    identity = torch.randn(1, 32)
    assert generator(x, identity).shape == x.shape
    with pytest.raises(ValueError, match="mask_enabled"):
        generator(x, identity, return_mask=True)
    assert _stylegan2().to_mask_first is not None


def test_mask_branch_shapes_stages_and_gradients() -> None:
    torch.manual_seed(222)
    generator = _stylegan2(mask_noise_blank_level=-1)
    for module in generator.modules():
        if hasattr(module, "inject_gain"):
            module.inject_gain.data.fill_(0.25)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    id_feat = torch.randn(2, 32, requires_grad=True)
    rgb, mask = generator(x, id_feat, return_mask=True)
    assert rgb.shape == (2, 3, 16, 16)
    assert mask.shape == (2, 1, 16, 16)
    assert mask.min().item() >= 0 and mask.max().item() <= 1
    with torch.no_grad():
        other_rgb, mask_logits = generator(x, id_feat, return_mask_logits=True)
        torch.testing.assert_close(rgb, other_rgb)
        torch.testing.assert_close(mask.float(), mask_logits.float().sigmoid())
    reference = torch.rand_like(mask).round()
    _, mask_logits = generator(x, id_feat, return_mask_logits=True)
    loss = rgb.square().mean() + F.binary_cross_entropy_with_logits(mask_logits.float(), reference.float())
    loss.backward()
    assert generator.to_mask_first.conv.weight.grad is not None
    assert generator.to_mask_first.conv.weight.grad.abs().sum() > 0
    assert generator.stages[-1].to_mask.conv.weight.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(id_feat.grad).all()


def test_mask_skip_accumulates_signed_fp32_logits_without_sigmoid() -> None:
    generator = _stylegan2().eval()
    x = torch.randn(1, generator.stages[0].to_mask.conv.in_ch, 8, 8)
    previous_logits = torch.full((1, 1, 4, 4), -4.0)
    style = torch.randn(1, 16)
    to_mask = generator.stages[0].to_mask
    with torch.no_grad():
        expected_logits = to_mask.conv(x, style).float() + to_mask.upsample(previous_logits)
        actual_logits = to_mask(x, style, previous_logits)
    assert actual_logits.dtype is torch.float32
    torch.testing.assert_close(actual_logits, expected_logits)
    # A negative skip must remain negative: only sigmoid for Mask guidance/output.
    assert (to_mask.upsample(previous_logits) < 0).all()
    assert not torch.allclose(actual_logits, actual_logits.sigmoid())


def test_mask_sigmoid_is_only_used_for_guidance_or_requested_output(monkeypatch) -> None:
    import models.stylegan2 as stylegan2_module

    model = _stylegan2(mask_noise_blank_level=9).eval()
    target = torch.randn(1, 3, 16, 16)
    identity = torch.randn(1, 32)
    calls: list[torch.dtype] = []
    sigmoid = stylegan2_module.mask_probability

    def observe(logits: torch.Tensor) -> torch.Tensor:
        calls.append(logits.dtype)
        return sigmoid(logits)

    monkeypatch.setattr(stylegan2_module, "mask_probability", observe)
    with torch.no_grad():
        model(target, identity)
        model(target, identity, return_mask_logits=True)
        assert not calls  # Neither the normal RGB path nor logits output needs sigmoid.
        model(target, identity, return_mask=True)
        assert calls == [torch.float32]  # Only the final probability.
        calls.clear()
        model.mask_noise_blank_level = -1
        model(target, identity, return_mask_logits=True)
        assert calls == [torch.float32, torch.float32]  # Guidance at 8x8 and 16x16.


def test_precomputed_sigmoid_is_only_used_for_guidance(monkeypatch) -> None:
    import swapface.split_identity as split_module

    full = _full(coarse_resolution=16, coarse_start_resolution=4, coarse_mask_noise_blank_level=9).eval()
    projection = StyleGAN2Projection(full.coarse).eval()
    client = PrecomputedStyleGAN2(full.coarse).eval()
    target = torch.randn(1, 3, 16, 16)
    identity = torch.randn(1, 32)
    calls = []
    sigmoid = split_module.mask_probability

    def observe(logits: torch.Tensor) -> torch.Tensor:
        calls.append(logits.dtype)
        return sigmoid(logits)

    monkeypatch.setattr(split_module, "mask_probability", observe)
    with torch.no_grad():
        conditioning = projection(identity)
        client(target, conditioning)
        assert not calls
        client.mask_noise_blank_level = -1
        client(target, conditioning)
        assert calls == [torch.float32, torch.float32]


def test_absolute_mask_level_indices_match_original_stylegan2() -> None:
    expected = {8: 1, 16: 3, 32: 5, 64: 7, 128: 9, 256: 11, 512: 13, 1024: 15}
    assert {resolution: stylegan2_mask_layer_index(resolution) for resolution in expected} == expected
    assert stylegan2_mask_layer_index(256) > 9
    assert stylegan2_mask_layer_index(128) <= 9


def test_mask_guidance_with_non_four_start_and_split_matches() -> None:
    # With start_resolution=32, 64x64 is still layer 7 (not relative layer 1).
    torch.manual_seed(33)
    full = _full(
        img_resolution=64,
        coarse_resolution=64,
        coarse_start_resolution=32,
        coarse_channels=[2, 2],
        coarse_mask_noise_blank_level=5,
    ).eval()
    for module in full.modules():
        if hasattr(module, "inject_gain"):
            module.inject_gain.data.fill_(1.0)
    generator = full.coarse.generator
    x = torch.randn(1, 3, 64, 64)
    identity = torch.randn(1, 32)
    projection = StyleGAN2Projection(full.coarse).eval()
    client = PrecomputedStyleGAN2(full.coarse).eval()
    with torch.no_grad():
        guided = full.coarse(x, identity)
        torch.testing.assert_close(guided, client(x, projection(identity)), rtol=1e-5, atol=1e-6)
        generator.mask_noise_blank_level = 7
        not_guided = full.coarse(x, identity)
    assert (guided - not_guided).abs().amax() > 1e-6


def test_mask_logits_not_available_if_mask_disabled() -> None:
    generator = _stylegan2(mask_enabled=False)
    x = torch.randn(1, 3, 16, 16)
    w = torch.randn(1, 32)
    with pytest.raises(ValueError, match="mask_enabled"):
        generator(x, w, return_mask_logits=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _stylegan2()(x, w, return_mask=True, return_mask_logits=True)


def test_mask_guidance_weighted_average_and_gradient() -> None:
    x = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]], requires_grad=True)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], requires_grad=True)
    result = mask_guided_target_features(x, mask, epsilon=1e-4, interpolation="nearest")
    expected = torch.tensor([[[[3.0, 3.0], [3.0, 7.0]]]])
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert mask.grad is not None and torch.isfinite(mask.grad).all()
    zero_mask = torch.zeros_like(mask)
    torch.testing.assert_close(mask_guided_target_features(x.detach(), zero_mask, epsilon=1e-4, interpolation="bilinear"), x.detach())


def test_guidance_layer_threshold_affects_rgb_not_mask_api() -> None:
    torch.manual_seed(7)
    generator = _stylegan2(mask_noise_blank_level=9).eval()
    for module in generator.modules():
        if hasattr(module, "inject_gain"):
            module.inject_gain.data.fill_(1.0)
    x = torch.randn(1, 3, 16, 16)
    w = torch.randn(1, 32)
    with torch.no_grad():
        plain_rgb, _ = generator(x, w, return_mask=True)
        generator.mask_noise_blank_level = -1
        guided_rgb, _ = generator(x, w, return_mask=True)
    assert (plain_rgb - guided_rgb).abs().max() > 1e-6


def test_coarse_mask_forward_and_split_projection_equal() -> None:
    torch.manual_seed(64)
    full = _full().eval()
    projection = StyleGAN2Projection(full.coarse).eval()
    client = PrecomputedStyleGAN2(full.coarse).eval()
    x = torch.randn(1, 3, 16, 16)
    identity = torch.randn(1, 32)
    with torch.no_grad():
        expected, resized, mask = full.coarse(x, identity, return_resize_in=True, return_mask=True)
        actual = client(x, projection(identity))
    assert resized.shape == expected.shape == (1, 3, 8, 8)
    assert mask.shape == (1, 1, 8, 8)
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
    assert projection.width == client.condition_width
    assert full(x, identity).shape == (1, 3, 16, 16)


def test_full_generator_returns_hq_resolution_mask_without_changing_rgb() -> None:
    torch.manual_seed(377)
    generator = _full().eval()
    target = torch.randn(1, 3, 16, 16)
    identity = torch.randn(1, 32)

    with torch.no_grad():
        plain_rgb = generator(target, identity)
        plain_rgb_and_coarse = generator(target, identity, return_coarse=True)
        rgb, coarse, hq_mask = generator(target, identity, return_coarse=True, return_mask=True)
        rgb_mask, hq_mask_without_coarse = generator(target, identity, return_mask=True)
        rgb_logits, hq_logits = generator(target, identity, return_mask_logits=True)
        _, coarse_mask = generator.coarse(target, identity, return_mask=True)
        _, coarse_logits = generator.coarse(target, identity, return_mask_logits=True)

    torch.testing.assert_close(rgb, plain_rgb)
    torch.testing.assert_close(rgb_mask, plain_rgb)
    torch.testing.assert_close(rgb_logits, plain_rgb)
    torch.testing.assert_close(plain_rgb_and_coarse[0], rgb)
    torch.testing.assert_close(plain_rgb_and_coarse[1], coarse)
    torch.testing.assert_close(hq_mask, hq_mask_without_coarse)
    assert hq_mask.shape == hq_logits.shape == (1, 1, 16, 16)
    assert hq_mask.dtype is torch.float32 and hq_logits.dtype is torch.float32
    torch.testing.assert_close(hq_mask, F.interpolate(coarse_mask, size=(16, 16), mode="bilinear", align_corners=False))
    torch.testing.assert_close(hq_logits, F.interpolate(coarse_logits, size=(16, 16), mode="bilinear", align_corners=False))
    assert 0 <= hq_mask.min() and hq_mask.max() <= 1


def test_full_generator_mask_output_validation() -> None:
    x = torch.randn(1, 3, 16, 16)
    identity = torch.randn(1, 32)
    generator = _full(coarse_mask_enabled=False)
    with pytest.raises(ValueError, match="mask_enabled"):
        generator(x, identity, return_mask=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _full()(x, identity, return_mask=True, return_mask_logits=True)
    assert generator(x, identity).shape == x.shape

    inswapper = FullGenerator(
        img_resolution=16,
        coarse_resolution=8,
        id_dim=32,
        coarse_type="inswapper",
        coarse_latent_resolution=4,
        coarse_num_latent=1,
        coarse_base_ch=8,
        coarse_max_ch=16,
        hq_bottleneck_resolution=4,
        hq_base_ch=2,
        hq_max_ch=8,
    )
    with pytest.raises(ValueError, match="StyleGAN2Coarse"):
        inswapper(x, identity, return_mask=True)


def test_mask_supervision_reuses_clean_face_parsing_and_coarse_id_gate() -> None:
    clean = torch.ones(2, 3, 8, 8)
    labels = torch.empty(2, 8, 8, dtype=torch.long)
    labels[:, :2] = 1  # face skin
    labels[:, 2:4] = 4  # excluded eye
    labels[:, 4:6] = 12  # lip: still allowed for Coarse IDLoss
    labels[:, 6:] = 17  # hair must not be replaced, even though ID gate permits it
    occlusion_gate = torch.ones(2, 1, 8, 8)
    occlusion_gate[:, :, :2, :4] = 0.2
    occlusion_gate[:, :, 4:6] = 0.5

    class Dataset:
        def next(self):
            return clean, clean, clean, torch.zeros(2, 2, 3), torch.zeros(2, dtype=torch.bool), clean

    class Parser:
        def predict_labels(self, images):
            assert torch.allclose(images, clean)
            return labels

        def get_buffer(self, name):
            assert name == "face_part_indices"
            return torch.tensor([1, 4, 12])

    class Occlusion:
        def apply(self, images):
            return images - 0.2, occlusion_gate

    trainer = Trainer.__new__(Trainer)
    trainer.dataset = Dataset()
    trainer.face_parser = Parser()
    trainer.occlusion = Occlusion()
    trainer.parsing_exclusion_indices = (torch.tensor([4]), torch.tensor([12]))
    trainer.enable_coarse_mask_loss = True
    _, augmented, _, _, _, _, (coarse_gate, hq_gate), _, teacher = trainer._next_training_batch()
    torch.testing.assert_close(augmented, clean - 0.2)
    face_mask = torch.isin(labels, torch.tensor([1, 4, 12])).unsqueeze(1).float()
    torch.testing.assert_close(teacher, face_mask * coarse_gate)
    assert teacher.shape == (2, 1, 8, 8)
    assert teacher[:, :, :2, :4].eq(0.2).all()  # preserve soft occlusion
    assert teacher[:, :, :2, 4:].eq(1).all()
    assert teacher[:, :, 2:4].eq(0).all()  # Coarse excludes eyes
    assert teacher[:, :, 4:6].eq(0.5).all()  # Coarse permits lips
    assert teacher[:, :, 6:].eq(0).all()  # hair excluded by base face mask
    assert hq_gate[:, :, 4:6].eq(0).all()  # HQ exclusions do not change Coarse Mask

    # When neither occlusion nor parsing exclusions are enabled, use the base face region.
    trainer.occlusion = None
    trainer.parsing_exclusion_indices = (None, None)
    _, clean_dst, _, _, _, _, (coarse_gate, _), _, teacher = trainer._next_training_batch()
    assert coarse_gate is None
    torch.testing.assert_close(clean_dst, clean)
    torch.testing.assert_close(teacher, face_mask)


def test_mask_bce_added_to_coarse_loss() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.enable_coarse_mask_loss = True
    trainer.coarse_mask_weight = 1.75
    trainer.geometry_augmentation_enabled = False
    for name in ("wfm", "gaze", "smirk", "vgg", "l1"):
        setattr(trainer, f"enable_coarse_{name}_loss", False)
    trainer.train_d_coarse = lambda x, *_args: x.mean(dim=(1, 2, 3)).unsqueeze(1)
    trainer.coarse_gan_loss = lambda scores: scores.sum() * 0
    trainer.coarse_identity_embeddings_forward = lambda x: x.mean(dim=(2, 3))
    trainer.prepare_identity_encoder_faces = lambda x, _theta: x
    trainer.coarse_id_loss = lambda a, b: a.sum() * 0
    trainer._log_identity_threshold_metrics = lambda *_args: None
    logged = {}
    trainer.log = lambda key, value, **_kwargs: logged.__setitem__(key, value)
    rgb = torch.randn(2, 3, 8, 8, requires_grad=True)
    prediction = torch.full((2, 1, 8, 8), math.log(4), requires_grad=True)
    teacher = torch.zeros(2, 1, 16, 16)
    loss = trainer._coarse_generator_stage_loss(
        rgb,
        rgb.detach(),
        rgb.detach(),
        None,
        None,
        rgb.detach().mean(dim=(2, 3)),
        None,
        None,
        coarse_mask_logits=prediction,
        coarse_mask_target=teacher,
    )
    assert math.isclose(loss.item(), -math.log(0.2) * 1.75, rel_tol=1e-5)
    assert "coarse_mask_loss" in logged
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    with pytest.raises(RuntimeError, match="Mask BCE"):
        trainer._coarse_generator_stage_loss(rgb, rgb.detach(), rgb.detach(), None, None, rgb.detach().mean(dim=(2, 3)), None, None)


def test_mask_bce_uses_area_resampling_for_soft_teacher() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.enable_coarse_mask_loss = True
    trainer.coarse_mask_weight = 1.0
    trainer.geometry_augmentation_enabled = False
    for name in ("wfm", "gaze", "smirk", "vgg", "l1"):
        setattr(trainer, f"enable_coarse_{name}_loss", False)
    trainer.train_d_coarse = lambda x, *_args: x.mean(dim=(1, 2, 3)).unsqueeze(1)
    trainer.coarse_gan_loss = lambda scores: scores.sum() * 0
    trainer.coarse_identity_embeddings_forward = lambda x: x.mean(dim=(2, 3))
    trainer.prepare_identity_encoder_faces = lambda x, _theta: x
    trainer.coarse_id_loss = lambda a, b: a.sum() * 0
    trainer._log_identity_threshold_metrics = lambda *_args: None
    trainer.log = lambda *_args, **_kwargs: None

    rgb = torch.zeros(1, 3, 2, 2)
    logits = torch.full((1, 1, 2, 2), 2.0, requires_grad=True)
    teacher = torch.zeros(1, 1, 4, 4)
    teacher[:, :, 0, 0] = 1.0
    teacher[:, :, 0, 1] = 0.5
    expected_soft = F.interpolate(teacher, size=(2, 2), mode="area")
    torch.testing.assert_close(expected_soft[0, 0, 0, 0], torch.tensor(0.375))
    loss = trainer._coarse_generator_stage_loss(
        rgb,
        rgb,
        rgb,
        None,
        None,
        rgb.mean((2, 3)),
        None,
        None,
        coarse_mask_logits=logits,
        coarse_mask_target=teacher,
    )
    torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(logits, expected_soft))
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_mask_config_validation() -> None:
    base = {"coarse_type": "stylegan2", "coarse_resolution": 128}
    normalized = _normalize_generator({"generator": base})
    assert normalized["coarse_mask_enabled"] is False
    assert normalized["coarse_mask_noise_blank_level"] == 9
    assert normalized["coarse_mask_guidance_epsilon"] == pytest.approx(1e-4)
    assert normalized["coarse_mask_guidance_interpolation"] == "bilinear"
    for field, invalid in (
        ("coarse_mask_enabled", "true"),
        ("coarse_mask_noise_blank_level", -2),
        ("coarse_mask_noise_blank_level", 1.5),
        ("coarse_mask_guidance_epsilon", 0),
        ("coarse_mask_guidance_epsilon", float("nan")),
        ("coarse_mask_guidance_interpolation", "bicubic"),
    ):
        with pytest.raises((TypeError, ValueError), match=field):
            _normalize_generator({"generator": base | {field: invalid}})
    with pytest.raises(ValueError, match="coarse_mask"):
        _normalize_generator({"generator": {"coarse_type": "inswapper", "coarse_mask_enabled": True}})
    # The sample TOML selects InSwapper; its historical normalized fields must
    # remain unchanged to allow resuming frozen pre-Mask runs.
    legacy = tomllib.loads((Path(__file__).parents[1] / "experiments/train.toml").read_text())
    resolved_legacy = resolve_train_config(legacy)
    assert resolved_legacy["loss"]["coarse"]["mask"] == {
        "weight": 1.0,
        "labels": [1, 2, 3, 4, 5, 6, 10, 11, 12, 13],
    }
    assert resolve_train_config(resolved_legacy) == resolved_legacy

    # StyleGAN2 uses the derived face × ID gate teacher; no duplicate labels.
    raw = tomllib.loads((Path(__file__).parents[1] / "experiments/train.toml").read_text())
    raw["generator"]["coarse_type"] = "stylegan2"
    resolved = resolve_train_config(raw)
    assert resolved["loss"]["coarse"]["mask"] == {"weight": 1.0}
    for field, invalid in (("weight", -1), ("labels", []), ("labels", [1, 1]), ("labels", [19])):
        invalid_raw = tomllib.loads((Path(__file__).parents[1] / "experiments/train.toml").read_text())
        invalid_raw["generator"]["coarse_type"] = "stylegan2"
        invalid_raw["loss"]["coarse"]["mask"][field] = invalid
        with pytest.raises((TypeError, ValueError), match="mask"):
            resolve_train_config(invalid_raw)

    for invalid in ([], [1, 1], [19]):
        invalid_legacy = tomllib.loads((Path(__file__).parents[1] / "experiments/train.toml").read_text())
        invalid_legacy["loss"]["coarse"]["mask"]["labels"] = invalid
        with pytest.raises((TypeError, ValueError), match="mask"):
            resolve_train_config(invalid_legacy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_mask_bce_is_safe_under_cuda_autocast() -> None:
    from torch.amp import autocast

    trainer = Trainer.__new__(Trainer)
    trainer.enable_coarse_mask_loss = True
    trainer.coarse_mask_weight = 1.0
    trainer.geometry_augmentation_enabled = False
    for name in ("wfm", "gaze", "smirk", "vgg", "l1"):
        setattr(trainer, f"enable_coarse_{name}_loss", False)
    trainer.train_d_coarse = lambda x, *_args: x.mean(dim=(1, 2, 3)).unsqueeze(1)
    trainer.coarse_gan_loss = lambda scores: scores.sum() * 0
    trainer.coarse_identity_embeddings_forward = lambda x: x.mean(dim=(2, 3))
    trainer.prepare_identity_encoder_faces = lambda x, _theta: x
    trainer.coarse_id_loss = lambda a, b: a.sum() * 0
    trainer._log_identity_threshold_metrics = lambda *_args: None
    trainer.log = lambda *_args, **_kwargs: None
    image = torch.randn(2, 3, 8, 8, device="cuda", requires_grad=True)
    # Incorrect saturated logits would yield zero gradient if sigmoid ran
    # under FP16 before BCE (sigmoid(10)=1, sigmoid(-20)=0).
    prediction = torch.empty((2, 1, 8, 8), device="cuda", dtype=torch.float16)
    prediction[:, :, :, :4] = 10.0
    prediction[:, :, :, 4:] = -20.0
    prediction.requires_grad_()
    teacher = torch.zeros((2, 1, 16, 16), device="cuda")
    teacher[:, :, :, 8:] = 1.0
    with autocast(device_type="cuda", dtype=torch.float16):
        loss = trainer._coarse_generator_stage_loss(
            image,
            image.detach(),
            image.detach(),
            None,
            None,
            image.detach().mean(dim=(2, 3)),
            None,
            None,
            coarse_mask_logits=prediction,
            coarse_mask_target=teacher,
        )
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert (prediction.grad[:, :, :, :4] > 0).all()
    assert (prediction.grad[:, :, :, 4:] < 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_intermediate_mask_head_retains_gradient_under_fp16_autocast() -> None:
    from torch.amp import autocast

    early = ToMask(4, 8, upsample=False, blur_kernel=(1, 3, 3, 1)).cuda()
    later = ToMask(4, 8, upsample=True, blur_kernel=(1, 3, 3, 1)).cuda()
    with torch.no_grad():
        early.conv.weight.fill_(5.0)
        early.conv.modulation.weight.zero_()
        early.conv.modulation.bias.fill_(1.0)
        later.conv.weight.zero_()
        later.conv.modulation.weight.zero_()
        later.conv.modulation.bias.fill_(1.0)

    with autocast(device_type="cuda", dtype=torch.float16):
        first_logits = early(torch.ones(1, 4, 4, 4, device="cuda"), torch.ones(1, 8, device="cuda"), None)
        final_logits = later(
            torch.ones(1, 4, 8, 8, device="cuda"),
            torch.ones(1, 8, device="cuda"),
            first_logits,
        )
        with autocast(device_type="cuda", enabled=False):
            loss = F.binary_cross_entropy_with_logits(final_logits, torch.zeros_like(final_logits))

    assert first_logits.dtype is torch.float32
    assert first_logits.min() >= 10.0  # FP16 sigmoid(10) would become exactly 1.
    assert final_logits.dtype is torch.float32
    loss.backward()
    assert early.conv.weight.grad is not None
    assert torch.isfinite(early.conv.weight.grad).all()
    assert early.conv.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_mask_fp16_precomputed_client_matches_generator() -> None:
    from torch.amp import autocast

    full = _full(coarse_resolution=16, coarse_start_resolution=4).cuda().eval()
    projection = StyleGAN2Projection(full.coarse).cuda().eval()
    client = PrecomputedStyleGAN2(full.coarse).cuda().eval()
    face = torch.randn(1, 3, 16, 16, device="cuda")
    identity = torch.randn(1, 32, device="cuda")
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
        rgb, mask = full.coarse(face, identity, return_mask=True)
        client_rgb = client(face, projection(identity))
    assert mask.dtype is torch.float32
    torch.testing.assert_close(rgb, client_rgb, atol=1e-3, rtol=1e-3)


def test_mask_precomputed_coarse_cpu_onnx_roundtrip(tmp_path) -> None:
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from swapface.export import OnnxExportOptions, export_to_onnx

    generator = _full().eval()
    projection = StyleGAN2Projection(generator.coarse).eval()
    client = PrecomputedStyleGAN2(generator.coarse).eval()
    face = torch.randn(1, 3, 16, 16)
    with torch.inference_mode():
        conditioning = projection(torch.randn(1, 32))
        expected = client(face, conditioning).numpy()
    path = export_to_onnx(
        client,
        (face, conditioning),
        output_dir=tmp_path,
        file_prefix="masked-stylegan2-client",
        metadata={},
        input_names=["target", "conditioning"],
        output_name="output",
        options=OnnxExportOptions(opset_version=18),
    )
    onnx.checker.check_model(onnx.load(path))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"target": face.numpy(), "conditioning": conditioning.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)


def test_mask_enabled_checkpoint_roundtrip(tmp_path) -> None:
    from misc.models.id_encoder import IDEncoderProvider
    from swapface.contracts import CHECKPOINT_VERSION
    from swapface.inference import load_generator

    torch.manual_seed(109)
    model = _full(id_dim=512).eval()
    checkpoint_path = tmp_path / "step_000000001.pth"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "face_alignment": "ffhq",
            "step": 1,
            "identity_encoders": {"generator": IDEncoderProvider.BLENDFACE.name},
            "net_g": {"network_cfg": model.network_cfg, "state_dict": model.state_dict()},
        },
        checkpoint_path,
    )
    loaded, provider, step = load_generator(checkpoint_path)
    assert provider is IDEncoderProvider.BLENDFACE and step == 1
    assert loaded.network_cfg == model.network_cfg
    assert loaded.coarse.generator.mask_enabled is True
    assert loaded.coarse.generator.to_mask_first is not None
    with torch.inference_mode():
        target = torch.randn(1, 3, 16, 16)
        identity = torch.randn(1, 512)
        expected = model(target, identity)
        actual = loaded(target, identity)
    torch.testing.assert_close(actual, expected)


def test_mask_full_model_cpu_onnx_roundtrip(tmp_path) -> None:
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from swapface.export import OnnxExportOptions, export_to_onnx

    generator = _full().eval()
    x = torch.randn(1, 3, 16, 16)
    w = torch.randn(1, 32)
    with torch.no_grad():
        expected = generator(x, w).numpy()
    path = export_to_onnx(
        generator,
        (x, w),
        output_dir=tmp_path,
        file_prefix="masked-stylegan2",
        metadata={},
        input_names=["target", "identity"],
        output_name="output",
        options=OnnxExportOptions(opset_version=18),
    )
    onnx.checker.check_model(onnx.load(path))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"target": x.numpy(), "identity": w.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
