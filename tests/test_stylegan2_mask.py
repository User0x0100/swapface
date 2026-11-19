"""StyleGAN2 single-stage Mask synthesis, logits, guidance and teacher tests."""

import pytest
import torch
import torch.nn.functional as F

from models.networks import Generator as FullGenerator
from models.stylegan2 import Generator, mask_guided_target_features, stylegan2_mask_layer_index
from swapface.config import _normalize_generator, resolve_train_config
from swapface.split_identity import build_split_models
from swapface.train import Trainer


def _stylegan2(**kwargs) -> Generator:
    cfg = {"img_resolution": 16, "id_dim": 32, "w_dim": 16, "mapping_layers": 1,
           "channels": (8, 16), "mask_enabled": True}
    return Generator(**(cfg | kwargs))


def _full(**kwargs) -> FullGenerator:
    cfg = {"img_resolution": 16, "id_dim": 32, "type": "stylegan2", "start_resolution": 4,
           "channels": [8, 16], "w_dim": 16, "mapping_layers": 1,
           "mask_enabled": True, "mask_noise_blank_level": -1}
    return FullGenerator(**(cfg | kwargs))


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

def test_absolute_mask_level_indices_match_original_stylegan2() -> None:
    expected = {8: 1, 16: 3, 32: 5, 64: 7, 128: 9, 256: 11, 512: 13, 1024: 15}
    assert {resolution: stylegan2_mask_layer_index(resolution) for resolution in expected} == expected
    assert stylegan2_mask_layer_index(256) > 9
    assert stylegan2_mask_layer_index(128) <= 9

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

def test_full_model_mask_logits_and_client_consistency() -> None:
    full = _full().eval()
    x, identity = torch.randn(1, 3, 16, 16), torch.randn(1, 32)
    with torch.no_grad():
        rgb = full(x, identity)
        other, mask = full(x, identity, return_mask=True)
        _, logits = full(x, identity, return_mask_logits=True)
    torch.testing.assert_close(rgb, other)
    torch.testing.assert_close(mask, logits.float().sigmoid())
    assert rgb.shape == (1, 3, 16, 16) and mask.shape == (1, 1, 16, 16)
    server, client = build_split_models(full, torch.nn.Identity())
    with torch.no_grad():
        torch.testing.assert_close(rgb, client(x, server(identity)), atol=1e-5, rtol=1e-5)


def test_mask_type_and_training_config() -> None:
    cfg = _normalize_generator({"generator": {"type": "stylegan2", "mask_enabled": True}})
    assert cfg["mask_enabled"]
    with pytest.raises(ValueError, match="未知字段"):
        _normalize_generator({"generator": {"type": "inswapper", "mask_enabled": True}})
    for field, invalid in (("mask_enabled", "true"), ("mask_noise_blank_level", -2),
                           ("mask_guidance_epsilon", 0), ("mask_guidance_interpolation", "bicubic")):
        with pytest.raises((TypeError, ValueError), match=field):
            _normalize_generator({"generator": {"type": "stylegan2", field: invalid}})
    import tomllib
    from pathlib import Path
    raw = tomllib.loads((Path(__file__).parents[1] / "experiments/train.toml").read_text())
    with pytest.raises((TypeError, ValueError), match="mask"):
        resolve_train_config(raw | {"loss": raw["loss"] | {"mask": {"weight": 1.0}}})
    # A disabled Mask must not create unused Mask loss configuration.
    disabled = dict(raw)
    disabled["generator"] = {"type": "stylegan2", "img_resolution": 512, "mask_enabled": False}
    assert "mask" not in resolve_train_config(disabled)["loss"]
    with pytest.raises(ValueError, match="mask"):
        resolve_train_config({**disabled, "loss": disabled["loss"] | {"mask": {"weight": 1.0}}})
    raw["generator"] = {"type": "stylegan2", "img_resolution": 512, "mask_enabled": True}
    raw["loss"]["mask"] = {"weight": 1.0}
    resolved = resolve_train_config(raw)
    assert resolved["loss"]["mask"] == {"weight": 1.0}


def test_mask_teacher_and_bce_gradient() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.enable_wfm_loss = False
    trainer.enable_mask_loss = True
    trainer.mask_weight = 2.0
    trainer.enable_gaze_loss = trainer.enable_smirk_loss = trainer.enable_l1_loss = trainer.enable_vgg_loss = False
    trainer.train_d = lambda x, *args: x.mean()
    trainer.gan_loss = lambda scores: scores * 0
    trainer.identity_embeddings_forward = lambda x: x.mean(dim=(2, 3))
    trainer.prepare_identity_encoder_faces = lambda x, theta_restore=None: x
    trainer.id_loss = lambda a, b: (a - b).mean() * 0
    trainer.id_loss.min_similarity = None
    trainer._log_identity_threshold_metrics = lambda *args: None
    trainer.log = lambda *args: None
    trainer.geometry_augmentation_enabled = False
    rgb = torch.zeros(2, 3, 8, 8, requires_grad=True)
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    teacher = torch.rand(2, 1, 8, 8)
    loss = trainer._generator_loss(rgb, rgb, rgb, torch.zeros(2, 2, 3), torch.ones(2),
                                   torch.zeros(2, 3), None, None, None, None, logits, teacher)
    expected = F.binary_cross_entropy_with_logits(logits.float(), F.interpolate(teacher, (4, 4), mode="area")) * 2.0
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_mask_bce_stays_stable_with_saturated_fp16_logits() -> None:
    """BCE must receive FP32 logits, not FP16 sigmoid probabilities."""
    from torch.amp import autocast

    trainer = Trainer.__new__(Trainer)
    trainer.enable_mask_loss = True
    trainer.mask_weight = 1.0
    trainer.enable_wfm_loss = False
    trainer.enable_gaze_loss = trainer.enable_smirk_loss = trainer.enable_vgg_loss = trainer.enable_l1_loss = False
    trainer.geometry_augmentation_enabled = False
    trainer.train_d = lambda x, *args: x.mean()
    trainer.gan_loss = lambda scores: scores * 0
    trainer.identity_embeddings_forward = lambda x: x.mean(dim=(2, 3))
    trainer.prepare_identity_encoder_faces = lambda x, theta_restore=None: x
    trainer.id_loss = lambda a, b: a.sum() * 0
    trainer.id_loss.min_similarity = None
    trainer._log_identity_threshold_metrics = lambda *args: None
    trainer.log = lambda *args, **kwargs: None

    generated = torch.zeros(2, 3, 8, 8, device="cuda", requires_grad=True)
    logits = torch.empty(2, 1, 8, 8, device="cuda", dtype=torch.float16)
    logits[:, :, :, :4] = 10.0
    logits[:, :, :, 4:] = -20.0
    logits.requires_grad_()
    teacher = torch.zeros(2, 1, 16, 16, device="cuda")
    teacher[:, :, :, 8:] = 1.0
    with autocast(device_type="cuda", dtype=torch.float16):
        loss = trainer._generator_loss(
            generated, generated.detach(), generated.detach(),
            torch.zeros(2, 2, 3, device="cuda"), torch.ones(2, device="cuda"),
            torch.zeros(2, 3, device="cuda"), None, None, None, None,
            logits, teacher,
        )
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert (logits.grad[:, :, :, :4] > 0).all()
    assert (logits.grad[:, :, :, 4:] < 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_early_mask_head_preserves_fp32_logits_and_fp16_gradients() -> None:
    from torch.amp import autocast

    from models.stylegan2 import ToMask

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
        first = early(torch.ones(1, 4, 4, 4, device="cuda"), torch.ones(1, 8, device="cuda"), None)
        last = later(torch.ones(1, 4, 8, 8, device="cuda"), torch.ones(1, 8, device="cuda"), first)
        with autocast(device_type="cuda", enabled=False):
            loss = F.binary_cross_entropy_with_logits(last, torch.zeros_like(last))
    assert first.dtype == torch.float32 and first.min() >= 10
    assert last.dtype == torch.float32
    loss.backward()
    assert early.conv.weight.grad is not None and torch.isfinite(early.conv.weight.grad).all()
    assert early.conv.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_fp16_stylegan2_mask_split_client_matches_full_generator() -> None:
    from torch.amp import autocast

    model = _full(mask_noise_blank_level=-1).cuda().eval()
    server, client = build_split_models(model, torch.nn.Identity())
    server, client = server.cuda().eval(), client.cuda().eval()
    target = torch.randn(1, 3, 16, 16, device="cuda")
    identity = torch.randn(1, 32, device="cuda")
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
        rgb, mask = model(target, identity, return_mask=True)
        other = client(target, server(identity))
    assert mask.dtype == torch.float32
    torch.testing.assert_close(other, rgb, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("inject_mode", ["concat", "add"])
def test_mask_guidance_client_onnxruntime_round_trip(inject_mode, tmp_path) -> None:
    """Mask guidance must survive an actual ONNX Runtime deployment path."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    from swapface.export import OnnxExportOptions, export_to_onnx

    generator = _full(mask_enabled=True, mask_noise_blank_level=-1,
                      target_inject_mode=inject_mode).eval()
    with torch.no_grad():
        for layer in generator.modules():
            if hasattr(layer, "inject_gain"):
                layer.inject_gain.fill_(0.25)
    server, client = build_split_models(generator, torch.nn.Identity())
    target = torch.randn(1, 3, 16, 16)
    source = torch.randn(1, 32)
    with torch.inference_mode():
        condition = server(source)
        expected = client(target, condition).numpy()
        torch.testing.assert_close(generator(target, source), client(target, condition))

    path = export_to_onnx(
        client.eval(), (target, condition), output_dir=tmp_path, file_prefix=f"masked-{inject_mode}",
        metadata={}, input_names=["faces", "identity_condition"], output_name="swapped_faces",
        options=OnnxExportOptions(opset_version=18),
    )
    onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(["swapped_faces"], {"faces": target.numpy(), "identity_condition": condition.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)
