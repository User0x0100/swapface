"""StyleGAN2 Coarse integration contracts: config, training, checkpoint and export."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from misc.models.id_encoder import IDEncoderProvider
from models.networks import Coarse, Generator, StyleGAN2Coarse
from swapface.config import _normalize_generator
from swapface.contracts import CHECKPOINT_VERSION
from swapface.inference import load_generator
from swapface.train import _branch_generator_mode


def small_config(*, coarse_type: str = "stylegan2") -> dict:
    return {
        "img_resolution": 16,
        "img_channels": 3,
        "id_dim": 32,
        "coarse_resolution": 8,
        "coarse_latent_resolution": 4,
        "coarse_num_latent": 1,
        "coarse_base_ch": 4,
        "coarse_max_ch": 8,
        "coarse_type": coarse_type,
        "coarse_w_dim": 32,
        "coarse_mapping_layers": 1,
        "coarse_channel_multiplier": 1,
        "hq_bottleneck_resolution": 4,
        "hq_base_ch": 2,
        "hq_max_ch": 8,
    }


def test_legacy_generator_config_remains_unchanged() -> None:
    legacy = small_config(coarse_type="inswapper")
    for key in ("coarse_type", "coarse_w_dim", "coarse_mapping_layers", "coarse_channel_multiplier"):
        legacy.pop(key)

    assert _normalize_generator({"generator": legacy}) == legacy
    model = Generator(**legacy)
    assert model.network_cfg == legacy
    assert isinstance(model.coarse, Coarse)


def test_stylegan2_coarse_config_and_validation() -> None:
    config = _normalize_generator({"generator": {"coarse_type": "stylegan2", "coarse_resolution": 128}})
    assert config["coarse_w_dim"] == 512
    assert config["coarse_mapping_layers"] == 8
    assert config["coarse_channel_multiplier"] == 1
    assert "coarse_target_inject_mode" not in config
    assert "coarse_target_inject_layers" not in config

    selective = _normalize_generator({"generator": {
        "coarse_type": "stylegan2",
        "coarse_resolution": 128,
        "coarse_target_inject_mode": "add",
        "coarse_target_inject_layers": [2, 0, 1],
    }})
    assert selective["coarse_target_inject_mode"] == "add"
    assert selective["coarse_target_inject_layers"] == [0, 1, 2]
    default_layers = _normalize_generator({"generator": {
        "coarse_type": "stylegan2",
        "coarse_target_inject_mode": "concat",
        "coarse_resolution": 128,
    }})
    assert default_layers["coarse_target_inject_layers"] == list(range(6))
    for invalid in ([0, 0], [6], [-1], [True], ["1"], "0,1"):
        with pytest.raises(ValueError, match="coarse_target_inject_layers"):
            _normalize_generator({"generator": {
                "coarse_type": "stylegan2", "coarse_target_inject_layers": invalid,
            }})
    with pytest.raises(ValueError, match="coarse_target_inject_mode"):
        _normalize_generator({"generator": {
            "coarse_type": "stylegan2", "coarse_target_inject_mode": "bad",
        }})

    for invalid in ("invalid", 2):
        with pytest.raises(ValueError, match="coarse_type"):
            _normalize_generator({"generator": {"coarse_type": invalid}})

    with pytest.raises(ValueError, match="2 的整数次幂"):
        _normalize_generator({"generator": {"coarse_type": "stylegan2", "coarse_resolution": 96}})
    with pytest.raises((ValueError, TypeError), match="coarse_mapping_layers"):
        _normalize_generator({"generator": {"coarse_type": "stylegan2", "coarse_mapping_layers": 0}})

    with pytest.raises(ValueError, match="unknown coarse_type"):
        Generator(**(small_config() | {"coarse_type": "invalid"}))


def test_stylegan2_forward_training_gradients_and_hq_boundary() -> None:
    torch.manual_seed(1)
    net = Generator(**small_config())
    assert isinstance(net.coarse, StyleGAN2Coarse)

    target = torch.randn(2, 3, 16, 16, requires_grad=True)
    identity = torch.randn(2, 32, requires_grad=True)
    # The new StyledConv starts with zero injection; activate it to verify
    # gradients reach the target encoder as training progresses.
    for module in net.coarse.modules():
        if hasattr(module, "inject_gain"):
            module.inject_gain.data.fill_(0.25)

    coarse, reference = net.coarse(target, identity, return_resize_in=True)
    assert coarse.shape == reference.shape == (2, 3, 8, 8)
    torch.testing.assert_close(reference, F.interpolate(target, size=8, mode="bilinear", align_corners=False))
    with torch.no_grad():
        expected_raw = net.coarse.generator(reference, identity)
    torch.testing.assert_close(coarse.detach(), expected_raw)
    assert torch.isfinite(coarse).all()
    coarse.mean().backward()
    assert target.grad is not None and target.grad.abs().sum() > 0
    assert identity.grad is not None and identity.grad.abs().sum() > 0
    assert net.coarse.generator.encoder.from_rgb.weight.grad is not None
    assert net.coarse.generator.encoder.from_rgb.weight.grad.abs().sum() > 0
    assert net.coarse.generator.conv4.conv.weight.grad is not None

    net.zero_grad(set_to_none=True)
    output, low = net(target.detach(), identity.detach(), return_coarse=True)
    assert output.shape == (2, 3, 16, 16)
    torch.testing.assert_close(low, net.coarse(target.detach(), identity.detach()))

    # HQ-only training must not update Coarse.
    net.zero_grad(set_to_none=True)
    with torch.no_grad():
        frozen_coarse = net.coarse(target.detach(), identity.detach())
    net.hq(target.detach(), frozen_coarse).square().mean().backward()
    assert all(p.grad is None for p in net.coarse.parameters())
    assert any(p.grad is not None for p in net.hq.parameters())


def test_stylegan2_coarse_does_not_squash_raw_rgb() -> None:
    from unittest.mock import patch

    net = StyleGAN2Coarse(img_resolution=8, id_dim=32, w_dim=32, mapping_layers=1)
    x = torch.randn(1, 3, 16, 16)
    identity = torch.randn(1, 32)
    raw = torch.tensor([2.0, -2.0, 0.5]).view(1, 3, 1, 1).expand(1, 3, 8, 8)
    with patch.object(net.generator, "forward", return_value=raw):
        output, resized = net(x, identity, return_resize_in=True)
    torch.testing.assert_close(output, raw)
    torch.testing.assert_close(resized, F.interpolate(x, size=8, mode="bilinear", align_corners=False))


def test_stylegan2_checkpoint_roundtrip_and_coarse_branch_guards(tmp_path) -> None:
    # Checkpoint inference requires the application's actual 512-D identity contract.
    config = small_config() | {"id_dim": 512}
    model = Generator(**config).eval()
    path = tmp_path / "step_000000001.pth"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "step": 1,
            "identity_encoders": {"generator": IDEncoderProvider.BLENDFACE.name},
            "net_g": {"network_cfg": model.network_cfg, "state_dict": model.state_dict()},
        },
        path,
    )
    loaded, provider, step = load_generator(path)
    assert provider is IDEncoderProvider.BLENDFACE and step == 1
    assert isinstance(loaded.coarse, StyleGAN2Coarse)
    assert loaded.network_cfg == model.network_cfg
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], weight)

    parent = {"net_g": {"network_cfg": model.network_cfg}, "identity_encoders": {"generator": "BLENDFACE"}}
    current = {"generator": config, "train": {"stage": "hq"}, "identity": {"provider": "BLENDFACE"}}
    assert _branch_generator_mode(parent, current) == "inherit"
    changed_hq = {"generator": config | {"img_resolution": 32}, "train": {"stage": "hq"}, "identity": {"provider": "BLENDFACE"}}
    assert _branch_generator_mode(parent, changed_hq) == "hq_rebuild"
    for key, val in (
        ("coarse_type", "inswapper"), ("coarse_mapping_layers", 2), ("coarse_channel_multiplier", 2),
        ("coarse_target_inject_mode", "add"), ("coarse_target_inject_layers", [0]),
    ):
        different = {"generator": config | {key: val}, "train": {"stage": "hq"}, "identity": {"provider": "BLENDFACE"}}
        with pytest.raises(ValueError, match="Coarse"):
            _branch_generator_mode(parent, different)

    # Inactive legacy options must not invalidate a StyleGAN2 Coarse checkpoint.
    unused_legacy = {"generator": config | {"coarse_base_ch": 32}, "train": {"stage": "hq"}, "identity": {"provider": "BLENDFACE"}}
    assert _branch_generator_mode(parent, unused_legacy) == "inherit"


@pytest.mark.parametrize("injection", [
    {},
    {"coarse_target_inject_mode": "concat", "coarse_target_inject_layers": [0]},
    {"coarse_target_inject_mode": "add", "coarse_target_inject_layers": [0]},
])
def test_stylegan2_strict_cpu_export_and_runtime(tmp_path, injection: dict) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from swapface.export import OnnxExportOptions, export_to_onnx

    torch.manual_seed(31)
    model = Generator(**(small_config() | injection)).eval()
    target = torch.randn(1, 3, 16, 16)
    identity = torch.randn(1, 32)
    with torch.inference_mode():
        expected = model(target, identity).numpy()

    path = export_to_onnx(
        model,
        (target, identity),
        output_dir=tmp_path,
        file_prefix="coarse-hq",
        metadata={},
        input_names=["target", "identity"],
        output_name="output",
        options=OnnxExportOptions(opset_version=18),
    )
    exported = onnx.load(path)
    onnx.checker.check_model(exported)
    assert all(node.domain in ("", "ai.onnx") for node in exported.graph.node)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=options)
    actual = session.run(None, {"target": target.numpy(), "identity": identity.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)


def test_swapface_export_forces_cpu_even_if_cuda_requested() -> None:
    from swapface import export

    model = Generator(**small_config())
    with (
        patch.object(export, "load_generator", return_value=(model, IDEncoderProvider.BLENDFACE, 1)),
        patch.object(export, "export_to_onnx", return_value=Path("/tmp/mock.onnx")) as mocked,
    ):
        export.export_swapface("mock.pth", device="cuda", batch_size=1)
    wrapped, inputs = mocked.call_args.args[:2]
    assert all(p.device.type == "cpu" for p in wrapped.parameters())
    assert all(t.device.type == "cpu" for t in inputs)


@pytest.mark.parametrize("mode", ["concat", "add"])
def test_selective_stylegan2_split_identity_matches_full_model(mode: str) -> None:
    from swapface.split_identity import PrecomputedStyleGAN2, StyleGAN2Projection

    config = small_config() | {
        "coarse_target_inject_mode": mode,
        "coarse_target_inject_layers": [0],
    }
    model = Generator(**config).eval()
    assert isinstance(model.coarse, StyleGAN2Coarse)
    projection = StyleGAN2Projection(model.coarse).eval()
    for layer in model.coarse.generator.modules():
        if hasattr(layer, "inject_gain"):
            layer.inject_gain.data.fill_(0.25)
    client = PrecomputedStyleGAN2(model.coarse).eval()
    target = torch.randn(1, 3, 16, 16)
    identity = torch.randn(1, 32)
    with torch.no_grad():
        conditioning = projection(identity)
        expected = model.coarse(target, identity)
        actual = client(target, conditioning)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_direct_constructor_records_canonical_layer_order() -> None:
    config = small_config() | {
        "coarse_target_inject_mode": "concat",
        "coarse_target_inject_layers": [1, 0],
    }
    model = Generator(**config)
    assert model.network_cfg["coarse_target_inject_layers"] == [0, 1]
    assert model.coarse.generator.target_inject_layers == (0, 1)
