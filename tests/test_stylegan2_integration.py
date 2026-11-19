"""Single-stage StyleGAN2 integration, architecture configuration and export parity."""

from pathlib import Path

import pytest
import torch
from torch import nn

from misc.models.id_encoder import IDEncoderProvider
from models.networks import Generator, StyleGAN2Swap
from swapface.config import _normalize_generator
from swapface.contracts import CHECKPOINT_VERSION
from swapface.inference import load_generator
from swapface.split_identity import build_split_models
from swapface.train import _canonical_generator_config


def config(**updates):
    values = {"type": "stylegan2", "img_resolution": 16, "id_dim": 512,
              "w_dim": 16, "mapping_layers": 1, "start_resolution": 4,
              "channels": [8, 16], "mask_enabled": False,
              "target_inject_mode": "concat", "target_inject_layers": [0, 1]}
    return _normalize_generator({"generator": values | updates})


def test_stylegan2_generator_respects_original_synthesis() -> None:
    cfg = config()
    g = Generator(**cfg)
    assert isinstance(g.core, StyleGAN2Swap)
    assert not hasattr(g, "hq")
    assert g.network_cfg == cfg
    identity = torch.randn(2, 512, requires_grad=True)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    for m in g.modules():
        if hasattr(m, "inject_gain"):
            m.inject_gain.data.fill_(.25)
    rgb = g(x, identity)
    assert rgb.shape == x.shape
    rgb.square().mean().backward()
    assert torch.count_nonzero(identity.grad) > 0
    assert torch.count_nonzero(x.grad) > 0
    assert _canonical_generator_config(g.network_cfg) == g.network_cfg


@pytest.mark.parametrize("mode", ["concat", "add"])
@pytest.mark.parametrize("mask", [False, True])
def test_server_client_parity(mode: str, mask: bool) -> None:
    g = Generator(**config(target_inject_mode=mode, mask_enabled=mask)).eval()
    with torch.no_grad():
        for layer in g.modules():
            if hasattr(layer, "inject_gain"):
                layer.inject_gain.fill_(.25)
    server, client = build_split_models(g, nn.Identity())
    face, identity = torch.randn(2, 3, 16, 16), torch.randn(2, 512)
    with torch.no_grad():
        torch.testing.assert_close(g(face, identity), client(face, server(identity)), atol=1e-5, rtol=1e-5)
    torch.export.export(client.eval(), (face, server(identity).detach()), strict=True)


def test_mask_enabled_checkpoint(tmp_path: Path) -> None:
    cfg = config(mask_enabled=True)
    g = Generator(**cfg)
    checkpoint = tmp_path / "step_000000012.pth"
    torch.save({"version": CHECKPOINT_VERSION, "face_alignment": "ffhq", "step": 12,
                "identity_encoders": {"generator": IDEncoderProvider.BLENDFACE.name},
                "net_g": {"network_cfg": g.network_cfg, "state_dict": g.state_dict()}}, checkpoint)
    loaded, _, step = load_generator(checkpoint)
    assert step == 12
    image, identity = torch.randn(1, 3, 16, 16), torch.randn(1, 512)
    with torch.no_grad():
        rgb, mask = loaded(image, identity, return_mask=True)
    assert rgb.shape == image.shape and mask.shape == (1, 1, 16, 16)
    assert torch.isfinite(mask).all()


@pytest.mark.parametrize("start", [3, 12, 32, 0])
def test_start_resolution_validation(start: int) -> None:
    with pytest.raises((ValueError, TypeError), match="start_resolution"):
        config(start_resolution=start)


@pytest.mark.parametrize("channels", [[], [8], [16, 8], [0, 16], [True, 16], [8, 16, 32]])
def test_channels_validation(channels) -> None:
    with pytest.raises((ValueError, TypeError), match="channels"):
        config(channels=channels)


@pytest.mark.parametrize("invalid", [[0, 0], [3], [-1], [True], "1"])
def test_injection_indices_validation(invalid) -> None:
    with pytest.raises((ValueError, TypeError), match="target_inject_layers"):
        config(target_inject_layers=invalid)


def test_architecture_specific_fields_rejected() -> None:
    with pytest.raises(ValueError, match="未知字段"):
        _normalize_generator({"generator": {"type": "inswapper", "mask_enabled": True}})
    with pytest.raises(ValueError, match="未知字段"):
        _normalize_generator({"generator": {"type": "stylegan2", "latent_resolution": 32}})


@pytest.mark.parametrize("field, invalid", [
    ("start_resolution", 0),
    ("start_resolution", -4),
    ("start_resolution", "4"),
    ("start_resolution", 3),
    ("start_resolution", 32),
    ("img_resolution", 0),
    ("img_resolution", "16"),
    ("img_resolution", 3),
])
def test_invalid_stylegan2_resolutions_fail_during_validation(field, invalid) -> None:
    raw = {"type": "stylegan2", "img_resolution": 16, "start_resolution": 4}
    raw[field] = invalid
    with pytest.raises((ValueError, TypeError), match=field):
        _normalize_generator({"generator": raw})


def test_default_target_injection_layers_follow_validated_resolution() -> None:
    for img, start, expected in ((16, 4, [0, 1, 2]), (16, 16, [0]), (32, 8, [0, 1, 2])):
        cfg = _normalize_generator({"generator": {"type": "stylegan2", "img_resolution": img, "start_resolution": start}})
        assert cfg["target_inject_layers"] == expected
        assert _normalize_generator({"generator": cfg}) == cfg
    cfg = _normalize_generator({"generator": {"type": "stylegan2", "img_resolution": 16,
                                                 "start_resolution": 4, "target_inject_layers": []}})
    assert cfg["target_inject_layers"] == []
