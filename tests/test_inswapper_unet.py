"""Single-stage InSwapper U-Net architecture, topology, and client split contract."""

from pathlib import Path

import pytest
import torch
from torch import nn

from models.networks import Generator, InSwapper
from swapface.config import _normalize_generator
from swapface.contracts import CHECKPOINT_VERSION
from swapface.inference import load_generator
from swapface.split_identity import build_split_models


@pytest.mark.parametrize("mode", ["cat", "add"])
@pytest.mark.parametrize("layers", [[], [0], [1], [0, 1]])
def test_skip_fusion_matches_selected_encoder_features(mode: str, layers: list[int]) -> None:
    torch.manual_seed(13)
    model = InSwapper(img_resolution=16, latent_resolution=4, num_latent=1, base_ch=8, max_ch=16, id_dim=8, skip_mode=mode, skip_layers=layers)
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for i, block in enumerate(model.encoder):
        handles.append(block.register_forward_hook(lambda _module, _args, output, i=i: captured.__setitem__(f"e{i}", output)))
    handles.append(model.from_rgb.register_forward_hook(lambda _module, _args, output: captured.__setitem__("from", output)))
    for i, block in enumerate(model.decoder):
        handles.append(block.register_forward_hook(lambda _module, _args, output, i=i: captured.__setitem__(f"d{i}", output)))
    handles.append(model.to_rgb.register_forward_pre_hook(lambda _module, args: captured.__setitem__("rgb_input", args[0])))
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    identity = torch.randn(2, 8, requires_grad=True)
    try:
        output, resized = model(x, identity, return_resize_in=True)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(resized, x)
    assert output.shape == (2, 3, 16, 16)
    # Deepest decoder skip comes from the last encoder output above bottleneck.
    expected_channels = model.from_rgb[2].out_channels * (2 if 1 in layers and mode == "cat" else 1)
    assert model.to_rgb[0].in_channels == expected_channels
    if layers and layers[-1] == 1:
        expected = torch.cat((captured["d1"], captured["from"]), dim=1) if mode == "cat" else captured["d1"] + captured["from"]
        torch.testing.assert_close(captured["rgb_input"], expected)
    else:
        torch.testing.assert_close(captured["rgb_input"], captured["d1"])
    output.square().mean().backward()
    assert torch.count_nonzero(x.grad) > 0
    assert torch.count_nonzero(identity.grad) > 0
    assert torch.count_nonzero(model.from_rgb[0].weight.grad) > 0


@pytest.mark.parametrize("mode", ["cat", "add"])
@pytest.mark.parametrize("layers", [[], [0], [1], [0, 1]])
def test_split_runtime_equals_generator(mode: str, layers: list[int]) -> None:
    generator = Generator(img_resolution=16, id_dim=512, latent_resolution=4, num_latent=1, base_ch=4, max_ch=8, skip_mode=mode, skip_layers=layers).eval()
    server, client = build_split_models(generator, nn.Identity())
    x, identity = torch.randn(1, 3, 16, 16), torch.randn(1, 512)
    with torch.no_grad():
        expected = generator(x, identity)
        actual = client(x, server(identity))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["cat", "add"])
def test_network_configuration_and_checkpoint_round_trip(mode: str, tmp_path: Path) -> None:
    raw = {
        "type": "inswapper",
        "img_resolution": 16,
        "id_dim": 512,
        "latent_resolution": 4,
        "num_latent": 1,
        "base_ch": 4,
        "max_ch": 8,
        "skip_mode": mode,
        "skip_layers": [1, 0],
    }
    normalized = _normalize_generator({"generator": raw})
    assert normalized["skip_layers"] == [0, 1]
    model = Generator(**normalized)
    assert model.network_cfg == normalized
    path = tmp_path / "step_000000003.pth"
    torch.save(
        {"version": CHECKPOINT_VERSION, "face_alignment": "ffhq", "step": 3, "identity_encoders": {"generator": "WEBFACE600K_ARCFACE_R50"}, "net_g": {"network_cfg": model.network_cfg, "state_dict": model.state_dict()}}, path
    )
    loaded, _, step = load_generator(path)
    assert step == 3 and loaded.network_cfg == model.network_cfg
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])


@pytest.mark.parametrize("invalid", [[0, 0], [2], [-1], [True], "1", [1.0]])
def test_skip_layer_validation(invalid) -> None:
    with pytest.raises((ValueError, TypeError), match="skip_layers"):
        _normalize_generator({"generator": {"type": "inswapper", "img_resolution": 16, "latent_resolution": 4, "skip_layers": invalid}})


def test_invalid_mode_and_strict_type_separation() -> None:
    with pytest.raises(ValueError, match="skip_mode"):
        _normalize_generator({"generator": {"type": "inswapper", "skip_mode": "concat"}})
    with pytest.raises(ValueError, match="unknown"):
        Generator(type="invalid")
    with pytest.raises(ValueError, match="未知字段"):
        _normalize_generator({"generator": {"type": "stylegan2", "skip_layers": [0]}})
    with pytest.raises(ValueError, match="未知字段"):
        _normalize_generator({"generator": {"type": "inswapper", "mask_enabled": True}})


@pytest.mark.parametrize("latent_resolution", [0, 1])
def test_reflection_padding_requires_at_least_two_bottleneck_pixels(latent_resolution: int) -> None:
    with pytest.raises(ValueError, match="latent_resolution"):
        _normalize_generator({"generator": {"img_resolution": 16, "latent_resolution": latent_resolution}})
    with pytest.raises(ValueError, match="latent_resolution"):
        InSwapper(img_resolution=16, latent_resolution=latent_resolution)


def test_two_pixel_bottleneck_is_valid() -> None:
    model = InSwapper(img_resolution=8, latent_resolution=2, num_latent=1, base_ch=4, max_ch=8, id_dim=8)
    with torch.no_grad():
        assert model(torch.randn(1, 3, 8, 8), torch.randn(1, 8)).shape == (1, 3, 8, 8)


@pytest.mark.parametrize("mode", ["cat", "add"])
@pytest.mark.parametrize("layers", [[], [0], [1], [2], [0, 2], [0, 1, 2]])
@pytest.mark.parametrize("client", [False, True])
def test_unused_skip_features_are_released_before_bottleneck(mode: str, layers: list[int], client: bool) -> None:
    """Inference should not keep unselected high-resolution Encoder tensors alive."""
    import weakref

    generator = Generator(img_resolution=32, latent_resolution=4, num_latent=1,
                          base_ch=4, max_ch=8, skip_mode=mode, skip_layers=layers).eval()
    if client:
        server, runtime = build_split_models(generator, nn.Identity())
        model = runtime.core
        conditioning = server(torch.randn(1, 512))
    else:
        model = generator.core
        conditioning = torch.randn(1, 512)

    refs: dict[int, weakref.ReferenceType[torch.Tensor]] = {}
    observed = []
    handles = []
    depth = len(model.encoder)

    def capture(index: int):
        def hook(_module, _args, output):
            refs[index] = weakref.ref(output)
        return hook

    handles.append(model.from_rgb.register_forward_hook(capture(depth - 1)))
    for encoder_index, block in enumerate(model.encoder[:-1]):
        handles.append(block.register_forward_hook(capture(depth - encoder_index - 2)))

    def inspect(_module, _args):
        observed.append({index for index, reference in refs.items() if reference() is not None})

    bottleneck = model.layers[0] if client else model.latent_space
    handles.append(bottleneck.register_forward_pre_hook(inspect))
    try:
        with torch.no_grad():
            result = model(torch.randn(1, 3, 32, 32), conditioning)
    finally:
        for handle in handles:
            handle.remove()
    assert result.shape == (1, 3, 32, 32)
    assert observed == [set(layers)]
