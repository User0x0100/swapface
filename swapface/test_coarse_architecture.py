"""InSwapper-style Coarse topology/gradient/checkpoint round-trip regression.

Run: .venv/bin/python -m swapface.test_coarse_architecture
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from models.networks import AdaIN, Coarse, Generator, HQRefiner, IDInject
from swapface.config import _normalize_generator
from swapface.contracts import CHECKPOINT_VERSION
from swapface.inference import load_generator


def _compact_config() -> dict[str, int]:
    return {
        "img_resolution": 16,
        "img_channels": 3,
        "id_dim": 512,
        "coarse_resolution": 8,
        "coarse_latent_resolution": 4,
        "coarse_num_latent": 2,
        "coarse_base_ch": 4,
        "coarse_max_ch": 16,
        "hq_bottleneck_resolution": 4,
        "hq_base_ch": 2,
        "hq_max_ch": 8,
    }


def _assert_topology_and_scaling() -> None:
    with torch.device("meta"):
        # This size configuration reproduces the InSwapper128 ONNX learnable
        # graph's exact layer-wise channels and trainable parameter count.
        full = Coarse(img_resolution=128, img_channels=3, latent_resolution=32,
                      num_latent=6, base_ch=256, max_ch=1024, id_dim=512)
    assert sum(p.numel() for p in full.parameters()) == 138_292_739

    cases = [(64, 256, 6), (96, 384, 4), (256, 1024, 6)]
    for base_ch, max_ch, num_latent in cases:
        with torch.device("meta"):
            network = Coarse(img_resolution=128, latent_resolution=32,
                             num_latent=num_latent, base_ch=base_ch,
                             max_ch=max_ch)
        assert len(network.latent_space.layers) == num_latent
        assert not hasattr(network, "w_space_map")
        assert network.from_rgb[0].out_channels == base_ch // 2
        assert network.from_rgb[2].out_channels == base_ch
        latent_ch = min(max_ch, base_ch * 4)
        assert network.encoder[-1][0].out_channels == latent_ch
        assert network.decoder[0][1].in_channels == latent_ch
        assert network.to_rgb[2].out_channels == 3
        for block in network.latent_space.layers:
            assert isinstance(block, IDInject)
            assert isinstance(block.act, nn.SiLU)
            assert block.conv0.padding_mode == block.conv1.padding_mode == "reflect"
            assert block.adain0.norm.eps == block.adain1.norm.eps == 1e-8
            assert isinstance(block.adain0, AdaIN) and isinstance(block.adain1, AdaIN)
            assert block.adain0.linear.in_features == 512
            assert block.adain0.linear.out_features == latent_ch * 2
            assert block.adain1.linear.out_features == latent_ch * 2
            assert not hasattr(block.adain0, "gamma_fc")
            assert not hasattr(block.adain0, "beta_fc")


def _assert_conditioning_and_gradients() -> None:
    torch.manual_seed(7)
    coarse = Coarse(img_resolution=8, latent_resolution=4, num_latent=2,
                    base_ch=4, max_ch=16, id_dim=8).eval()
    x = torch.randn(2, 3, 8, 8)
    src = torch.randn(2, 8, requires_grad=True)
    identities_seen: list[torch.Tensor] = []
    handles = []
    for layer in coarse.latent_space.layers:
        for adain in (layer.adain0, layer.adain1):
            handles.append(adain.linear.register_forward_pre_hook(
                lambda _mod, inputs: identities_seen.append(inputs[0])))
    try:
        y = coarse(x, src)
    finally:
        for handle in handles:
            handle.remove()
    assert len(identities_seen) == 4
    assert all(t is src for t in identities_seen)
    assert y.shape == (2, 3, 8, 8)
    assert bool(torch.isfinite(y).all())
    (y.square().mean()).backward()
    assert src.grad is not None and torch.count_nonzero(src.grad) > 0
    for layer in coarse.latent_space.layers:
        for adain in (layer.adain0, layer.adain1):
            assert adain.linear.weight.grad is not None
            assert torch.isfinite(adain.linear.weight.grad).all()

    # The shared source vector should affect the output of both style
    # projections, not a learned intermediate WSpaceMap.
    with torch.inference_mode():
        y0 = coarse(x, torch.zeros_like(src))
        y1 = coarse(x, torch.ones_like(src))
    assert not torch.allclose(y0, y1)


def _assert_linear_rgb_outputs() -> None:
    """Coarse and HQ RGB heads must not squash the raw output."""
    coarse = Coarse(img_resolution=8, latent_resolution=4, num_latent=2,
                    base_ch=4, max_ch=16, id_dim=8).eval()
    hq = HQRefiner(img_resolution=16, coarse_resolution=8,
                   bottleneck_resolution=4, base_ch=2, max_ch=8).eval()
    assert isinstance(coarse.to_rgb[-1], nn.Conv2d)
    assert isinstance(hq.to_rgb, nn.Conv2d)

    expected = torch.tensor([2.0, -2.0, 0.5]).reshape(1, 3, 1, 1)
    with torch.no_grad():
        for head in (coarse.to_rgb[-1], hq.to_rgb):
            head.weight.zero_()
            head.bias.copy_(expected.flatten())
        coarse_out = coarse(torch.randn(1, 3, 8, 8), torch.randn(1, 8))
        hq_out = hq(torch.randn(1, 3, 16, 16), coarse_out)
    torch.testing.assert_close(coarse_out, expected.expand_as(coarse_out))
    torch.testing.assert_close(hq_out, expected.expand_as(hq_out))


def _assert_id_inject_residual() -> None:
    """The final AdaIN result is added directly, without an activation."""
    block = IDInject(channels=4, w_dim=8).eval()
    x = torch.randn(2, 4, 4, 4)
    identity = torch.randn(2, 8)
    with torch.no_grad():
        residual = block.act(block.adain0(block.conv0(x), identity))
        residual = block.adain1(block.conv1(residual), identity)
        torch.testing.assert_close(block(x, identity), x + residual)


def _assert_checkpoint_and_config() -> None:
    cfg = _compact_config()
    model = Generator(**cfg).eval()
    assert _normalize_generator({"generator": cfg}) == cfg
    with TemporaryDirectory() as tmp:
        checkpoint_path = Path(tmp) / "gen.pth"
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "step": 1,
            "identity_encoders": {"generator": "WEBFACE600K_ARCFACE_R50"},
            "net_g": {"network_cfg": cfg, "state_dict": model.state_dict()},
        }
        torch.save(checkpoint, checkpoint_path)
        loaded, _, step = load_generator(checkpoint_path)
        assert step == 1
        assert loaded.network_cfg == cfg


def main() -> None:
    torch.set_num_threads(2)
    _assert_topology_and_scaling()
    _assert_conditioning_and_gradients()
    _assert_id_inject_residual()
    _assert_linear_rgb_outputs()
    _assert_checkpoint_and_config()
    print("PASS: InSwapper-style Coarse structure, configurable dimensions, direct conditioning, gradients and checkpoint round-trip")


if __name__ == "__main__":
    main()
