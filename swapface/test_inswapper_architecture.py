"""Identity-bottleneck architecture and checkpoint smoke test.

Run: python -m swapface.test_inswapper_architecture
"""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F
from torch import nn

from models.networks import AdaIN, Generator, IDInject, InSwapper, WSpaceMap
from swapface.config import _normalize_generator
from swapface.contracts import CHECKPOINT_VERSION
from swapface.inference import load_generator


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    mapping = WSpaceMap(512)
    torch.manual_seed(2026)
    expected = nn.Sequential(nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU())
    nn.init.zeros_(expected[0].bias)
    nn.init.zeros_(expected[2].bias)
    assert len(mapping) == 4
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(mapping.state_dict()[key], value, rtol=0, atol=0)
    identity = F.normalize(torch.randn(2, 512), dim=1)
    w = mapping(identity)
    torch.testing.assert_close(w, expected(identity * math.sqrt(512)), rtol=0, atol=0)
    assert w.shape == identity.shape
    assert not torch.allclose(w[0], w[1])
    for mode in ("cat", "add"):
        for skips in ([], [0], [1], [0, 1]):
            net = InSwapper(img_resolution=16, latent_resolution=4, num_latent=2, base_ch=4, max_ch=16,
                            id_dim=512, skip_mode=mode, skip_layers=skips)
            assert isinstance(net.w_space_map, WSpaceMap)
            assert len(net.latent_space.layers) == 2
            for block in net.latent_space.layers:
                assert isinstance(block, IDInject)
                assert isinstance(block.adain0, AdaIN)
                assert isinstance(block.act, nn.SiLU)
                assert block.conv0.padding_mode == block.conv1.padding_mode == "reflect"
            identity = torch.randn(1, 512, requires_grad=True)
            image = torch.randn(1, 3, 16, 16, requires_grad=True)
            output = net(image, identity)
            assert output.shape == (1, 3, 16, 16)
            output.square().mean().backward()
            assert identity.grad is not None and torch.count_nonzero(identity.grad) > 0
            assert image.grad is not None and torch.count_nonzero(image.grad) > 0
            assert nn.Conv2d is type(net.to_rgb[-1])

    cfg = _normalize_generator({"generator": {
        "type": "inswapper", "img_resolution": 16, "latent_resolution": 4,
        "num_latent": 2, "base_ch": 4, "max_ch": 16, "skip_layers": [0, 1],
    }})
    assert "mapping_layers" not in cfg
    try:
        _normalize_generator({"generator": {**cfg, "mapping_layers": 2}})
        raise AssertionError("InSwapper mapping_layers must not be configurable")
    except ValueError:
        pass
    model = Generator(**cfg)
    assert model.network_cfg == cfg
    with TemporaryDirectory() as temp:
        path = Path(temp) / "step_000000001.pth"
        torch.save({
            "version": CHECKPOINT_VERSION, "step": 1, "face_alignment": "ffhq",
            "identity_encoders": {"generator": "BLENDFACE"},
            "net_g": {"network_cfg": cfg, "state_dict": model.state_dict()},
        }, path)
        loaded, _, step = load_generator(path)
        assert step == 1 and loaded.network_cfg == model.network_cfg
    print("PASS: InSwapper single-stage U-Net, AdaIN bottleneck, skips, gradients, checkpoint")


if __name__ == "__main__":
    main()
