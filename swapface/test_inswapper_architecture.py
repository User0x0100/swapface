"""Identity-bottleneck architecture and checkpoint smoke test.

Run: python -m swapface.test_inswapper_architecture
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from models.networks import AdaIN, Generator, IDInject, InSwapper, WSpaceMap
from swapface.config import _normalize_generator
from swapface.contracts import CHECKPOINT_VERSION
from swapface.inference import load_generator


def main() -> None:
    torch.set_num_threads(2)
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
