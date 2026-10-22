"""遮挡增强、OCC 监督区域与 ID 梯度门控的轻量回归：uv run --no-sync python -m swapface.test_occlusion。"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .occlusion import OcclusionAugmentor, gate_identity_gradient


def main() -> None:
    x = torch.randn((2, 3, 16, 16), requires_grad=True)
    mask = torch.ones((2, 1, 16, 16))
    mask[:, :, :, :8] = 0.0
    masked = gate_identity_gradient(x, mask)
    torch.testing.assert_close(masked, x)
    masked.sum().backward()
    torch.testing.assert_close(x.grad, mask.expand_as(x))

    def visible_face(images: torch.Tensor) -> torch.Tensor:
        assert images.shape[1:] == (3, 224, 224)
        return torch.ones((images.shape[0], 1, 224, 224), device=images.device)

    torch.manual_seed(51)
    clean = torch.full((4, 3, 64, 64), -0.9)
    with tempfile.TemporaryDirectory() as tmp:
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[4:28, 4:28, :3] = (245, 30, 30)
        rgba[4:28, 4:28, 3] = 255
        Image.fromarray(rgba, "RGBA").save(Path(tmp) / "object.png")
        aug = OcclusionAugmentor(1.0, tmp, torch.device("cpu"), visible_face)
        occluded, gate = aug.apply(clean)
        assert occluded.shape == clean.shape and gate.shape == (4, 1, 64, 64)
        assert not torch.equal(occluded, clean)
        assert bool((gate < 0.5).any())
        assert bool((gate > 0.99).any())
        assert bool((occluded >= -1).all() and (occluded <= 1).all())

    # Trainer 的遮挡适配层不修改既有 DataLoader 六项契约。
    from .train import Trainer

    class DummyLoader:
        def next(self):
            src = torch.ones_like(clean)
            return src, clean, clean, torch.zeros(4, 2, 3), torch.ones(4, dtype=torch.bool), torch.zeros(4, 3, 112, 112)

    trainer = Trainer.__new__(Trainer)
    trainer.dataset = DummyLoader()
    trainer.occlusion = aug
    batch = trainer._next_training_batch()
    assert len(batch) == 7 and torch.equal(batch[0], torch.ones_like(clean))
    assert torch.equal(batch[2], clean) and torch.equal(batch[5], torch.zeros(4, 3, 112, 112))
    assert batch[6].shape == (4, 1, 64, 64) and not torch.equal(batch[1], clean)
    trainer.occlusion = None
    clean_batch = trainer._next_training_batch()
    assert torch.equal(clean_batch[1], clean) and clean_batch[6] is None

    # 无素材时也能使用 RandOcc 路径，不依赖外部文件。
    aug = OcclusionAugmentor(1.0, "", torch.device("cpu"), visible_face)
    random_dst, random_gate = aug.apply(clean)
    assert not torch.equal(random_dst, clean)
    assert bool((random_gate < 0.5).any())

    # clean 样本未经 OCC 推理且无修改；0<prob<1 下验证独立样本选择。
    torch.manual_seed(0)
    aug = OcclusionAugmentor(0.5, "", torch.device("cpu"), visible_face)
    out, gate = aug.apply(torch.zeros((20, 3, 32, 32)))
    untouched = (out == 0).flatten(1).all(dim=1)
    assert bool(untouched.any() and (~untouched).any())
    assert bool((gate[untouched] == 1).all())
    from .config import load_train_config, resolve_train_config

    existing = load_train_config("experiments/train.toml")
    assert "occlusion" not in existing["data"]
    config = {**existing, "data": {**existing["data"], "occlusion": {"prob": 0.2}}}
    resolved = resolve_train_config(config)
    assert resolved["data"]["occlusion"] == {"prob": 0.2, "assets_dir": ""}
    assert resolve_train_config(resolved) == resolved
    print("PASS: identity gradient gate, RGBA/procedural occlusion, OCC selection and canonical config")


if __name__ == "__main__":
    main()
