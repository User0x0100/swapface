"""遮挡增强、OCC 监督区域与 ID 梯度门控的轻量回归：uv run --no-sync python -m swapface.test_occlusion。"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import DEFAULT_OCCLUSION_CONFIG
from .occlusion import OcclusionAugmentor, gate_identity_gradient

GEOMETRY = {key: value for key, value in DEFAULT_OCCLUSION_CONFIG.items() if key.endswith("_pct") or key.endswith("_deg")}



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
        aug = OcclusionAugmentor(1.0, tmp, torch.device("cpu"), visible_face, **GEOMETRY)
        occluded, gate = aug.apply(clean)
        assert occluded.shape == clean.shape and gate.shape == (4, 1, 64, 64)
        assert not torch.equal(occluded, clean)
        assert bool((gate < 0.5).any())
        assert bool((gate > 0.99).any())
        assert bool((occluded >= -1).all() and (occluded <= 1).all())

    # 固定尺寸/中心/旋转时，百分比应精确决定被覆盖的位置与宽度。
    # 100% 尺寸不表示 100% Alpha 覆盖；还受原始素材的形状和透明区域影响。
    with tempfile.TemporaryDirectory() as tmp:
        image = np.zeros((32, 32, 4), dtype=np.uint8)
        image[:, :, :3] = 255
        image[:, :, 3] = 255
        Image.fromarray(image, "RGBA").save(Path(tmp) / "square.png")
        def area_and_center(size_pct: float, offset_pct: float):
            overlay = OcclusionAugmentor(
                1.0, tmp, torch.device("cpu"), visible_face,
                size_range_pct=(size_pct, size_pct),
                center_x_range_pct=(offset_pct, offset_pct),
                center_y_range_pct=(0, 0),
                rotation_range_deg=(0, 0),
            )
            out, _ = overlay.apply(torch.full((1, 3, 100, 100), -1.0))
            pixels = (out[0, 0] > 0.9).nonzero()
            return int(pixels[:, 1].max() - pixels[:, 1].min() + 1), float(pixels[:, 1].float().mean())
        w20, center20 = area_and_center(20, 0)
        w80, center80 = area_and_center(80, 0)
        _, center_shifted = area_and_center(20, 25)
        assert 17 <= w20 <= 21 and 76 <= w80 <= 81, (w20, w80)
        assert abs(center20 - 49.5) <= 1 and abs(center80 - 49.5) <= 1
        assert abs(center_shifted - center20 - 25) <= 1.5

    # 旋转角度使用 degree 而非 radian；固定 90° 应交换矩形的长/短边。
    with tempfile.TemporaryDirectory() as tmp:
        rgba = np.zeros((16, 32, 4), dtype=np.uint8)
        rgba[:, :, :3] = 255
        rgba[:, :, 3] = 255
        Image.fromarray(rgba, "RGBA").save(Path(tmp) / "bar.png")
        def bounding_size(degrees: int):
            aug_rotation = OcclusionAugmentor(
                1.0, tmp, torch.device("cpu"), visible_face,
                size_range_pct=(80, 80),
                center_x_range_pct=(0, 0),
                center_y_range_pct=(0, 0),
                rotation_range_deg=(degrees, degrees),
            )
            out, _ = aug_rotation.apply(torch.full((1, 3, 100, 100), -1.0))
            coords = (out[0, 0] > 0.9).nonzero()
            return int(coords[:, 0].max() - coords[:, 0].min()), int(coords[:, 1].max() - coords[:, 1].min())
        unrotated = bounding_size(0)
        rotated = bounding_size(90)
        assert unrotated[0] < unrotated[1] and rotated[0] > rotated[1], (unrotated, rotated)

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
    aug = OcclusionAugmentor(1.0, "", torch.device("cpu"), visible_face, **GEOMETRY)
    random_dst, random_gate = aug.apply(clean)
    assert not torch.equal(random_dst, clean)
    assert bool((random_gate < 0.5).any())

    # clean 样本未经 OCC 推理且无修改；0<prob<1 下验证独立样本选择。
    torch.manual_seed(0)
    aug = OcclusionAugmentor(0.5, "", torch.device("cpu"), visible_face, **GEOMETRY)
    out, gate = aug.apply(torch.zeros((20, 3, 32, 32)))
    untouched = (out == 0).flatten(1).all(dim=1)
    assert bool(untouched.any() and (~untouched).any())
    assert bool((gate[untouched] == 1).all())
    from .config import load_train_config, resolve_train_config

    existing = load_train_config("experiments/train.toml")
    assert "occlusion" not in existing["data"]
    config = {**existing, "data": {**existing["data"], "occlusion": {"prob": 0.2}}}
    resolved = resolve_train_config(config)
    assert resolved["data"]["occlusion"] == {**DEFAULT_OCCLUSION_CONFIG, "prob": 0.2}
    assert resolve_train_config(resolved) == resolved

    import copy
    invalid = (
        ("size_range_pct", [0, 50]),
        ("size_range_pct", [-10, 50]),
        ("size_range_pct", [80, 20]),
        ("size_range_pct", [float("nan"), 50]),
        ("center_x_range_pct", [15, -15]),
        ("center_y_range_pct", [1]),
        ("rotation_range_deg", [20, -20]),
        ("rotation_range_deg", [0, float("inf")]),
    )
    for field, value in invalid:
        bad = copy.deepcopy(config)
        bad["data"]["occlusion"][field] = value
        try:
            resolve_train_config(bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"无效范围意外通过: {field}={value}")
    print("PASS: identity gradients, RGBA/procedural overlays, percent geometry/degree rotation, config validation")


if __name__ == "__main__":
    main()
