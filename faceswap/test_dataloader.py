"""无需 DALI/GPU 的训练数据管线回归检查：uv run python -m faceswap.test_dataloader"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from faceswap.dataloader_common import DEFAULT_DATALOADER_CONFIG, build_image_pools
from faceswap.dataloader_native import NativeTrainingDataLoader, make_affine_thetas


def _write_image(path: Path, value: int, size: tuple[int, int] = (48, 40)) -> None:
    height, width = size
    y, x = np.mgrid[:height, :width]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (x + value) % 256
    image[..., 1] = (y + value * 3) % 256
    image[..., 2] = (x + y + value * 7) % 256
    Image.fromarray(image).save(path, quality=95)


def main() -> None:
    # 固定参数下必须精确复现旧 DALI 的 normalized canonical->augmented theta。
    _, theta_restore = make_affine_thetas(
        batch_size=1,
        img_resolution=16,
        rotation_range=(10.0, 10.0),
        scale_factor_range=(1.1, 1.1),
        tx_range=(0.125, 0.125),
        ty_range=(-0.0625, -0.0625),
        device=torch.device("cpu"),
    )
    expected = torch.tensor([[[1.0832886, 0.1910130, 0.23285627], [-0.1910130, 1.0832886, -0.11826712]]])
    torch.testing.assert_close(theta_restore, expected, atol=2e-7, rtol=0)

    with tempfile.TemporaryDirectory(prefix="faceswap-dataloader-") as temporary:
        root = Path(temporary)
        src_a, src_b, dst = root / "src_a", root / "src_b", root / "dst"
        src_a.mkdir()
        src_b.mkdir()
        dst.mkdir()
        for index in range(4):
            _write_image(src_a / f"{index}.jpg", index)
        for index in range(9):
            _write_image(src_b / f"{index}.jpg", 20 + index)
        for index in range(6):
            _write_image(dst / f"{index}.jpg", 40 + index)

        # 与旧 DALI external_source 共用同一采样权重公式。
        _, _, info = build_image_pools([(src_a, 0.0), (src_b, 1.0)])
        weights = [item[2] for item in info]
        expected_ratio = (9**0.5 * 2.0) / (4**0.5)
        assert abs(weights[1] / weights[0] - expected_ratio) < 1e-12

        config = dict(DEFAULT_DATALOADER_CONFIG)
        config.update(
            py_num_workers=0,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            flip_prob=0.5,
            rotation_range=(-3.0, 3.0),
            scale_factor_range=(0.95, 1.05),
            tx_range=(-0.05, 0.05),
            ty_range=(-0.05, 0.05),
        )
        loader = NativeTrainingDataLoader(
            batch_size=2,
            device=torch.device("cpu"),
            img_resolution=32,
            src=[str(src_a), (str(src_b), 1.0)],
            dst=[str(dst)],
            **config,
        )
        src, target, theta = loader.next()
        assert src.shape == target.shape == (2, 3, 32, 32)
        assert theta.shape == (2, 2, 3)
        assert src.dtype == target.dtype == theta.dtype == torch.float32
        assert -1.0 <= float(src.min()) <= float(src.max()) <= 1.0
        assert -1.0 <= float(target.min()) <= float(target.max()) <= 1.0
        assert torch.isfinite(theta).all()

    # Trainer 的配置解析不再依赖导入 nvidia.dali；ROCm 环境可以在未安装 DALI 时导入训练入口。
    import sys

    assert "nvidia.dali.plugin.pytorch" not in sys.modules
    from faceswap import train

    assert "nvidia.dali.plugin.pytorch" not in sys.modules
    assert set(train.DEFAULT_DATALOADER_CONFIG) == set(DEFAULT_DATALOADER_CONFIG)
    print("PASS: portable dataloader contract, affine semantics, sampling weights and lazy DALI import")


if __name__ == "__main__":
    main()
