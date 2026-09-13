"""无需 DALI/GPU 的训练数据管线回归检查：uv run python -m faceswap.test_dataloader"""

import builtins
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from faceswap.dataloader import DEFAULT_DATALOADER_CONFIG, TrainingDataLoader
from faceswap.dataloader_common import build_image_pools
from faceswap.dataloader_native import make_affine_thetas


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
        # CPU 也通过统一 facade 走 native 后端，并验证实际输出契约。
        loader = TrainingDataLoader(
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

        # 无损、通道值不同的图像直接保护 RGB 顺序与归一化，随机图像的值域检查无法发现 BGR 回归。
        rgb_src, rgb_dst = root / "rgb_src", root / "rgb_dst"
        for folder, color in ((rgb_src, (255, 128, 0)), (rgb_dst, (0, 64, 255))):
            folder.mkdir()
            Image.new("RGB", (8, 8), color).save(folder / "color.png")
        config.update(brightness=0.0, contrast=0.0, saturation=0.0, flip_prob=0.0, rotation_range=(0.0, 0.0), scale_factor_range=(1.0, 1.0), tx_range=(0.0, 0.0), ty_range=(0.0, 0.0))
        rgb_loader = TrainingDataLoader(batch_size=1, device=torch.device("cpu"), img_resolution=8, src=[rgb_src], dst=[rgb_dst], **config)
        src, target, theta = rgb_loader.next()
        for actual, color in ((src, (255, 128, 0)), (target, (0, 64, 255))):
            expected = (torch.tensor(color, dtype=torch.float32) / 127.5 - 1.0).view(1, 3, 1, 1).expand(1, 3, 8, 8)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(theta, torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), atol=0, rtol=0)

    # backend selector 必须真正区分同一个 CUDA device 上的 HIP 与 NVIDIA build。
    selected: list[str] = []

    class FakeNativeLoader:
        def __init__(self, **_kwargs) -> None:
            selected.append("native")

    class FakeDaliLoader:
        def __init__(self, **_kwargs) -> None:
            selected.append("dali")

    native_module = ModuleType("faceswap.dataloader_native")
    native_module._NativeTrainingDataLoader = FakeNativeLoader
    dali_module = ModuleType("faceswap.dataloader_dali")
    dali_module._DALITrainingDataLoader = FakeDaliLoader
    with patch.dict(sys.modules, {"faceswap.dataloader_native": native_module, "faceswap.dataloader_dali": dali_module}):
        with patch("faceswap.dataloader.torch.version.hip", "6.4.0"):
            TrainingDataLoader(batch_size=1, device=torch.device("cuda"), img_resolution=32, src=[], dst=[])
        with patch("faceswap.dataloader.torch.version.hip", None):
            TrainingDataLoader(batch_size=1, device=torch.device("cuda"), img_resolution=32, src=[], dst=[])
    assert selected == ["native", "dali"]

    # Trainer 的配置解析不再依赖导入 nvidia.dali；ROCm 环境可以在未安装 DALI 时导入训练入口。
    assert "nvidia.dali.plugin.pytorch" not in sys.modules

    # ROCm 训练环境不安装 CUDA-only 依赖；训练入口必须仍可独立导入。
    blocked_optional_imports = ("nvidia.dali", "torchcodec", "onnxruntime", "xformers", "tensorrt")
    real_import = builtins.__import__

    def import_without_cuda_optional(name, *args, **kwargs):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked_optional_imports):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_cuda_optional):
        from faceswap import train

    assert "nvidia.dali.plugin.pytorch" not in sys.modules
    assert "torchcodec" not in sys.modules
    assert set(train.DEFAULT_DATALOADER_CONFIG) == set(DEFAULT_DATALOADER_CONFIG)
    train_source = Path(train.__file__).read_text(encoding="utf-8")
    assert "DALIGenericIterator" not in train_source
    assert "data_backend" not in train_source
    assert "dataloader_native" not in train_source
    assert "dataloader_dali" not in train_source
    print("PASS: unified dataloader facade, affine semantics, sampling weights and lazy DALI import")


if __name__ == "__main__":
    main()
