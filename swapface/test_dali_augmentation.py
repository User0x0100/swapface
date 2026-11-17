"""NVIDIA DALI 增强独立跳过路径的端到端回归：python -m swapface.test_dali_augmentation。"""

import gc
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import DEFAULT_DATA_CONFIG
from .dataloader import TrainingDataLoader
from .dataloader_common import ImageDecoderBackend


def main() -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        print("SKIP: NVIDIA CUDA is unavailable")
        return
    try:
        import nvidia.dali  # noqa: F401
    except ImportError:
        print("SKIP: NVIDIA DALI is unavailable")
        return

    with tempfile.TemporaryDirectory(prefix="swapface-dali-augmentation-") as tmp:
        root = Path(tmp)
        src_dir, dst_dir = root / "src", root / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()
        y, x = np.mgrid[:32, :32]
        data = np.stack((x * 7, y * 7, (x + y) * 3), axis=-1).astype(np.uint8)
        Image.fromarray(data).save(src_dir / "src.png")
        Image.fromarray(data).save(dst_dir / "dst.png")

        config = {**DEFAULT_DATA_CONFIG["loader"], **DEFAULT_DATA_CONFIG["augmentation"], **DEFAULT_DATA_CONFIG["sampling"]}
        config.update(
            decoder_backend=ImageDecoderBackend.CPU, num_threads=2, py_num_workers=1, prefetch_queue_depth=1, reader_prefetch_queue_depth=1,
            brightness=0.0, contrast=0.0, saturation=0.0, flip_prob=0.0, same_prob=0.0,
            rotation_range=(0.0, 0.0), scale_factor_range=(1.0, 1.0), tx_range=(0.0, 0.0), ty_range=(0.0, 0.0),
        )
        identity_theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device="cuda")

        def load_one(**changes):
            loader = TrainingDataLoader(batch_size=2, device=torch.device("cuda"), img_resolution=32, src=[src_dir], dst=[dst_dir], **(config | changes))
            batch = loader.next()
            del loader
            gc.collect()
            return batch

        # 无随机几何：DALI constant 必须返回 (B, 2, 3)，且 target/canonical 完全相同。
        src, target, canonical, theta, _, _ = load_one()
        assert target.shape == canonical.shape == (2, 3, 32, 32)
        torch.testing.assert_close(target, canonical, atol=0, rtol=0)
        torch.testing.assert_close(theta, identity_theta.expand(2, -1, -1), atol=0, rtol=0)
        original = torch.from_numpy(data.copy()).permute(2, 0, 1).float().to("cuda").unsqueeze(0).div(127.5).sub(1)
        torch.testing.assert_close(target, original.expand(2, -1, -1, -1), atol=1e-4, rtol=0)

        # 仅水平翻转，保留 identity 恢复矩阵，且 target/canonical 不被额外插值。
        _, flipped, flipped_canonical, flipped_theta, _, _ = load_one(flip_prob=1.0)
        torch.testing.assert_close(flipped, flipped_canonical, atol=0, rtol=0)
        torch.testing.assert_close(flipped_theta, identity_theta.expand(2, -1, -1), atol=0, rtol=0)
        torch.testing.assert_close(flipped, original.flip(-1).expand(2, -1, -1, -1), atol=1e-4, rtol=0)

        # 仅启用颜色不引入随机仿射；仅启用几何则 canonical 必须保留原图。
        _, colored, colored_canonical, colored_theta, _, _ = load_one(brightness=0.2)
        torch.testing.assert_close(colored, colored_canonical, atol=0, rtol=0)
        torch.testing.assert_close(colored_theta, identity_theta.expand(2, -1, -1), atol=0, rtol=0)

        _, shifted, shifted_canonical, shifted_theta, _, _ = load_one(tx_range=(0.25, 0.25))
        torch.testing.assert_close(shifted_canonical, original.expand(2, -1, -1, -1), atol=1e-4, rtol=0)
        assert not torch.equal(shifted, shifted_canonical)
        assert not torch.equal(shifted_theta, identity_theta.expand(2, -1, -1))

        # 单独使用固定角度/缩放也应正常工作，其他恒等参数不能生成零宽随机范围。
        _, rotated, rotated_canonical, rotated_theta, _, _ = load_one(rotation_range=(10.0, 10.0), scale_factor_range=(1.1, 1.1))
        torch.testing.assert_close(rotated_canonical, original.expand(2, -1, -1, -1), atol=1e-4, rtol=0)
        assert not torch.equal(rotated, rotated_canonical)
        assert not torch.equal(rotated_theta, identity_theta.expand(2, -1, -1))

        # 固定的 FFHQ->ArcFace 坐标变换仍需执行（不属于随机几何增强）。
        _, aligned, aligned_canonical, aligned_theta, _, _ = load_one(alignment="arcface")
        torch.testing.assert_close(aligned, aligned_canonical, atol=0, rtol=0)
        torch.testing.assert_close(aligned_theta, identity_theta.expand(2, -1, -1), atol=0, rtol=0)
        assert not torch.equal(aligned, original.expand(2, -1, -1, -1))

    print("PASS: NVIDIA DALI color/flip/geometry fast paths, identity theta and fixed ArcFace alignment")


if __name__ == "__main__":
    main()
