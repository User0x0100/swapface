"""训练数据管线共享的配置、路径扫描与采样权重逻辑。"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor

from misc.face_alignment import ffhq_to_arcface_112

type ImagePath = str | os.PathLike[str]
type FloatRange = tuple[float, float]
type ImageSource = ImagePath | tuple[ImagePath, float] | tuple[ImagePath, float, str]


@dataclass(frozen=True, slots=True)
class LocalImagePool:
    root: str
    file_names: tuple[str, ...]
    alignment: str = "ffhq"


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


class ImageDecoderBackend(Enum):
    """DALI 图像解码后端配置；ROCm 原生管线固定使用 CPU 解码。"""

    MIXED = "mixed"
    CPU = "cpu"


def validate_range(name: str, value: FloatRange) -> FloatRange:
    if len(value) != 2:
        raise ValueError(f"{name} 必须包含两个值，实际为 {value!r}")
    low, high = float(value[0]), float(value[1])
    if not np.isfinite((low, high)).all() or low > high:
        raise ValueError(f"{name} 必须是有限且递增的范围，实际为 {value!r}")
    return low, high


def scan_image_files(directory: ImagePath) -> LocalImagePool:
    folder = Path(directory).resolve(strict=True)
    with os.scandir(folder) as entries:
        file_names = tuple(entry.name for entry in entries if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS))
    if not file_names:
        raise FileNotFoundError(f"目录中没有支持的图片文件: {folder}")
    return LocalImagePool(str(folder), file_names)


def build_image_pools(sources: Sequence[ImageSource]) -> tuple[tuple[LocalImagePool, ...], ndarray, tuple[tuple[str, int, float], ...]]:
    if isinstance(sources, (str, os.PathLike)):
        raise TypeError(f"图片源必须是路径序列；单个目录请写成 [{os.fspath(sources)!r}]")
    if not sources:
        raise ValueError("图片源列表不能为空")

    pools: list[LocalImagePool] = []
    counts: list[int] = []
    adjustments: list[float] = []

    for source in sources:
        if isinstance(source, (str, os.PathLike)):
            path, adjustment, alignment = source, 0.0, "ffhq"
        elif isinstance(source, tuple) and len(source) in (2, 3) and isinstance(source[0], (str, os.PathLike)):
            path, adjustment = source[:2]
            alignment = source[2] if len(source) == 3 else "ffhq"
        else:
            raise TypeError(f"图片源必须为路径、(路径, 权重调整) 或 (路径, 权重调整, 对齐方式)，实际为 {source!r}")

        adjustment = float(adjustment)
        if not np.isfinite(adjustment):
            raise ValueError(f"权重调整必须为有限数值，实际为 {adjustment!r}")

        if alignment not in ("ffhq", "arcface"):
            raise ValueError(f"alignment 必须为 ffhq 或 arcface，实际为 {alignment!r}")
        scanned = scan_image_files(path)
        pool = LocalImagePool(scanned.root, scanned.file_names, alignment)
        pools.append(pool)
        counts.append(len(pool.file_names))
        adjustments.append(adjustment)

    # source_weight ∝ sqrt(file_count) * 2**adjustment。
    log_weights = 0.5 * np.log(np.asarray(counts, dtype=np.float64)) + np.asarray(adjustments, dtype=np.float64) * np.log(2.0)
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0

    info = tuple((pool.root, count, float(weight)) for pool, count, weight in zip(pools, counts, weights, strict=True))
    return tuple(pools), cdf, info


def prepare_source_identity_faces(src: Tensor, arcface_faces: Tensor, src_arcface_mask: Tensor, ffhq_grid: Tensor) -> Tensor:
    """FFHQ 使用原裁剪；ArcFace 使用独立保留的 112 像素，避免展示分辨率往返缩放。"""
    ffhq_faces = ffhq_to_arcface_112(src, ffhq_grid)
    return torch.where(src_arcface_mask[:, None, None, None], arcface_faces, ffhq_faces)


def print_image_pools(title: str, info: tuple[tuple[str, int, float], ...]) -> None:
    print(title + ":")
    for label, count, weight in info:
        print(f"    {label}\n        count: {count:<7d} weight: {weight:<6.3f}")
