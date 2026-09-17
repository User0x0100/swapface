"""训练数据管线共享的配置、路径扫描与采样权重逻辑。"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray

type ImagePath = str | os.PathLike[str]
type FloatRange = tuple[float, float]
type ImageSource = ImagePath | tuple[ImagePath, float]


@dataclass(frozen=True, slots=True)
class LocalImagePool:
    root: str
    file_names: tuple[str, ...]


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
DATALOADER_RESERVED_KEYS = frozenset({"batch_size", "device", "device_id", "img_resolution", "src", "dst"})


class ImageDecoderBackend(Enum):
    """DALI 图像解码后端配置；ROCm 原生管线固定使用 CPU 解码。"""

    MIXED = "mixed"
    CPU = "cpu"


# 数据管线默认配置。
DEFAULT_DATALOADER_CONFIG: dict[str, Any] = {
    "num_threads": 16,
    "prefetch_queue_depth": 4,
    "py_num_workers": 8,
    "py_start_method": "spawn",
    "reader_prefetch_queue_depth": 2,
    "decoder_backend": ImageDecoderBackend.MIXED,
    "decoder_hw_load": 0.75,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "flip_prob": 0.5,
    "same_prob": 0.2,
    "rotation_range": (-10.0, 10.0),
    "scale_factor_range": (1.0 / 1.3, 1.25),
    "tx_range": (-0.15, 0.15),
    "ty_range": (-0.15, 0.15),
}


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
            path, adjustment = source, 0.0
        elif isinstance(source, tuple) and len(source) == 2 and isinstance(source[0], (str, os.PathLike)):
            path, adjustment = source
        else:
            raise TypeError(f"图片源必须为路径或 (路径, 权重调整)，实际为 {source!r}")

        adjustment = float(adjustment)
        if not np.isfinite(adjustment):
            raise ValueError(f"权重调整必须为有限数值，实际为 {adjustment!r}")

        pool = scan_image_files(path)
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


def print_image_pools(title: str, info: tuple[tuple[str, int, float], ...]) -> None:
    print(title + ":")
    for label, count, weight in info:
        print(f"    {label}\n        count: {count:<7d} weight: {weight:<6.3f}")
