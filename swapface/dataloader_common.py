"""训练数据管线共享的配置、路径扫描与采样权重逻辑。"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from misc.face_alignment import FFHQ_TO_ARCFACE_112_AFFINE_512, ffhq_to_arcface_112, ffhq_to_arcface_affine


# (left, top, right, bottom)，相对原始图像尺寸；供 Pillow 和 DALI 共享。
# 只使用轴对齐 ROI 来近似固定 ArcFace 的微小旋转：最大角点输出坐标误差每轴 < 0.01 px。
# ROI 使用像素边界，而 affine 使用像素中心；必须针对真实输出分辨率换算。
def _axis_aligned_roi(
    affine: tuple[tuple[float, float, float], tuple[float, float, float]], output_size: int, input_size: int
) -> tuple[float, float, float, float]:
    if min(input_size, output_size) <= 0:
        raise ValueError("input_size/output_size 必须为正数")
    (a, b, tx), (c, d, ty) = affine
    if max(abs(b / a), abs(c / d)) > 2e-4:
        raise ValueError("固定对齐变换不再适合使用轴对齐 ROI")
    det = a * d - b * c
    inv_a, inv_b, inv_c, inv_d = d / det, -b / det, -c / det, a / det
    output_center = (output_size - 1) / 2
    center_x = inv_a * (output_center - tx) + inv_b * (output_center - ty) + 0.5
    center_y = inv_c * (output_center - tx) + inv_d * (output_center - ty) + 0.5
    half_x, half_y = inv_a * output_size / 2, inv_d * output_size / 2
    return ((center_x - half_x) / input_size, (center_y - half_y) / input_size, (center_x + half_x) / input_size, (center_y + half_y) / input_size)


def generation_alignment_rois(output_size: int) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """返回 FFHQ→ArcFace128 与 ArcFace112→ArcFace128 的原图直采样 ROI。"""
    ffhq = _axis_aligned_roi(ffhq_to_arcface_affine(512, output_size), output_size, 512)
    scale = output_size / 128.0
    arcface_affine = ((scale, 0.0, 8.0 * scale), (0.0, scale, 0.0))
    arcface = _axis_aligned_roi(arcface_affine, output_size, 112)
    return ffhq, arcface


FFHQ_TO_ARCFACE112_ROI = _axis_aligned_roi(FFHQ_TO_ARCFACE_112_AFFINE_512, 112, 512)


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


def is_identity_geometry(rotation_range: FloatRange, scale_factor_range: FloatRange, tx_range: FloatRange, ty_range: FloatRange) -> bool:
    """几何增强是否严格等于恒等变换；与翻转和色彩增强无关。"""
    return rotation_range == (0.0, 0.0) and scale_factor_range == (1.0, 1.0) and tx_range == (0.0, 0.0) and ty_range == (0.0, 0.0)


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
