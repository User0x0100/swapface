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
type ImageSource = ImagePath | tuple[ImagePath, float] | tuple[ImagePath, float, str] | tuple[ImagePath, float, str, str]


@dataclass(frozen=True, slots=True)
class ImageGroup:
    root: str
    file_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImageIndex:
    root: str
    layout: str
    groups: tuple[ImageGroup, ...]
    count: int
    multi_view_indices: tuple[int, ...]
    multi_view_count: int


@dataclass(frozen=True, slots=True)
class LocalImagePool:
    index: ImageIndex
    alignment: str
    adjustment: float


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


def _scan_group(folder: str) -> ImageGroup:
    with os.scandir(folder) as entries:
        names = tuple(entry.name for entry in entries if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS))
    return ImageGroup(folder, names)


def _scan_image_index(root: str, layout: str) -> ImageIndex:
    if layout == "flat":
        groups = (_scan_group(root),)
    else:
        with os.scandir(root) as entries:
            groups = tuple(_scan_group(entry.path) for entry in entries if entry.is_dir())
    groups = tuple(group for group in groups if group.file_names)
    if not groups:
        raise FileNotFoundError(f"目录中没有支持的图片文件: {root}")
    multi = tuple(i for i, group in enumerate(groups) if layout == "identity" and len(group.file_names) >= 2)
    return ImageIndex(root, layout, groups, sum(len(group.file_names) for group in groups), multi,
                      sum(len(groups[i].file_names) for i in multi))


def _source_cdf(counts: Sequence[int], adjustments: Sequence[float]) -> tuple[ndarray, ndarray]:
    # source_weight ∝ sqrt(file_count) * 2**adjustment。
    log_weights = 0.5 * np.log(np.asarray(counts, dtype=np.float64)) + np.asarray(adjustments, dtype=np.float64) * np.log(2.0)
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return cdf, weights


def _pool_index(rng: np.random.Generator, cdf: ndarray) -> int:
    return min(int(np.searchsorted(cdf, rng.random(), side="right")), len(cdf) - 1)


def _other_index(rng: np.random.Generator, count: int, excluded: int) -> int:
    index = int(rng.integers(count - 1))
    return index + (index >= excluded)


def build_image_pools(sources: Sequence[ImageSource], cache: dict[tuple[str, str], ImageIndex] | None = None
                      ) -> tuple[tuple[LocalImagePool, ...], ndarray, tuple[tuple[str, int, float], ...]]:
    if isinstance(sources, (str, os.PathLike)):
        raise TypeError(f"图片源必须是路径序列；单个目录请写成 [{os.fspath(sources)!r}]")
    if not sources:
        raise ValueError("图片源列表不能为空")
    if cache is None:
        cache = {}

    pools: list[LocalImagePool] = []
    for source in sources:
        if isinstance(source, (str, os.PathLike)):
            path, adjustment, alignment, layout = source, 0.0, "ffhq", "flat"
        elif isinstance(source, tuple) and len(source) in (2, 3, 4) and isinstance(source[0], (str, os.PathLike)):
            path, adjustment = source[:2]
            alignment = source[2] if len(source) >= 3 else "ffhq"
            layout = source[3] if len(source) == 4 else "flat"
        else:
            raise TypeError(f"无效图片源: {source!r}")

        adjustment = float(adjustment)
        if not np.isfinite(adjustment):
            raise ValueError(f"权重调整必须为有限数值，实际为 {adjustment!r}")
        if alignment not in ("ffhq", "arcface"):
            raise ValueError(f"alignment 必须为 ffhq 或 arcface，实际为 {alignment!r}")
        if layout not in ("flat", "identity"):
            raise ValueError(f"layout 必须为 flat 或 identity，实际为 {layout!r}")
        root = str(Path(path).resolve(strict=True))
        key = (root, layout)
        if key not in cache:
            cache[key] = _scan_image_index(root, layout)
        pools.append(LocalImagePool(cache[key], alignment, adjustment))

    counts = [pool.index.count for pool in pools]
    cdf, weights = _source_cdf(counts, [pool.adjustment for pool in pools])
    info = tuple((pool.index.root, count, float(weight)) for pool, count, weight in zip(pools, counts, weights, strict=True))
    return tuple(pools), cdf, info


class ImagePairSampler:
    """两种后端共用的路径采样；same 表示已知同身份重建监督。"""

    def __init__(self, src: Sequence[ImageSource], dst: Sequence[ImageSource], same_prob: float, cross_view_prob: float = 0.75) -> None:
        if not 0.0 <= same_prob <= 1.0 or not 0.0 <= cross_view_prob <= 1.0:
            raise ValueError("same_prob/cross_view_prob 必须位于 [0, 1]")
        self.same_prob = same_prob
        self.cross_view_prob = cross_view_prob
        cache: dict[tuple[str, str], ImageIndex] = {}
        self.src_pools, self.src_cdf, src_info = build_image_pools(src, cache)
        self.dst_pools, self.dst_cdf, dst_info = build_image_pools(dst, cache)
        if any(pool.alignment != "ffhq" for pool in self.dst_pools):
            raise ValueError("dst 数据池必须使用 ffhq 对齐")
        print_image_pools("SRC Sources", src_info, self.src_pools)
        print_image_pools("DST Sources", dst_info, self.dst_pools)

        self.cross_pools = tuple(pool for pool in self.dst_pools if pool.index.multi_view_indices)
        self.cross_cdf = np.empty(0)
        if self.cross_pools and same_prob > 0.0 and cross_view_prob > 0.0:
            counts = [pool.index.multi_view_count for pool in self.cross_pools]
            self.cross_cdf, weights = _source_cdf(counts, [pool.adjustment for pool in self.cross_pools])
            print_image_pools("Cross-view eligible DST Sources",
                              tuple((pool.index.root, count, float(weight)) for pool, count, weight in zip(self.cross_pools, counts, weights, strict=True)),
                              self.cross_pools)

        # 仅当目标数据集只有一个已知身份时，才需要从 SRC 池排除整个同根数据集。
        # 预计算条件 CDF；运行时无需拒绝采样或重复扫描。
        self.src_alternatives: dict[str, tuple[tuple[LocalImagePool, ...], ndarray]] = {}
        if same_prob < 1.0:
            src_roots = {pool.index.root for pool in self.src_pools if pool.index.layout == "identity"}
            for pool in self.dst_pools:
                index = pool.index
                if index.layout != "identity" or len(index.groups) != 1 or index.root not in src_roots or index.root in self.src_alternatives:
                    continue
                alternatives = tuple(source for source in self.src_pools if source.index is not index)
                if not alternatives:
                    raise ValueError(f"无法从 SRC 池构造不同身份样本: {index.root}")
                cdf, _ = _source_cdf([source.index.count for source in alternatives], [source.adjustment for source in alternatives])
                self.src_alternatives[index.root] = alternatives, cdf

        effective_cross_prob = same_prob * cross_view_prob if self.cross_cdf.size else 0.0
        print("Effective sampling probabilities:")
        print(f"    swap: {1.0 - same_prob:.3%}")
        print(f"    same_image: {same_prob - effective_cross_prob:.3%}")
        print(f"    cross_view: {effective_cross_prob:.3%}")
        if same_prob > 0.0 and cross_view_prob > 0.0 and not self.cross_pools and any(pool.index.layout == "identity" for pool in self.dst_pools):
            print("    No eligible cross-view DST identities; falling back to same-image reconstruction.")

    @staticmethod
    def _sample_file(rng: np.random.Generator, group: ImageGroup) -> str:
        return os.path.join(group.root, group.file_names[int(rng.integers(len(group.file_names)))])

    @staticmethod
    def _sample_group(rng: np.random.Generator, index: ImageIndex) -> int:
        # flat 池只有一组；不增加 RNG 调用，保留旧采样序列。
        return int(rng.integers(len(index.groups))) if index.layout == "identity" else 0

    def _sample_cross_view(self, rng: np.random.Generator) -> tuple[str, str, bool, bool]:
        index = self.cross_pools[_pool_index(rng, self.cross_cdf)].index
        group = index.groups[index.multi_view_indices[int(rng.integers(len(index.multi_view_indices)))]]
        count = len(group.file_names)
        src_idx = int(rng.integers(count))
        dst_idx = _other_index(rng, count, src_idx)
        return os.path.join(group.root, group.file_names[src_idx]), os.path.join(group.root, group.file_names[dst_idx]), True, False

    def sample(self, rng: np.random.Generator) -> tuple[str, str, bool, bool]:
        # 跨视角启用时先选择采样模式，避免构造随后会被丢弃的普通 DST 路径。
        # 禁用时保留历史 RNG 顺序：先采样 DST，再决定是否同图重建。
        same = bool(rng.random() < self.same_prob) if self.cross_cdf.size else None
        if same and rng.random() < self.cross_view_prob:
            return self._sample_cross_view(rng)

        dst_index = self.dst_pools[_pool_index(rng, self.dst_cdf)].index
        dst_group = self._sample_group(rng, dst_index)
        dst_path = self._sample_file(rng, dst_index.groups[dst_group])
        if same is None:
            same = bool(rng.random() < self.same_prob)
        if same:
            return dst_path, dst_path, True, False

        pools, cdf = self.src_alternatives.get(dst_index.root, (self.src_pools, self.src_cdf)) if dst_index.layout == "identity" else (self.src_pools, self.src_cdf)
        src_pool = pools[_pool_index(rng, cdf)]
        src_index = src_pool.index
        if src_index is dst_index and src_index.layout == "identity":
            src_group = _other_index(rng, len(src_index.groups), dst_group)
        else:
            src_group = self._sample_group(rng, src_index)
        src_path = self._sample_file(rng, src_index.groups[src_group])
        return src_path, dst_path, False, src_pool.alignment == "arcface"


def effective_source_alignments(src: Sequence[ImageSource], same_prob: float) -> set[str]:
    """重建样本的 source 来自 FFHQ DST；仅考虑实际可能采样的对齐类型。"""
    alignments = {entry[2] if isinstance(entry, tuple) and len(entry) >= 3 else "ffhq" for entry in src} if same_prob < 1.0 else set()
    if same_prob > 0.0:
        alignments.add("ffhq")
    return alignments


def source_identity_mode(src: Sequence[ImageSource], same_prob: float, alignment: str) -> str:
    """FFHQ 生成模式可复用展示图；ArcFace 生成模式始终从原图直接生成 teacher。"""
    if alignment == "arcface":
        return "direct"
    alignments = effective_source_alignments(src, same_prob)
    if alignments == {"ffhq"}:
        return "ffhq"
    if alignments == {"arcface"}:
        return "direct"
    return "mixed"


def prepare_source_identity_faces(src: Tensor, arcface_faces: Tensor, src_arcface_mask: Tensor, ffhq_grid: Tensor) -> Tensor:
    """FFHQ 使用原裁剪；ArcFace 使用独立保留的 112 像素，避免展示分辨率往返缩放。"""
    ffhq_faces = ffhq_to_arcface_112(src, ffhq_grid)
    return torch.where(src_arcface_mask[:, None, None, None], arcface_faces, ffhq_faces)


def print_image_pools(title: str, info: tuple[tuple[str, int, float], ...], pools: tuple[LocalImagePool, ...]) -> None:
    print(title + ":")
    for (label, count, weight), pool in zip(info, pools, strict=True):
        print(f"    {label}\n        count: {count:<7d} weight: {weight:<6.3f}")
        if pool.index.layout == "identity":
            print(f"        identities: {len(pool.index.groups)} multi_view_identities: {len(pool.index.multi_view_indices)}")
