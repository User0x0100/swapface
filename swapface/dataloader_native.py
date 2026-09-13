"""不依赖厂商 SDK 的 PyTorch 训练数据管线，主要用于 ROCm。"""

import math
import os
from collections.abc import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from .dataloader_common import FloatRange, ImageDecoderBackend, ImageSource, LocalImagePool, build_image_pools, print_image_pools, validate_range


class _RandomImagePairDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """在 CPU worker 中随机采样、解码并 Lanczos resize 成固定尺寸。"""

    def __init__(self, src: Sequence[ImageSource], dst: Sequence[ImageSource], img_resolution: int) -> None:
        super().__init__()
        if img_resolution <= 0:
            raise ValueError(f"img_resolution 必须为正数，实际为 {img_resolution}")
        self.img_resolution = img_resolution
        self.src_pools, self.src_cdf, src_info = build_image_pools(src)
        self.dst_pools, self.dst_cdf, dst_info = build_image_pools(dst)
        print_image_pools("SRC Sources", src_info)
        print_image_pools("DST Sources", dst_info)

    @staticmethod
    def _sample_path(rng: np.random.Generator, pools: tuple[LocalImagePool, ...], cdf: np.ndarray) -> str:
        pool_index = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(pools) - 1)
        pool = pools[pool_index]
        file_name = pool.file_names[int(rng.integers(len(pool.file_names)))]
        return os.path.join(pool.root, file_name)

    def _decode_resize(self, path: str) -> Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (self.img_resolution, self.img_resolution):
                image = image.resize((self.img_resolution, self.img_resolution), resample=Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        # DataLoader 会为每个 worker 设置独立 torch seed；以它初始化 NumPy RNG，
        # 避免 spawn/fork worker 复制出相同采样序列。
        rng = np.random.default_rng(torch.initial_seed())
        while True:
            src_path = self._sample_path(rng, self.src_pools, self.src_cdf)
            dst_path = self._sample_path(rng, self.dst_pools, self.dst_cdf)
            yield self._decode_resize(src_path), self._decode_resize(dst_path)


def _uniform(batch_size: int, value_range: FloatRange, device: torch.device) -> Tensor:
    low, high = value_range
    if low == high:
        return torch.full((batch_size,), low, device=device, dtype=torch.float32)
    return torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(low, high)


def make_affine_thetas(
    batch_size: int,
    img_resolution: int,
    rotation_range: FloatRange,
    scale_factor_range: FloatRange,
    tx_range: FloatRange,
    ty_range: FloatRange,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """返回 augmentation output->input theta 与恢复用 canonical->augmented theta。"""
    rotation_range = validate_range("rotation_range", rotation_range)
    scale_factor_range = validate_range("scale_factor_range", scale_factor_range)
    tx_range = validate_range("tx_range", tx_range)
    ty_range = validate_range("ty_range", ty_range)
    if scale_factor_range[0] <= 0.0:
        raise ValueError(f"scale_factor_range 必须为正数范围，实际为 {scale_factor_range}")

    # 现有 DALI 图先采样 angle，再传给 transforms.rotation(angle=-angle)。
    angle = -_uniform(batch_size, rotation_range, device) * (math.pi / 180.0)
    scale = _uniform(batch_size, scale_factor_range, device)
    tx = _uniform(batch_size, tx_range, device) * img_resolution
    ty = _uniform(batch_size, ty_range, device) * img_resolution

    cos = angle.cos() * scale
    sin = angle.sin() * scale
    center = img_resolution * 0.5

    # 像素坐标 src->dst：T * C * S * R * C^-1。
    affine = torch.empty((batch_size, 2, 3), device=device, dtype=torch.float32)
    affine[:, 0, 0] = cos
    affine[:, 0, 1] = -sin
    affine[:, 1, 0] = sin
    affine[:, 1, 1] = cos
    affine[:, 0, 2] = center + tx - cos * center + sin * center
    affine[:, 1, 2] = center + ty - sin * center - cos * center

    # 与 DALI norm_to_pixel -> src_to_dst -> pixel_to_norm 完全相同的
    # align_corners=False 坐标变换。线性部分不变，只转换 translation。
    offset = center - 0.5
    norm_offset = 1.0 / img_resolution - 1.0
    theta_restore = torch.empty_like(affine)
    theta_restore[:, :, :2] = affine[:, :, :2]
    theta_restore[:, 0, 2] = (2.0 / img_resolution) * (affine[:, 0, 0] * offset + affine[:, 0, 1] * offset + affine[:, 0, 2]) + norm_offset
    theta_restore[:, 1, 2] = (2.0 / img_resolution) * (affine[:, 1, 0] * offset + affine[:, 1, 1] * offset + affine[:, 1, 2]) + norm_offset

    # grid_sample 需要 augmented output -> canonical input，因此取恢复 theta 的逆。
    a, b = theta_restore[:, 0, 0], theta_restore[:, 0, 1]
    c, d = theta_restore[:, 1, 0], theta_restore[:, 1, 1]
    x, y = theta_restore[:, 0, 2], theta_restore[:, 1, 2]
    det = a * d - b * c
    theta_augment = torch.empty_like(theta_restore)
    theta_augment[:, 0, 0] = d / det
    theta_augment[:, 0, 1] = -b / det
    theta_augment[:, 1, 0] = -c / det
    theta_augment[:, 1, 1] = a / det
    theta_augment[:, 0, 2] = -(theta_augment[:, 0, 0] * x + theta_augment[:, 0, 1] * y)
    theta_augment[:, 1, 2] = -(theta_augment[:, 1, 0] * x + theta_augment[:, 1, 1] * y)
    return theta_augment, theta_restore


def _color_twist(images: Tensor, brightness: float, contrast: float, saturation: float) -> Tensor:
    batch_size = images.shape[0]
    device = images.device
    brightness_factor = _uniform(batch_size, (1.0 - brightness, 1.0 + brightness), device).view(-1, 1, 1, 1)
    contrast_factor = _uniform(batch_size, (1.0 - contrast, 1.0 + contrast), device).view(-1, 1, 1, 1)
    saturation_factor = _uniform(batch_size, (1.0 - saturation, 1.0 + saturation), device).view(-1, 1, 1, 1)

    # DALI ColorTwist 的无 hue 路径：BT.601 luminance 饱和度变换，contrast 以 0.5
    # 为中心，最后乘 brightness。输入保持 DALI 当前的 0..255 float 语义。
    gray = images[:, 0:1] * 0.299 + images[:, 1:2] * 0.587 + images[:, 2:3] * 0.114
    images = gray + saturation_factor * (images - gray)
    images = images * contrast_factor + (1.0 - contrast_factor) * 0.5
    return images * brightness_factor


class _NativeTrainingDataLoader:
    """CPU 并行解码 + GPU 批量增强的数据加载器。"""

    def __init__(
        self,
        *,
        batch_size: int,
        device: torch.device,
        img_resolution: int,
        src: Sequence[ImageSource],
        dst: Sequence[ImageSource],
        num_threads: int,
        prefetch_queue_depth: int,
        py_num_workers: int,
        py_start_method: str,
        reader_prefetch_queue_depth: int,
        decoder_backend: ImageDecoderBackend,
        decoder_hw_load: float,
        brightness: float,
        contrast: float,
        saturation: float,
        flip_prob: float,
        rotation_range: FloatRange,
        scale_factor_range: FloatRange,
        tx_range: FloatRange,
        ty_range: FloatRange,
    ) -> None:
        del num_threads, prefetch_queue_depth, decoder_backend, decoder_hw_load
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数")
        if py_num_workers < 0:
            raise ValueError("py_num_workers 不能为负数")
        if reader_prefetch_queue_depth <= 0:
            raise ValueError("reader_prefetch_queue_depth 必须为正数")
        if min(brightness, contrast, saturation) < 0.0:
            raise ValueError("brightness/contrast/saturation 不能为负数")
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError("flip_prob 必须位于 [0, 1]")

        self.device = device
        self.img_resolution = img_resolution
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.flip_prob = flip_prob
        self.rotation_range = validate_range("rotation_range", rotation_range)
        self.scale_factor_range = validate_range("scale_factor_range", scale_factor_range)
        self.tx_range = validate_range("tx_range", tx_range)
        self.ty_range = validate_range("ty_range", ty_range)
        if self.scale_factor_range[0] <= 0.0:
            raise ValueError("scale_factor_range 必须为正数范围")

        dataset = _RandomImagePairDataset(src, dst, img_resolution)
        loader_kwargs: dict[str, object] = {
            "batch_size": batch_size,
            "num_workers": py_num_workers,
            "pin_memory": True,
            "drop_last": True,
        }
        if py_num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=reader_prefetch_queue_depth,
                multiprocessing_context=py_start_method,
            )
        self.loader = DataLoader(dataset, **loader_kwargs)
        self.iterator = iter(self.loader)
        print(f"PyTorch 数据管线：CPU Pillow/Lanczos 解码缩放，workers={py_num_workers}, prefetch_factor={reader_prefetch_queue_depth}；批量增强在 {device} 执行")

    @torch.no_grad()
    def next(self) -> tuple[Tensor, Tensor, Tensor]:
        src, dst = next(self.iterator)
        src = src.to(device=self.device, dtype=torch.float32, non_blocking=True)
        dst = dst.to(device=self.device, dtype=torch.float32, non_blocking=True)
        batch_size = src.shape[0]

        if self.flip_prob > 0.0:
            src_flip = torch.rand((batch_size, 1, 1, 1), device=self.device) < self.flip_prob
            dst_flip = torch.rand((batch_size, 1, 1, 1), device=self.device) < self.flip_prob
            src = torch.where(src_flip, src.flip(-1), src)
            dst = torch.where(dst_flip, dst.flip(-1), dst)

        dst = _color_twist(dst, self.brightness, self.contrast, self.saturation)
        theta_augment, theta_restore = make_affine_thetas(
            batch_size,
            self.img_resolution,
            self.rotation_range,
            self.scale_factor_range,
            self.tx_range,
            self.ty_range,
            self.device,
        )
        grid = F.affine_grid(theta_augment, dst.shape, align_corners=False)
        dst = F.grid_sample(dst, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        src = src.clamp_(0.0, 255.0).div_(127.5).sub_(1.0)
        dst = dst.clamp_(0.0, 255.0).div_(127.5).sub_(1.0)
        return src, dst, theta_restore
