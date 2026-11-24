"""不依赖厂商 SDK 的 PyTorch 训练数据管线，主要用于 ROCm。"""

import math
from collections.abc import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from misc.face_alignment import ffhq_to_arcface_112, make_ffhq_to_arcface_112_grid, make_ffhq_to_arcface_grid

from .dataloader_common import (
    FFHQ_TO_ARCFACE112_ROI,
    FloatRange,
    ImageDecoderBackend,
    ImagePairSampler,
    ImageSource,
    generation_alignment_rois,
    is_identity_geometry,
    prepare_source_identity_faces,
    source_identity_mode,
    validate_range,
)


def _resize_roi(image: Image.Image, output_size: int, roi: tuple[float, float, float, float]) -> Image.Image:
    """从原图直接以 Lanczos 抗混叠采样目标坐标；越界处复制边缘像素。"""
    left, top, right, bottom = (roi[0] * image.width, roi[1] * image.height, roi[2] * image.width, roi[3] * image.height)
    pad_left, pad_top = max(0, math.ceil(-left)), max(0, math.ceil(-top))
    pad_right, pad_bottom = max(0, math.ceil(right - image.width)), max(0, math.ceil(bottom - image.height))
    if pad_left or pad_top or pad_right or pad_bottom:
        # 原有 grid_sample(padding_mode="border") 的边缘复制语义；仅 ArcFace 原生源图会触发。
        pixels = np.pad(np.asarray(image), ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
        image = Image.fromarray(pixels)
        left, right = left + pad_left, right + pad_left
        top, bottom = top + pad_top, bottom + pad_top
    return image.resize((output_size, output_size), resample=Image.Resampling.LANCZOS, box=(left, top, right, bottom))


def _image_to_tensor(image: Image.Image) -> Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)


class _RandomImagePairDataset(IterableDataset[tuple[Tensor, Tensor, bool, bool, Tensor, bool, bool]]):
    """在 CPU worker 中随机采样、解码并 Lanczos resize 成固定尺寸。"""

    def __init__(self, src: Sequence[ImageSource], dst: Sequence[ImageSource], img_resolution: int, same_prob: float, alignment: str = "ffhq", gpu_align_ffhq: bool = False,
                 cross_view_prob: float = 0.75) -> None:
        super().__init__()
        if img_resolution <= 0:
            raise ValueError(f"img_resolution 必须为正数，实际为 {img_resolution}")
        if not 0.0 <= same_prob <= 1.0:
            raise ValueError(f"same_prob 必须位于 [0, 1]，实际为 {same_prob}")
        if alignment not in ("ffhq", "arcface"):
            raise ValueError(f"生成对齐标准无效：{alignment!r}")
        self.img_resolution = img_resolution
        self.alignment = alignment
        self.generation_rois = generation_alignment_rois(img_resolution) if alignment == "arcface" else None
        # 原图恰好等于生成尺寸且输出 >=512 时，固定对齐可留到 GPU：
        # 无需提前缩小，不会发生二次采样，避免 CPU 同尺寸 Lanczos 的高额成本。
        self.gpu_align_ffhq = gpu_align_ffhq
        self.sampler = ImagePairSampler(src, dst, same_prob, cross_view_prob)
        self.identity_mode = source_identity_mode(src, same_prob, alignment)
        # 纯 FFHQ 不携带 112 图像；混合模式保留固定形状以供默认 collate。
        self.identity_placeholder = torch.empty(0, dtype=torch.uint8) if self.identity_mode == "ffhq" else (
            torch.zeros((3, 112, 112), dtype=torch.uint8) if self.identity_mode == "mixed" else None
        )

    def _resize_generation(self, image: Image.Image, source_alignment: str) -> tuple[Image.Image, bool]:
        if self.gpu_align_ffhq and source_alignment == "ffhq" and image.size == (self.img_resolution, self.img_resolution):
            return image, True
        if self.alignment == "arcface":
            assert self.generation_rois is not None
            roi = self.generation_rois[0] if source_alignment == "ffhq" else self.generation_rois[1]
            return _resize_roi(image, self.img_resolution, roi), False
        if image.size != (self.img_resolution, self.img_resolution):
            return image.resize((self.img_resolution, self.img_resolution), resample=Image.Resampling.LANCZOS), False
        return image, False

    def _decode_resize_with_meta(self, path: str) -> tuple[Tensor, bool]:
        with Image.open(path) as opened:
            image, needs_gpu_align = self._resize_generation(opened.convert("RGB"), "ffhq")
            return _image_to_tensor(image), needs_gpu_align

    def _decode_resize(self, path: str) -> Tensor:
        # 保留原有辅助接口，测试及 FFHQ 路径仅需图片。
        return self._decode_resize_with_meta(path)[0]

    def _decode_source(self, path: str, arcface: bool) -> tuple[Tensor, Tensor, bool]:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            display, needs_gpu_align = self._resize_generation(image, "arcface" if arcface else "ffhq")
            if arcface:
                identity = image if image.size == (112, 112) else image.resize((112, 112), resample=Image.Resampling.LANCZOS)
            elif self.alignment == "arcface":
                # 112 teacher 必须直接从原始 FFHQ 图生成，不能从生成用的 128/256 图再次裁剪。
                identity = _resize_roi(image, 112, FFHQ_TO_ARCFACE112_ROI)
            else:
                return _image_to_tensor(display), self.identity_placeholder, False
            return _image_to_tensor(display), _image_to_tensor(identity), needs_gpu_align

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, bool, bool, Tensor, bool, bool]]:
        # DataLoader 会为每个 worker 设置独立 torch seed；以它初始化 NumPy RNG。
        rng = np.random.default_rng(torch.initial_seed())
        while True:
            src_path, dst_path, same, arcface = self.sampler.sample(rng)
            if same and src_path == dst_path:
                if self.alignment == "arcface":
                    image, identity, needs_gpu_align = self._decode_source(dst_path, False)
                else:
                    image, identity, needs_gpu_align = self._decode_resize(dst_path), self.identity_placeholder, False
                yield image, image, True, False, identity, needs_gpu_align, needs_gpu_align
            else:
                display, identity, src_gpu_align = self._decode_source(src_path, arcface)
                dst, dst_gpu_align = self._decode_resize_with_meta(dst_path)
                yield display, dst, same, arcface, identity, src_gpu_align, dst_gpu_align


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

    # 与 DALI 一致：旋转中心为 H/2，而 align_corners=False 的像素中心为 (H-1)/2。
    # 直接在归一化坐标构造矩阵，保留这半像素偏移及相同的随机参数采样顺序。
    angle = -_uniform(batch_size, rotation_range, device) * (math.pi / 180.0)
    scale = _uniform(batch_size, scale_factor_range, device)
    tx = _uniform(batch_size, tx_range, device)
    ty = _uniform(batch_size, ty_range, device)
    cos, sin = angle.cos(), angle.sin()
    a, b = cos * scale, sin * scale
    x = 2.0 * tx + (1.0 - a + b) / img_resolution
    y = 2.0 * ty + (1.0 - b - a) / img_resolution
    theta_restore = torch.stack((a, -b, x, b, a, y), dim=1).reshape(batch_size, 2, 3)

    # grid_sample 需要 output->input；相似变换的逆线性部分为 R^T / scale。
    p, q = cos / scale, sin / scale
    theta_augment = torch.stack((p, q, -p * x - q * y, -q, p, q * x - p * y), dim=1).reshape(batch_size, 2, 3)
    return theta_augment, theta_restore


def _color_twist(images: Tensor, brightness: float, contrast: float, saturation: float) -> Tensor:
    batch_size = images.shape[0]
    device = images.device
    brightness_factor = _uniform(batch_size, (1.0 - brightness, 1.0 + brightness), device).view(-1, 1, 1, 1)
    contrast_factor = _uniform(batch_size, (1.0 - contrast, 1.0 + contrast), device).view(-1, 1, 1, 1)
    saturation_factor = _uniform(batch_size, (1.0 - saturation, 1.0 + saturation), device).view(-1, 1, 1, 1)

    # BT.601 luminance 饱和度变换；0..255 像素的对比度中心为 127.5，最后乘 brightness。
    gray = images[:, 0:1] * 0.299 + images[:, 1:2] * 0.587 + images[:, 2:3] * 0.114
    images = gray + saturation_factor * (images - gray)
    images = images * contrast_factor + (1.0 - contrast_factor) * 127.5
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
        same_prob: float,
        rotation_range: FloatRange,
        scale_factor_range: FloatRange,
        tx_range: FloatRange,
        ty_range: FloatRange,
        alignment: str = "ffhq",
        cross_view_prob: float = 0.75,
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
        if not 0.0 <= same_prob <= 1.0:
            raise ValueError("same_prob 必须位于 [0, 1]")

        self.device = device
        self.img_resolution = img_resolution
        if alignment not in ("ffhq", "arcface"):
            raise ValueError(f"无效的生成对齐标准：{alignment!r}")
        self.alignment = alignment
        self.identity_mode = source_identity_mode(src, same_prob, alignment)
        self.source_ffhq_grid = make_ffhq_to_arcface_112_grid(img_resolution, batch_size, device) if self.identity_mode != "direct" else None
        gpu_align_ffhq = alignment == "arcface" and device.type == "cuda" and img_resolution >= 512
        self.generation_grid = make_ffhq_to_arcface_grid(img_resolution, img_resolution, batch_size, device) if gpu_align_ffhq else None
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
        self.apply_color_twist = any(value != 0.0 for value in (brightness, contrast, saturation))
        self.apply_geometry = not is_identity_geometry(self.rotation_range, self.scale_factor_range, self.tx_range, self.ty_range)
        # 仅在禁用随机几何增强时创建一次恒等矩阵，避免每个 batch 重复构造。
        self.identity_theta_restore = (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            if not self.apply_geometry else None
        )

        dataset = _RandomImagePairDataset(src, dst, img_resolution, same_prob, alignment, gpu_align_ffhq, cross_view_prob)
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
        description = f"PyTorch 数据管线：CPU Pillow/Lanczos 解码缩放，workers={py_num_workers}, prefetch_factor={reader_prefetch_queue_depth}；批量增强在 {device} 执行"
        if gpu_align_ffhq:
            description += "；同尺寸高分辨率 FFHQ 对齐使用 GPU Bicubic"
        print(description)

    @torch.no_grad()
    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        src, dst, same_mask, src_arcface_mask, arcface_faces, src_gpu_align, dst_gpu_align = next(self.iterator)
        src = src.to(device=self.device, dtype=torch.float32, non_blocking=True)
        dst = dst.to(device=self.device, dtype=torch.float32, non_blocking=True)
        same_mask = same_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.identity_mode == "mixed":
            src_arcface_mask = src_arcface_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.identity_mode != "ffhq":
            arcface_faces = arcface_faces.to(device=self.device, dtype=torch.float32, non_blocking=True)
        batch_size = src.shape[0]

        if self.generation_grid is not None:
            # 原图恰好就是目标尺寸时，在 GPU 数据加载阶段执行一次高质量固定仿射。
            # 对非同尺寸 FFHQ 或 ArcFace 原生 SRC，CPU 已一次 Lanczos 采样，不能再变换。
            if src_gpu_align.any():
                aligned_src = F.grid_sample(src, self.generation_grid, mode="bicubic", padding_mode="border", align_corners=False).clamp_(0.0, 255.0)
                src = aligned_src if src_gpu_align.all() else torch.where(src_gpu_align.to(self.device)[:, None, None, None], aligned_src, src)
            if dst_gpu_align.any():
                aligned_dst = F.grid_sample(dst, self.generation_grid, mode="bicubic", padding_mode="border", align_corners=False).clamp_(0.0, 255.0)
                dst = aligned_dst if dst_gpu_align.all() else torch.where(dst_gpu_align.to(self.device)[:, None, None, None], aligned_dst, dst)

        if self.flip_prob > 0.0:
            src_flip = torch.rand((batch_size, 1, 1, 1), device=self.device) < self.flip_prob
            dst_flip = torch.rand((batch_size, 1, 1, 1), device=self.device) < self.flip_prob
            src = torch.where(src_flip, src.flip(-1), src)
            if self.identity_mode != "ffhq":
                arcface_faces = torch.where(src_flip, arcface_faces.flip(-1), arcface_faces)
            dst = torch.where(dst_flip, dst.flip(-1), dst)

        if self.apply_color_twist:
            dst = _color_twist(dst, self.brightness, self.contrast, self.saturation)
        # 固定 FFHQ->ArcFace 对齐已在 CPU Lanczos 或 GPU Bicubic 快速路径完成。
        dst_canonical = dst
        if self.apply_geometry:
            theta_augment, theta_restore = make_affine_thetas(
                batch_size, self.img_resolution, self.rotation_range, self.scale_factor_range, self.tx_range, self.ty_range, self.device
            )
            grid = F.affine_grid(theta_augment, dst_canonical.shape, align_corners=False)
            dst = F.grid_sample(dst_canonical, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
            dst = dst.clamp_(0.0, 255.0).div_(127.5).sub_(1.0)
        else:
            assert self.identity_theta_restore is not None
            theta_restore = self.identity_theta_restore

        src = src.clamp_(0.0, 255.0).div_(127.5).sub_(1.0)
        dst_canonical = dst_canonical.clamp_(0.0, 255.0).div_(127.5).sub_(1.0)
        if not self.apply_geometry:
            # 无仿射时直接共用 canonical，避免重采样和原地重复归一化。
            dst = dst_canonical
        if self.identity_mode != "ffhq":
            arcface_faces = arcface_faces.clamp_(0.0, 255.0).div_(127.5).sub_(1.0)
        if self.identity_mode == "direct":
            source_identity_faces = arcface_faces
        elif self.identity_mode == "ffhq":
            source_identity_faces = ffhq_to_arcface_112(src, self.source_ffhq_grid)
        else:
            source_identity_faces = prepare_source_identity_faces(src, arcface_faces, src_arcface_mask, self.source_ffhq_grid)
        return src, dst, dst_canonical, theta_restore, same_mask, source_identity_faces
