"""高吞吐、可配置的 DALI 人脸交换训练数据管线。

数据流:
    1. ``src`` 与 ``dst`` 默认独立采样；按 ``same_prob`` 采样同图或跨视角同身份重建。
    2. ArcFace 模式从解码原图直接使用 Lanczos ROI resize，FFHQ 模式保持普通 resize；随后在 GPU 上执行随机水平翻转。
    3. ``dst`` 额外执行亮度、对比度、饱和度增强，并保留几何增强前的 ``dst_canonical``。
    4. ``dst`` 再执行随机仿射增强。
    5. 图像统一归一化到 ``[-1, 1]`` 并转换为 NCHW。
    6. 同时返回 ``theta_restore``，供 PyTorch ``affine_grid``/``grid_sample``
       将生成结果映射回仿射增强前的目标坐标系。

图片源:
    每个源可以是路径或 ``(path, adjustment, alignment, layout)`` 的 2~4 项前缀元组。alignment 默认为 ffhq；arcface 仅用于 src。layout 默认为 flat，identity 表示一级子目录为身份。数据源首先按
    ``sqrt(file_count) * 2**adjustment`` 分配采样概率，再在选中的文件夹内均匀
    采样图片（identity 布局先均匀选择身份）。训练数据只从本地目录读取，不包含网络访问或远程数据源状态。
    ``adjustment=1`` 表示将该数据源的基础权重翻倍，``-1`` 表示减半。

性能原则:
    Python 侧只负责目录扫描和读取压缩图像字节；解码后的 resize、flip、颜色增强、
    随机仿射、归一化和布局转换均保留在 DALI 图中。随机仿射矩阵也由 DALI 原生算子
    生成，避免逐样本 Python/OpenCV 回调和小矩阵 CPU→GPU 拷贝。

输出契约:
    ``src``: ``float32``，形状 ``(B, 3, H, W)``，RGB，值域 ``[-1, 1]``。
    ``dst``: ``float32``，形状 ``(B, 3, H, W)``，RGB，值域 ``[-1, 1]``。
    ``dst_canonical``: ``float32``，与 ``dst`` 相同的 flip/color，但未做几何仿射。
    ``theta_restore``: ``float32``，形状 ``(B, 2, 3)``，可直接传给
    ``torch.nn.functional.affine_grid(..., align_corners=False)``。
    ``same_mask``: ``bool``，形状 ``(B,)``，标记 src/dst 是否属于同身份重建样本（含跨视角）。
    ``src_arcface_mask``: ``bool``，形状 ``(B,)``，标记 src 是否已经使用 ArcFace 对齐；same pair 为 False。
    ``src_arcface_faces``: ArcFace 模式下是从原图直接采样的 112×112 身份输入；FFHQ 模式下保留原有 ArcFace 源图分支。
    FFHQ 生成模式保留 PyTorch 侧原有身份裁剪；ArcFace 生成模式直接返回解码原图产生的身份图。
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy import ndarray
from nvidia.dali import Pipeline, fn, pipeline_def
from nvidia.dali.math import clamp
from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

from misc.face_alignment import ffhq_to_arcface_112, make_ffhq_to_arcface_112_grid

from .dataloader_common import (
    FFHQ_TO_ARCFACE112_ROI,
    FloatRange,
    ImageDecoderBackend,
    ImagePairSampler,
    ImageSource,
    effective_source_alignments,
    generation_alignment_rois,
    is_identity_geometry,
    prepare_source_identity_faces,
    source_identity_mode,
    validate_range,
)


class _RandomImagePairSource:
    """为 DALI parallel external_source 提供本地 source/target 编码图像。"""

    def __init__(self, src: Sequence[ImageSource], dst: Sequence[ImageSource], same_prob: float,
                 cross_view_prob: float = 0.75, include_arcface_mask: bool = True) -> None:
        self.sampler = ImagePairSampler(src, dst, same_prob, cross_view_prob)
        self.include_arcface_mask = include_arcface_mask
        self.rng = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("rng", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.rng = None

    def __call__(self, _sample_info) -> tuple[ndarray, ...]:
        # fork 不经过反序列化；在每个 worker 第一次调用时独立初始化。
        if self.rng is None:
            self.rng = np.random.default_rng()
        src_path, dst_path, same, arcface = self.sampler.sample(self.rng)
        dst = np.fromfile(dst_path, dtype=np.uint8)
        src = dst if same and src_path == dst_path else np.fromfile(src_path, dtype=np.uint8)
        outputs = (src, dst, np.asarray(same, dtype=np.bool_))
        if self.include_arcface_mask:
            outputs += (np.asarray(arcface, dtype=np.bool_),)
        return outputs


def _constant_or_random_uniform(value_range: FloatRange):
    """DALI 不允许零宽的 uniform 范围；固定参数使用常量而非伪随机扰动。"""
    if value_range[0] == value_range[1]:
        return fn.constant(fdata=float(value_range[0]), shape=[], dtype=DALIDataType.FLOAT, device="cpu")
    return fn.random.uniform(range=value_range)


def _random_affine_matrices(img_resolution: int, rotation_range: FloatRange, scale_factor_range: FloatRange, tx_range: FloatRange, ty_range: FloatRange):
    rotation_range = validate_range("rotation_range", rotation_range)
    scale_factor_range = validate_range("scale_factor_range", scale_factor_range)
    tx_range = validate_range("tx_range", tx_range)
    ty_range = validate_range("ty_range", ty_range)

    if scale_factor_range[0] <= 0.0:
        raise ValueError(f"scale_factor_range 必须为正数范围，实际为 {scale_factor_range}")

    center = img_resolution * 0.5
    angle = _constant_or_random_uniform(rotation_range)
    scale = _constant_or_random_uniform(scale_factor_range)
    tx = _constant_or_random_uniform(tx_range) * img_resolution
    ty = _constant_or_random_uniform(ty_range) * img_resolution

    # DALI 与 OpenCV 的二维旋转正方向相反，因此 angle 取负即可保持旧数据增强语义。
    rotation = fn.transforms.rotation(angle=-angle, center=(center, center))
    scaling = fn.transforms.scale(scale=fn.stack(scale, scale), center=(center, center))
    translation = fn.transforms.translation(offset=fn.stack(tx, ty))
    src_to_dst = fn.transforms.combine(rotation, scaling, translation)

    # PyTorch affine_grid(align_corners=False) 需要 normalized output -> input 坐标矩阵。
    norm_to_pixel = fn.transforms.combine(
        fn.transforms.scale(scale=(center, center)),
        fn.transforms.translation(offset=(center - 0.5, center - 0.5)),
    )
    pixel_to_norm = fn.transforms.combine(
        fn.transforms.scale(scale=(2.0 / img_resolution, 2.0 / img_resolution)),
        fn.transforms.translation(offset=(1.0 / img_resolution - 1.0, 1.0 / img_resolution - 1.0)),
    )
    theta_restore = fn.transforms.combine(norm_to_pixel, src_to_dst, pixel_to_norm)
    return src_to_dst, fn.copy(theta_restore, device="gpu")


def _decode(encoded, backend: ImageDecoderBackend, decoder_hw_load: float):
    if not isinstance(backend, ImageDecoderBackend):
        raise TypeError(f"decoder_backend 必须为 ImageDecoderBackend，实际为 {type(backend).__name__}")
    if not 0.0 <= decoder_hw_load <= 1.0:
        raise ValueError(f"decoder_hw_load 必须位于 [0, 1]，实际为 {decoder_hw_load}")

    decoded = fn.decoders.image(encoded, device=backend.value, output_type=DALIImageType.RGB, hw_decoder_load=decoder_hw_load)
    if backend is ImageDecoderBackend.CPU:
        decoded = fn.copy(decoded, device="gpu")
    return decoded


def _normalize_chw(image, pixel_max: float = 255.0):
    image = clamp(image, lo=0.0, hi=pixel_max)
    midpoint = pixel_max * 0.5
    return fn.crop_mirror_normalize(
        image,
        device="gpu",
        dtype=DALIDataType.FLOAT,
        output_layout="CHW",
        mean=(midpoint, midpoint, midpoint),
        std=(midpoint, midpoint, midpoint),
    )


def _roi_in_dali_order(roi: tuple[float, float, float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    left, top, right, bottom = roi
    return (top, left), (bottom, right)


def _select_source_roi(arcface_mask, ffhq_roi: tuple[float, float, float, float], arcface_roi: tuple[float, float, float, float]):
    """每张 src 只运行一个 resize；ROI 根据原图对齐类型逐样本选择。"""
    ffhq_start, ffhq_end = _roi_in_dali_order(ffhq_roi)
    arc_start, arc_end = _roi_in_dali_order(arcface_roi)
    selected = fn.cast(arcface_mask, dtype=DALIDataType.FLOAT)
    start = fn.stack(*(ffhq_start[i] + selected * (arc_start[i] - ffhq_start[i]) for i in range(2)))
    end = fn.stack(*(ffhq_end[i] + selected * (arc_end[i] - ffhq_end[i]) for i in range(2)))
    return start, end


@pipeline_def
def create_dataloader_pipeline(
    img_resolution: int,
    src: Sequence[ImageSource],
    dst: Sequence[ImageSource],
    reader_prefetch_queue_depth: int = 2,
    decoder_backend: ImageDecoderBackend = ImageDecoderBackend.MIXED,
    decoder_hw_load: float = 0.75,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    flip_prob: float = 0.5,
    same_prob: float = 0.2,
    rotation_range: FloatRange = (-10.0, 10.0),
    scale_factor_range: FloatRange = (1.0 / 1.3, 1.25),
    tx_range: FloatRange = (-0.15, 0.15),
    ty_range: FloatRange = (-0.15, 0.15),
    alignment: str = "ffhq",
    cross_view_prob: float = 0.75,
):
    """构建训练用 DALI pipeline。

    参数:
        img_resolution: 最终训练图像的正方形边长。
        src: 身份来源本地图像池。必须是 ``ImageSource`` 序列；可包含路径或
            ``(路径, adjustment)``。单个本地目录也应写成 ``[path]``。
        dst: 目标本地图像池，格式与 ``src`` 相同；与 ``src`` 独立采样。
        reader_prefetch_queue_depth: parallel ``external_source`` 每个 Python worker
            可提前准备的 batch 数。只影响编码文件读取阶段。
        decoder_backend: 图像 decoder backend。默认 ``MIXED``。
        decoder_hw_load: mixed decoder 可交给专用 JPEG HW decoder 的负载比例。
            该参数只在支持对应硬件路径的平台上实际生效。
        brightness: ``dst`` 亮度随机乘数的最大偏移；``0.2`` 对应 ``[0.8, 1.2]``。
        contrast: ``dst`` 对比度随机乘数的最大偏移。
        saturation: ``dst`` 饱和度随机乘数的最大偏移。
        flip_prob: ``src`` 和 ``dst`` 各自独立水平翻转的概率。
        same_prob: 同身份重建的总概率。
        cross_view_prob: 同身份重建中使用跨视角数据的条件概率。
        rotation_range: ``dst`` 随机旋转角范围，单位为度。
        scale_factor_range: ``dst`` **实际缩放倍数**范围，例如 ``(0.8, 1.2)``。
            不再使用旧式“缩放偏移”语义。
        tx_range: ``dst`` 水平平移范围，相对于图像宽度，例如 ``0.1`` 表示 10%。
        ty_range: ``dst`` 垂直平移范围，相对于图像高度。

    返回:
        五个基础 DataNode：``src``、``dst``、``dst_canonical``、``theta_restore``、``same_mask``；按身份模式追加混合掩码与独立 112 图像。
        图像输出均为 RGB/NCHW/float32/``[-1, 1]``；``dst_canonical`` 保留与 ``dst``
        相同的 flip/color，但未做几何仿射；``theta_restore`` 为 ``(2, 3)`` 每样本矩阵，
        batch 经 DALIGenericIterator 后形状为 ``(B, 2, 3)``；``same_mask`` 为逐样本 bool。

    说明:
        ``@pipeline_def`` 还接受 ``batch_size``、``device_id``、``num_threads``、
        ``prefetch_queue_depth``、``py_num_workers`` 等 DALI Pipeline 构造参数；这些参数
        不属于本函数签名，但可以在调用 ``create_dataloader_pipeline(...)`` 时直接传入。
    """
    if img_resolution <= 0:
        raise ValueError(f"img_resolution 必须为正数，实际为 {img_resolution}")
    if alignment not in ("ffhq", "arcface"):
        raise ValueError(f"生成对齐标准无效：{alignment!r}")
    if reader_prefetch_queue_depth <= 0:
        raise ValueError(f"reader_prefetch_queue_depth 必须为正数，实际为 {reader_prefetch_queue_depth}")
    for name, value in (("brightness", brightness), ("contrast", contrast), ("saturation", saturation), ("flip_prob", flip_prob), ("same_prob", same_prob)):
        if not np.isfinite(value):
            raise ValueError(f"{name} 必须为有限数值，实际为 {value}")
    if min(brightness, contrast, saturation) < 0.0:
        raise ValueError("brightness/contrast/saturation 不能为负数")
    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError(f"flip_prob 必须位于 [0, 1]，实际为 {flip_prob}")
    if not 0.0 <= same_prob <= 1.0:
        raise ValueError(f"same_prob 必须位于 [0, 1]，实际为 {same_prob}")
    rotation_range = validate_range("rotation_range", rotation_range)
    scale_factor_range = validate_range("scale_factor_range", scale_factor_range)
    tx_range = validate_range("tx_range", tx_range)
    ty_range = validate_range("ty_range", ty_range)
    if scale_factor_range[0] <= 0.0:
        raise ValueError("scale_factor_range 必须为正数范围")
    apply_geometry = not is_identity_geometry(rotation_range, scale_factor_range, tx_range, ty_range)

    parallel = Pipeline.current().py_num_workers > 0
    identity_mode = source_identity_mode(src, same_prob, alignment)
    source_alignments = effective_source_alignments(src, same_prob)
    include_arcface_mask = identity_mode == "mixed" or (alignment == "arcface" and len(source_alignments) > 1)
    raw_outputs = fn.external_source(
        source=_RandomImagePairSource(src, dst, same_prob, cross_view_prob, include_arcface_mask),
        num_outputs=4 if include_arcface_mask else 3,
        device="cpu",
        parallel=parallel,
        **({"prefetch_queue_depth": reader_prefetch_queue_depth} if parallel else {}),
        dtype=(DALIDataType.UINT8, DALIDataType.UINT8, DALIDataType.BOOL) + ((DALIDataType.BOOL,) if include_arcface_mask else ()),
        ndim=(1, 1, 0) + ((0,) if include_arcface_mask else ()),
        batch=False,
    )
    src_raw, dst_raw, same_mask = raw_outputs[:3]
    if include_arcface_mask:
        src_arcface_mask = raw_outputs[3]

    src_image = _decode(src_raw, decoder_backend, decoder_hw_load)
    dst_image = _decode(dst_raw, decoder_backend, decoder_hw_load)

    if alignment == "arcface":
        # ROI 以原图尺寸为基准，直接完成固定 FFHQ->ArcFace128 和输出尺寸的抗混叠采样。
        # 112 teacher 分支也从原图独立采样，绝不经过生成图缩放。
        # 纯 FFHQ 或纯 ArcFace（且无 same pair）使用静态 ROI，避免逐样本构造动态参数。
        ffhq_roi, native_arcface_roi = generation_alignment_rois(img_resolution)
        if source_alignments == {"ffhq"}:
            src_start, src_end = _roi_in_dali_order(ffhq_roi)
            id_start, id_end = _roi_in_dali_order(FFHQ_TO_ARCFACE112_ROI)
        elif source_alignments == {"arcface"}:
            # same pair 的 src 来自 FFHQ dst；只要 same_prob>0，就不能使用 ArcFace 静态 ROI。
            src_start, src_end = _roi_in_dali_order(native_arcface_roi)
            id_start, id_end = (0.0, 0.0), (1.0, 1.0)
        else:
            src_start, src_end = _select_source_roi(src_arcface_mask, ffhq_roi, native_arcface_roi)
            id_start, id_end = _select_source_roi(src_arcface_mask, FFHQ_TO_ARCFACE112_ROI, (0.0, 0.0, 1.0, 1.0))
        src_arcface_faces = fn.resize(src_image, device="gpu", size=112, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3,
                                      roi_start=id_start, roi_end=id_end, roi_relative=True, antialias=True)
        src_image = fn.resize(src_image, device="gpu", size=img_resolution, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3,
                              roi_start=src_start, roi_end=src_end, roi_relative=True, antialias=True)
        dst_start, dst_end = _roi_in_dali_order(ffhq_roi)
        dst_image = fn.resize(dst_image, device="gpu", size=img_resolution, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3,
                              roi_start=dst_start, roi_end=dst_end, roi_relative=True, antialias=True)
    else:
        # FFHQ 快速路径保持原样，无固定对齐仿射和额外重采样。
        if identity_mode != "ffhq":
            src_arcface_faces = fn.resize(src_image, device="gpu", size=112, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)
        src_image = fn.resize(src_image, device="gpu", size=img_resolution, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)
        dst_image = fn.resize(dst_image, device="gpu", size=img_resolution, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)

    if flip_prob > 0.0:
        src_flip = fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.INT32)
        src_image = fn.flip(src_image, device="gpu", horizontal=src_flip)
        if identity_mode != "ffhq":
            src_arcface_faces = fn.flip(src_arcface_faces, device="gpu", horizontal=src_flip)
        dst_image = fn.flip(dst_image, device="gpu", horizontal=fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.INT32))

    dst_pixel_max = 255.0
    if any(value != 0.0 for value in (brightness, contrast, saturation)):
        # DALI float color_twist 以 0.5 为对比度中心；后续几何增强沿用单位值域。
        dst_pixel_max = 1.0
        dst_image = fn.color_twist(
            dst_image * (1.0 / 255.0),
            device="gpu",
            brightness=fn.random.uniform(range=(1.0 - brightness, 1.0 + brightness)) if brightness else 1.0,
            contrast=fn.random.uniform(range=(1.0 - contrast, 1.0 + contrast)) if contrast else 1.0,
            saturation=fn.random.uniform(range=(1.0 - saturation, 1.0 + saturation)) if saturation else 1.0,
        )

    dst_canonical = _normalize_chw(dst_image, dst_pixel_max)
    if apply_geometry:
        affine_matrix, theta_restore = _random_affine_matrices(img_resolution, rotation_range, scale_factor_range, tx_range, ty_range)
        # WarpAffine 仅支持常量填充/边缘复制；Perspective 支持镜像反射。
        # 将原 2x3 仿射矩阵补为 3x3 单应矩阵，保持原几何增强语义。
        perspective_matrix = fn.cat(
            fn.copy(affine_matrix, device="gpu"),
            fn.constant(fdata=[0.0, 0.0, 1.0], shape=(1, 3), dtype=DALIDataType.FLOAT, device="gpu"),
            axis=0,
        )
        dst_image = fn.experimental.warp_perspective(dst_image, perspective_matrix, device="gpu", inverse_map=False,
                                                     border_mode="reflect")
        dst_output = _normalize_chw(dst_image, dst_pixel_max)
    else:
        # 恒等矩阵作为 DALI 常量输出，跳过随机仿射计算、warp 和重复归一化。
        theta_restore = fn.constant(fdata=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], shape=[2, 3], dtype=DALIDataType.FLOAT, device="gpu")
        dst_output = dst_canonical

    outputs = (_normalize_chw(src_image), dst_output, dst_canonical, theta_restore, fn.copy(same_mask, device="gpu"))
    if identity_mode == "mixed":
        outputs += (fn.copy(src_arcface_mask, device="gpu"),)
    if identity_mode != "ffhq":
        outputs += (_normalize_chw(src_arcface_faces),)
    return outputs


class _DALITrainingDataLoader:
    """将 DALI pipeline/iterator 适配到统一 ``next()`` 接口。"""

    def __init__(
        self,
        *,
        batch_size: int,
        device,
        img_resolution: int,
        src: Sequence[ImageSource],
        dst: Sequence[ImageSource],
        alignment: str = "ffhq",
        **config,
    ) -> None:
        import torch
        from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

        if alignment not in ("ffhq", "arcface"):
            raise ValueError(f"生成对齐标准无效：{alignment!r}")
        self.alignment = alignment
        self.identity_mode = source_identity_mode(src, config.get("same_prob", 0.2), alignment)
        device_id = device.index if device.index is not None else torch.cuda.current_device()
        pipe = create_dataloader_pipeline(
            batch_size=batch_size,
            device_id=device_id,
            img_resolution=img_resolution,
            src=src,
            dst=dst,
            alignment=alignment,
            **config,
        )
        self._output_map = ["src", "dst", "dst_canonical", "theta_restore", "same_mask"]
        if self.identity_mode == "mixed":
            self._output_map.append("src_arcface_mask")
        if self.identity_mode != "ffhq":
            self._output_map.append("src_arcface_faces")
        self._iterator = DALIGenericIterator(
            pipelines=pipe,
            output_map=self._output_map,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP,
        )
        self._source_ffhq_grid = make_ffhq_to_arcface_112_grid(img_resolution, batch_size, device) if self.identity_mode != "direct" else None

    def next(self):
        data = self._iterator.next()[0]
        if self.identity_mode == "direct":
            identity = data["src_arcface_faces"]
        elif self.identity_mode == "ffhq":
            identity = ffhq_to_arcface_112(data["src"], self._source_ffhq_grid)
        else:
            identity = prepare_source_identity_faces(data["src"], data["src_arcface_faces"], data["src_arcface_mask"], self._source_ffhq_grid)
        return data["src"], data["dst"], data["dst_canonical"], data["theta_restore"], data["same_mask"], identity
