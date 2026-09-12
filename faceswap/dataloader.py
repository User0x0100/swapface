"""高吞吐、可配置的 DALI 人脸交换训练数据管线。

数据流:
    1. ``src`` 与 ``dst`` 从两个独立图片池随机采样编码图像。
    2. DALI 解码后在 GPU 上执行 resize 与随机水平翻转。
    3. ``dst`` 额外执行亮度、对比度、饱和度和随机仿射增强。
    4. 图像统一归一化到 ``[-1, 1]`` 并转换为 NCHW。
    5. 同时返回 ``theta_restore``，供 PyTorch ``affine_grid``/``grid_sample``
       将生成结果映射回仿射增强前的目标坐标系。

图片源:
    每个源可以是路径，也可以是 ``(path, adjustment)``。文件夹首先按
    ``sqrt(file_count) * 2**adjustment`` 分配采样概率，再在选中的文件夹内均匀
    采样图片。训练数据只从本地目录读取，不包含网络访问或远程数据源状态。
    ``adjustment=1`` 表示将该数据源的基础权重翻倍，``-1`` 表示减半。

性能原则:
    Python 侧只负责目录扫描和读取压缩图像字节；解码后的 resize、flip、颜色增强、
    仿射、归一化和布局转换均保留在 DALI 图中。随机仿射矩阵也由 DALI 原生算子
    生成，避免逐样本 Python/OpenCV 回调和小矩阵 CPU→GPU 拷贝。

输出契约:
    ``src``: ``float32``，形状 ``(B, 3, H, W)``，RGB，值域 ``[-1, 1]``。
    ``dst``: ``float32``，形状 ``(B, 3, H, W)``，RGB，值域 ``[-1, 1]``。
    ``theta_restore``: ``float32``，形状 ``(B, 2, 3)``，可直接传给
    ``torch.nn.functional.affine_grid(..., align_corners=False)``。
"""

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy import ndarray
from nvidia.dali import fn, pipeline_def
from nvidia.dali.math import clamp
from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

from .dataloader_common import FloatRange, ImageDecoderBackend, ImageSource, LocalImagePool, build_image_pools, print_image_pools, validate_range


class _RandomImagePairSource:
    """为 DALI parallel external_source 提供本地 source/target 编码图像。"""

    def __init__(self, src: Sequence[ImageSource], dst: Sequence[ImageSource]) -> None:
        self.src_pools, self.src_cdf, src_info = build_image_pools(src)
        self.dst_pools, self.dst_cdf, dst_info = build_image_pools(dst)
        self.rng = np.random.default_rng()
        print_image_pools("SRC Sources", src_info)
        print_image_pools("DST Sources", dst_info)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("rng", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.rng = np.random.default_rng()

    def _sample_encoded(self, pools: tuple[LocalImagePool, ...], cdf: ndarray) -> ndarray:
        pool_index = min(int(np.searchsorted(cdf, self.rng.random(), side="right")), len(pools) - 1)
        pool = pools[pool_index]
        file_name = pool.file_names[int(self.rng.integers(len(pool.file_names)))]
        return np.fromfile(os.path.join(pool.root, file_name), dtype=np.uint8)

    def __call__(self, _sample_info) -> tuple[ndarray, ndarray]:
        return self._sample_encoded(self.src_pools, self.src_cdf), self._sample_encoded(self.dst_pools, self.dst_cdf)


def _random_affine_matrices(img_resolution: int, rotation_range: FloatRange, scale_factor_range: FloatRange, tx_range: FloatRange, ty_range: FloatRange):
    rotation_range = validate_range("rotation_range", rotation_range)
    scale_factor_range = validate_range("scale_factor_range", scale_factor_range)
    tx_range = validate_range("tx_range", tx_range)
    ty_range = validate_range("ty_range", ty_range)

    if scale_factor_range[0] <= 0.0:
        raise ValueError(f"scale_factor_range 必须为正数范围，实际为 {scale_factor_range}")

    center = img_resolution * 0.5
    angle = fn.random.uniform(range=rotation_range)
    scale = fn.random.uniform(range=scale_factor_range)
    tx = fn.random.uniform(range=tx_range) * img_resolution
    ty = fn.random.uniform(range=ty_range) * img_resolution

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


def _normalize_chw(image):
    image = clamp(image, lo=0.0, hi=255.0)
    return fn.crop_mirror_normalize(
        image,
        device="gpu",
        dtype=DALIDataType.FLOAT,
        output_layout="CHW",
        mean=(127.5, 127.5, 127.5),
        std=(127.5, 127.5, 127.5),
    )


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
    rotation_range: FloatRange = (-10.0, 10.0),
    scale_factor_range: FloatRange = (1.0 / 1.3, 1.25),
    tx_range: FloatRange = (-0.15, 0.15),
    ty_range: FloatRange = (-0.15, 0.15),
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
        rotation_range: ``dst`` 随机旋转角范围，单位为度。
        scale_factor_range: ``dst`` **实际缩放倍数**范围，例如 ``(0.8, 1.2)``。
            不再使用旧式“缩放偏移”语义。
        tx_range: ``dst`` 水平平移范围，相对于图像宽度，例如 ``0.1`` 表示 10%。
        ty_range: ``dst`` 垂直平移范围，相对于图像高度。

    返回:
        三个 DALI DataNode：``src``、``dst`` 和 ``theta_restore``。图像输出均为
        RGB/NCHW/float32/``[-1, 1]``；``theta_restore`` 为 ``(2, 3)`` 每样本矩阵，
        batch 经 DALIGenericIterator 后形状为 ``(B, 2, 3)``。

    说明:
        ``@pipeline_def`` 还接受 ``batch_size``、``device_id``、``num_threads``、
        ``prefetch_queue_depth``、``py_num_workers`` 等 DALI Pipeline 构造参数；这些参数
        不属于本函数签名，但可以在调用 ``create_dataloader_pipeline(...)`` 时直接传入。
    """
    if img_resolution <= 0:
        raise ValueError(f"img_resolution 必须为正数，实际为 {img_resolution}")
    if reader_prefetch_queue_depth <= 0:
        raise ValueError(f"reader_prefetch_queue_depth 必须为正数，实际为 {reader_prefetch_queue_depth}")
    for name, value in (("brightness", brightness), ("contrast", contrast), ("saturation", saturation), ("flip_prob", flip_prob)):
        if not np.isfinite(value):
            raise ValueError(f"{name} 必须为有限数值，实际为 {value}")
    if min(brightness, contrast, saturation) < 0.0:
        raise ValueError("brightness/contrast/saturation 不能为负数")
    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError(f"flip_prob 必须位于 [0, 1]，实际为 {flip_prob}")

    src_raw, dst_raw = fn.external_source(
        source=_RandomImagePairSource(src, dst),
        num_outputs=2,
        device="cpu",
        parallel=True,
        prefetch_queue_depth=reader_prefetch_queue_depth,
        dtype=DALIDataType.UINT8,
        ndim=1,
        batch=False,
    )

    src_image = _decode(src_raw, decoder_backend, decoder_hw_load)
    dst_image = _decode(dst_raw, decoder_backend, decoder_hw_load)

    src_image = fn.resize(src_image, device="gpu", size=img_resolution, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)
    dst_image = fn.resize(dst_image, device="gpu", size=img_resolution, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)

    src_image = fn.flip(src_image, device="gpu", horizontal=fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.INT32))
    dst_image = fn.flip(dst_image, device="gpu", horizontal=fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.INT32))

    dst_image = fn.color_twist(
        dst_image,
        device="gpu",
        brightness=fn.random.uniform(range=(1.0 - brightness, 1.0 + brightness)),
        contrast=fn.random.uniform(range=(1.0 - contrast, 1.0 + contrast)),
        saturation=fn.random.uniform(range=(1.0 - saturation, 1.0 + saturation)),
    )

    affine_matrix, theta_restore = _random_affine_matrices(img_resolution, rotation_range, scale_factor_range, tx_range, ty_range)
    dst_image = fn.warp_affine(dst_image, affine_matrix, device="gpu", inverse_map=False, fill_value=-1.0)

    return _normalize_chw(src_image), _normalize_chw(dst_image), theta_restore
