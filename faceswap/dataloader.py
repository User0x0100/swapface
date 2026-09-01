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
    采样图片。``adjustment=1`` 表示将该文件夹的基础权重翻倍，``-1`` 表示减半。

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
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray
from nvidia.dali import fn, pipeline_def
from nvidia.dali.math import clamp
from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

type ImagePath = str | os.PathLike[str]
type ImageSource = ImagePath | tuple[ImagePath, float]
type ImageFolder = tuple[str, tuple[str, ...]]
type FloatRange = tuple[float, float]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
DATALOADER_RESERVED_KEYS = frozenset({"batch_size", "device_id", "img_resolution", "src", "dst"})


class ImageDecoderBackend(Enum):
    """DALI 图像解码后端。

    ``MIXED`` 是默认高性能路径，解码结果直接驻留 GPU；即使 GPU 没有专用 JPEG
    硬件解码单元，也可使用 nvJPEG 的 mixed backend。``CPU`` 主要用于兼容性、
    定位 decoder 问题或做 A/B 对照，解码后会显式复制到 GPU。
    """

    MIXED = "mixed"
    CPU = "cpu"


# 默认配置同时包含两类参数：
# - DALI Pipeline 构造参数：num_threads、prefetch_queue_depth、py_num_workers、
#   py_start_method，由 @pipeline_def 生成的工厂直接消费。
# - 图内数据参数：reader_prefetch_queue_depth、decoder、颜色/几何增强参数，
#   传入 create_dataloader_pipeline() 的函数体。
# Trainer 会先将用户覆盖项合并到这份完整默认配置，因此日志中的 dataloader_cfg
# 就是本次实验实际生效的数据管线配置。
#
# prefetch_queue_depth 控制整个 DALI pipeline 的 CPU/GPU 预取深度；
# reader_prefetch_queue_depth 仅控制 parallel external_source 的 Python worker 预取。
# 两者都不是越大越快，尤其高分辨率/大 batch 时 pipeline 预取会直接增加显存占用。
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
    "rotation_range": (-10.0, 10.0),
    "scale_factor_range": (1.0 / 1.3, 1.25),
    "tx_range": (-0.15, 0.15),
    "ty_range": (-0.15, 0.15),
}


def _validate_range(name: str, value: FloatRange) -> FloatRange:
    if len(value) != 2:
        raise ValueError(f"{name} 必须包含两个值，实际为 {value!r}")
    low, high = float(value[0]), float(value[1])
    if not np.isfinite((low, high)).all() or low > high:
        raise ValueError(f"{name} 必须是有限且递增的范围，实际为 {value!r}")
    return low, high


def _scan_image_files(directory: ImagePath) -> ImageFolder:
    folder = Path(directory).resolve(strict=True)
    with os.scandir(folder) as entries:
        file_names = tuple(entry.name for entry in entries if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS))
    if not file_names:
        raise FileNotFoundError(f"目录中没有支持的图片文件: {folder}")
    return str(folder), file_names


class _RandomImagePairSource:
    """为 DALI parallel external_source 提供独立的 source/target 编码图像。

    初始化时只扫描一次目录并预计算文件夹采样 CDF。worker 热路径只执行一次
    CDF 查找、一次文件索引随机采样和一次 ``np.fromfile``，不做目录排序或图像解码。
    ``rng`` 不参与 pickle；每个 spawn worker 在反序列化时创建独立 RNG。
    """

    def __init__(self, src: Sequence[ImageSource], dst: Sequence[ImageSource]) -> None:
        self.src_files, self.src_cdf, src_info = self._build_pool(src)
        self.dst_files, self.dst_cdf, dst_info = self._build_pool(dst)
        self.rng = np.random.default_rng()
        self._print_pool("SRC Folders", src_info)
        self._print_pool("DST Folders", dst_info)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("rng", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.rng = np.random.default_rng()

    @staticmethod
    def _build_pool(sources: Sequence[ImageSource]) -> tuple[tuple[ImageFolder, ...], ndarray, tuple[tuple[str, int, float], ...]]:
        if isinstance(sources, (str, os.PathLike)):
            raise TypeError(f"图片源必须是路径序列；单个目录请写成 [{os.fspath(sources)!r}]")
        if not sources:
            raise ValueError("图片源列表不能为空")

        folders: list[ImageFolder] = []
        paths: list[str] = []
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

            resolved_path, file_names = _scan_image_files(path)
            folders.append((resolved_path, file_names))
            paths.append(resolved_path)
            counts.append(len(file_names))
            adjustments.append(adjustment)

        # 保留旧规则：folder_weight ∝ sqrt(file_count) * 2**adjustment。
        # 在 log 域归一化，避免极端 adjustment 导致上溢/下溢。
        log_weights = 0.5 * np.log(np.asarray(counts, dtype=np.float64)) + np.asarray(adjustments, dtype=np.float64) * np.log(2.0)
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0

        info = tuple((path, count, float(weight)) for path, count, weight in zip(paths, counts, weights))
        return tuple(folders), cdf, info

    @staticmethod
    def _print_pool(title: str, info: tuple[tuple[str, int, float], ...]) -> None:
        print(title + ":")
        for path, count, weight in info:
            print(f"    {path}\n        count: {count:<7d} weight: {weight:<6.3f}")

    def _sample_encoded(self, folders: tuple[ImageFolder, ...], cdf: ndarray) -> ndarray:
        folder_index = min(int(np.searchsorted(cdf, self.rng.random(), side="right")), len(folders) - 1)
        root, file_names = folders[folder_index]
        file_name = file_names[int(self.rng.integers(len(file_names)))]
        return np.fromfile(os.path.join(root, file_name), dtype=np.uint8)

    def __call__(self, _sample_info) -> tuple[ndarray, ndarray]:
        return self._sample_encoded(self.src_files, self.src_cdf), self._sample_encoded(self.dst_files, self.dst_cdf)


def _random_affine_matrices(img_resolution: int, rotation_range: FloatRange, scale_factor_range: FloatRange, tx_range: FloatRange, ty_range: FloatRange):
    rotation_range = _validate_range("rotation_range", rotation_range)
    scale_factor_range = _validate_range("scale_factor_range", scale_factor_range)
    tx_range = _validate_range("tx_range", tx_range)
    ty_range = _validate_range("ty_range", ty_range)

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
        src: 身份来源图像池。必须是序列，元素为路径或 ``(路径, adjustment)``；
            单个目录也应写成 ``[path]``，而不是直接传入字符串。
        dst: 目标图像池，格式与 ``src`` 相同；与 ``src`` 独立采样。
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
