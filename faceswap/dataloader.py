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
    采样图片。也支持 Hugging Face 与 ModelScope 远程图片池，图片按需下载并复用各自 Hub cache。
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

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from huggingface_hub import HfApi, hf_hub_download, set_client_factory
from huggingface_hub import constants as hf_constants
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import disable_progress_bars
from modelscope_hub import HubApi as ModelScopeHubApi
from modelscope_hub.errors import HubError as ModelScopeHubError
from numpy import ndarray
from nvidia.dali import fn, pipeline_def
from nvidia.dali.math import clamp
from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

type ImagePath = str | os.PathLike[str]
type FloatRange = tuple[float, float]


def _configure_huggingface_hub(proxy: str | None = None) -> None:
    """关闭训练数据下载的 Xet/CAS 路径并配置 HTTP client。

    FFHQ 镜像的部分 Xet reconstruction 在并发 worker 下可能返回 CAS 404；训练只需要
    普通 Hub 文件下载与本地 cache，因此强制使用标准 HTTP 路径。
    """
    hf_constants.HF_HUB_DISABLE_XET = True
    if proxy is not None:
        proxy = proxy.strip()
        if not proxy:
            raise ValueError("huggingface_proxy 不能为空字符串；不使用代理时请删除该配置")
    set_client_factory(lambda: httpx.Client(proxy=proxy, follow_redirects=True, timeout=None))


def _disable_modelscope_progress_bars() -> None:
    """仅在当前 DALI worker 中关闭 ModelScope Hub 的下载进度条。

    modelscope-hub 0.4.0 在 ``_download`` 中直接绑定 ``tqdm``；如果在模块导入后才设置
    ``TQDM_DISABLE``，tqdm 已经读取过环境变量，因此不会生效。worker 是独立进程，
    在这里覆盖其下载模块使用的 tqdm，不会影响训练主进程的进度条。
    """
    os.environ["TQDM_DISABLE"] = "1"
    import modelscope_hub._download as modelscope_download
    from tqdm.auto import tqdm as tqdm_auto

    def silent_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return tqdm_auto(*args, **kwargs)

    modelscope_download.tqdm = silent_tqdm


_configure_huggingface_hub()


@dataclass(frozen=True, slots=True)
class HuggingFaceImageSource:
    """Hugging Face dataset 仓库中的远程图片池。

    仓库只在初始化时读取一次图片文件清单；训练时随机选择图片并通过
    ``hf_hub_download`` 按需下载。已访问文件复用 Hugging Face 本地 cache，
    不需要预先下载整个数据集。
    """

    repo_id: str
    revision: str = "main"
    path_prefix: str = ""
    adjustment: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelScopeImageSource:
    """ModelScope dataset 仓库中的远程图片池。

    仓库在训练进程初始化时枚举一次图片文件；训练时通过 ``download_file``
    按需下载单张图片并复用 ModelScope cache。DALI worker 继承已经构建好的
    文件清单，不会在每个 worker 中重复枚举远程仓库。
    """

    repo_id: str
    revision: str = "master"
    path_prefix: str = ""
    adjustment: float = 0.0


type ImageSource = ImagePath | tuple[ImagePath, float] | HuggingFaceImageSource | ModelScopeImageSource


@dataclass(frozen=True, slots=True)
class _LocalImagePool:
    root: str
    file_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _HuggingFaceImagePool:
    repo_id: str
    revision: str
    file_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ModelScopeImagePool:
    repo_id: str
    revision: str
    file_names: tuple[str, ...]
    cache_dir: str | None


type ImagePool = _LocalImagePool | _HuggingFaceImagePool | _ModelScopeImagePool

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
REMOTE_DOWNLOAD_ATTEMPTS = 5
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
    "huggingface_proxy": None,
    "modelscope_cache_dir": None,
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


def _scan_image_files(directory: ImagePath) -> _LocalImagePool:
    folder = Path(directory).resolve(strict=True)
    with os.scandir(folder) as entries:
        file_names = tuple(entry.name for entry in entries if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS))
    if not file_names:
        raise FileNotFoundError(f"目录中没有支持的图片文件: {folder}")
    return _LocalImagePool(str(folder), file_names)


def _hf_manifest_cache_path(repo_id: str, revision: str, path_prefix: str) -> Path:
    key = hashlib.sha256(f"{repo_id}\0{revision}\0{path_prefix}".encode()).hexdigest()[:24]
    return Path(HF_HUB_CACHE) / "swap-image-manifests" / f"{key}.json"


@cache
def _scan_huggingface_image_files(repo_id: str, revision: str, path_prefix: str) -> tuple[str, tuple[str, ...]]:
    repo_id = repo_id.strip()
    revision = revision.strip()
    path_prefix = path_prefix.strip("/")
    if not repo_id:
        raise ValueError("Hugging Face repo_id 不能为空")
    if not revision:
        raise ValueError("Hugging Face revision 不能为空")

    api = HfApi()
    repo_info = api.dataset_info(repo_id=repo_id, revision=revision)
    resolved_revision = repo_info.sha
    if not resolved_revision:
        raise RuntimeError(f"无法解析 Hugging Face dataset revision：{repo_id}@{revision}")

    cache_path = _hf_manifest_cache_path(repo_id, resolved_revision, path_prefix)
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            cached_files = json.load(file)
        if isinstance(cached_files, list) and cached_files and all(isinstance(name, str) for name in cached_files):
            return resolved_revision, tuple(cached_files)
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass

    if path_prefix:
        entries = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", revision=resolved_revision, path_in_repo=path_prefix, recursive=True)
        file_names = tuple(path for entry in entries if isinstance(path := getattr(entry, "path", None), str) and path.lower().endswith(IMAGE_EXTENSIONS))
    else:
        file_names = tuple(path for path in api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=resolved_revision) if path.lower().endswith(IMAGE_EXTENSIONS))

    if not file_names:
        location = f"/{path_prefix}" if path_prefix else ""
        raise FileNotFoundError(f"Hugging Face dataset 中没有支持的图片文件：{repo_id}@{revision}{location}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(file_names, file, ensure_ascii=False)
        temp_path.replace(cache_path)
    except OSError:
        temp_path.unlink(missing_ok=True)

    return resolved_revision, file_names


@cache
def _scan_modelscope_image_files(repo_id: str, revision: str, path_prefix: str) -> tuple[str, ...]:
    """枚举 ModelScope dataset 当前 revision 下的独立图片文件。

    ModelScope 当前没有像 Hugging Face 那样可廉价解析并缓存的 immutable commit SHA，
    因此这里只做进程内缓存：同一训练进程中的 src/dst 复用清单，而下一次训练启动
    会重新枚举，从而能看到 ``master`` 上新上传的图片。
    """
    repo_id = repo_id.strip()
    revision = revision.strip()
    path_prefix = path_prefix.strip("/")
    if not repo_id:
        raise ValueError("ModelScope repo_id 不能为空")
    if not revision:
        raise ValueError("ModelScope revision 不能为空")

    prefix = f"{path_prefix}/" if path_prefix else ""
    files = ModelScopeHubApi().list_repo_files(repo_id, "dataset", revision=revision, recursive=True)
    file_names = tuple(path for entry in files if isinstance(path := getattr(entry, "path", None), str) and path.startswith(prefix) and path.lower().endswith(IMAGE_EXTENSIONS))
    if not file_names:
        location = f"/{path_prefix}" if path_prefix else ""
        raise FileNotFoundError(f"ModelScope dataset 中没有支持的图片文件：{repo_id}@{revision}{location}")
    return file_names


class _RandomImagePairSource:
    """为 DALI parallel external_source 提供独立的 source/target 编码图像。

    本地目录初始化时只扫描一次；远程源初始化时只获取一次图片清单。worker 热路径
    只负责随机选图并读取压缩字节；Hugging Face/ModelScope 图片均通过各自 Hub cache
    按需下载。``rng`` 与 ModelScope HTTP client 不参与 pickle；每个 spawn worker 在
    反序列化时创建独立实例。
    """

    def __init__(
        self,
        src: Sequence[ImageSource],
        dst: Sequence[ImageSource],
        huggingface_proxy: str | None = None,
        modelscope_cache_dir: str | None = None,
    ) -> None:
        self.huggingface_proxy = huggingface_proxy
        self.modelscope_cache_dir = modelscope_cache_dir
        _configure_huggingface_hub(huggingface_proxy)
        self.src_pools, self.src_cdf, src_info = self._build_pool(src, modelscope_cache_dir)
        self.dst_pools, self.dst_cdf, dst_info = self._build_pool(dst, modelscope_cache_dir)
        self.rng = np.random.default_rng()
        self.modelscope_api = ModelScopeHubApi()
        self._print_pool("SRC Sources", src_info)
        self._print_pool("DST Sources", dst_info)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("rng", None)
        state.pop("modelscope_api", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        _configure_huggingface_hub(self.huggingface_proxy)
        # Hugging Face / ModelScope 单文件 cache miss 都可能输出进度条；DALI worker
        # 是独立进程，因此只在 worker 内关闭，避免污染主进程中的模型权重下载进度。
        disable_progress_bars()
        _disable_modelscope_progress_bars()
        self.rng = np.random.default_rng()
        self.modelscope_api = ModelScopeHubApi()

    @staticmethod
    def _build_pool(sources: Sequence[ImageSource], modelscope_cache_dir: str | None) -> tuple[tuple[ImagePool, ...], ndarray, tuple[tuple[str, int, float], ...]]:
        if isinstance(sources, (str, os.PathLike)):
            raise TypeError(f"图片源必须是路径/远程源序列；单个目录请写成 [{os.fspath(sources)!r}]")
        if not sources:
            raise ValueError("图片源列表不能为空")

        pools: list[ImagePool] = []
        labels: list[str] = []
        counts: list[int] = []
        adjustments: list[float] = []

        for source in sources:
            if isinstance(source, HuggingFaceImageSource):
                adjustment = float(source.adjustment)
                resolved_revision, file_names = _scan_huggingface_image_files(source.repo_id, source.revision, source.path_prefix)
                pool = _HuggingFaceImagePool(source.repo_id, resolved_revision, file_names)
                prefix = source.path_prefix.strip("/")
                location = f"/{prefix}" if prefix else ""
                label = f"hf://{source.repo_id}@{resolved_revision[:12]}{location}"
            elif isinstance(source, ModelScopeImageSource):
                adjustment = float(source.adjustment)
                file_names = _scan_modelscope_image_files(source.repo_id, source.revision, source.path_prefix)
                pool = _ModelScopeImagePool(source.repo_id, source.revision, file_names, modelscope_cache_dir)
                prefix = source.path_prefix.strip("/")
                location = f"/{prefix}" if prefix else ""
                label = f"modelscope://{source.repo_id}@{source.revision}{location}"
            else:
                if isinstance(source, (str, os.PathLike)):
                    path, adjustment = source, 0.0
                elif isinstance(source, tuple) and len(source) == 2 and isinstance(source[0], (str, os.PathLike)):
                    path, adjustment = source
                else:
                    raise TypeError(f"图片源必须为路径、(路径, 权重调整)、HuggingFaceImageSource 或 ModelScopeImageSource，实际为 {source!r}")
                adjustment = float(adjustment)
                pool = _scan_image_files(path)
                label = pool.root
                file_names = pool.file_names

            if not np.isfinite(adjustment):
                raise ValueError(f"权重调整必须为有限数值，实际为 {adjustment!r}")

            pools.append(pool)
            labels.append(label)
            counts.append(len(file_names))
            adjustments.append(adjustment)

        # 保留旧规则：source_weight ∝ sqrt(file_count) * 2**adjustment。
        # 在 log 域归一化，避免极端 adjustment 导致上溢/下溢。
        log_weights = 0.5 * np.log(np.asarray(counts, dtype=np.float64)) + np.asarray(adjustments, dtype=np.float64) * np.log(2.0)
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0

        info = tuple((label, count, float(weight)) for label, count, weight in zip(labels, counts, weights))
        return tuple(pools), cdf, info

    @staticmethod
    def _print_pool(title: str, info: tuple[tuple[str, int, float], ...]) -> None:
        print(title + ":")
        for label, count, weight in info:
            print(f"    {label}\n        count: {count:<7d} weight: {weight:<6.3f}")

    def _sample_encoded(self, pools: tuple[ImagePool, ...], cdf: ndarray) -> ndarray:
        pool_index = min(int(np.searchsorted(cdf, self.rng.random(), side="right")), len(pools) - 1)
        pool = pools[pool_index]

        if isinstance(pool, _LocalImagePool):
            file_name = pool.file_names[int(self.rng.integers(len(pool.file_names)))]
            return np.fromfile(os.path.join(pool.root, file_name), dtype=np.uint8)

        last_error: Exception | None = None
        for _ in range(REMOTE_DOWNLOAD_ATTEMPTS):
            file_name = pool.file_names[int(self.rng.integers(len(pool.file_names)))]
            try:
                if isinstance(pool, _HuggingFaceImagePool):
                    image_path = hf_hub_download(repo_id=pool.repo_id, filename=file_name, repo_type="dataset", revision=pool.revision)
                else:
                    image_path = self.modelscope_api.download_file(
                        pool.repo_id,
                        "dataset",
                        file_name,
                        revision=pool.revision,
                        cache_dir=pool.cache_dir,
                    )
                return np.fromfile(image_path, dtype=np.uint8)
            except (HfHubHTTPError, ModelScopeHubError, OSError, RuntimeError) as exc:
                last_error = exc

        backend = "Hugging Face" if isinstance(pool, _HuggingFaceImagePool) else "ModelScope"
        raise RuntimeError(f"{backend} 数据源连续 {REMOTE_DOWNLOAD_ATTEMPTS} 次下载失败：{pool.repo_id}@{pool.revision}") from last_error

    def __call__(self, _sample_info) -> tuple[ndarray, ndarray]:
        return self._sample_encoded(self.src_pools, self.src_cdf), self._sample_encoded(self.dst_pools, self.dst_cdf)


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
    huggingface_proxy: str | None = None,
    modelscope_cache_dir: str | None = None,
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
        src: 身份来源图像池。必须是 ``ImageSource`` 序列；可包含本地路径、
            ``(路径, adjustment)``、``HuggingFaceImageSource`` 或
            ``ModelScopeImageSource``。单个本地目录也应写成 ``[path]``。
        dst: 目标图像池，格式与 ``src`` 相同；与 ``src`` 独立采样。
        reader_prefetch_queue_depth: parallel ``external_source`` 每个 Python worker
            可提前准备的 batch 数。只影响编码文件读取阶段。
        huggingface_proxy: Hugging Face 在线数据源使用的 HTTP(S) 代理，例如
            ``http://127.0.0.1:7890``。不配置时直接连接。
        modelscope_cache_dir: ModelScope 单文件下载 cache 目录；留空使用 SDK 默认 cache。
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
        source=_RandomImagePairSource(src, dst, huggingface_proxy, modelscope_cache_dir),
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
