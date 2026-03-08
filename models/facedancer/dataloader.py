import random
from typing import List, Tuple, Union
import numpy as np
from numpy import ndarray
from scipy.stats import norm
import cv2
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.types import Constant, DALIDataType, DALIImageType, DALIInterpType
from nvidia.dali.math import clamp
from misc.utils import ImageFolder


class RndWarpPars(object):
    def __init__(self, size: int, rotation_range=[-1.0, 1.0], scale_range=[-0.0, 0.0], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05], trunc_val=2.5):

        self.size = size
        self.rotation_range = rotation_range
        self.tx_range = tx_range
        self.ty_range = ty_range
        self.scale_range = scale_range
        self.trunc_val = trunc_val
        self.rng = np.random.default_rng(None)

    def random_normal(self, shape):

        size = np.prod(shape)
        out = np.empty(size, dtype=np.float32)

        accept_prob = norm.cdf(self.trunc_val) - norm.cdf(-self.trunc_val)

        filled = 0
        while filled < size:

            remaining = size - filled
            batch_size = int(remaining / accept_prob * 1.2) + 100

            samples = self.rng.normal(size=batch_size)
            valid_mask = np.abs(samples) <= self.trunc_val
            valid_samples = samples[valid_mask] / self.trunc_val

            n_valid = min(len(valid_samples), remaining)
            out[filled : filled + n_valid] = valid_samples[:n_valid]
            filled += n_valid

        return out.reshape(shape).astype(np.float32)

    def __call__(self, sample_info):
        _ = sample_info

        w = self.size

        cell_size = [w // 2, w // 4, w // 8][self.rng.integers(0, 3)]
        cell_count = w // cell_size + 1

        grid = np.linspace(0, w, cell_count)
        mapx = np.broadcast_to(grid, (cell_count, cell_count)).copy()
        mapy = mapx.T.copy()

        noise = self.random_normal((cell_count - 2, cell_count - 2))
        mapx[1:-1, 1:-1] += noise * (cell_size * 0.24)

        noise = self.random_normal((cell_count - 2, cell_count - 2))
        mapy[1:-1, 1:-1] += noise * (cell_size * 0.24)

        half = cell_size // 2

        dsize = (w + cell_size,) * 2
        mapx = cv2.resize(mapx, dsize)
        mapy = cv2.resize(mapy, dsize)

        mapx = mapx[half:-half, half:-half]
        mapy = mapy[half:-half, half:-half]

        rotation = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
        scale = np.random.uniform(1 / (1 - self.scale_range[0]), 1 + self.scale_range[1])
        tx = np.random.uniform(self.tx_range[0], self.tx_range[1])
        ty = np.random.uniform(self.ty_range[0], self.ty_range[1])

        transform_matrix = cv2.getRotationMatrix2D((w // 2, w // 2), rotation, scale)
        transform_matrix[:, 2] += (tx * w, ty * w)

        return mapx.astype(np.float32), mapy.astype(np.float32), transform_matrix.astype(np.float32)


class SampleReader(object):
    """
    DALI external_source 数据提供器。

    负责从多个源域(src)与目标域(dst)文件夹中按权重随机采样图像文件，
    并返回原始字节流（未解码）。

    设计目标：
        - 支持多文件夹加权采样（用于域平衡 / 数据集规模不均衡问题）
        - 支持随机采样或顺序采样（由 ImageFolder 控制）
        - 通过 sqrt(count) 权重缩放，降低超大数据集的主导效应
        - 提供额外指数权重调整 (2**adj)，用于人为增强某些域

    Args:
        src:
            List[str] 或 List[(folder, adj_weight)]
            源域图像文件夹列表。
        dst:
            List[str] 或 List[(folder, adj_weight)]
            目标域图像文件夹列表。
        random_sampling:
            是否在文件夹内部随机采样文件。
            False 时使用 ImageFolder 内部顺序采样策略。

    Returns:
        __call__ 返回:
            src: ndarray(uint8)  源图像原始字节流
            dst: ndarray(uint8)  目标图像原始字节流

    注意:
        - 返回的是 bytes buffer，不是解码后的图像。
        - 解码由 DALI GPU mixed decoder 完成。
        - 多线程安全依赖 ImageFolder.sample() 实现。
    """

    def __init__(
        self,
        src: Union[List[str], List[Tuple[str, float]]],
        dst: Union[List[str], List[Tuple[str, float]]],
        random_sampling: bool = True,
    ):
        self.random_sampling = random_sampling
        self.src_folders, self.src_weight = self.collect_folder(src)
        self.dst_folders, self.dst_weight = self.collect_folder(dst)
        self.print_folders_info()

    def __call__(self, sample_info) -> tuple[ndarray, ndarray]:
        _ = sample_info

        dst_folder = random.choices(self.dst_folders, weights=self.dst_weight, k=1)[0]
        dst_file_path = dst_folder.sample()
        dst = np.fromfile(dst_file_path, dtype=np.uint8)

        src_folder = random.choices(self.src_folders, weights=self.src_weight, k=1)[0]
        src_file_path = src_folder.sample()
        src = np.fromfile(src_file_path, dtype=np.uint8)

        return src, dst

    def collect_folder(self, folder_weight_list: Union[List[str], List[Tuple[str, float]]]):
        """
        构建 ImageFolder 列表并计算采样权重。

        权重策略:
            1. 基础权重 = sqrt(file_count) / sum(sqrt(file_count))
               → 防止大数据集垄断采样概率。
            2. 调整权重 = base_weight * 2**adj
               → adj 为人工调节因子（指数缩放）。
            3. 归一化得到最终采样权重。

        Args:
            folder_weight_list:
                List[str]  → 无额外权重调整
                List[(folder, adj)] → adj 为 log2 scale 调整量

        Returns:
            readers: List[ImageFolder]
            weights: List[float] 归一化采样概率

        Raises:
            ValueError: 当所有文件夹为空时抛出。

        设计动机:
            - sqrt scaling 常用于 dataset balancing（类似 CLIP / diffusion dataset sampling）
            - adj 提供人为 domain emphasis capability
        """
        readers: List[ImageFolder] = []
        file_counts: List[int] = []
        adjustments: List[float] = []

        if isinstance(folder_weight_list[0], str):
            for folder in folder_weight_list:
                reader = ImageFolder(folder=folder, random_sampling=self.random_sampling)
                readers.append(reader)
                file_counts.append(len(reader))
                adjustments.append(0.0)
        else:
            for folder, adj in folder_weight_list:
                reader = ImageFolder(folder=folder, random_sampling=self.random_sampling)
                readers.append(reader)
                file_counts.append(len(reader))
                adjustments.append(adj)

        total_files = sum(file_counts)
        if total_files == 0:
            raise ValueError("所有文件夹为空，无法分配权重")

        sqrt_counts = [count**0.5 for count in file_counts]
        sqrt_sum = sum(sqrt_counts)
        base_weights = [sc / sqrt_sum for sc in sqrt_counts]

        adjusted_weights = [w * (2**adj) for w, adj in zip(base_weights, adjustments)]

        weight_sum = sum(adjusted_weights)
        if weight_sum == 0:
            weights = [1.0 / len(adjusted_weights)] * len(adjusted_weights)
        else:
            weights = [w / weight_sum for w in adjusted_weights]

        return readers, weights

    def print_folders_info(self):
        """
        打印源域与目标域文件夹信息，包括：
            - 路径
            - 文件数量
            - 最终采样权重

        仅用于调试和实验日志记录。
        使用 ANSI 绿色高亮文件夹路径。
        """

        def dump(title, folders: List[ImageFolder] | None, weights: list[float]):

            count_w = 7
            GREEN = "\033[32m"
            RESET = "\033[0m"

            lines = [f"{title}:"]
            if folders is None:
                lines.append("None")
            else:
                for f, w in zip(folders, weights):
                    lines.append(f"    {GREEN}{f.folder}{RESET}")
                    lines.append(f"        count: {f.len:<{count_w}d} weight: {w:<6.3f}")
            print("\n".join(lines))

        dump("SRC Folders", self.src_folders, self.src_weight)
        dump("DST Folders", self.dst_folders, self.dst_weight)


@pipeline_def(enable_conditionals=True)
def datasetloader(
    resize: int,
    src: List[Tuple[str, float]],
    dst: List[Tuple[str, float]],
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    flip_prob: float = 0.5,
    same_image_prob: float = 0.2,
    random_sampling: bool = True,
    rndwarp: bool = False,
):
    """
    Face Swapping 数据加载 DALI Pipeline。

    Pipeline stages:
        1. external_source → CPU 读取原始字节
        2. mixed decoder → GPU 硬件 JPEG decode
        3. resize → GPU Lanczos3
        4. random flip → source & target independently
        5. color augmentation → target only
        6. clamp + normalize → [-1, 1]
        7. transpose → CHW
        8. same-image sampling (identity training trick)

    Args:
        resize:
            输出分辨率 (H=W=resize)
        src / dst:
            [(folder, adj_weight)] 列表
        brightness / contrast / saturation:
            ColorJitter 幅度，均匀采样 [1-x, 1+x]
        flip_prob:
            水平翻转概率
        same_image_prob:
            以 dst 覆盖 src 的概率，用于 identity reconstruction loss
        random_sampling:
            是否随机采样文件

    Returns:
        src: Tensor [3, H, W] float32 in [-1, 1]
        dst: Tensor [3, H, W] float32 in [-1, 1]
        is_same: Tensor[1] float32
            1.0 → src == dst (identity case)
            0.0 → normal swap case

    关键设计点:
        - mixed decoder 减少 CPU bottleneck
        - color jitter 仅作用于 dst（模拟真实视频 domain shift）
        - same_image_prob 用于稳定 GAN identity preservation
        - enable_conditionals=True 允许 pipeline-level if 分支

    性能注意:
        - hw_decoder_load=0.75 避免 GPU decode 饱和
        - no_copy=True 避免 CPU→GPU redundant memcpy
        - prefetch_queue_depth=2 提高 pipeline overlap
    """

    external_source = fn.external_source(
        source=SampleReader(src, dst, random_sampling),
        num_outputs=2,
        device="cpu",
        no_copy=True,
        parallel=True,
        prefetch_queue_depth=2,
        dtype=DALIDataType.UINT8,
        batch=False,
    )
    if rndwarp:
        random_warp_params = fn.external_source(source=RndWarpPars(resize), num_outputs=3, device="gpu", no_copy=False, parallel=False, dtype=DALIDataType.FLOAT, batch=False)

    src_raw, dst_raw = external_source

    src = fn.decoders.image(src_raw, device="mixed", output_type=DALIImageType.RGB, hw_decoder_load=0.75)
    src = fn.resize(src, device="gpu", size=resize, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)
    if fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.BOOL):
        src = fn.flip(src, device="gpu")

    dst = fn.decoders.image(dst_raw, device="mixed", output_type=DALIImageType.RGB, hw_decoder_load=0.75)
    dst = fn.resize(dst, device="gpu", size=resize, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)
    if fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.BOOL):
        dst = fn.flip(dst, device="gpu")

    dst = fn.color_twist(
        dst,
        device="gpu",
        brightness=fn.random.uniform(range=[1.0 - brightness, 1.0 + brightness]),
        contrast=fn.random.uniform(range=[1.0 - contrast, 1.0 + contrast]),
        saturation=fn.random.uniform(range=[1.0 - saturation, 1.0 + saturation]),
    )

    if fn.random.coin_flip(probability=same_image_prob, dtype=DALIDataType.BOOL):
        is_same = Constant(value=1.0, device="gpu", dtype=DALIDataType.FLOAT, shape=[1])
        src = fn.copy(dst, device="gpu")
        if rndwarp:
            mapx, mapy, transform_matrix = random_warp_params[0:]
            dst = fn.experimental.remap(dst, mapx, mapy)
            dst = fn.reinterpret(dst, layout="HWC")
            dst = fn.warp_affine(dst, transform_matrix)
    else:
        is_same = Constant(value=0.0, device="gpu", dtype=DALIDataType.FLOAT, shape=[1])

    src = clamp(src, lo=0.0, hi=255.0)
    src = fn.normalize(src, device="gpu", mean=127.5, stddev=127.5)
    src = fn.transpose(src, device="gpu", perm=[2, 0, 1])

    dst = clamp(dst, lo=0.0, hi=255.0)
    dst = fn.normalize(dst, device="gpu", mean=127.5, stddev=127.5)
    dst = fn.transpose(dst, device="gpu", perm=[2, 0, 1])

    return src, dst, is_same
