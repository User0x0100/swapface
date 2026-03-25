from pathlib import Path
import random
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


class IdentityPairReader:
    """
    从多层目录结构中读取“同一身份的两张不同图片”。

    目录结构要求：

        root/
            group1/
                id_0000/
                    img1.jpg
                    img2.jpg
                id_0001/
                    ...
            group2/
                id_xxxx/
                    ...

    设计说明：

    1. identity 定义
        每个 id_xxxx 文件夹视为一个 identity

    2. 最小样本数
        仅保留图片数量 >= 2 的 identity

    3. 采样策略
        - 先按 identity 权重采样一个文件夹
        - 再从该 identity 中采样两张不同图片
    """

    def __init__(self, roots: list[str], random_sampling: bool = True):
        self.random_sampling = random_sampling
        self.folders: list[ImageFolder] = []

        for root in roots:
            root = Path(root)
            for group in root.iterdir():
                if not group.is_dir():
                    continue
                for identity in group.iterdir():
                    if not identity.is_dir():
                        continue
                    reader = ImageFolder(folder=identity, random_sampling=self.random_sampling)
                    if len(reader) >= 2:  # 至少两张
                        self.folders.append(reader)

        if len(self.folders) == 0:
            raise ValueError("没有有效 identity 数据")

        counts = [len(x) for x in self.folders]
        weights = [c**0.5 for c in counts]
        s = sum(weights)
        self.weights = [w / s for w in weights]

    def __call__(self, sample_info) -> tuple[ndarray, ndarray]:
        _ = sample_info

        folder = random.choices(self.folders, weights=self.weights, k=1)[0]

        x = np.fromfile(folder.sample(), dtype=np.uint8)
        x1 = np.fromfile(folder.sample(), dtype=np.uint8)

        return x, x1


class SampleReader(object):
    def __init__(
        self,
        src: list[str] | list[tuple[str, float]],
        dst: list[str] | list[tuple[str, float]],
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

    def collect_folder(self, folder_weight_list: list[str] | list[tuple[str, float]]) -> tuple[list[ImageFolder], list[float]]:

        readers: list[ImageFolder] = []
        file_counts: list[int] = []
        adjustments: list[float] = []

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

        def dump(title, folders: list[ImageFolder] | None, weights: list[float]):

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
    src: list[tuple[str, float]],
    dst: list[tuple[str, float]],
    identity_root: list[str],
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    flip_prob: float = 0.5,
    same_image_prob: float = 0.2,
    random_sampling: bool = True,
    rndwarp: bool = False,
):

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

    identity_sampler = fn.external_source(
        source=IdentityPairReader(identity_root, random_sampling),
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

    # 是否采样同一身份
    if fn.random.coin_flip(probability=same_image_prob, dtype=DALIDataType.BOOL):
        is_same = Constant(value=1.0, device="gpu", dtype=DALIDataType.FLOAT, shape=[1])
        src_raw, dst_raw = identity_sampler

    else:
        is_same = Constant(value=0.0, device="gpu", dtype=DALIDataType.FLOAT, shape=[1])
        src_raw, dst_raw = external_source

    src = fn.decoders.image(src_raw, device="mixed", output_type=DALIImageType.RGB, hw_decoder_load=0.75)
    src = fn.resize(src, device="gpu", size=resize, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)
    if fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.BOOL):
        src = fn.flip(src, device="gpu")

    dst = fn.decoders.image(dst_raw, device="mixed", output_type=DALIImageType.RGB, hw_decoder_load=0.75)
    dst = fn.resize(dst, device="gpu", size=resize, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)
    if fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.BOOL):
        dst = fn.flip(dst, device="gpu")

    if rndwarp:
        mapx, mapy, transform_matrix = random_warp_params[0:]
        dst = fn.experimental.remap(dst, mapx, mapy)
        dst = fn.reinterpret(dst, layout="HWC")
        dst = fn.warp_affine(dst, transform_matrix)

    dst = fn.color_twist(
        dst,
        device="gpu",
        brightness=fn.random.uniform(range=[1.0 - brightness, 1.0 + brightness]),
        contrast=fn.random.uniform(range=[1.0 - contrast, 1.0 + contrast]),
        saturation=fn.random.uniform(range=[1.0 - saturation, 1.0 + saturation]),
    )

    src = clamp(src, lo=0.0, hi=255.0)
    src = fn.normalize(src, device="gpu", mean=127.5, stddev=127.5)
    src = fn.transpose(src, device="gpu", perm=[2, 0, 1])

    dst = clamp(dst, lo=0.0, hi=255.0)
    dst = fn.normalize(dst, device="gpu", mean=127.5, stddev=127.5)
    dst = fn.transpose(dst, device="gpu", perm=[2, 0, 1])

    return src, dst, is_same
