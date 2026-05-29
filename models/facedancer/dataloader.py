import os
from pathlib import Path

import numpy as np
from numpy import ndarray
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.types import Constant, DALIDataType, DALIImageType, DALIInterpType
from nvidia.dali.math import clamp
from misc.utils import ImageFolder


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

    def __init__(self, roots: list[str]):
        self.folders: list[ImageFolder] = []

        for root in roots:
            root = Path(root)
            for group in root.iterdir():
                if not group.is_dir():
                    continue
                for identity in group.iterdir():
                    if not identity.is_dir():
                        continue
                    reader = ImageFolder(identity)
                    if len(reader) >= 2:  # 至少两张
                        self.folders.append(reader)

        if len(self.folders) == 0:
            raise ValueError("没有有效 identity 数据")

        counts = [len(x) for x in self.folders]
        weights = [c**0.5 for c in counts]
        s = sum(weights)
        self.weights = [w / s for w in weights]

        self.rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["rng"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        seed = int.from_bytes(os.urandom(8), "little")
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample_info) -> tuple[ndarray, ndarray]:
        _ = sample_info

        folder_idx = self.rng.choice(len(self.folders), p=self.weights)
        folder = self.folders[folder_idx]

        x = np.fromfile(folder.sample(), dtype=np.uint8)
        x1 = np.fromfile(folder.sample(), dtype=np.uint8)

        return x, x1


class SampleReader(object):
    def __init__(
        self,
        src: list[str] | list[tuple[str, float]],
        dst: list[str] | list[tuple[str, float]],
    ):

        self.src_folders, self.src_weight = self.collect_folder(src)
        self.dst_folders, self.dst_weight = self.collect_folder(dst)
        self.print_folders_info()

        self.rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["rng"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        seed = int.from_bytes(os.urandom(8), "little")
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample_info) -> tuple[ndarray, ndarray]:
        _ = sample_info

        src_idx = self.rng.choice(len(self.src_folders), p=self.src_weight)
        src_folder = self.src_folders[src_idx]
        src_file_path = src_folder.sample()
        src = np.fromfile(src_file_path, dtype=np.uint8)

        dst_idx = self.rng.choice(len(self.dst_folders), p=self.dst_weight)
        dst_folder = self.dst_folders[dst_idx]
        dst_file_path = dst_folder.sample()
        dst = np.fromfile(dst_file_path, dtype=np.uint8)

        return src, dst

    def collect_folder(self, folder_weight_list: list[str] | list[tuple[str, float]]) -> tuple[list[ImageFolder], list[float]]:

        readers: list[ImageFolder] = []
        file_counts: list[int] = []
        adjustments: list[float] = []

        if isinstance(folder_weight_list[0], str):
            for folder in folder_weight_list:
                reader = ImageFolder(folder)
                readers.append(reader)
                file_counts.append(len(reader))
                adjustments.append(0.0)
        else:
            for folder, adj in folder_weight_list:
                reader = ImageFolder(folder)
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
                    lines.append(f"        count: {len(f):<{count_w}d} weight: {w:<6.3f}")
            print("\n".join(lines))

        dump("SRC Folders", self.src_folders, self.src_weight)
        dump("DST Folders", self.dst_folders, self.dst_weight)


@pipeline_def(enable_conditionals=True)
def datasetloader(
    img_resolution: int,
    src: list[tuple[str, float]],
    dst: list[tuple[str, float]],
    identity_root: list[str] | None = None,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    flip_prob: float = 0.5,
    same_prob: float = 0.2,
    hw_decoder: bool = True,
):

    external_source = fn.external_source(
        source=SampleReader(src, dst),
        num_outputs=2,
        device="cpu",
        no_copy=True,
        parallel=True,
        prefetch_queue_depth=2,
        dtype=DALIDataType.UINT8,
        batch=False,
    )

    if identity_root is None:
        identity_sampler = external_source
        same_prob = 0.0

    else:
        identity_sampler = fn.external_source(
            source=IdentityPairReader(identity_root),
            num_outputs=2,
            device="cpu",
            no_copy=True,
            parallel=True,
            prefetch_queue_depth=2,
            dtype=DALIDataType.UINT8,
            batch=False,
        )

    if fn.random.coin_flip(probability=same_prob, dtype=DALIDataType.BOOL):
        is_same = Constant(value=1.0, device="gpu", dtype=DALIDataType.FLOAT, shape=[1])
        src_raw, dst_raw = identity_sampler

    else:
        is_same = Constant(value=0.0, device="gpu", dtype=DALIDataType.FLOAT, shape=[1])
        src_raw, dst_raw = external_source

    if hw_decoder:
        src = fn.decoders.image(src_raw, device="mixed", output_type=DALIImageType.RGB, hw_decoder_load=0.75)
    else:
        src = fn.decoders.image(src_raw, device="cpu", output_type=DALIImageType.RGB, hw_decoder_load=0.75)
        src = fn.copy(src, device="gpu")

    src = fn.resize(src, device="gpu", size=img_resolution, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)
    if fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.BOOL):
        src = fn.flip(src, device="gpu")

    if hw_decoder:
        dst = fn.decoders.image(dst_raw, device="mixed", output_type=DALIImageType.RGB, hw_decoder_load=0.75)
    else:
        dst = fn.decoders.image(dst_raw, device="cpu", output_type=DALIImageType.RGB, hw_decoder_load=0.75)
        dst = fn.copy(dst, device="gpu")

    dst = fn.resize(dst, device="gpu", size=img_resolution, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)
    if fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.BOOL):
        dst = fn.flip(dst, device="gpu")

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
