import random
from typing import List, Tuple, Union
import numpy as np
from numpy import ndarray
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.types import Constant, DALIDataType, DALIImageType, DALIInterpType
from nvidia.dali.math import clamp
from misc.utils import ImageFolder


class SampleReader(object):
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

    src_raw, dst_raw = external_source

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

    dst = clamp(dst, lo=0.0, hi=255.0)
    dst = fn.normalize(dst, device="gpu", mean=127.5, stddev=127.5)
    dst = fn.transpose(dst, device="gpu", perm=[2, 0, 1])

    if fn.random.coin_flip(probability=same_image_prob, dtype=DALIDataType.BOOL):
        is_same = Constant(value=1.0, device="gpu", dtype=DALIDataType.FLOAT, shape=[1])
        src = dst
    else:
        is_same = Constant(value=0.0, device="gpu", dtype=DALIDataType.FLOAT, shape=[1])
        src = fn.decoders.image(src_raw, device="mixed", output_type=DALIImageType.RGB, hw_decoder_load=0.75)
        src = fn.resize(src, device="gpu", size=resize, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)
        if fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.BOOL):
            src = fn.flip(src, device="gpu")

        src = clamp(src, lo=0.0, hi=255.0)
        src = fn.normalize(src, device="gpu", mean=127.5, stddev=127.5)
        src = fn.transpose(src, device="gpu", perm=[2, 0, 1])

    return src, dst, is_same
