import os

import cv2
import numpy as np
from numpy import ndarray
from nvidia.dali import fn, pipeline_def
from nvidia.dali.math import clamp
from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

from misc.utils import ImageDirectory


class RndAffinePars:
    def __init__(
        self,
        img_resolution: int,
        rotation_range: tuple[int | float, ...] = (-10.0, 10.0),
        scale_range: tuple[int | float, ...] = (-0.25, 0.25),
        tx_range: tuple[int | float, ...] = (-0.05, 0.05),
        ty_range: tuple[int | float, ...] = (-0.05, 0.05),
    ):
        self.img_resolution = img_resolution
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.tx_range = tx_range
        self.ty_range = ty_range
        self.rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["rng"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        seed = int.from_bytes(os.urandom(8), "little")
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _to_3x3(m: ndarray) -> ndarray:
        out = np.eye(3, dtype=np.float64)
        out[:2, :] = m.astype(np.float64)
        return out

    @staticmethod
    def _invert_affine(m: ndarray) -> ndarray:
        m3 = RndAffinePars._to_3x3(m)
        return np.linalg.inv(m3)[:2, :].astype(np.float32)

    @staticmethod
    def _pixel_mtx_to_torch_theta(m: ndarray, h: int, w: int, align_corners: bool = False) -> ndarray:
        """
        m: output_pixel -> input_pixel 的 2x3 像素坐标矩阵。
        返回 PyTorch affine_grid 可直接使用的 2x3 theta。
        """

        m3 = RndAffinePars._to_3x3(m)

        if align_corners:
            px_to_norm = np.array(
                [
                    [2.0 / (w - 1), 0.0, -1.0],
                    [0.0, 2.0 / (h - 1), -1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            norm_to_px = np.array(
                [
                    [(w - 1) / 2.0, 0.0, (w - 1) / 2.0],
                    [0.0, (h - 1) / 2.0, (h - 1) / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

        else:
            px_to_norm = np.array(
                [
                    [2.0 / w, 0.0, 1.0 / w - 1.0],
                    [0.0, 2.0 / h, 1.0 / h - 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            norm_to_px = np.array(
                [
                    [w / 2.0, 0.0, w / 2.0 - 0.5],
                    [0.0, h / 2.0, h / 2.0 - 0.5],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

        theta = px_to_norm @ m3 @ norm_to_px
        return theta[:2, :].astype(np.float32)

    def __call__(self, sample_info=None) -> tuple[ndarray, ndarray]:
        _ = sample_info

        w = self.img_resolution
        h = self.img_resolution

        rotation = self.rng.uniform(self.rotation_range[0], self.rotation_range[1])
        scale = self.rng.uniform(1 / (1 - self.scale_range[0]), 1 + self.scale_range[1])
        tx = self.rng.uniform(self.tx_range[0], self.tx_range[1])
        ty = self.rng.uniform(self.ty_range[0], self.ty_range[1])

        # OpenCV/DALI source -> destination 像素坐标矩阵
        src_to_dst = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), rotation, scale)
        src_to_dst[:, 2] += (tx * w, ty * h)
        src_to_dst = src_to_dst.astype(np.float32)

        # destination -> source，用于 PyTorch 复现 warp：
        # warped = grid_sample(base, theta_warp)
        # dst_to_src = self._invert_affine(src_to_dst)

        # theta_warp = self._pixel_mtx_to_torch_theta(dst_to_src, h=h, w=w, align_corners=False)

        # source -> destination，用于 PyTorch 还原：
        # restored = grid_sample(warped, theta_restore)
        theta_restore = self._pixel_mtx_to_torch_theta(src_to_dst, h=h, w=w, align_corners=False)

        dali_mtx = src_to_dst.astype(np.float32)

        return dali_mtx, theta_restore


class SampleReader:
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

    def collect_folder(self, folder_weight_list: list[str] | list[tuple[str, float]]) -> tuple[list[ImageDirectory], list[float]]:

        readers: list[ImageDirectory] = []
        file_counts: list[int] = []
        adjustments: list[float] = []

        if isinstance(folder_weight_list[0], str):
            for folder in folder_weight_list:
                reader = ImageDirectory(folder)
                readers.append(reader)
                file_counts.append(len(reader))
                adjustments.append(0.0)
        else:
            for folder, adj in folder_weight_list:
                reader = ImageDirectory(folder)
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

        def dump(title, folders: list[ImageDirectory] | None, weights: list[float]):

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
    hw_decoder: bool = True,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    flip_prob: float = 0.5,
    rotation_range: tuple[int | float, ...] = (-10.0, 10.0),
    scale_range: tuple[int | float, ...] = (-0.25, 0.25),
    tx_range: tuple[int | float, ...] = (-0.05, 0.05),
    ty_range: tuple[int | float, ...] = (-0.05, 0.05),
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

    rnd_affine_params = fn.external_source(
        source=RndAffinePars(img_resolution, rotation_range, scale_range, tx_range, ty_range),
        num_outputs=2,
        device="gpu",
        no_copy=False,
        parallel=False,
        dtype=DALIDataType.FLOAT,
        batch=False,
    )

    src_raw, dst_raw = external_source

    image_decoders_device = "mixed" if hw_decoder else "cpu"

    src = fn.decoders.image(src_raw, device=image_decoders_device, output_type=DALIImageType.RGB, hw_decoder_load=0.75)
    if hw_decoder:
        src = fn.copy(src, device="gpu")

    dst = fn.decoders.image(dst_raw, device="mixed", output_type=DALIImageType.RGB, hw_decoder_load=0.75)
    if hw_decoder:
        dst = fn.copy(dst, device="gpu")

    src = fn.resize(src, device="gpu", size=img_resolution, dtype=DALIDataType.FLOAT, interp_type=DALIInterpType.INTERP_LANCZOS3)
    if fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.BOOL):
        src = fn.flip(src, device="gpu")

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

    dali_mtx, theta_restore = rnd_affine_params
    dst = fn.warp_affine(dst, dali_mtx, inverse_map=False, fill_value=-1.0)

    src = clamp(src, lo=0.0, hi=255.0)
    src = fn.normalize(src, device="gpu", mean=127.5, stddev=127.5)
    src = fn.transpose(src, device="gpu", perm=[2, 0, 1])

    dst = clamp(dst, lo=0.0, hi=255.0)
    dst = fn.normalize(dst, device="gpu", mean=127.5, stddev=127.5)
    dst = fn.transpose(dst, device="gpu", perm=[2, 0, 1])

    return src, dst, theta_restore
