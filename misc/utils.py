import os
import re
import time
from collections.abc import Iterator
from functools import wraps
from pathlib import Path
from typing import Literal, Self, overload

import numpy as np
import torch
from torch import Tensor
from torchvision.io import decode_image
from torchvision.io.image import ImageReadMode


class Timer:
    def __init__(self, name: str | None = None, verbose: bool = True) -> None:
        self.name = name
        self.verbose = verbose
        self.start: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> Self:
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start is None:
            raise RuntimeError("Timer was not started")
        self.elapsed = time.perf_counter() - self.start
        if self.verbose:
            label = self.name or "Block"
            print(f"[{label}] Elapsed: {self.format_time(self.elapsed)}")

    def __call__(self, func):
        """使 Timer 实例可作为装饰器使用"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            label = self.name or func.__name__
            with Timer(name=label, verbose=self.verbose):
                return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds >= 1:
            return f"{seconds:.3f} s"
        elif seconds >= 1e-3:
            return f"{seconds * 1e3:.3f} ms"
        else:
            return f"{seconds * 1e6:.3f} µs"


class ImageDirectory:
    """
    一个轻量的图片目录封装器，提供确定性排序、随机采样和张量迭代功能。

    文件按"自然排序"（natural sort）规则排列，即数字部分按数值大小比较，
    而非字典序（例如 img2.jpg < img10.jpg）。

    本类支持 pickle 序列化（multiprocessing DataLoader worker 场景），
    `__getstate__` / `__setstate__` 会在反序列化时为每个 worker 独立重建 RNG。

    Attributes:
        folder (Path): 图片目录的绝对/相对路径。
        files  (list[str]): 按自然排序后的文件名列表（不含目录前缀）。
        rng    (numpy.random.Generator): 实例私有的随机数生成器。

    Raises:
        FileNotFoundError: 目录下不存在任何匹配 `file_filters` 的文件时抛出。
    """

    DEFAULT_FILTERS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

    def __init__(self, directory: str | Path, file_filters: tuple[str, ...] | None = None) -> None:
        """
        初始化 。

        Args:
            folder      (str):           图片目录路径。
            file_filters (tuple[str, ...]): 允许的文件扩展名（不区分大小写）。
                                          默认".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"。

        Raises:
            FileNotFoundError: 目录下无匹配文件。
        """

        self.directory = Path(directory).resolve(strict=True)

        if file_filters is None:
            file_filters = self.DEFAULT_FILTERS
        filters = tuple(ext.lower() for ext in file_filters)

        self.file_names = sorted(
            [entry.name for entry in self.directory.iterdir() if entry.is_file() and entry.name.lower().endswith(filters)],
            key=self._natural_sort_key,
        )

        if not self.file_names:
            raise FileNotFoundError(f"没有在目录：{directory}下找到任何图片文件，file_filters：{file_filters}")

        self.rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))

    @staticmethod
    def _natural_sort_key(file_name: str) -> list:
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", file_name)]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["rng"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        seed = int.from_bytes(os.urandom(8), "little")
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.file_names)

    def __iter__(self) -> Iterator[Path]:
        for f in self.file_names:
            yield self.directory / f

    @overload
    def __getitem__(self, index: int) -> Path: ...

    @overload
    def __getitem__(self, index: slice) -> list[Path]: ...

    def __getitem__(self, index: int | slice) -> Path | list[Path]:
        if isinstance(index, slice):
            return [self.directory / f for f in self.file_names[index]]
        return self.directory / self.file_names[index]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"ImageDirectory(path='{self.directory}', count={len(self)})"

    def sample(self) -> Path:
        """
        从文件列表中均匀随机采样一张图片的路径。

        Returns:
            Path: 随机选中的图片完整路径。

        Note:
            使用实例私有的 ``numpy`` RNG，多进程环境下各 worker
            的随机序列相互独立（见 ``__setstate__``）。
        """
        return self[int(self.rng.integers(len(self)))]

    @staticmethod
    def _decode(path: Path, device: torch.device | str, dtype: torch.dtype) -> Tensor:
        return decode_image(str(path), ImageReadMode.RGB).to(device=device, dtype=dtype)

    @overload
    def sample_tensor(
        self,
        return_stem: Literal[False] = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float,
    ) -> Tensor: ...

    @overload
    def sample_tensor(
        self,
        return_stem: Literal[True],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float,
    ) -> tuple[Tensor, str]: ...

    def sample_tensor(
        self,
        return_stem: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float,
    ) -> Tensor | tuple[Tensor, str]:
        """
        随机采样一张图片并解码为 RGB 张量。

        Args:
            return_stem (bool):                是否同时返回文件名 stem。
            device (torch.device | str):  目标设备，默认 ``"cpu"``。
            dtype  (torch.dtype):         目标数据类型，默认 ``torch.float``。

        Returns:
            Tensor:               ``return_stem=False`` 时，形状为 ``[3, H, W]``，
                                  值域为 ``[0.0, 255.0]``（float32 直接转换自 uint8，
                                  **未归一化**，如需 [0,1] 请手动除以 255）。
            tuple[Tensor, str]:   ``return_stem=True`` 时，额外返回 ``(tensor, stem)``，
                                  ``stem`` 为不含扩展名的文件名字符串。

        Note:
            底层使用 ``torchvision.io.decode_image``，直接从磁盘解码，
            避免经过 PIL 的中间转换，性能更优。
        """
        path = self.sample()
        image = self._decode(path, device=device, dtype=dtype)

        if return_stem:
            return image, path.stem

        return image

    def iter_tensors(
        self,
        return_stem: bool = False,
        device: torch.device | str = "cpu",
        dtype=torch.float,
    ) -> Iterator[Tensor] | Iterator[tuple[Tensor, str]]:
        """
        按自然排序顺序逐一产出 RGB 张量（惰性求值，不预加载到内存）。

        Args:
            return_stem (bool):                是否同时 yield 文件名 stem。
            device (torch.device | str):  目标设备，默认 ``"cpu"``。
            dtype  (torch.dtype):         目标数据类型，默认 ``torch.float``。

        Yields:
            Tensor:               ``return_stem=False`` 时，形状 ``[3, H, W]``，
                                  值域 ``[0.0, 255.0]``（**未归一化**）。
            tuple[Tensor, str]:   ``return_stem=True`` 时，产出 ``(tensor, stem)``。

        Note:
            适合在单线程脚本中遍历全量数据；若需并行预取，
            建议封装为 ``torch.utils.data.Dataset`` 并配合 ``DataLoader``。
        """
        for path in self:
            image = self._decode(path, device=device, dtype=dtype)
            if return_stem:
                yield image, path.stem
            else:
                yield image

    def iter_tensor_batches(
        self,
        batch_size: int,
        return_stem: bool = False,
        device: torch.device | str = "cpu",
        dtype=torch.float,
    ) -> Iterator[list[Tensor] | tuple[list[Tensor], list[str]]]:
        """
        按自然排序顺序以批次为单位产出 RGB 张量列表（惰性求值）。

        最后一个批次可能不足 ``batch_size``（自动处理尾部余量）。

        Args:
            batch_size (int):             每批图片数量，建议为 2 的幂次（如 8、16、32）。
            return_stem (bool):      是否同时返回文件名 stem 列表。
            device     (torch.device | str): 目标设备，默认 ``"cpu"``。
            dtype      (torch.dtype):     目标数据类型，默认 ``torch.float``。

        Yields:
            list[Tensor]:                       ``return_stem=False`` 时，长度 ≤ ``batch_size``
                                                的张量列表，每项形状 ``[3, H, W]``，
                                                值域 ``[0.0, 255.0]``（**未归一化**）。
            tuple[list[Tensor], list[str]]:     ``return_stem=True`` 时，额外返回对应文件名列表。

        Note:
            由于图片尺寸不一定相同，本方法返回 **列表** 而非堆叠的 4D 张量；
            若所有图片尺寸一致，可在调用处使用 ``torch.stack(images)`` 合并。

        Example::

            folder = ImageDirectory("/data/faces")
            for batch, names in folder.iter_tensor_batches(32, return_stem=True, device="cuda"):
                # batch: list of [3, H, W] tensors  (len <= 32)
                x = torch.stack(batch) / 255.0      # → [B, 3, H, W], 归一化
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        for i in range(0, len(self), batch_size):
            batch_paths = self[i : min(i + batch_size, len(self))]
            images = [self._decode(path, device=device, dtype=dtype) for path in batch_paths]

            if return_stem:
                yield images, [path.stem for path in batch_paths]
            else:
                yield images
