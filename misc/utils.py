import os
import random
import time
from functools import wraps
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch import Tensor
from torchvision.io import decode_image
from torchvision.io.image import ImageReadMode
from torchvision.transforms.v2.functional import to_dtype
import re


class Timer:
    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose = verbose
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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


class ImageFolder:
    def __init__(self, folder: str, file_filters: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"), random_sampling: bool = True):

        def natural_sort_key(filename):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", filename)]

        self.folder = Path(folder)
        self.files = [f for f in os.listdir(folder) if f.lower().endswith(file_filters)]
        self.files = sorted(self.files, key=natural_sort_key)
        self.len = len(self.files)
        self.idx = 0
        self._seq_idx = 0
        self.random_sampling = random_sampling

        if self.len == 0:
            raise FileNotFoundError(f"没有在目录：{folder}下找到任何图片文件，file_filters：{file_filters}")

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self) -> Path:
        if self.idx >= self.len:
            raise StopIteration
        fp = self.folder.joinpath(self.files[self.idx])
        self.idx += 1
        return fp

    def __getitem__(self, index: int) -> Path:
        if 0 <= index < self.len:
            fp = self.folder.joinpath(self.files[index])
            return fp
        else:
            raise IndexError

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ImageFolder(path='{self.folder}', nb={self.len}, random_sampling={self.random_sampling})"

    def sample(self, loop: bool = True) -> Path:
        """
        :return: 返回文件路径
        """

        if self.random_sampling:
            fp = self[random.randint(0, self.len - 1)]
        else:
            if self.idx >= self.len and loop:
                self.idx = 0
            fp = next(self)

        return fp

    def sample_tensor(self, label: bool = False, loop: bool = True) -> Tensor | tuple[Tensor, str]:
        """
        :param label: 是否返回对应的文件名
        :return: 位于cpu的 CHW RGB [0 ~ 255] torch.uint8
        """

        fp = self.sample(loop=loop)

        image = decode_image(fp, ImageReadMode.RGB)
        image = to_dtype(inpt=image, dtype=torch.uint8, scale=True)

        if label:
            return image, fp.name

        return image

    def iter_tensor(self, label: bool = False) -> Iterator[Tensor] | Iterator[tuple[Tensor, str]]:
        """
        顺序迭代返回tensor

        :param label: 是否返回对应的文件名
        :yield: Tensor 或 (Tensor, str)
            Tensor 为 CHW RGB [0~255] torch.uint8
        """
        indices = list(range(self.len))
        if self.random_sampling:
            random.shuffle(indices)

        for i in indices:
            fp = self[i]
            image = decode_image(fp, ImageReadMode.RGB)
            image = to_dtype(inpt=image, dtype=torch.uint8, scale=True)
            if label:
                yield image, fp.name
            else:
                yield image

    def iter_batch_tensor(self, batch_size: int, label: bool = False):
        """
        批量迭代返回tensor

        :param batch_size: 批大小
        :param label: 是否返回文件名
        :yield: (images, labels) 如果label=True, 否则只返回 images
            images 为 List[Tensor], 每个tensor为 CHW RGB [0~255] torch.uint8
        """
        indices = list(range(self.len))
        if self.random_sampling:
            random.shuffle(indices)

        for start in range(0, self.len, batch_size):
            batch_indices = indices[start : start + batch_size]
            images: list[torch.Tensor] = []
            if label:
                labels: list[str] = []

            for i in batch_indices:
                fp = self[i]
                img = decode_image(fp, ImageReadMode.RGB)
                img = to_dtype(inpt=img, dtype=torch.uint8, scale=True)
                images.append(img)

                if label:
                    labels.append(fp.name)

            yield (images, labels) if label else images
