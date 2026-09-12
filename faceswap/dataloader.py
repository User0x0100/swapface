"""训练数据管线统一入口。

训练代码只依赖 :class:`TrainingDataLoader` 与这里导出的配置类型；具体使用
NVIDIA DALI 还是可移植 PyTorch 实现由本模块内部根据当前 PyTorch 后端选择。
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from .dataloader_common import DATALOADER_RESERVED_KEYS, DEFAULT_DATALOADER_CONFIG, ImageDecoderBackend, ImageSource


class TrainingDataLoader:
    """统一训练数据加载接口，屏蔽 DALI/PyTorch 后端差异。"""

    def __init__(
        self,
        *,
        batch_size: int,
        device: torch.device,
        img_resolution: int,
        src: Sequence[ImageSource],
        dst: Sequence[ImageSource],
        **config,
    ) -> None:
        if device.type != "cuda" or torch.version.hip is not None:
            from .dataloader_native import _NativeTrainingDataLoader

            loader_type = _NativeTrainingDataLoader
        else:
            from .dataloader_dali import _DALITrainingDataLoader

            loader_type = _DALITrainingDataLoader

        self._loader = loader_type(
            batch_size=batch_size,
            device=device,
            img_resolution=img_resolution,
            src=src,
            dst=dst,
            **config,
        )

    def next(self) -> tuple[Tensor, Tensor, Tensor]:
        """返回 ``src``、``dst`` 与 ``theta_restore`` 三个目标设备 tensor。"""
        return self._loader.next()


__all__ = [
    "DATALOADER_RESERVED_KEYS",
    "DEFAULT_DATALOADER_CONFIG",
    "ImageDecoderBackend",
    "ImageSource",
    "TrainingDataLoader",
]
