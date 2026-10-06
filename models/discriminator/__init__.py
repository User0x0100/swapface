from collections.abc import Mapping
from enum import Enum
from typing import Any

from torch import nn

from .network import (
    INSTANCE_NORM_RESIDUAL_MIN_RESOLUTION,
    FIRMinibatchStdDiscriminator,
    InstanceNormResidualDiscriminator,
)


class DiscriminatorType(Enum):
    FIR_MINIBATCH_STD = "fir_minibatch_std"
    INSTANCE_NORM_RESIDUAL = "instance_norm_residual"


def build_discriminator(config: Mapping[str, Any]) -> nn.Module:
    cfg = dict(config)
    discriminator_type = DiscriminatorType[str(cfg.pop("type"))]
    if discriminator_type is DiscriminatorType.FIR_MINIBATCH_STD:
        model = FIRMinibatchStdDiscriminator(**cfg)
    elif discriminator_type is DiscriminatorType.INSTANCE_NORM_RESIDUAL:
        model = InstanceNormResidualDiscriminator(**cfg)
    else:
        raise AssertionError(f"未处理的判别器类型：{discriminator_type}")
    model.network_cfg = dict(config)
    return model


# 兼容仍直接导入旧泛型名称的外部代码；训练配置和新代码使用明确类型名。
Discriminator = FIRMinibatchStdDiscriminator

__all__ = [
    "INSTANCE_NORM_RESIDUAL_MIN_RESOLUTION",
    "Discriminator",
    "DiscriminatorType",
    "FIRMinibatchStdDiscriminator",
    "InstanceNormResidualDiscriminator",
    "build_discriminator",
]
