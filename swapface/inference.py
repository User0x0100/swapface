"""当前训练 checkpoint 与 ONNX 推理共用的输入约定。"""

from pathlib import Path

import torch
from torch import Tensor

from misc.face_alignment import FFHQ_TO_ARCFACE_112_AFFINE_512
from misc.models.id_encoder import IDEncoderProvider, get_alignment_template
from models.networks import Generator

from .contracts import CHECKPOINT_VERSION


def load_generator(checkpoint_path: str | Path) -> tuple[Generator, IDEncoderProvider, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        version = checkpoint["version"]
    except KeyError as error:
        raise ValueError("checkpoint 缺少 version") from error
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"仅支持当前 train.py 保存的 v{CHECKPOINT_VERSION} checkpoint")
    try:
        provider = IDEncoderProvider[checkpoint["identity_encoders"]["generator"]]
    except (KeyError, TypeError) as error:
        raise ValueError("checkpoint 缺少有效的 Generator 身份编码器配置") from error
    config = checkpoint["net_g"]["network_cfg"]
    if config["img_channels"] != 3 or config["id_dim"] != 512:
        raise ValueError("当前推理要求 RGB 图像和 512 维身份特征")
    model = Generator(**config)
    # train.py 将用于推理的 EMA 权重存入 net_g；training_state.net_g 是训练权重。
    model.load_state_dict(checkpoint["net_g"]["state_dict"])
    try:
        completed_step = checkpoint["step"]
    except KeyError as error:
        raise ValueError("checkpoint 缺少 step") from error
    if not isinstance(completed_step, int) or isinstance(completed_step, bool) or completed_step <= 0:
        raise ValueError(f"checkpoint.step 必须是正整数：{completed_step!r}")
    return model.eval(), provider, completed_step


def get_ffhq_alignment_template(output_size: int, device: torch.device) -> Tensor:
    """反解训练端的 canonical 映射，得到检测器五点对应的 FFHQ 目标坐标。"""
    if output_size <= 0:
        raise ValueError("output_size 必须为正数")
    affine = torch.tensor(FFHQ_TO_ARCFACE_112_AFFINE_512, device=device, dtype=torch.float32)
    arcface = torch.as_tensor(get_alignment_template(112), device=device, dtype=torch.float32)
    # 五点对齐遵循训练的固定映射；若需完整 FFHQ 68 点裁剪，需更换关键点检测器。
    return torch.linalg.solve(affine[:, :2], (arcface - affine[:, 2]).T).T * (output_size / 512.0)
