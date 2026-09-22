from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

from ...models import MODEL_REPOSITORY_ID
from .adaface import IR_101
from .iresnet import iresnet50, iresnet100
from .qcface_iresnet import qcface_iresnet100
from .vit import vit_b, vit_l, vit_s

ID_ENCODER_INPUT_SIZE = (112, 112)
ID_ALIGNMENT_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def get_alignment_template(output_size: int = 112) -> np.ndarray:
    if output_size <= 0:
        raise ValueError("output_size must be greater than 0")
    if output_size % 112 == 0:
        ratio = float(output_size) / 112.0
        diff_x = 0
    else:
        ratio = float(output_size) / 128.0
        diff_x = 8.0 * ratio
    dst = ID_ALIGNMENT_TEMPLATE * ratio
    dst[:, 0] += diff_x
    return dst


@dataclass(frozen=True, slots=True)
class IDEncoderProviderConfig:
    backbone: Callable[[], nn.Module]
    weight_filename: str
    input_mean: tuple[float, float, float] | None = None
    input_std: tuple[float, float, float] | None = None


class IDEncoderProvider(Enum):
    BLENDFACE = IDEncoderProviderConfig(iresnet100, "blendface.pth")
    MS1MV3_ARCFACE_R50_FP16 = IDEncoderProviderConfig(iresnet50, "ms1mv3_arcface_r50_fp16.pth")
    MS1MV3_ARCFACE_R100_FP16 = IDEncoderProviderConfig(iresnet100, "ms1mv3_arcface_r100_fp16.pth")
    MS1MV2_TRANSFACE_S = IDEncoderProviderConfig(vit_s, "ms1mv2_model_TransFace_S.pt")
    MS1MV2_TRANSFACE_B = IDEncoderProviderConfig(vit_b, "ms1mv2_model_TransFace_B.pt")
    MS1MV2_TRANSFACE_L = IDEncoderProviderConfig(vit_l, "ms1mv2_model_TransFace_L.pt")
    MS1MV2_ADAFACE_R100 = IDEncoderProviderConfig(IR_101, "adaface_ir101_ms1mv2.ckpt")
    MS1MV3_ADAFACE_R100 = IDEncoderProviderConfig(IR_101, "adaface_ir101_ms1mv3.ckpt")
    QCFACE_ARC_IR100 = IDEncoderProviderConfig(
        qcface_iresnet100,
        "qcface_arc_ir100.pth",
        input_mean=(0.5312, 0.4265, 0.3753),
        input_std=(0.2873, 0.2555, 0.2496),
    )
    GLINT360K_TOPOFR_R100 = IDEncoderProviderConfig(iresnet100, "glint360k_r100_topofr.pth")



class IDEncoder(nn.Module):
    """面部身份特征编码器

    使用预训练的 IResNet 模型提取面部身份特征向量。

    Args:
        provider: 模型提供者，支持 ArcFace BlendFace TransFace

    Example:
        >>> encoder = IDEncoder(IDEncoderProvider.BLENDFACE)
        >>> features = encoder(face_images)
    """

    def __init__(self, provider: IDEncoderProvider = IDEncoderProvider.BLENDFACE) -> None:
        super().__init__()

        self.provider = provider
        weight_path = hf_hub_download(repo_id=MODEL_REPOSITORY_ID, filename=provider.value.weight_filename)
        self.backbone = provider.value.backbone()

        mean = provider.value.input_mean
        std = provider.value.input_std
        if (mean is None) != (std is None):
            raise ValueError(f"{provider.name} must define both input_mean and input_std")
        self.register_buffer("input_mean", None if mean is None else torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", None if std is None else torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        state_dict = torch.load(weight_path, weights_only=True, map_location="cpu")
        self.backbone.load_state_dict(state_dict)
        self.backbone.eval().requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x: 输入图像张量 [B, C, H, W]，值域 [-1, 1]，RGB

        Returns:
            归一化的 FP32 特征向量 [B, 512]
        """

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            if x.shape[2:] != ID_ENCODER_INPUT_SIZE:
                x = F.interpolate(x, ID_ENCODER_INPUT_SIZE, mode="bilinear", align_corners=False)
            if self.input_mean is not None and self.input_std is not None:
                x = (x + 1.0) * 0.5
                x = (x - self.input_mean) / self.input_std
            features = self.backbone(x)
            return F.normalize(features, p=2, dim=1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    id_encoder = IDEncoder(IDEncoderProvider.MS1MV3_ADAFACE_R100).to(device)

    id_encoder = torch.compile(id_encoder, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")

    x = torch.randn((2, 3, 112, 112), device=device, dtype=torch.float)
    feat = id_encoder(x)
    print(feat.shape)
