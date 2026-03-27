import numpy as np
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .iresnet import iresnet50, iresnet100
from .vit import vit_s, vit_b, vit_l, VisionTransformer
from ...models import REPO_ID


INPUT_SIZE = (112, 112)
ALIGN_LANDMARKS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def get_align_landmarks(dst_size: int = 112) -> np.ndarray:
    if dst_size % 112 == 0:
        ratio = float(dst_size) / 112.0
        diff_x = 0
    else:
        ratio = float(dst_size) / 128.0
        diff_x = 8.0 * ratio
    dst = ALIGN_LANDMARKS * ratio
    dst[:, 0] += diff_x
    return dst


class PROVIDER(Enum):
    BLENDFACE = {"backbone": iresnet100, "weight": "blendface.pth"}
    MS1MV3_ARCFACE_R50_FP16 = {"backbone": iresnet50, "weight": "ms1mv3_arcface_r50_fp16.pth"}
    MS1MV3_ARCFACE_R100_FP16 = {"backbone": iresnet100, "weight": "ms1mv3_arcface_r100_fp16.pth"}
    MS1MV2_TRANSFACE_S = {"backbone": vit_s, "weight": "ms1mv2_model_TransFace_S.pt"}
    MS1MV2_TRANSFACE_B = {"backbone": vit_b, "weight": "ms1mv2_model_TransFace_B.pt"}
    MS1MV2_TRANSFACE_L = {"backbone": vit_l, "weight": "ms1mv2_model_TransFace_L.pt"}


class IDEncoder(nn.Module):
    """面部身份特征编码器

    使用预训练的 IResNet 模型提取面部身份特征向量。

    Args:
        provider: 模型提供者，支持 ArcFace 和 BlendFace

    Example:
        >>> encoder = IDEncoder(PROVIDER.BLENDFACE)
        >>> features = encoder(face_images)
    """

    def __init__(self, provider: PROVIDER = PROVIDER.BLENDFACE) -> None:
        super().__init__()

        weight = hf_hub_download(repo_id=REPO_ID, filename=provider.value["weight"])
        self.backbone = provider.value["backbone"]()

        weight = torch.load(weight, weights_only=True, map_location=torch.device("cpu"))
        self.backbone.load_state_dict(weight)

        self.backbone.eval()
        self.backbone.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x: 输入图像张量 [B, C, H, W]，值域 [-1, 1]，RGB

        Returns:
            归一化的特征向量 [B, 512]
        """

        if x.shape[2:] != INPUT_SIZE:
            x = F.interpolate(x, INPUT_SIZE, mode="bilinear", align_corners=False)
        id = self.backbone(x)

        return F.normalize(id, p=2, dim=1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idencoder = IDEncoder(PROVIDER.MS1MV2_TRANSFACE_S).to(device)

    idencoder = torch.compile(idencoder, fullgraph=True, dynamic=False, options={"max_autotune": True, "epilogue_fusion": True})

    x = torch.randn((2, 3, 112, 112), device=device, dtype=torch.float)
    feat = idencoder(x)
    print(feat.shape)
