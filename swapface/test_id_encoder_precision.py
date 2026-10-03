"""无需下载权重的 ID 编码器混合精度回归检查：uv run --no-sync python -m swapface.test_id_encoder_precision"""

import torch
from torch import nn

from misc.models.id_encoder.model import IDEncoder


def main() -> None:
    encoder = IDEncoder.__new__(IDEncoder)
    nn.Module.__init__(encoder)
    encoder.backbone = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 8),
    ).eval()
    encoder.register_buffer("input_mean", None, persistent=False)
    encoder.register_buffer("input_std", None, persistent=False)

    faces = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        fp32_embeddings = encoder(faces)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            bf16_embeddings = encoder(faces)

    assert fp32_embeddings.dtype == torch.float32
    assert bf16_embeddings.dtype == torch.bfloat16

    print("PASS: ID encoder follows the caller autocast precision without forcing FP32")


if __name__ == "__main__":
    main()
