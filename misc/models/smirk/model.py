"""SMIRK encoder only; adapted from georgeretsi/smirk (MIT, Copyright 2024 George Retsinas).

Upstream: https://github.com/georgeretsi/smirk/blob/main/src/smirk_encoder.py
No FLAME, renderer or Pytorch3D dependency is required by the training loss.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

# The project's leehx/model_hub currently does not host SMIRK. SkyReels-A1
# mirrors the official SMIRK_em1.pt byte-for-byte (verified against Google Drive).
# Revision and SHA-256 pin the original model independently of mutable main.
SMIRK_REPOSITORY_ID = "Skywork/SkyReels-A1"
SMIRK_FILENAME = "extra_models/smirk/SMIRK_em1.pt"
SMIRK_REVISION = "6cee256d5ea38dbbe87ff026566c95b6401d3124"
SMIRK_SHA256 = "26b234e3cc31a5de226bcba4321bb5d7343a2ad48234c028b57a1c2f8c2d22d0"


def _download_smirk_checkpoint() -> Path:
    """Fetch into the standard HF Hub cache; check identity before deserializing."""
    path = Path(hf_hub_download(repo_id=SMIRK_REPOSITORY_ID, filename=SMIRK_FILENAME, revision=SMIRK_REVISION))
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    if sha.hexdigest() != SMIRK_SHA256:
        raise ValueError(f"SMIRK checkpoint SHA-256 校验失败：{path}")
    return path


class _ImageHead(nn.Module):
    def __init__(self, backbone_name: str, num_outputs: int):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("启用 SMIRK 损失需要安装 timm（uv sync --group cuda 或 rocm）") from exc
        # SMIRK checkpoint must provide all model weights. Do not implicitly fetch
        # unrelated ImageNet weights, which would mask an incomplete checkpoint.
        self.encoder = timm.create_model(backbone_name, pretrained=False, features_only=True)
        channels = self.encoder.feature_info[-1]["num_chs"]
        self.output = nn.Linear(channels, num_outputs)

    def forward(self, images: Tensor) -> Tensor:
        features = self.encoder(images)[-1]
        return self.output(F.adaptive_avg_pool2d(features, 1).flatten(1))


class _PoseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        head = _ImageHead("tf_mobilenetv3_small_minimal_100", 6)
        self.encoder = head.encoder
        self.pose_cam_layers = nn.Sequential(head.output)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        feat = F.adaptive_avg_pool2d(self.encoder(images)[-1], 1).flatten(1)
        params = self.pose_cam_layers(feat)
        return {"pose_params": params[:, :3], "cam": params[:, 3:]}


def _straight_through_clamp(value: Tensor, minimum: float, maximum: float) -> Tensor:
    """Retain upstream physical output range while preserving input gradients.

    The frozen teacher is evaluated under no_grad for references. On the
    generated branch, conventional clamp/relu would have a zero derivative
    outside their ranges, so recovery from a bad prediction would be blocked.
    """
    bounded = value.clamp(minimum, maximum)
    return value + (bounded - value).detach()


class _ExpressionEncoder(nn.Module):
    def __init__(self, n_exp: int = 50):
        super().__init__()
        head = _ImageHead("tf_mobilenetv3_large_minimal_100", n_exp + 5)
        self.encoder = head.encoder
        self.expression_layers = nn.Sequential(head.output)
        self.n_exp = n_exp

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        feat = F.adaptive_avg_pool2d(self.encoder(images)[-1], 1).flatten(1)
        params = self.expression_layers(feat)
        jaw = torch.cat(
            (
                _straight_through_clamp(params[:, self.n_exp + 2 : self.n_exp + 3], 0.0, float("inf")),
                _straight_through_clamp(params[:, self.n_exp + 3 : self.n_exp + 5], -0.2, 0.2),
            ),
            dim=1,
        )
        return {
            "expression_params": params[:, : self.n_exp],
            "eyelid_params": _straight_through_clamp(params[:, self.n_exp : self.n_exp + 2], 0.0, 1.0),
            "jaw_params": jaw,
        }


class SMIRKEncoder(nn.Module):
    """SMIRK state dict matches upstream ``smirk_encoder.*`` keys exactly."""

    def __init__(self) -> None:
        super().__init__()
        path = _download_smirk_checkpoint()
        self.pose_encoder = _PoseEncoder()
        self.expression_encoder = _ExpressionEncoder()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError("SMIRK checkpoint 必须为 state_dict")
        prefix = "smirk_encoder."
        state = {key.removeprefix(prefix): value for key, value in checkpoint.items() if key.startswith((prefix + "pose_encoder.", prefix + "expression_encoder."))}
        if not state:
            raise ValueError("SMIRK checkpoint 缺少 smirk_encoder.* 权重；拒绝随机初始化教师")
        self.load_state_dict(state, strict=True)
        self.eval().requires_grad_(False)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        outputs = self.pose_encoder(images)
        outputs.update(self.expression_encoder(images))
        return outputs
