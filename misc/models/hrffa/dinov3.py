"""DINOv3 ViT-L/16 runtime loader for the HRFFA teacher.

The official DINOv3 implementation is cloned into PyTorch's hub cache on first use.
DINOv3 source code and pretrained weights are intentionally not vendored here.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

_DINOV3_GIT = "https://github.com/facebookresearch/dinov3"
_DINOV3_CACHE_DIR = "facebookresearch_dinov3_main"


class Dinov3ViTL16Backbone(nn.Module):
    """Official DINOv3 ViT-L/16 wrapped to match HRFFA teacher state_dict keys."""

    embed_dim = 1024

    def __init__(self, patch_instance_norm: bool = True) -> None:
        super().__init__()
        hub_dir = Path(torch.hub.get_dir()) / _DINOV3_CACHE_DIR
        if not hub_dir.exists():
            hub_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", _DINOV3_GIT, str(hub_dir)], check=True)
        if str(hub_dir) not in sys.path:
            sys.path.insert(0, str(hub_dir))

        backbones = importlib.import_module("dinov3.hub.backbones")

        # clean_v3 contains the complete fine-tuned backbone state_dict. Constructing
        # without pretrained weights avoids downloading another ~1 GB DINOv3 checkpoint.
        self.inner = backbones.dinov3_vitl16(pretrained=False)

        if patch_instance_norm:
            self.patch_in = nn.InstanceNorm2d(self.embed_dim, affine=True)

            def _patch_norm(_module, _inputs, output: Tensor) -> Tensor:
                output = output.permute(0, 3, 1, 2)
                output = self.patch_in(output)
                return output.permute(0, 2, 3, 1)

            self.inner.patch_embed.register_forward_hook(_patch_norm)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.inner.get_intermediate_layers(images, n=1, reshape=True, return_class_token=True)
        patch, cls = features[0]
        return patch, cls
