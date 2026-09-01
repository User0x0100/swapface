"""HRFFA ViT-T/16 学生骨干网络。

该实现按 HRFFA 上游 ``vit_tiny.py`` 的推理结构整理，仅保留加载最终训练
checkpoint 所需的模块。输入位置编码使用二维 RoPE，因此推理时不需要绝对位置
嵌入插值。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Rope2D(nn.Module):
    """轴分离的二维旋转位置编码。"""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim 必须能被 4 整除，实际为 {head_dim}")
        self.head_dim = head_dim
        self.register_buffer(
            "periods", torch.empty(head_dim // 4, dtype=torch.float32), persistent=True
        )
        self._cache: dict[tuple[int, int, str], tuple[Tensor, Tensor]] = {}

    def forward(self, height: int, width: int) -> tuple[Tensor, Tensor]:
        """返回指定 patch 网格对应的 ``(sin, cos)`` RoPE 张量。"""
        key = (height, width, str(self.periods.device))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        periods = self.get_buffer("periods")
        rows = (
            torch.arange(0.5, float(height), device=periods.device, dtype=torch.float32)
            / height
        )
        cols = (
            torch.arange(0.5, float(width), device=periods.device, dtype=torch.float32)
            / width
        )
        coordinates = torch.stack(
            torch.meshgrid(rows, cols, indexing="ij"), dim=-1
        ).flatten(0, 1)
        coordinates = coordinates.mul(2.0).sub(1.0)
        angles = 2.0 * math.pi * coordinates[:, :, None] / periods.float()[None, None]
        angles = angles.flatten(1, 2).tile(2)
        cached = (torch.sin(angles), torch.cos(angles))
        self._cache[key] = cached
        return cached


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    dtype = x.dtype
    x_float = x.float()
    return ((x_float * cos) + (_rotate_half(x_float) * sin)).to(dtype)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} 必须能被 num_heads={num_heads} 整除")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        rope: tuple[Tensor, Tensor],
        prefix_tokens: int,
    ) -> Tensor:
        batch_size, token_count, channels = x.shape
        q, k, v = (
            tensor.reshape(
                batch_size, token_count, self.num_heads, self.head_dim
            ).transpose(1, 2)
            for tensor in self.qkv(x).split(channels, dim=-1)
        )

        sin, cos = rope
        q = torch.cat(
            (q[:, :, :prefix_tokens], _apply_rope(q[:, :, prefix_tokens:], sin, cos)),
            dim=2,
        )
        k = torch.cat(
            (k[:, :, :prefix_tokens], _apply_rope(k[:, :, prefix_tokens:], sin, cos)),
            dim=2,
        )

        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).reshape(batch_size, token_count, channels)
        return self.proj(output)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        x = x + self.attn(self.norm1(x), rope, prefix_tokens=1)
        return x + self.mlp(self.norm2(x))


class ViTTiny(nn.Module):
    """HRFFA ViT-T/16 backbone。

    最终 HRFFA checkpoint 已包含训练完成后的全部 backbone 参数，因此这里不加载
    ImageNet 初始化权重。``forward`` 返回 patch 特征和 CLS 特征。
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        patch_size: int = 16,
        patch_instance_norm: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = nn.ModuleDict(
            {"proj": nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)}
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.rope_embed = Rope2D(embed_dim // num_heads)
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads) for _ in range(depth)]
        )
        self.patch_in = (
            nn.InstanceNorm2d(embed_dim, affine=True) if patch_instance_norm else None
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        patch = self.patch_embed["proj"](x)
        if self.patch_in is not None:
            patch = self.patch_in(patch)

        batch_size, channels, height, width = patch.shape
        tokens = patch.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)

        rope = self.rope_embed(height, width)
        for block in self.blocks:
            tokens = block(tokens, rope)

        cls_output = tokens[:, 0]
        patch_output = (
            tokens[:, 1:].transpose(1, 2).reshape(batch_size, channels, height, width)
        )
        return patch_output, cls_output
