"""HRFFA 高角度鲁棒人脸关键点模型的 PyTorch 推理封装。

模型来源：PINTO0309/High-Angle_Robust_Fast_FaceAlignment。
当前默认使用 ViT-T/16、256×256 输入的学生模型。HRFFA 的训练输入是整头部
(head crop)，而不是传统紧致人脸 crop；推荐外部检测器使用头部框，并按长边扩展
5% 后构造正方形 crop，与上游训练几何保持一致。
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

from ...models import MODEL_REPOSITORY_ID, ImageInputRange
from .vit import ViTTiny


class HRFFAScheme(Enum):
    """HRFFA checkpoint 内置的关键点拓扑。"""

    IBUG68 = "ibug68"
    WFLW98 = "wflw98"
    COFW29 = "cofw29"


SCHEME_LANDMARK_COUNTS: dict[HRFFAScheme, int] = {
    HRFFAScheme.IBUG68: 68,
    HRFFAScheme.WFLW98: 98,
    HRFFAScheme.COFW29: 29,
}


class HRFFAVisibility(Enum):
    """HRFFA 每个关键点的三分类可见性标签。"""

    OUTSIDE_IMAGE = 0
    OCCLUDED = 1
    VISIBLE = 2


class _HRFFABackbone(nn.Module):
    """保持上游 state_dict 键结构的 ViT-T backbone wrapper。"""

    def __init__(self) -> None:
        super().__init__()
        self.inner = ViTTiny(
            embed_dim=192,
            depth=12,
            num_heads=3,
            patch_size=16,
            patch_instance_norm=True,
        )
        self.embed_dim = 192

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.inner(x)


def _sincos_pos_embed_2d(dim: int, height: int, width: int, device: torch.device) -> Tensor:
    """生成二维正弦/余弦位置编码，输出形状为 ``(H*W, dim)``。"""
    if dim % 4 != 0:
        raise ValueError(f"dim 必须能被 4 整除，实际为 {dim}")

    quarter = dim // 4
    omega = torch.arange(quarter, device=device) / quarter
    omega = 1.0 / (10000**omega)
    rows, cols = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    output: list[Tensor] = []
    for positions in (cols, rows):
        angles = positions.reshape(-1, 1).float() * omega[None]
        output.extend((torch.sin(angles), torch.cos(angles)))
    return torch.cat(output, dim=1)


class _HRFFANetwork(nn.Module):
    """与 HRFFA ViT-T 学生 checkpoint 严格匹配的原生 PyTorch 网络。"""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _HRFFABackbone()
        d_model = 256

        self.input_proj = nn.Linear(self.backbone.embed_dim, d_model)
        self.cls_proj = nn.Linear(self.backbone.embed_dim, d_model)
        self.queries = nn.ParameterDict({scheme.value: nn.Parameter(torch.empty(count, d_model, dtype=torch.float32)) for scheme, count in SCHEME_LANDMARK_COUNTS.items()})

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3, norm=nn.LayerNorm(d_model))
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.vis_head = nn.Linear(d_model, 3)
        self.pose_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 8),
        )
        self._pos_cache: dict[tuple[int, int, int, str], Tensor] = {}

    def _position_embedding(self, height: int, width: int, dim: int, device: torch.device) -> Tensor:
        key = (height, width, dim, str(device))
        cached = self._pos_cache.get(key)
        if cached is None:
            cached = _sincos_pos_embed_2d(dim, height, width, device)
            self._pos_cache[key] = cached
        return cached

    def forward(self, images: Tensor, scheme: HRFFAScheme) -> tuple[Tensor, Tensor]:
        patch, _cls = self.backbone(images)
        batch_size, channels, height, width = patch.shape
        memory = self.input_proj(patch.reshape(batch_size, channels, height * width).transpose(1, 2))
        memory = memory + self._position_embedding(height, width, memory.shape[-1], memory.device)[None]

        queries = self.queries[scheme.value][None].expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        points = self.coord_head(decoded)
        visibility_logits = self.vis_head(decoded)
        return points, visibility_logits


class HRFFALandmarkModel(nn.Module):
    """HRFFA ViT-T/256 整头部关键点模型。

    输入是已经裁好的**正方形整头部 crop**，不是传统紧致人脸 crop。模型内部会将
    任意正方形输入缩放到 256×256，并按学生模型的 center05 规范转换到 ``[-1, 1]``。
    输出关键点会恢复到调用方输入 crop 的像素坐标，因此输入尺寸不必固定为 256。

    参数:
        scheme: 关键点拓扑，默认 ``HRFFAScheme.IBUG68``。
        input_range: 调用方输入张量值域枚举。

    输出:
        默认返回 ``(N, K, 2)`` 像素坐标；设置 ``return_visibility=True`` 时返回
        ``(landmarks, visibility)``，其中 visibility 形状为 ``(N, K)``，类别含义为
        0=图像外、1=遮挡、2=可见。

    说明:
        上游训练 crop 几何为：以头部框中心为中心，取框长边的 ``1.1`` 倍正方形区域，
        即每侧约 5% padding。若输入 crop 几何不同，关键点精度可能下降。
    """

    INPUT_SIZE = 256
    WEIGHT_FILENAME = "HRFFA/student_s256_96gb_r2_best_e0449_0.007970.pt"

    def __init__(self, scheme: HRFFAScheme = HRFFAScheme.IBUG68, input_range: ImageInputRange = ImageInputRange.ZERO_TO_ONE) -> None:
        super().__init__()
        if not isinstance(input_range, ImageInputRange):
            raise TypeError(f"input_range 必须为 ImageInputRange，实际为 {type(input_range).__name__}")

        self.scheme = scheme
        self.input_range = input_range
        self.network = _HRFFANetwork()

        checkpoint_path = hf_hub_download(
            repo_id=MODEL_REPOSITORY_ID,
            filename=self.WEIGHT_FILENAME,
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint.get("ema")
        if not isinstance(state_dict, dict):
            raise TypeError("HRFFA checkpoint 的 ema 必须为 state_dict")
        self.network.load_state_dict(state_dict, strict=True)
        self.eval().requires_grad_(False)

    def _prepare_input(self, images: Tensor) -> tuple[Tensor, int]:
        if images.ndim != 4:
            raise ValueError(f"images 必须为 NCHW，实际 shape={tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"images 必须为 RGB 三通道，实际 C={images.shape[1]}")
        height, width = images.shape[-2:]
        if height != width:
            raise ValueError(f"HRFFA 输入必须为正方形 crop，实际 H={height}, W={width}")

        x = images.float()
        match self.input_range:
            case ImageInputRange.ZERO_TO_255:
                x = x.div(127.5).sub(1.0)
            case ImageInputRange.ZERO_TO_ONE:
                x = x.mul(2.0).sub(1.0)
            case ImageInputRange.MINUS_ONE_TO_ONE:
                pass

        if height != self.INPUT_SIZE:
            x = F.interpolate(
                x,
                size=(self.INPUT_SIZE, self.INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return x, height

    @torch.inference_mode()
    def forward(self, images: Tensor, return_visibility: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """预测整头部 crop 的关键点。

        参数:
            images: ``(N, 3, H, H)`` RGB 张量。
            return_visibility: 是否同时返回三分类可见性标签。

        返回:
            关键点像素坐标 ``(N, K, 2)``；若请求可见性，则额外返回 ``(N, K)``
            的整型类别标签。
        """
        x, original_size = self._prepare_input(images)
        normalized_points, visibility_logits = self.network(x, self.scheme)
        landmarks = normalized_points * float(original_size)

        if return_visibility:
            visibility = visibility_logits.argmax(dim=-1)
            return landmarks, visibility
        return landmarks
