"""HRFFA 高角度鲁棒人脸关键点模型的 PyTorch 推理封装。

模型来源：PINTO0309/High-Angle_Robust_Fast_FaceAlignment。
当前默认使用 ViT-T/16、256×256 输入的学生模型。HRFFA 的训练输入是整头部
(head crop)，而不是传统紧致人脸 crop；推荐外部检测器使用头部框，并按长边扩展
5% 后构造正方形 crop，与上游训练几何保持一致。
"""

from __future__ import annotations

import io
import tarfile
from enum import Enum
from pathlib import Path
from urllib.request import urlopen

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

from ...models import MODEL_REPOSITORY_ID, ImageInputRange
from .dinov3 import Dinov3ViTL16Backbone
from .vit import ViTTiny


class HRFFAModelVariant(Enum):
    """HRFFA inference model variant."""

    VITT_256 = "vitt-256"
    VITL_320 = "vitl-320"


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


class _HRFFAStudentBackbone(nn.Module):
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
    """HRFFA point-query head shared by student and teacher variants."""

    def __init__(self, backbone: nn.Module, backbone_dim: int, decoder_layers: int) -> None:
        super().__init__()
        self.backbone = backbone
        d_model = 256

        self.input_proj = nn.Linear(backbone_dim, d_model)
        self.cls_proj = nn.Linear(backbone_dim, d_model)
        self.queries = nn.ParameterDict({scheme.value: nn.Parameter(torch.empty(count, d_model, dtype=torch.float32)) for scheme, count in SCHEME_LANDMARK_COUNTS.items()})

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers, norm=nn.LayerNorm(d_model))
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
    """HRFFA whole-head landmark model.

    ``VITT_256`` is the lightweight student and remains the default. ``VITL_320`` is
    the highest-accuracy clean_v3 teacher using the official DINOv3 ViT-L/16 runtime
    implementation. Both accept an arbitrary square RGB crop and restore predicted
    coordinates to the caller's original pixel size.

    The teacher checkpoint is downloaded directly from the upstream HRFFA GitHub
    release into ``~/.cache/swap/hrffa`` on first use. The split release archive is
    streamed and only the first ``clean_v3_best_*.pt`` member is retained.
    """

    STUDENT_INPUT_SIZE = 256
    TEACHER_INPUT_SIZE = 320
    STUDENT_WEIGHT_FILENAME = "HRFFA/student_s256_96gb_r2_best_e0449_0.007970.pt"
    TEACHER_RELEASE_BASE = "https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment/releases/download/weights"
    TEACHER_RELEASE_PARTS = tuple(f"clean_v3.tar.gz.part{i:02d}" for i in range(4))

    def __init__(
        self,
        scheme: HRFFAScheme = HRFFAScheme.IBUG68,
        input_range: ImageInputRange = ImageInputRange.ZERO_TO_ONE,
        variant: HRFFAModelVariant = HRFFAModelVariant.VITT_256,
        teacher_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(input_range, ImageInputRange):
            raise TypeError(f"input_range 必须为 ImageInputRange，实际为 {type(input_range).__name__}")
        if not isinstance(variant, HRFFAModelVariant):
            raise TypeError(f"variant 必须为 HRFFAModelVariant，实际为 {type(variant).__name__}")

        self.scheme = scheme
        self.input_range = input_range
        self.variant = variant

        match variant:
            case HRFFAModelVariant.VITT_256:
                self.input_size = self.STUDENT_INPUT_SIZE
                self.network = _HRFFANetwork(_HRFFAStudentBackbone(), backbone_dim=192, decoder_layers=3)
                checkpoint_path = hf_hub_download(repo_id=MODEL_REPOSITORY_ID, filename=self.STUDENT_WEIGHT_FILENAME)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            case HRFFAModelVariant.VITL_320:
                self.input_size = self.TEACHER_INPUT_SIZE
                self.network = _HRFFANetwork(Dinov3ViTL16Backbone(patch_instance_norm=True), backbone_dim=1024, decoder_layers=4)
                checkpoint_path = Path(teacher_checkpoint).expanduser() if teacher_checkpoint is not None else self._download_teacher_checkpoint()
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        state_dict = checkpoint.get("ema") or checkpoint.get("model")
        if not isinstance(state_dict, dict):
            raise TypeError("HRFFA checkpoint 必须包含 ema 或 model state_dict")
        self.network.load_state_dict(state_dict, strict=True)
        self.eval().requires_grad_(False)

    @classmethod
    def _download_teacher_checkpoint(cls) -> Path:
        cache_dir = Path.home() / ".cache" / "swap" / "hrffa"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = sorted(cache_dir.glob("clean_v3_best_*.pt"))
        if cached:
            return cached[-1]

        class _MultipartStream(io.RawIOBase):
            def __init__(self) -> None:
                super().__init__()
                self.part_index = 0
                self.response = None

            def readable(self) -> bool:
                return True

            def readinto(self, buffer) -> int:
                view = memoryview(buffer)
                written = 0
                while written < len(view):
                    if self.response is None:
                        if self.part_index >= len(cls.TEACHER_RELEASE_PARTS):
                            break
                        url = f"{cls.TEACHER_RELEASE_BASE}/{cls.TEACHER_RELEASE_PARTS[self.part_index]}"
                        self.response = urlopen(url)
                        self.part_index += 1
                    chunk = self.response.read(len(view) - written)
                    if not chunk:
                        self.response.close()
                        self.response = None
                        continue
                    view[written : written + len(chunk)] = chunk
                    written += len(chunk)
                return written

            def close(self) -> None:
                if self.response is not None:
                    self.response.close()
                    self.response = None
                super().close()

        stream = _MultipartStream()
        buffered = io.BufferedReader(stream, buffer_size=8 * 1024 * 1024)
        try:
            with tarfile.open(fileobj=buffered, mode="r|gz") as archive:
                for member in archive:
                    name = Path(member.name).name
                    if not member.isfile() or not (name.startswith("clean_v3_best_") and name.endswith(".pt")):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"无法读取 teacher checkpoint: {member.name}")
                    output = cache_dir / name
                    temporary = output.with_suffix(output.suffix + ".part")
                    with temporary.open("wb") as file:
                        while chunk := extracted.read(8 * 1024 * 1024):
                            file.write(chunk)
                    temporary.replace(output)
                    return output
        finally:
            buffered.close()
        raise FileNotFoundError("clean_v3 release archive 中未找到 clean_v3_best_*.pt")

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
                x = x.div(255.0)
            case ImageInputRange.ZERO_TO_ONE:
                pass
            case ImageInputRange.MINUS_ONE_TO_ONE:
                x = x.add(1.0).mul(0.5)

        match self.variant:
            case HRFFAModelVariant.VITT_256:
                x = x.sub(0.5).div(0.5)
            case HRFFAModelVariant.VITL_320:
                mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
                std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
                x = x.sub(mean).div(std)

        if height != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False, antialias=True)
        return x, height

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """返回输入尺寸下的关键点坐标与 visibility logits。"""
        x, original_size = self._prepare_input(images)
        normalized_points, visibility_logits = self.network(x, self.scheme)
        landmarks = normalized_points * float(original_size)
        return landmarks, visibility_logits
