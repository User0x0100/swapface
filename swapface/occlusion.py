"""在线合成遮挡和 ID Loss 可见区域梯度门控；只依赖 PyTorch/Pillow。

自然遮挡使用离线裁剪的 RGBA 素材；没有素材时采用 GPU 随机纹理形状，
不在每步训练中执行 OpenCV、文件 I/O 或新的人脸检测。
"""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


def gate_identity_gradient(images: Tensor, mask: Tensor | None) -> Tensor:
    """前向严格等于 images，反向仅对 ID 分支施加空间梯度门控。"""
    if mask is None:
        return images
    detached = images.detach()
    return detached + (images - detached) * mask.detach()


class OcclusionAugmentor:
    """在 Generator 前向前为目标人脸合成遮挡，并生成独立的身份梯度掩码。"""

    def __init__(
        self,
        probability: float,
        assets_dir: str,
        device: torch.device,
        mask_model: Callable[[Tensor], Tensor] | None = None,
        asset_size: int = 192,
        *,
        size_range_pct: tuple[float, float],
        center_x_range_pct: tuple[float, float],
        center_y_range_pct: tuple[float, float],
        rotation_range_deg: tuple[float, float],
    ) -> None:
        if not 0.0 < probability <= 1.0:
            raise ValueError("occlusion.prob 必须在 (0, 1] 内")
        self.probability = probability
        self.device = device
        self.size_range_pct = tuple(size_range_pct)
        self.center_x_range_pct = tuple(center_x_range_pct)
        self.center_y_range_pct = tuple(center_y_range_pct)
        self.rotation_range_deg = tuple(rotation_range_deg)
        if mask_model is None:
            from misc.models.face_mask import FaceMasker

            # 当前 OCC checkpoint 在 eval() 会输出全零，按已有 swapper 的 train 模式运行。
            # 全程 no_grad + 冻结参数；BatchNorm 的 batch 统计依旧参与推理。
            mask_model = FaceMasker(mode="occlusion").to(device).train().requires_grad_(False)
        self.mask_model = mask_model
        self.assets: Tensor | None = None
        if assets_dir:
            root = Path(assets_dir).expanduser().resolve(strict=True)
            paths = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".webp"})
            if not paths:
                raise ValueError(f"遮挡目录没有 PNG/WebP RGBA 图片：{root}")
            images = []
            for path in paths:
                with Image.open(path) as image:
                    if "A" not in image.getbands():
                        raise ValueError(f"遮挡素材缺少 Alpha 通道：{path}")
                    rgba = image.convert("RGBA")
                    bounds = rgba.getchannel("A").getbbox()
                    if bounds is None:
                        raise ValueError(f"遮挡素材完全透明：{path}")
                    rgba = rgba.crop(bounds)
                    # thumbnail() 只会缩小，不会放大小素材；会令实际覆盖率远小于配置值。
                    # 将素材最长边统一映射到 asset_size，使 size_range_pct 有一致的几何含义。
                    factor = asset_size / max(rgba.size)
                    rgba = rgba.resize((max(1, round(rgba.width * factor)), max(1, round(rgba.height * factor))), Image.Resampling.LANCZOS)
                    square = Image.new("RGBA", (asset_size, asset_size))
                    square.paste(rgba, ((asset_size - rgba.width) // 2, (asset_size - rgba.height) // 2))
                    images.append(torch.from_numpy(np.asarray(square, dtype=np.uint8).copy()).permute(2, 0, 1))
            # 缓存预乘 Alpha 的 RGB：几何重采样时一起插值，避免透明区 RGB=0
            # 造成半透明轮廓颜色二次衰减（黑边/暗边）。Alpha 通道本身不变。
            self.assets = torch.stack(images).to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32).div_(255)
            self.assets[:, :3].mul_(self.assets[:, 3:4])

    @torch.no_grad()
    def apply(self, dst: Tensor) -> tuple[Tensor, Tensor]:
        """返回带遮挡 dst，以及 ID 梯度门控 (B,1,H,W)。非遮挡样本门控恒为 1。"""
        batch, _, height, width = dst.shape
        selected = (torch.rand(batch, device=dst.device) < self.probability).nonzero().flatten()
        gate = torch.ones((batch, 1, height, width), device=dst.device, dtype=dst.dtype)
        if selected.numel() == 0:
            return dst, gate

        clean = dst.index_select(0, selected)
        count = clean.shape[0]
        # 素材画布长边相对于训练图像边长的百分比；中心偏移以图像中心为 0。
        # affine_grid 的坐标范围为 [-1,1]，因此中心偏移百分比乘以 2/100。
        scale = torch.empty((count,), device=dst.device).uniform_(*self.size_range_pct).mul_(0.01)
        angle = torch.empty((count,), device=dst.device).uniform_(*self.rotation_range_deg).deg2rad_()
        cx = torch.empty((count,), device=dst.device).uniform_(*self.center_x_range_pct).mul_(0.02)
        cy = torch.empty((count,), device=dst.device).uniform_(*self.center_y_range_pct).mul_(0.02)
        c, s = angle.cos(), angle.sin()
        # 从目标像素归一化坐标映射到素材的 [-1,1] 区域。
        theta = torch.zeros((count, 2, 3), device=dst.device)
        theta[:, 0, 0], theta[:, 0, 1] = c / scale, s / scale
        theta[:, 1, 0], theta[:, 1, 1] = -s / scale, c / scale
        theta[:, 0, 2] = -(c * cx + s * cy) / scale
        theta[:, 1, 2] = (s * cx - c * cy) / scale
        grid = F.affine_grid(theta, clean.shape, align_corners=False)

        if self.assets is not None:
            indices = torch.randint(self.assets.shape[0], (count,), device=dst.device)
            rgba = F.grid_sample(self.assets.index_select(0, indices).float(), grid, padding_mode="zeros", align_corners=False)
            alpha = rgba[:, 3:4].to(dst.dtype)
            # 素材采用预乘 Alpha 的 [0,1] RGB。转换到 [-1,1] 后的预乘颜色
            # 为 (2*RGBpremul - alpha)，因此无需再对纹理重复乘以 alpha。
            premultiplied = rgba[:, :3].to(dst.dtype).mul(2).sub(alpha)
            occluded = clean * (1 - alpha) + premultiplied
        else:
            # RandOcc：单一 GPU 网格上的不规则椭圆 + 随机颜色/低频纹理。
            x, y = grid[..., 0:1], grid[..., 1:2]
            radius = (x.square() + y.square()).clamp_min(1e-8).sqrt()
            direction = torch.atan2(y, x)
            phase = torch.rand((count, 1, 1, 1), device=dst.device) * 6.2831853
            boundary = 0.80 + 0.12 * torch.sin(direction * 5 + phase) + 0.07 * torch.sin(direction * 3 - phase)
            alpha = ((boundary - radius) * 35).clamp(0, 1).permute(0, 3, 1, 2).to(dst.dtype)
            color = torch.rand((count, 3, 1, 1), device=dst.device, dtype=dst.dtype).mul(1.5).sub(0.75)
            noise = F.interpolate(torch.rand((count, 3, 8, 8), device=dst.device, dtype=dst.dtype), size=(height, width), mode="bilinear", align_corners=False)
            texture = (color + (noise - 0.5) * 0.35).clamp(-1, 1)
            occluded = clean + alpha * (texture - clean)
        result = dst.index_copy(0, selected, occluded)

        # OCC 只推理实际遮挡的干净目标，避免遮挡本身影响人脸可见区域判定。
        # 已知的合成 Alpha 精确排除新增遮挡；OCC 负责排除背景/原始遮挡区域。
        faces_224 = F.interpolate(clean.float(), size=(224, 224), mode="bilinear", align_corners=False)
        visible_224 = self.mask_model(faces_224)
        visible = F.interpolate(visible_224.float(), size=(height, width), mode="bilinear", align_corners=False).clamp(0, 1)
        allowed = visible * (1 - alpha.float())
        allowed = F.avg_pool2d(allowed, kernel_size=3, stride=1, padding=1).to(dst.dtype)
        gate = gate.index_copy(0, selected, allowed)
        return result, gate
