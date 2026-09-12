from math import ceil
from pathlib import Path
from typing import overload

import torch
import torch.nn.functional as F
from torch import Tensor

from .models.id_encoder import get_alignment_template
from .models.retinaface import RetinaFace, extract_landmarks
from .utils import ImageDirectory

# FFHQ 512 canonical coordinates -> InsightFace ArcFace 112 canonical coordinates.
# The linear part is scaled at runtime for the actual square FFHQ input resolution;
# translation remains in the 112x112 destination coordinate system.
FFHQ_TO_ARCFACE_112_AFFINE_512 = (
    (0.29209004227630586, 4.4496195577163076e-05, -18.93014028585668),
    (-4.4496195577163076e-05, 0.29209004227630586, -17.86002120510068),
)


def make_ffhq_to_arcface_112_grid(input_size: int, batch_size: int, device: torch.device) -> Tensor:
    """预计算标准 FFHQ aligned 图像到 112x112 canonical 坐标系的采样网格。

    该网格在固定训练分辨率、batch size 和 device 下完全不变，应在训练初始化阶段
    构造一次并复用，避免在每个 iteration 重复创建 affine/theta/grid。
    """
    if input_size <= 0:
        raise ValueError(f"input_size 必须为正数，实际为 {input_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正数，实际为 {batch_size}")

    affine = torch.tensor(FFHQ_TO_ARCFACE_112_AFFINE_512, device=device, dtype=torch.float32)
    affine[:, :2] *= 512.0 / float(input_size)
    theta = affine_to_grid_theta(affine, (input_size, input_size), 112, align_corners=False)
    theta = theta.expand(batch_size, -1, -1)
    return F.affine_grid(theta, (batch_size, 3, 112, 112), align_corners=False)


def transform_sampling_grid(grid: Tensor, theta: Tensor) -> Tensor:
    """将 normalized output->input 仿射变换作用到已有采样网格。

    ``grid`` 中的点先落在 canonical FFHQ normalized 坐标系，``theta`` 再将这些点映射
    到实际输入图像坐标系。这样可把“恢复到 FFHQ + FFHQ->ArcFace”合并为一次采样。
    """
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(f"grid 必须为 [B,H,W,2]，实际为 {tuple(grid.shape)}")
    if theta.ndim != 3 or theta.shape[1:] != (2, 3):
        raise ValueError(f"theta 必须为 [B,2,3]，实际为 {tuple(theta.shape)}")
    if grid.shape[0] != theta.shape[0]:
        raise ValueError(f"grid 与 theta batch size 不一致：{grid.shape[0]} != {theta.shape[0]}")
    if grid.device != theta.device:
        raise ValueError(f"grid 与 theta 必须位于同一 device：{grid.device} != {theta.device}")

    with torch.amp.autocast(device_type=grid.device.type, enabled=False):
        flat_grid = grid.float().reshape(grid.shape[0], -1, 2)
        theta = theta.float()
        transformed = torch.bmm(flat_grid, theta[:, :, :2].transpose(1, 2))
        transformed.add_(theta[:, :, 2].unsqueeze(1))
        return transformed.reshape_as(grid)


def ffhq_to_arcface_112(images: Tensor, grid: Tensor, *, padding_mode: str = "border") -> Tensor:
    """使用预计算网格将 FFHQ aligned 方形图像映射到 112x112 canonical 坐标系。"""
    if images.ndim != 4:
        raise ValueError(f"images must be NCHW, got shape={tuple(images.shape)}")
    if images.shape[2] != images.shape[3]:
        raise ValueError(f"FFHQ aligned input must be square, got {images.shape[2]}x{images.shape[3]}")
    if grid.ndim != 4 or grid.shape[0] != images.shape[0] or grid.shape[1:] != (112, 112, 2):
        raise ValueError(f"grid shape 必须为 ({images.shape[0]}, 112, 112, 2)，实际为 {tuple(grid.shape)}")
    if grid.device != images.device:
        raise ValueError(f"grid 与 images 必须位于同一 device：{grid.device} != {images.device}")
    if padding_mode not in {"zeros", "border", "reflection"}:
        raise ValueError(f"不支持的 padding_mode：{padding_mode}")

    # grid_sample 要求 input/grid dtype 一致；固定几何网格保留 FP32 精度。
    # 即使外层训练启用了 BF16 autocast，这一步也显式以 FP32 执行。
    with torch.amp.autocast(device_type=images.device.type, enabled=False):
        return F.grid_sample(images.float(), grid, mode="bilinear", padding_mode=padding_mode, align_corners=False)


def center_crop_and_resize(images: Tensor, crop_fraction: float = 0.102) -> Tensor:
    """Crop the same fraction from each edge and resize back to the input size."""
    if images.ndim != 4:
        raise ValueError(f"img must be NCHW, got shape={tuple(images.shape)}")
    if not 0.0 <= crop_fraction < 0.5:
        raise ValueError(f"crop_fraction must be in [0, 0.5), got {crop_fraction}")

    _, _, height, width = images.shape
    crop_h = int(height * crop_fraction)
    crop_w = int(width * crop_fraction)

    cropped = images[:, :, crop_h : height - crop_h, crop_w : width - crop_w]

    return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)


def compute_similarity_transform(src_points: Tensor, dst_points: Tensor) -> Tensor:
    """
    计算两组点之间的相似变换 (similarity transform)

    Args:
        src_points: [B, N, 2] 源点集
        dst_points: [B, N, 2] 目标点集

    Returns:
        affine: [B, 2, 3] 仿射变换矩阵，形式为 [sR | t]
    """
    p_c = src_points.mean(dim=1, keepdim=True)
    q_c = dst_points.mean(dim=1, keepdim=True)
    p_prime = src_points - p_c
    q_prime = dst_points - q_c

    H = torch.bmm(p_prime.transpose(1, 2), q_prime)

    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)

    det = torch.det(torch.bmm(V, U.transpose(-2, -1)))
    sign = det.sign()

    V_corrected = V.clone()
    V_corrected[..., -1] *= sign.unsqueeze(-1)
    S_corrected = S.clone()
    S_corrected[:, -1] *= sign
    R = torch.bmm(V_corrected, U.transpose(-2, -1))

    sigma_p = p_prime.pow(2).sum(dim=[1, 2])
    s = S_corrected.sum(dim=1) / sigma_p

    Rp_c = torch.bmm(R, p_c.transpose(1, 2)).squeeze(2)
    t = q_c.squeeze(1) - s[:, None] * Rp_c

    linear_part = s[:, None, None] * R
    translation_part = t.unsqueeze(-1)
    affine = torch.cat([linear_part, translation_part], dim=2)
    return affine


def affine_to_grid_theta(
    src_to_dst_affine: Tensor,
    in_hw: tuple[int, int],
    output_size: int,
    align_corners: bool = False,
) -> Tensor:

    if src_to_dst_affine.dim() == 2:
        src_to_dst_affine = src_to_dst_affine.unsqueeze(0)

    B = src_to_dst_affine.shape[0]
    device = src_to_dst_affine.device
    dtype = src_to_dst_affine.dtype

    H_in, W_in = in_hw
    H_out, W_out = output_size, output_size

    # 预计算缩放和偏移参数
    if align_corners:
        # norm = 2 * pix / (size - 1) - 1
        scale_in_x, scale_in_y = 2.0 / (W_in - 1), 2.0 / (H_in - 1)
        scale_out_x, scale_out_y = (W_out - 1) / 2.0, (H_out - 1) / 2.0
        offset_in_x, offset_in_y = -1.0, -1.0
        offset_out_x, offset_out_y = 0.5 * (W_out - 1), 0.5 * (H_out - 1)
    else:
        # norm = (2 * pix + 1) / size - 1
        scale_in_x, scale_in_y = 2.0 / W_in, 2.0 / H_in
        scale_out_x, scale_out_y = W_out / 2.0, H_out / 2.0
        offset_in_x, offset_in_y = (1.0 / W_in) - 1.0, (1.0 / H_in) - 1.0
        offset_out_x, offset_out_y = W_out / 2.0 - 0.5, H_out / 2.0 - 0.5

    # 获取 src->dst 变换的逆 (dst->src)
    # src_to_dst_affine: [[a, b, tx], [c, d, ty]]
    # 计算2x2部分的逆
    a, b = src_to_dst_affine[:, 0, 0], src_to_dst_affine[:, 0, 1]
    c, d = src_to_dst_affine[:, 1, 0], src_to_dst_affine[:, 1, 1]
    tx, ty = src_to_dst_affine[:, 0, 2], src_to_dst_affine[:, 1, 2]

    det = a * d - b * c
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    inv_tx = -(inv_a * tx + inv_b * ty)
    inv_ty = -(inv_c * tx + inv_d * ty)

    # 构建最终的theta矩阵
    # theta = scale_in * inv_affine * scale_out
    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)

    theta[:, 0, 0] = scale_in_x * (inv_a * scale_out_x)
    theta[:, 0, 1] = scale_in_x * (inv_b * scale_out_y)
    theta[:, 0, 2] = scale_in_x * (inv_a * offset_out_x + inv_b * offset_out_y + inv_tx) + offset_in_x

    theta[:, 1, 0] = scale_in_y * (inv_c * scale_out_x)
    theta[:, 1, 1] = scale_in_y * (inv_d * scale_out_y)
    theta[:, 1, 2] = scale_in_y * (inv_c * offset_out_x + inv_d * offset_out_y + inv_ty) + offset_in_y

    return theta


def make_alignment_grid_theta(
    source_landmarks: Tensor,
    alignment_template: Tensor,
    input_hw: tuple[int, int],
    output_size: int,
) -> Tensor:
    """只计算对齐采样所需的 output->input normalized theta，不执行图像重采样。"""
    if output_size <= 0:
        raise ValueError("output_size must be greater than 0")
    if source_landmarks.ndim != 3 or source_landmarks.shape[-1] != 2:
        raise ValueError(f"source_landmarks must have shape [N, P, 2], got {tuple(source_landmarks.shape)}")
    if alignment_template.ndim != 2 or alignment_template.shape[-1] != 2:
        raise ValueError(f"alignment_template must have shape [P, 2], got {tuple(alignment_template.shape)}")
    if source_landmarks.shape[1] != alignment_template.shape[0]:
        raise ValueError("source_landmarks and alignment_template must contain the same number of landmarks")
    if source_landmarks.shape[0] == 0:
        return source_landmarks.new_empty((0, 2, 3))

    src_to_dst_affine = compute_similarity_transform(
        source_landmarks,
        alignment_template.unsqueeze(0).expand(source_landmarks.shape[0], -1, -1),
    )
    return affine_to_grid_theta(src_to_dst_affine, input_hw, output_size, align_corners=False)


def invert_grid_theta(theta: Tensor) -> Tensor:
    """
    计算 grid theta 的逆变换

    Args:
        theta: [B, 2, 3] 归一化的仿射变换矩阵

    Returns:
        inv_theta: [B, 2, 3] 逆变换矩阵
    """
    # 提取2x2线性部分和平移部分
    A = theta[..., :2, :2]  # [B, 2, 2]
    t = theta[..., :2, 2]  # [B, 2]

    # 计算2x2矩阵的逆
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]  # [B]

    # 构建逆矩阵
    inv_A = torch.zeros_like(A)
    inv_A[..., 0, 0] = A[..., 1, 1] / det
    inv_A[..., 0, 1] = -A[..., 0, 1] / det
    inv_A[..., 1, 0] = -A[..., 1, 0] / det
    inv_A[..., 1, 1] = A[..., 0, 0] / det

    # 计算逆平移：-A^(-1) * t
    inv_t = -torch.matmul(inv_A, t.unsqueeze(-1)).squeeze(-1)  # [B, 2]

    # 组合成逆变换矩阵
    inv_theta = torch.cat([inv_A, inv_t.unsqueeze(-1)], dim=-1)  # [B, 2, 3]

    return inv_theta


def align_faces(
    images: tuple[Tensor, ...] | list[Tensor] | Tensor,
    source_landmarks: Tensor,
    segment_lengths: list[int],
    alignment_template: Tensor,
    output_size: int,
) -> tuple[Tensor, Tensor]:
    """
    对输入图片进行人脸对齐（仿射变换），输出对齐后的图片和仿射矩阵。

    参数:
        images: 每一张尺寸相同的图片为: Tensor(B,C,H,W), 图片之间尺寸不同的为: [Tensor(C,H,W)]
        source_landmarks: 每张图像检测到的对应关键点 [N, P, 2]: K:对应图像检测到的面部数量，(P, 2)单张面部的关键点
        alignment_template: 对齐目标模板坐标点 [P,2]
        output_size: 对齐后输出图片的尺寸

    返回:
        aligned_faces: 对齐后的人脸图片列表，每个元素形状为[face_count, C, H, W]
        grid_theta: 对应的仿射变换矩阵列表，每个元素形状为[N, 2, 3]

    异常:
        TypeError:
            输入类型不是torch.Tensor或list[Tensor]时抛出。

    说明:
        - 所有关键点都为像素坐标
        - 支持批量处理不同尺寸的图片。
        - 每张图片可能检测到多个人脸，返回每个人脸的对齐结果。
    """
    if output_size <= 0:
        raise ValueError("output_size must be greater than 0")
    if source_landmarks.ndim != 3 or source_landmarks.shape[-1] != 2:
        raise ValueError(f"source_landmarks must have shape [N, P, 2], got {tuple(source_landmarks.shape)}")
    if alignment_template.ndim != 2 or alignment_template.shape[-1] != 2:
        raise ValueError(f"alignment_template must have shape [P, 2], got {tuple(alignment_template.shape)}")
    if source_landmarks.shape[1] != alignment_template.shape[0]:
        raise ValueError("source_landmarks and alignment_template must contain the same number of landmarks")

    image_count = images.shape[0] if isinstance(images, Tensor) else len(images)
    if image_count == 0:
        raise ValueError("images must not be empty")
    if len(segment_lengths) != image_count:
        raise ValueError(f"segment_lengths length ({len(segment_lengths)}) must match image count ({image_count})")
    if sum(segment_lengths) != source_landmarks.shape[0]:
        raise ValueError("sum(segment_lengths) must match source_landmarks.shape[0]")

    aligned_batches: list[Tensor] = []
    grid_theta_batches: list[Tensor] = []
    for src_point, image in zip(torch.split(source_landmarks, segment_lengths), images):
        face_count = src_point.size(0)
        if face_count == 0:
            continue

        C, H, W = image.shape
        image = image.unsqueeze(0).expand(face_count, -1, -1, -1)

        grid_theta = make_alignment_grid_theta(src_point, alignment_template, (H, W), output_size)

        grid = F.affine_grid(grid_theta, [face_count, C, output_size, output_size], align_corners=False)
        faces = F.grid_sample(image, grid, align_corners=False, mode="bilinear")

        aligned_batches.append(faces)
        grid_theta_batches.append(grid_theta)

    if not aligned_batches:
        first_image = images[0]
        return (
            first_image.new_empty((0, first_image.shape[0], output_size, output_size)),
            source_landmarks.new_empty((0, 2, 3)),
        )

    return torch.cat(aligned_batches, dim=0), torch.cat(grid_theta_batches, dim=0)


@overload
def restore_faces_to_original(
    original_images: Tensor,
    aligned_faces: Tensor,
    grid_theta: Tensor,
    segment_lengths: list[int],
) -> Tensor: ...


@overload
def restore_faces_to_original(
    original_images: tuple[Tensor, ...] | list[Tensor],
    aligned_faces: Tensor,
    grid_theta: Tensor,
    segment_lengths: list[int],
) -> list[Tensor]: ...


def restore_faces_to_original(
    original_images: tuple[Tensor, ...] | list[Tensor] | Tensor,
    aligned_faces: Tensor,
    grid_theta: Tensor,
    segment_lengths: list[int],
) -> Tensor | list[Tensor]:

    image_count = original_images.shape[0] if isinstance(original_images, Tensor) else len(original_images)
    if len(segment_lengths) != image_count:
        raise ValueError(f"segment_lengths length ({len(segment_lengths)}) must match image count ({image_count})")
    if sum(segment_lengths) != aligned_faces.shape[0] or aligned_faces.shape[0] != grid_theta.shape[0]:
        raise ValueError("segment_lengths, aligned_faces, and grid_theta face counts must match")

    result: list[Tensor] = []
    for theta, original_image, face in zip(
        torch.split(grid_theta, segment_lengths),
        original_images,
        torch.split(aligned_faces, segment_lengths),
    ):
        face_count = theta.size(0)
        if face_count == 0:
            result.append(original_image)
            continue

        C, H, W = original_image.shape
        inverse_theta = invert_grid_theta(theta)

        original_image = original_image.unsqueeze(0)

        grid = F.affine_grid(inverse_theta, [face_count, C, H, W], align_corners=False)
        faces_on_canvas = F.grid_sample(face, grid, padding_mode="zeros", align_corners=False, mode="bicubic")

        # affine_grid is independent of the channel count, so reuse it for masks.
        sum_faces = faces_on_canvas.sum(dim=0, keepdim=True)
        ones = torch.ones_like(face[:, :1])
        masks_on_canvas = F.grid_sample(ones, grid, padding_mode="zeros", align_corners=False, mode="bicubic")

        # 各像素被覆盖的次数（权重图），避免除以零
        weight = masks_on_canvas.sum(dim=0, keepdim=True).clamp(min=1e-6)  # [1, 1, H, W]

        # 加权平均后的人脸区域
        avg_faces = sum_faces / weight  # [1, C, H, W]

        # 仅在有人脸覆盖的区域替换原图，其余保留原图像素
        binary_mask = (weight > 0.5).float()  # [1, 1, H, W]，广播到 C 通道
        original_image.copy_(avg_faces * binary_mask + original_image * (1.0 - binary_mask))

        # 恢复原图的 batch 维度（squeeze 掉之前 unsqueeze_ 的维度）
        result.append(original_image.squeeze_(0))

    if isinstance(original_images, Tensor):
        # 每一帧已通过 copy_ 写回输入，无需再次 stack 一份全分辨率视频。
        return original_images

    return result


class _FaceExtractorBase:
    def __init__(
        self,
        batch_size: int = 16,
        output_size: int = 512,
        confidence_threshold: float = 0.99,
        iou_threshold: float = 0.2,
        min_face_size: tuple[int, int] = (256, 256),
        device: str | torch.device = "cuda",
        use_mobilenet: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if output_size <= 0:
            raise ValueError("output_size must be greater than 0")

        self.device = torch.device(device)
        self.face_detector = RetinaFace(use_mobilenet=use_mobilenet).to(device=self.device).eval()

        alignment_template = get_alignment_template(output_size)
        self.alignment_template = torch.as_tensor(alignment_template, device=self.device, dtype=torch.float32)

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.min_face_size = min_face_size
        self.batch_size = batch_size
        self.output_size = output_size

    def align(self, images: Tensor | list[Tensor] | tuple[Tensor, ...]) -> tuple[Tensor, Tensor, list[int]]:
        detections, segment_lengths = self.face_detector.detect(
            images,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            min_face_size=self.min_face_size,
        )
        source_landmarks = extract_landmarks(detections)
        aligned_faces, grid_theta = align_faces(
            images,
            source_landmarks,
            segment_lengths,
            self.alignment_template,
            self.output_size,
        )
        return aligned_faces, grid_theta, segment_lengths


class VideoFaceExtractor(_FaceExtractorBase):
    def __init__(
        self,
        video_path: str | Path,
        batch_size: int = 16,
        output_size: int = 512,
        confidence_threshold: float = 0.99,
        iou_threshold: float = 0.2,
        min_face_size: tuple[int, int] = (256, 256),
        device: str | torch.device = "cuda",
        use_mobilenet: bool = False,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            output_size=output_size,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            min_face_size=min_face_size,
            device=device,
            use_mobilenet=use_mobilenet,
        )
        if not Path(video_path).exists():
            raise FileNotFoundError(video_path)

        from torchcodec.decoders import VideoDecoder

        self.decoder = VideoDecoder(video_path, device=self.device)

    @property
    def frame_count(self) -> int:
        return len(self.decoder)

    def __len__(self) -> int:
        return ceil(self.frame_count / self.batch_size)

    def __iter__(self):
        total_frames = self.frame_count

        for i in range(0, total_frames, self.batch_size):
            j = min(i + self.batch_size, total_frames)

            batch_frames = self.decoder[i:j].data.to(device=self.device, dtype=torch.float)  # [0.0, 255.0]
            aligned_faces, grid_theta, segment_lengths = self.align(batch_frames)
            frame_count = j - i
            yield batch_frames, aligned_faces, grid_theta, segment_lengths, frame_count


class ImageDirectoryFaceExtractor(_FaceExtractorBase):
    def __init__(
        self,
        folder: str | Path,
        batch_size: int = 16,
        output_size: int = 512,
        confidence_threshold: float = 0.99,
        iou_threshold: float = 0.2,
        min_face_size: tuple[int, int] = (256, 256),
        device: str | torch.device = "cuda",
        use_mobilenet: bool = False,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            output_size=output_size,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            min_face_size=min_face_size,
            device=device,
            use_mobilenet=use_mobilenet,
        )

        self.image_folder = ImageDirectory(folder)

    @property
    def image_count(self) -> int:
        return len(self.image_folder)

    def __len__(self) -> int:
        return ceil(self.image_count / self.batch_size)

    def __iter__(self):
        for images, stems in self.image_folder.iter_tensor_batches(
            batch_size=self.batch_size,
            return_stem=True,
            device=self.device,
            dtype=torch.float,
        ):
            aligned_faces, grid_theta, segment_lengths = self.align(images)
            image_count = len(segment_lengths)
            yield images, stems, aligned_faces, grid_theta, segment_lengths, image_count


if __name__ == "__main__":
    pass
    # FaceAlign("/opt/share/deepfake/dataset_1/vggface2_hq512/", batch_size=64).process()
