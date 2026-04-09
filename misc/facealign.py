from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import torch
import torch.nn.functional as F
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from .models.idencoder import get_align_landmarks
from .models.retinaface import RetinaFace, get_pts


from misc.utils import ImageFolder


class AffineSmoother:
    """
    对 [sR | t] 形式的仿射矩阵做 EMA 平滑。
    支持多人脸追踪（按 face_id 维护独立状态）。
    """

    def __init__(self, alpha: float = 0.7):
        # alpha 越大越跟随当前帧，越小越平滑
        self.alpha = alpha
        self.state: dict[int, Tensor] = {}  # face_id -> [2, 3]

    def update(self, face_id: int, M: Tensor) -> Tensor:
        """
        M: [2, 3]
        返回平滑后的 [2, 3]
        """
        if face_id not in self.state:
            self.state[face_id] = M.clone()
            return M
        smoothed = self.alpha * M + (1 - self.alpha) * self.state[face_id]
        self.state[face_id] = smoothed
        return smoothed

    def reset(self, face_id: int | None = None):
        if face_id is None:
            self.state.clear()
        else:
            self.state.pop(face_id, None)


def zoom_in(img: torch.Tensor, zoom: float = 0.2) -> torch.Tensor:
    """
    img  : [B, C, H, W]
    zoom : 裁掉的比例，0.2 表示四周各裁 10%
    """
    _, _, H, W = img.shape
    ch = int(H * zoom / 2)
    cw = int(W * zoom / 2)

    cropped = img[:, :, ch : H - ch, cw : W - cw]

    return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)


def compute_similarity_transform(points_x: torch.Tensor, points_y: torch.Tensor) -> torch.Tensor:
    """
    计算两组点之间的相似变换 (similarity transform)

    Args:
        points_x: [B, N, 2] 源点集
        points_y: [B, N, 2] 目标点集

    Returns:
        affine: [B, 2, 3] 仿射变换矩阵，形式为 [sR | t]
    """
    p_c = points_x.mean(dim=1, keepdim=True)
    q_c = points_y.mean(dim=1, keepdim=True)
    p_prime = points_x - p_c
    q_prime = points_y - q_c

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


def normalize_theta(affine_src2dst, in_hw, out_hw, align_corners=True):

    if affine_src2dst.dim() == 2:
        affine_src2dst = affine_src2dst.unsqueeze(0)

    B = affine_src2dst.shape[0]
    device = affine_src2dst.device
    dtype = affine_src2dst.dtype

    H_in, W_in = in_hw
    H_out, W_out = out_hw, out_hw

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
    # affine_src2dst: [[a, b, tx], [c, d, ty]]
    # 计算2x2部分的逆
    a, b = affine_src2dst[:, 0, 0], affine_src2dst[:, 0, 1]
    c, d = affine_src2dst[:, 1, 0], affine_src2dst[:, 1, 1]
    tx, ty = affine_src2dst[:, 0, 2], affine_src2dst[:, 1, 2]

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


def invert_normalized_theta(theta: torch.Tensor) -> torch.Tensor:
    """
    计算 normalized_theta 的逆变换

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


class FaceAligner:
    def __init__(self, smooth_alpha: float = 0.5):
        self.smoother = AffineSmoother(alpha=smooth_alpha)

    def __call__(
        self,
        images: tuple[Tensor] | list[Tensor] | Tensor,
        src_pts: Tensor,
        src_pts_offset: list[int],
        dst_pts: Tensor,
        out_size: int,
    ):
        """
        对输入图片进行人脸对齐（仿射变换），输出对齐后的图片和仿射矩阵。

        参数:
            images: 每一张尺寸相同的图片为: Tensor(B,C,H,W), 图片之间尺寸不同的为: [Tensor(C,H,W)]
            src_pts: 每张图像检测到的对应关键点 [N, P, 2]: K:对应图像检测到的面部数量，(P, 2)单张面部的关键点
            dst_pts: 对齐目标模板坐标点 [P,2]
            out_size: 对齐后输出图片的尺寸

        返回:
            aligned_list: 对齐后的人脸图片列表，每个元素形状为[N, C, H, W]
            norm_theta: 对应的仿射变换矩阵列表，每个元素形状为[N, 2, 3]

        异常:
            TypeError:
                输入类型不是torch.Tensor或list[Tensor]时抛出。

        说明:
            - 所有关键点都为像素坐标
            - 支持批量处理不同尺寸的图片。
            - 每张图片可能检测到多个人脸，返回每个人脸的对齐结果。
        """
        aligned: list[Tensor] = []
        norm_theta: list[Tensor] = []
        for src_point, image in zip(torch.split(src_pts, src_pts_offset), images):
            N = src_point.size(0)
            if N == 0:
                continue

            C, H, W = image.shape
            image = image.unsqueeze(0).expand(N, -1, -1, -1)

            mats = compute_similarity_transform(src_point, dst_pts.unsqueeze(0).expand(N, -1, -1))
            mats = normalize_theta(mats, (H, W), out_size, align_corners=True)

            for idx in range(N):
                mats[idx : idx + 1] = self.smoother.update(idx, mats[idx : idx + 1])

            grid = F.affine_grid(mats, (N, C, out_size, out_size), align_corners=True)
            faces = F.grid_sample(image, grid, align_corners=True, mode="bilinear")

            aligned.append(faces)
            norm_theta.append(mats)

        if len(aligned) == 0:
            aligned = torch.zeros((0, 3, out_size, out_size), dtype=torch.float, device=src_point.device)
            norm_theta = torch.zeros((0, 2, 3), dtype=torch.float, device=src_point.device)
        else:
            aligned = torch.cat(aligned, dim=0)
            norm_theta = torch.cat(norm_theta, dim=0)

        return aligned, norm_theta


def face_align_batch(
    images: tuple[Tensor] | list[Tensor] | Tensor,
    src_pts: Tensor,
    src_pts_offset: list[int],
    dst_pts: Tensor,
    out_size: int,
) -> tuple[Tensor, Tensor]:
    """
    对输入图片进行人脸对齐（仿射变换），输出对齐后的图片和仿射矩阵。

    参数:
        images: 每一张尺寸相同的图片为: Tensor(B,C,H,W), 图片之间尺寸不同的为: [Tensor(C,H,W)]
        src_pts: 每张图像检测到的对应关键点 [N, P, 2]: K:对应图像检测到的面部数量，(P, 2)单张面部的关键点
        dst_pts: 对齐目标模板坐标点 [P,2]
        out_size: 对齐后输出图片的尺寸

    返回:
        aligned_list: 对齐后的人脸图片列表，每个元素形状为[N, C, H, W]
        norm_theta: 对应的仿射变换矩阵列表，每个元素形状为[N, 2, 3]

    异常:
        TypeError:
            输入类型不是torch.Tensor或list[Tensor]时抛出。

    说明:
        - 所有关键点都为像素坐标
        - 支持批量处理不同尺寸的图片。
        - 每张图片可能检测到多个人脸，返回每个人脸的对齐结果。
    """

    aligned: list[Tensor] = []
    norm_theta: list[Tensor] = []
    for src_point, image in zip(torch.split(src_pts, src_pts_offset), images):
        N = src_point.size(0)
        if N == 0:
            continue

        C, H, W = image.shape
        image = image.unsqueeze(0).expand(N, -1, -1, -1)

        mats = compute_similarity_transform(src_point, dst_pts.unsqueeze(0).expand(N, -1, -1))
        mats = normalize_theta(mats, (H, W), out_size, align_corners=True)

        grid = F.affine_grid(mats, (N, C, out_size, out_size), align_corners=True)
        faces = F.grid_sample(image, grid, align_corners=True, mode="bilinear")

        aligned.append(faces)
        norm_theta.append(mats)

    if len(aligned) == 0:
        aligned = torch.zeros((0, 3, out_size, out_size), dtype=torch.float, device=src_point.device)
        norm_theta = torch.zeros((0, 2, 3), dtype=torch.float, device=src_point.device)
    else:
        aligned = torch.cat(aligned, dim=0)
        norm_theta = torch.cat(norm_theta, dim=0)

    return aligned, norm_theta


def restore_faces_to_original(
    org_images: tuple[Tensor] | list[Tensor] | Tensor,
    aligned: list[Tensor],
    norm_theta: list[Tensor],
    offset: list[int],
) -> Tensor | list[Tensor]:

    result: list[Tensor] = []
    for mat, org_image, face in zip(
        torch.split(norm_theta, offset),
        org_images,
        torch.split(aligned, offset),
    ):
        N = mat.size(0)
        if N == 0:
            result.append(org_image)
            continue

        C, H, W = org_image.shape
        inv_aff_mat = invert_normalized_theta(mat)

        org_image = org_image.unsqueeze(0)

        grid = F.affine_grid(inv_aff_mat, [N, C, H, W], align_corners=False)
        faces_on_canvas = F.grid_sample(face, grid, padding_mode="zeros", align_corners=False, mode="bicubic")

        # 聚合多张人脸到单张
        sum_faces = faces_on_canvas.sum(dim=0, keepdim=True)  # [1, C, H, W]

        # 生成每张人脸的 mask（非零区域），用于计算覆盖权重
        ones = torch.ones(N, 1, face.shape[2], face.shape[3], device=face.device, dtype=face.dtype)
        mask_grid = F.affine_grid(inv_aff_mat, [N, 1, H, W], align_corners=False)
        masks_on_canvas = F.grid_sample(ones, mask_grid, padding_mode="zeros", align_corners=False, mode="bicubic")

        # 各像素被覆盖的次数（权重图），避免除以零
        weight = masks_on_canvas.sum(dim=0, keepdim=True).clamp(min=1e-6)  # [1, 1, H, W]

        # 加权平均后的人脸区域
        avg_faces = sum_faces / weight  # [1, C, H, W]

        # 仅在有人脸覆盖的区域替换原图，其余保留原图像素
        binary_mask = (weight > 0.5).float()  # [1, 1, H, W]，广播到 C 通道
        org_image.copy_(avg_faces * binary_mask + org_image * (1.0 - binary_mask))

        # 恢复原图的 batch 维度（squeeze 掉之前 unsqueeze_ 的维度）
        result.append(org_image.squeeze_(0))

    if isinstance(org_images, Tensor):
        result = torch.stack(result, dim=0)

    return result


def save_image(img: Tensor, fp: Path):
    img = img[[2, 1, 0], :, :]  # RGB -> BGR
    img = img.permute(1, 2, 0)  # CHW -> HWC
    img_cpu = img.to(device="cpu", dtype=torch.uint8).numpy()
    cv2.imwrite(fp, img_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3])


class FaceAlignBase:
    def __init__(
        self,
        batch_size: int = 16,
        align_size: int = 512,
        conf_thresh: float = 0.99,
        iou_thresh: float = 0.2,
        min_box_size: tuple[int, int] = (256, 256),
        device: str = "cuda",
    ) -> None:

        self.device = torch.device(device)

        self.FaceDetector = RetinaFace().to(device=device)

        dst_pts = get_align_landmarks(align_size)
        self.dst_pts = torch.tensor(dst_pts, device=device)

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.min_box_size = min_box_size
        self.batch_size = batch_size
        self.align_size = align_size

        self.aligner = FaceAligner(0.5)

    def align(self, images: Tensor | list[Tensor]) -> tuple[Tensor, Tensor, list[int]]:
        detected, offset = self.FaceDetector.detector(images, conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh, min_box_size=self.min_box_size)
        src_pts = get_pts(detected)
        align_face, norm_theta = self.aligner(images, src_pts, offset, self.dst_pts, self.align_size)
        return align_face, norm_theta, offset


class FaceExtractorVideo(FaceAlignBase):
    def __init__(
        self,
        vfp: str | Path,
        batch_size: int = 16,
        align_size: int = 512,
        conf_thresh: float = 0.99,
        iou_thresh: float = 0.2,
        min_box_size: tuple[int, int] = (256, 256),
        device: str = "cuda",
    ):
        super().__init__(
            batch_size=batch_size,
            align_size=align_size,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            min_box_size=min_box_size,
            device=device,
        )
        if not Path(vfp).exists():
            raise FileNotFoundError(vfp)

        self.decoder = VideoDecoder(vfp, device=self.device.type)

    def __len__(self):
        return len(self.decoder)

    def __iter__(self):

        total = len(self)

        for i in range(0, total, self.batch_size):
            j = min(i + self.batch_size, total)

            batch_frames = self.decoder[i:j].data.to(device=self.device, dtype=torch.float)  # [0.0, 255.0]
            align_face, norm_theta, offset = self.align(batch_frames)
            nb = j - i
            yield batch_frames, align_face, norm_theta, offset, nb


class FaceExtractorFolder(FaceAlignBase):
    def __init__(
        self,
        fp: str,
        batch_size: int = 16,
        align_size: int = 512,
        conf_thresh: float = 0.99,
        iou_thresh: float = 0.2,
        min_box_size: tuple[int, int] = (256, 256),
        device: str = "cuda",
    ):
        super().__init__(
            batch_size=batch_size,
            align_size=align_size,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            min_box_size=min_box_size,
            device=device,
        )

        self.fp = ImageFolder(fp)

    def __len__(self):
        return len(self.fp)

    def __iter__(self):
        for images, fn in self.fp.iter_batch_tensor(batch_size=self.batch_size, label=True, device=self.device, dtype=torch.float):
            align_face, norm_theta, offset = self.align(images)
            nb = len(offset)
            yield images, fn, align_face, norm_theta, offset, nb


if __name__ == "__main__":
    pass
    # FaceAlign("/opt/share/deepfake/dataset_1/vggface2_hq512/", batch_size=64).process()
