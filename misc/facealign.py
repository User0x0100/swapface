import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from torch import autocast, nn, Tensor
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm


from .models.idencoder import get_align_landmarks
from .models.retinaface import RetinaFace, get_pts, batch_resize_and_pad_varsize, iter_detections


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
    H_out, W_out = out_hw

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


def face_align_batch(
    images: tuple[Tensor] | list[Tensor] | Tensor, src_pts: Tensor, src_pts_offset: list[int], dst_pts: Tensor, out_size: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """
    对输入图片进行人脸对齐（仿射变换），输出对齐后的图片和仿射矩阵。

    参数:
        images: 每一张尺寸相同的图片为: Tensor(B,C,H,W), 图片之间尺寸不同的为: [Tensor(C,H,W)]
        src_pts: 每张图像检测到的对应关键点 [N, P, 2]: K:对应图像检测到的面部数量，(P, 2)单张面部的关键点
        dst_pts: 对齐目标模板坐标点 [P,2]
        out_size: 对齐后输出图片的尺寸 (height, width)

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
    for src_point, image in zip(iter_detections(src_pts, src_pts_offset), images):

        N = src_point.size(0)
        if N == 0:
            continue

        C, H, W = image.shape
        image = image.unsqueeze(0).expand(N, -1, -1, -1)

        mats = compute_similarity_transform(src_point, dst_pts.unsqueeze(0).expand(N, -1, -1))
        mats = normalize_theta(mats, (H, W), out_size, align_corners=True)

        grid = F.affine_grid(mats, (N, C, *out_size), align_corners=True)
        faces = F.grid_sample(image, grid, align_corners=True, mode="bilinear")

        aligned.append(faces)
        norm_theta.append(mats)

    if len(aligned) == 0:
        aligned = torch.zeros((0, 3, *out_size), dtype=torch.float, device=src_point.device)
        norm_theta = torch.zeros((0, 3, *out_size), dtype=torch.float, device=src_point.device)
    else:
        aligned = torch.cat(aligned, dim=0)
        norm_theta = torch.cat(norm_theta, dim=0)

    return aligned, norm_theta


def restore_faces_to_original(org_images: tuple[Tensor] | list[Tensor] | Tensor, aligned: list[Tensor], norm_theta: list[Tensor], offset: list[int]) -> Tensor | list[Tensor]:

    result: list[Tensor] = []
    for mat, org_image, face in zip(iter_detections(norm_theta, offset), org_images, iter_detections(aligned, offset)):

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
        result = torch.cat(result, dim=0)

    return result


def extract_alignface_from_video(vfp: str, batch_size: int, align_size: tuple[int, int], device: torch.device):
    """
    batch_size: 每次取出多少帧来进行面部检测与对齐，每帧可能包含多张面部，所以生成器返回的Tensor.size(0) != batch_size
    生成器返回的Tensor: Tensor(N, C, H, W), N: 对应帧检测到的面部数量, 值域: [0.0 ~ 255.0]

    """

    detector = RetinaFace().to(device=device)
    dst_pts = get_align_landmarks(align_size)
    dst_pts = torch.tensor(dst_pts, device=device)

    decoder = VideoDecoder(vfp, device=device.type)
    num_frames = decoder.metadata.num_frames

    for i in range(0, num_frames, batch_size):
        j = min(i + batch_size, num_frames)
        chunk = decoder.get_frames_in_range(i, j).data.to(device=device, dtype=torch.float)  # [0.0~255.0]
        detected, offset = detector.detector(chunk)
        src_pts = get_pts(detected)
        align_face, norm_theta = face_align_batch(chunk, src_pts, offset, dst_pts, (align_size, align_size))

        yield align_face, norm_theta, offset


class FaceAlign(nn.Module):
    def __init__(self, from_normalized: bool = False, from_unit_range: bool = False, from_rgb: bool = True):
        """
        Args:
            from_normalized: 输入值域是否为 [-1, 1]
            from_unit_range: 输入值域是否为 [0, 1]
            from_rgb: 输入图像是否为RGB
        """
        super().__init__()

        self.detector = RetinaFace(from_normalized, from_unit_range, from_rgb)

        dst_pts = get_align_landmarks(align_size)
        dst_pts = torch.tensor(dst_pts, device=device)
        self.register_buffer("dst_pts", dst_pts, persistent=False)

        self.eval()
        self.requires_grad_(False)

    def device(self) -> torch.device:
        return self.dst_pts.device

    def extract_face_from_video(self, vfp: str, batch_size: int):

        decoder = VideoDecoder(vfp, device="cuda")
        num_frames = decoder.metadata.num_frames
        device = self.device()

        for i in range(0, num_frames, batch_size):
            j = min(i + batch_size, num_frames)
            chunk = decoder.get_frames_in_range(i, j).data.to(device=device, dtype=torch.float)
            self.detector(chunk)


def tensor2cv_8uc3_bgr(image: torch.Tensor, value_range=(-1, 1), swap_rb_ch: bool = True) -> np.ndarray:
    """
    将形如 [C, H, W] 或 [B, C, H, W] 的图像 Tensor 转为 OpenCV 使用的 uint8 BGR 格式。

    参数:
    - image: torch.Tensor, 值域应在 value_range 范围内。
    - value_range: tuple, 指定输入值的范围，例如 (-1, 1), (0, 1), (0, 255)。

    返回:
    - np.ndarray, 形状为 [B, H, W, 3]，类型为 uint8，BGR 排列。
    """
    image = image.clone()

    # 自动补 batch 维度
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

    # 解包范围
    min_val, max_val = value_range
    assert max_val > min_val, "Invalid value_range"

    # 标准化到 0~1 再 *255
    image = (image - min_val) / (max_val - min_val)
    image = image.clamp(0, 1).mul(255)

    # RGB -> BGR
    if swap_rb_ch:
        image = image[:, [2, 1, 0], ...]

    # 转为 NHWC 格式并 uint8
    image = image.permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()

    return image


class FaceAlign(nn.Module):

    def __init__(self, from_normalized: bool = False, from_unit_range: bool = False, swap_rb_ch: bool = True):
        """
        Args:
            from_normalized: 如果为 True，则假设输入为 [-1, 1]，转换为 [0, 255]
            from_unit_range: 如果为 True，则假设输入为 [0, 1]，转换为 [0, 255]
        """
        super().__init__()
        self.register_buffer("device", torch.zeros(1), persistent=False)

        self.detector = RetinaFace.Model(from_normalized=from_normalized, from_unit_range=from_unit_range, swap_rb_ch=swap_rb_ch)
        self.eval()
        self.requires_grad_(False)

    def get_device(self) -> torch.device:
        return self.device.device

    @staticmethod
    def align_with(images: list[torch.Tensor] | torch.Tensor, src_pts5_pixel: torch.Tensor, idxs: list[int], out_size: tuple[int, int], dst_pts5_pixel: torch.Tensor):

        mats = compute_similarity_transform(src_pts5_pixel, dst_pts5_pixel.unsqueeze(0).expand(src_pts5_pixel.shape[0], -1, -1))

        aligned_list = []
        norm_theata = []
        slic_n = 0
        for image, N in zip(images, idxs):

            if N == 0:
                continue

            image = image.unsqueeze(0).expand(N, -1, -1, -1)
            C = image.shape[1]

            mat = normalize_theta(mats[slic_n : slic_n + N], image.shape[-2:], out_size, align_corners=False)

            grid = F.affine_grid(mat, (N, C, *out_size), align_corners=False)
            aligned = F.grid_sample(image, grid, align_corners=False, mode="bicubic")

            aligned_list.append(aligned)
            norm_theata.append(mat)
            slic_n += N

        if len(aligned_list) == 0:
            return torch.empty((0, 3, *out_size), device=images[0].device), torch.empty((0, 2, 3), device=images[0].device)

        aligned = torch.cat(aligned_list, dim=0)
        norm_theata = torch.cat(norm_theata, dim=0)

        return aligned, norm_theata

    @torch.no_grad()
    def align(self, images: list[torch.Tensor] | torch.Tensor, out_size: tuple[int, int], dst_pts5_pixel: torch.Tensor, conf_thresh: float = 0.85, min_box_size: int = 0):
        """
        对输入图片进行人脸对齐（仿射变换），输出对齐后的图片和仿射矩阵。

        参数:
            images (list[torch.Tensor] | torch.Tensor):
                输入图片，可以是单张图片（Tensor, 形状为[C, H, W]）或多张图片的列表（每张为Tensor，尺寸可不同）。
            out_size (tuple[int, int]):
                对齐后输出图片的尺寸 (height, width)。
            dst_pts5_pixel (torch.Tensor):
                目标五点关键点坐标，形状为[5, 2]，像素坐标。

        返回:
            aligned_list (List[torch.Tensor]):
                对齐后的人脸图片列表，每个元素形状为[N, C, H, W]，N为检测到的人脸数。
            norm_theta (List[torch.Tensor]):
                对应的仿射变换矩阵列表，每个元素形状为[N, 2, 3]。

        异常:
            TypeError:
                输入类型不是torch.Tensor或list[torch.Tensor]时抛出。

        说明:
            - 支持批量处理不同尺寸的图片。
            - 每张图片可能检测到多个人脸，返回每个人脸的对齐结果。
        """

        results, idxs = self.detector.detector(images, conf_thresh, min_box_size)  # results [N, 15]

        all_pts5 = RetinaFace.get_5pts(results)  # [N, 5, 2]
        aligned, norm_theata = FaceAlign.align_with(images, all_pts5, idxs, out_size, dst_pts5_pixel)
        return aligned, norm_theata, idxs

    @autocast(device_type="cuda")
    @staticmethod
    def restore_faces_to_original(
        align_faces: torch.Tensor, aff_matrix: torch.Tensor, org_images: list[torch.Tensor] | torch.Tensor, idxs: list[int], face_masks: torch.Tensor | None = None
    ):
        inv_aff_mat = invert_normalized_theta(aff_matrix)

        if face_masks is None:
            device = align_faces.device
            dtype = align_faces.dtype
            T, _, h, w = align_faces.shape
            face_masks = torch.ones((T, 1, h, w), dtype=dtype, device=device)

        slic_n = 0
        outputs = []
        for N, org_image in zip(idxs, org_images):

            if N == 0:
                outputs.append(org_image)
                continue

            org_image = org_image.unsqueeze(0)

            grid = F.affine_grid(inv_aff_mat[slic_n : slic_n + N], [N] + list(org_image.shape[1:]), align_corners=False)

            faces_on_canvas = F.grid_sample(align_faces[slic_n : slic_n + N], grid, padding_mode="zeros", align_corners=False, mode="bicubic")
            mask_on_canvas = F.grid_sample(face_masks[slic_n : slic_n + N], grid, padding_mode="zeros", align_corners=False, mode="nearest")

            # 聚合多张人脸到单张
            sum_faces = (faces_on_canvas * mask_on_canvas).sum(dim=0, keepdim=True)  # [1, C, H, W]
            sum_weights = mask_on_canvas.sum(dim=0, keepdim=True).clamp_min(1e-6)  # [1, 1, H, W]
            composite = sum_faces / sum_weights  # [1, C, H, W]

            acc_mask = mask_on_canvas.max(dim=0, keepdim=True).values  # [1, 1, H, W]

            # 现在维度匹配：[1,C,H,W] * [1,1,H,W] + [1,C,H,W] * [1,1,H,W]
            restored = org_image * (1 - acc_mask) + composite * acc_mask  # [1, C, H, W]

            outputs.append(restored.squeeze(0))  # 移除批次维度
            slic_n += N

        return outputs

    @staticmethod
    def video_face_ext_save_worker(q: mp.Queue, wfp: str, sf_prefix: str, value_range):

        while True:
            idxs, slic, aligned = q.get()

            if idxs is None:
                break

            aligned = tensor2cv_8uc3_bgr(aligned, value_range)

            slic_align_n = 0
            for n, frame_idx in zip(idxs, range(*slic)):

                faces = aligned[slic_align_n : slic_align_n + n]

                for face_num, face in enumerate(faces):
                    save_path = Path(wfp) / f"{sf_prefix}{str(frame_idx)}_{str(face_num)}.png"
                    cv2.imwrite(save_path, face, [cv2.IMWRITE_PNG_COMPRESSION, 1])

                slic_align_n += n
        q.close()

    def video_face_ext(
        self,
        vfp: str,
        dst_pts_pixel: torch.Tensor,
        wfp: str | None = None,
        sf_prefix: str = "",
        out_size: tuple[int, int] = [512, 512],
        batch_size: int = 8,
        conf_thresh: float = 0.85,
        min_box_size: int = 256,
    ):
        fn = Path(vfp).stem
        folder = Path(vfp).parent
        if wfp is None:
            wfp = Path(folder) / f"{fn}_align_results/"

        Path(wfp).mkdir(exist_ok=True, parents=True)

        device = self.get_device()
        self.detector.from_normalized = False
        self.detector.from_unit_range = False

        video_reader = VideoDecoder(vfp, device="cuda")
        num_frames = video_reader.metadata.num_frames_from_content

        mp.set_start_method("spawn", force=True)

        q = mp.Queue(maxsize=10)
        p = mp.Process(target=self.video_face_ext_save_worker, args=(q, wfp, sf_prefix, (0, 255)), daemon=True)
        p.start()

        pbar = tqdm(range(num_frames), unit="frame")
        pbar.set_description(fn)

        slices = [(i, min(i + batch_size, num_frames)) for i in range(0, num_frames, batch_size)]

        try:
            for slic in slices:
                pbar.n = slic[1]
                pbar.refresh()

                chunk = video_reader.get_frames_in_range(*slic).data.to(device=device, dtype=torch.float)

                aligned, _, idxs = self.align(list(chunk.unbind(dim=0)), out_size, dst_pts_pixel, conf_thresh, min_box_size)

                q.put((idxs, slic, aligned))

        except KeyboardInterrupt:
            q.put((None,) * 3)
            return

        q.put((None,) * 3)
        p.join()

    @staticmethod
    def align_folder_save_worker(q: mp.Queue, wfp: str, sf_prefix: str, value_range):
        while True:
            aligned, idxs, fn = q.get()

            if aligned is None:
                break

            aligned = tensor2cv_8uc3_bgr(aligned, value_range)

            slic_align_n = 0
            for n, name in zip(idxs, fn):

                faces = aligned[slic_align_n : slic_align_n + n]
                for idx, face in enumerate(faces):

                    save_path = Path(wfp) / f"{sf_prefix}{Path(name).stem}_{str(idx)}.png"
                    cv2.imwrite(save_path, face, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                slic_align_n += n

        q.close()

    def align_folder(
        self,
        fp: str,
        dst_pts_pixel: torch.Tensor,
        sfp: str | None = None,
        sf_prefix: str = "",
        batch_size: int = 32,
        out_size: tuple[int, int] = [512, 512],
        conf_thresh: float = 0.85,
        min_box_size: int = 256,
    ):

        device = self.get_device()
        self.detector.from_normalized = False
        self.detector.from_unit_range = False

        if sfp is None:
            sfp = Path(fp) / "align_results/"
        Path(sfp).mkdir(exist_ok=True, parents=True)

        mp.set_start_method("spawn", force=True)
        q = mp.Queue(maxsize=10)
        p = mp.Process(target=self.align_folder_save_worker, args=(q, sfp, sf_prefix, (0, 255)), daemon=True)
        p.start()

        dataset = ImageFolder(fp)

        pbar = tqdm(range(dataset.len), unit="frame")
        pbar.set_description(fp)
        try:
            for images, fns in dataset.iter_batch_with_tensor(batch_size, True):

                pbar.update(len(images))

                images = [image.to(device=device, dtype=torch.float) for image in images]

                aligned, _, idxs = self.align(images, out_size, dst_pts_pixel, conf_thresh, min_box_size)

                q.put((aligned, idxs, fns))
        except KeyboardInterrupt:
            q.put((None,) * 3)
            return

        q.put((None,) * 3)


if __name__ == "__main__":
    import torchvision.utils as utils
    from .models.idencoder import get_align_landmarks
    from .models.retinaface import get_pts, batch_resize_and_pad_varsize

    from misc.utils import ImageFolder, Timer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    align_size = 256

    model = RetinaFace(True).to(device=device)

    dataset = ImageFolder("/home/liaohaixun/swap/IDAssets")
    # dataset = ImageFolder("/opt/share/deepfake/dataset_1/ffhq_1024")

    images_org = [dataset.sample_tensor().float().to(device=device) for _ in range(batch_size)]

    # images_org = torch.stack(images_org)

    detected = model.detector(images_org)
    src_pts = get_pts(detected)

    dst_pts = get_align_landmarks(align_size)
    dst_pts = torch.tensor(dst_pts, device=device)

    faces, mats = face_align_batch(images_org, src_pts, dst_pts, (align_size, align_size))
    restore_images = restore_faces_to_original(images_org, faces, mats)

    faces = torch.cat(faces, dim=0)
    utils.save_image(faces, "faces.png", normalize=True, value_range=(0, 255))

    restore_images, _ = batch_resize_and_pad_varsize(restore_images, 1024, 1024)

    utils.save_image(restore_images, "restore_images.png", normalize=True, value_range=(0, 255))
