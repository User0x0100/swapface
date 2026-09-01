from collections import OrderedDict
from math import ceil
from typing import Literal, TypedDict

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from torchvision import models, ops
from torchvision.models import _utils

from ...models import MODEL_REPOSITORY_ID, ImageInputRange
from .net import FPN, SSH, BboxHead, ClassHead, LandmarkHead, MobileNetV1


class RetinaFaceConfig(TypedDict):
    backbone: Literal["mobilenet0.25", "resnet50"]
    min_sizes: list[list[int]]
    steps: list[int]
    variance: list[float]
    clip: bool
    image_size: int
    return_layers: dict[str, str]
    in_channel: int
    out_channel: int


MOBILENET_CONFIG: RetinaFaceConfig = {
    "backbone": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "image_size": 640,
    "return_layers": {"stage1": "1", "stage2": "2", "stage3": "3"},
    "in_channel": 32,
    "out_channel": 64,
}

RESNET50_CONFIG: RetinaFaceConfig = {
    "backbone": "resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "image_size": 840,
    "return_layers": {"layer2": "1", "layer3": "2", "layer4": "3"},
    "in_channel": 256,
    "out_channel": 256,
}


class RetinaFace(nn.Module):
    """
    RetinaFace 人脸检测器。

    支持 MobileNetV1-0.25 和 ResNet-50 两种 backbone，权重从 HuggingFace Hub
    自动下载。输入图像可来自多种归一化约定，内部统一转换为 BGR、像素值
    [0, 255] 减均值后送入网络。

    Args:
        use_mobilenet: 是否使用 MobileNetV1-0.25 backbone；否则使用 ResNet-50。
        input_range: 输入张量值域。
        color_order: 输入通道顺序：rgb / bgr。


    https://github.com/biubug6/Pytorch_Retinaface

    """

    def __init__(
        self,
        use_mobilenet: bool = False,
        input_range: ImageInputRange = ImageInputRange.ZERO_TO_255,
        color_order: Literal["rgb", "bgr"] = "rgb",
    ) -> None:
        super().__init__()
        if not isinstance(input_range, ImageInputRange):
            raise TypeError(f"input_range 必须为 ImageInputRange，实际为 {type(input_range).__name__}")
        self.input_range = input_range
        self.color_order = color_order

        self.register_buffer(
            "mean",
            torch.tensor([104.0, 117.0, 123.0], dtype=torch.float).view(1, 3, 1, 1),
            persistent=False,
        )  # bgr order

        config = MOBILENET_CONFIG if use_mobilenet else RESNET50_CONFIG

        match config["backbone"]:
            case "mobilenet0.25":
                checkpoint_path = hf_hub_download(repo_id=MODEL_REPOSITORY_ID, filename="mobilenet0.25_Final.pth")
                backbone = MobileNetV1()
            case "resnet50":
                checkpoint_path = hf_hub_download(repo_id=MODEL_REPOSITORY_ID, filename="Resnet50_Final.pth")
                backbone = models.resnet50(weights=None)

        self.body = _utils.IntermediateLayerGetter(backbone, config["return_layers"])
        in_channels_stage2 = config["in_channel"]
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = config["out_channel"]
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.class_heads = self._make_class_heads(fpn_num=3, inchannels=config["out_channel"])
        self.box_heads = self._make_box_heads(fpn_num=3, inchannels=config["out_channel"])
        self.landmark_heads = self._make_landmark_heads(fpn_num=3, inchannels=config["out_channel"])

        state_dict: dict[str, Tensor] = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        self.load_state_dict(state_dict)

        self.config = config
        self._priors_cache: OrderedDict[tuple[int, int, str], Tensor] = OrderedDict()

    def _make_class_heads(self, fpn_num: int = 3, inchannels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        heads = nn.ModuleList()
        for _ in range(fpn_num):
            heads.append(ClassHead(inchannels, anchor_num))
        return heads

    def _make_box_heads(self, fpn_num: int = 3, inchannels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        heads = nn.ModuleList()
        for _ in range(fpn_num):
            heads.append(BboxHead(inchannels, anchor_num))
        return heads

    def _make_landmark_heads(self, fpn_num: int = 3, inchannels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        heads = nn.ModuleList()
        for _ in range(fpn_num):
            heads.append(LandmarkHead(inchannels, anchor_num))
        return heads

    @torch.inference_mode()
    def detect(
        self,
        images: list[Tensor] | tuple[Tensor, ...] | Tensor,
        confidence_threshold: float = 0.9,
        iou_threshold: float = 0.5,
        min_face_size: tuple[int, int] = (0, 0),
    ) -> tuple[Tensor, list[int]]:
        """
        RetinaFace 人脸检测。

        支持单张或多张不同尺寸图像输入，内部完成：
        - resize + padding（仅限 list/tuple 输入）
        - 归一化 / 通道转换
        - backbone + FPN + SSH 前向传播
        - bbox / score / landmark 预测
        - decode + threshold + size filter + NMS

        Args:
            images:
                Tensor[B, C, H, W] | list[Tensor[C, H, W]] | tuple[Tensor[C, H, W]]
                支持 batch Tensor 或变尺寸 list/tuple 输入。
                传入 Tensor 时假定所有图像尺寸相同，不执行 resize；
                传入 list/tuple 时自动等比缩放并 padding 到 config["image_size"]。
            confidence_threshold:
                人脸置信度阈值，低于该值的候选框会被过滤。
            iou_threshold:
                NMS IoU 阈值。
            min_face_size:
                (min_w, min_h)，宽或高小于该值的检测框会被丢弃。

        Returns:
            detections: Tensor[M, 15]
                拼接 batch 后的检测结果，列布局为：
                [score, x1, y1, x2, y2, lx0, ly0, lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4]
                其中 score 为人脸置信度，(x1,y1,x2,y2) 为 bbox 像素坐标，
                (lx_i, ly_i) 为 5 个面部关键点（左眼、右眼、鼻尖、左嘴角、右嘴角）。

            segment_lengths: list[int]
                长度等于 batch size，第 i 个元素表示第 i 张图像的检测框数量。
                可用于将 detections 反拆回 per-image 结构：

                    detections, segment_lengths = detect(images)
                    per_image = torch.split(detections, segment_lengths)   # list[Tensor[Ni, 15]]

        Raises:
            TypeError: 若 images 既非 Tensor 也非 Tensor 的 list/tuple。
        """

        if isinstance(images, Tensor):
            batch, scales = images, None
        elif isinstance(images, (list, tuple)) and all(isinstance(x, Tensor) for x in images):
            batch, scales = _resize_and_pad_images(images, self.config["image_size"], self.config["image_size"])
        else:
            raise TypeError("images 必须为 Tensor 或 tuple/list[Tensor]")

        # Preprocess out-of-place so detect() never mutates caller-owned tensors.
        match self.input_range:
            case ImageInputRange.MINUS_ONE_TO_ONE:
                batch = batch.add(1.0).mul(127.5)
            case ImageInputRange.ZERO_TO_ONE:
                batch = batch.mul(255.0)
            case ImageInputRange.ZERO_TO_255:
                pass

        if self.color_order == "rgb":
            batch = batch[:, [2, 1, 0]]

        batch = batch.sub(self.get_buffer("mean"))

        out = self.body(batch)
        fpn = self.fpn(out)

        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        box_regression = torch.cat([self.box_heads[i](feature) for i, feature in enumerate(features)], dim=1)
        class_logits = torch.cat([self.class_heads[i](feature) for i, feature in enumerate(features)], dim=1)
        landmark_regression = torch.cat(
            [self.landmark_heads[i](feature) for i, feature in enumerate(features)],
            dim=1,
        )

        height, width = batch.shape[-2:]
        detections = self._postprocess(
            box_regression,
            class_logits,
            landmark_regression,
            (height, width),
            scales=scales,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            min_face_size=min_face_size,
        )

        return detections

    def _postprocess(
        self,
        box_regression: Tensor,
        class_logits: Tensor,
        landmark_regression: Tensor,
        image_shape: tuple[int, int],
        confidence_threshold: float,
        iou_threshold: float,
        scales: Tensor | None,
        min_face_size: tuple[int, int],
    ) -> tuple[Tensor, list[int]]:
        """Decode RetinaFace outputs, filter candidates, and apply per-image NMS."""
        batch_size = box_regression.shape[0]
        device = box_regression.device
        height, width = image_shape

        priors = self._get_priors(height, width, device=device)
        if scales is None:
            scales = torch.ones((batch_size, 1), dtype=torch.float, device=device)

        inverse_scales = scales.reciprocal()
        variances = self.config["variance"]
        detections: list[Tensor] = []
        segment_lengths: list[int] = []

        for batch_index in range(batch_size):
            scores = class_logits[batch_index, :, 1]
            confidence_mask = scores > confidence_threshold
            if not confidence_mask.any():
                segment_lengths.append(0)
                continue

            filtered_priors = priors[confidence_mask]
            filtered_scores = scores[confidence_mask]
            filtered_landmark_regression = landmark_regression[batch_index][confidence_mask]

            boxes = decode_boxes(box_regression[batch_index][confidence_mask], filtered_priors, variances)
            boxes[:, 0::2] *= width * inverse_scales[batch_index]
            boxes[:, 1::2] *= height * inverse_scales[batch_index]

            box_widths = boxes[:, 2] - boxes[:, 0]
            box_heights = boxes[:, 3] - boxes[:, 1]
            size_mask = (box_widths >= min_face_size[0]) & (box_heights >= min_face_size[1])
            if not size_mask.any():
                segment_lengths.append(0)
                continue

            boxes = boxes[size_mask]
            filtered_scores = filtered_scores[size_mask]
            filtered_priors = filtered_priors[size_mask]
            landmarks = decode_landmarks(filtered_landmark_regression[size_mask], filtered_priors, variances)
            landmarks[:, 0::2] *= width * inverse_scales[batch_index]
            landmarks[:, 1::2] *= height * inverse_scales[batch_index]

            keep = ops.nms(boxes, filtered_scores, iou_threshold)
            detection = torch.cat(
                [filtered_scores[keep].unsqueeze(1), boxes[keep], landmarks[keep]],
                dim=1,
            )
            detections.append(detection)
            segment_lengths.append(detection.shape[0])

        if not detections:
            return box_regression.new_empty((0, 15)), segment_lengths
        return torch.cat(detections, dim=0), segment_lengths

    def _get_priors(self, height: int, width: int, device: torch.device | str = "cpu") -> Tensor:
        """
        生成给定输入分辨率下的 prior anchor grid。

        Anchor 按 FPN 层级、空间位置（row-major）、anchor 尺寸三层顺序遍历，
        坐标以相对图像宽高的归一化值表示（cx, cy, w, h），与 decode_boxes /
        decode_landmarks 的输入约定一致。

        结果以 (height, width, device) 为 key 缓存在 _priors_cache（LRU，最多 10 种分辨率），
        避免重复计算。

        Args:
            height: 输入图像高度（像素）。
            width: 输入图像宽度（像素）。
            device: 目标设备。

        Returns:
            Tensor[N, 4]，每行为一个 anchor (cx, cy, sw, sh)，
            cx/cy/sw/sh 均为归一化值（相对图像宽/高）。
            N = sum(ceil(height/step) * ceil(width/step) * len(min_sizes[k]) for k in levels)
        """
        device_obj = torch.device(device)
        key = (height, width, str(device_obj))
        if key in self._priors_cache:
            self._priors_cache.move_to_end(key)
            return self._priors_cache[key]

        min_sizes, steps, clip = (
            self.config["min_sizes"],
            self.config["steps"],
            self.config["clip"],
        )

        level_grids: list[Tensor] = []
        for min_sizes_k, step in zip(min_sizes, steps):
            feature_h = ceil(height / step)
            feature_w = ceil(width / step)
            ys = (torch.arange(feature_h, device=device_obj, dtype=torch.float32) + 0.5) * (step / height)
            xs = (torch.arange(feature_w, device=device_obj, dtype=torch.float32) + 0.5) * (step / width)
            cy, cx = torch.meshgrid(ys, xs, indexing="ij")
            centers = torch.stack((cx, cy), dim=-1).reshape(-1, 1, 2)

            sizes = torch.tensor(
                [[size / width, size / height] for size in min_sizes_k],
                device=device_obj,
                dtype=torch.float32,
            )
            centers = centers.expand(-1, sizes.shape[0], -1)
            sizes = sizes.unsqueeze(0).expand(centers.shape[0], -1, -1)
            level_grids.append(torch.cat((centers, sizes), dim=-1).reshape(-1, 4))

        grid = torch.cat(level_grids, dim=0)

        if clip:
            grid.clamp_(max=1, min=0)

        # 添加到缓存
        self._priors_cache[key] = grid
        self._priors_cache.move_to_end(key)

        # 如果超过缓存大小，丢弃最旧的
        if len(self._priors_cache) > 10:
            self._priors_cache.popitem(last=False)

        return grid


def _resize_and_pad_images(images: tuple[Tensor, ...] | list[Tensor], output_height: int, output_width: int) -> tuple[Tensor, Tensor]:
    """
    将一组不同尺寸的图像等比例缩放后 padding 到固定大小，左上角对齐。

    缩放因子取宽高方向的最小值，保证缩放后图像不超出目标尺寸；
    空白区域以 0 填充。

    Args:
        images: list 或 tuple，每个元素为 Tensor[C, H, W]，允许不同 H/W。
        output_height: 输出高度（像素）。
        output_width: 输出宽度（像素）。

    Returns:
        output: Tensor[N, C, output_height, output_width]，缩放 + padding 后的图像 batch。
        scales: Tensor[N, 1]，每张图像的缩放因子 scale = min(output_height/H, output_width/W)。
                用于将检测坐标从缩放后图像空间还原到原始图像空间（除以 scale）。
    """
    if not images:
        raise ValueError("images must not be empty")
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_height and output_width must be greater than 0")

    batch_size = len(images)
    device = images[0].device
    dtype = images[0].dtype
    channels = images[0].shape[0]
    if any(image.ndim != 3 or image.shape[0] != channels or image.device != device or image.dtype != dtype for image in images):
        raise ValueError("all images must be CHW tensors with matching channels, device, and dtype")

    # 输出张量初始化
    output = torch.zeros((batch_size, channels, output_height, output_width), device=device, dtype=dtype)
    scales = torch.zeros((batch_size, 1), device=device, dtype=dtype)

    for index, image in enumerate(images):
        height, width = image.shape[-2:]
        scale = min(output_height / height, output_width / width)
        scales[index] = scale

        new_height = int(height * scale)
        new_width = int(width * scale)

        resized_image = F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        output[index, :, :new_height, :new_width] = resized_image

    return output, scales


def decode_boxes(box_regression: Tensor, priors: Tensor, variances: list[float]) -> torch.Tensor:
    """
    将网络输出的 bbox 回归偏移解码为 (x1, y1, x2, y2) 格式的归一化坐标。

    编码约定（与 RetinaFace 训练时一致）：
        dx = (cx_gt - cx_prior) / (variance[0] * w_prior)
        dy = (cy_gt - cy_prior) / (variance[0] * h_prior)
        dw = log(w_gt / w_prior) / variance[1]
        dh = log(h_gt / h_prior) / variance[1]

    Args:
        box_regression: Tensor[N, 4]，回归偏移 (dx, dy, dw, dh)。
        priors:    Tensor[N, 4]，prior anchor (cx, cy, w, h)，归一化坐标。
        variances: list[float, float]，[variance_xy, variance_wh]。

    Returns:
        Tensor[N, 4]，解码后的 bbox (x1, y1, x2, y2)，归一化坐标。
        乘以图像宽高后得到像素坐标。
    """
    # 预分配输出
    boxes = box_regression.new_empty(box_regression.size(0), 4)

    # 中心点
    cx = priors[:, 0] + box_regression[:, 0] * variances[0] * priors[:, 2]
    cy = priors[:, 1] + box_regression[:, 1] * variances[0] * priors[:, 3]

    w = priors[:, 2] * torch.exp(box_regression[:, 2] * variances[1])
    h = priors[:, 3] * torch.exp(box_regression[:, 3] * variances[1])

    # 半宽高
    half_w = w * 0.5
    half_h = h * 0.5

    # 左上右下
    boxes[:, 0] = cx - half_w
    boxes[:, 1] = cy - half_h
    boxes[:, 2] = cx + half_w
    boxes[:, 3] = cy + half_h

    return boxes


def decode_landmarks(landmark_regression: Tensor, priors: Tensor, variances: list[float]) -> torch.Tensor:
    """
    将网络输出的关键点回归偏移解码为归一化坐标。

    编码约定：
        dx_i = (x_gt_i - cx_prior) / (variance[0] * w_prior)
        dy_i = (y_gt_i - cy_prior) / (variance[0] * h_prior)

    Args:
        landmark_regression: Tensor[N, 10]，5 个关键点的回归偏移，x/y 交替排列。
        priors:    Tensor[N, 4]，prior anchor (cx, cy, w, h)，归一化坐标。
        variances: list[float, float]，仅使用 variances[0]（xy 方向方差）。

    Returns:
        Tensor[N, 10]，解码后的关键点归一化坐标，x/y 交替排列。
        乘以图像宽高后得到像素坐标。
    """
    count = landmark_regression.size(0)
    landmarks = landmark_regression.new_empty(count, 10)

    cx = priors[:, 0]
    cy = priors[:, 1]
    w = priors[:, 2] * variances[0]
    h = priors[:, 3] * variances[0]

    landmarks[:, 0::2] = cx.unsqueeze(1) + landmark_regression[:, 0::2] * w.unsqueeze(1)
    landmarks[:, 1::2] = cy.unsqueeze(1) + landmark_regression[:, 1::2] * h.unsqueeze(1)

    return landmarks


@torch.inference_mode()
def draw_detections(
    images: Tensor | tuple[Tensor] | list[Tensor],
    detections: Tensor,
    segment_lengths: list[int],
    color=(255.0, 0.0, 0.0),
    thickness=3,
    point_size=7,
) -> Tensor | list[Tensor]:
    """
    在图像上绘制检测结果（bbox + 5 关键点）。

    Args:
        images:
            Tensor[B, C, H, W] | list[Tensor[C, H, W]] | tuple[Tensor[C, H, W]]
            输入图像，不会被原地修改（内部 clone）。
        detections:
            Tensor[M, 15]，detect() 返回的拼接检测结果。
        segment_lengths:
            list[int]，detect() 返回的每张图像检测框数量，用于拆分 detections。
        color:
            绘制颜色，(R, G, B) 浮点值，与输入图像值域一致。
            默认为红色 (255, 0, 0)。
        thickness:
            bbox 边框线宽（像素）。
        point_size:
            关键点方块边长（像素）。

    Returns:
        与输入类型一致：
        - 输入为 Tensor → 返回 Tensor[B, C, H, W]
        - 输入为 list/tuple → 返回 list[Tensor[C, H, W]]
    """

    if isinstance(images, Tensor):
        new_images = images.clone()
        color_tensor = torch.tensor(color, device=new_images.device).view(3, 1, 1)

    elif isinstance(images, (tuple, list)):
        new_images = [image.clone() for image in images]
        color_tensor = torch.tensor(color, device=new_images[0].device).view(3, 1, 1)

    for boxes, image in zip(torch.split(detections, segment_lengths), new_images):
        if boxes.size(0) == 0:
            continue
        H, W = image.shape[-2:]

        for box in boxes:
            x1, y1, x2, y2 = box[1:5].int()

            # 确保坐标不超出边界
            x1 = max(0, min(x1.item(), W - 1))
            x2 = max(0, min(x2.item(), W - 1))
            y1 = max(0, min(y1.item(), H - 1))
            y2 = max(0, min(y2.item(), H - 1))

            # 画上边框
            image[:, y1 : y1 + thickness, x1:x2] = color_tensor
            # 画下边框
            image[:, y2 - thickness : y2, x1:x2] = color_tensor
            # 画左边框
            image[:, y1:y2, x1 : x1 + thickness] = color_tensor
            # 画右边框
            image[:, y1:y2, x2 - thickness : x2] = color_tensor

            # 绘制关键点
            keypoints = box[5:15].view(5, 2)  # 提取5个关键点
            for kp in keypoints:
                kx = kp[0].int().item()  # x 坐标
                ky = kp[1].int().item()  # y 坐标
                kx = max(0, min(kx, W - 1))  # 限制在图像边界内
                ky = max(0, min(ky, H - 1))
                half_size = point_size // 2
                kx_start = max(0, kx - half_size)
                kx_end = min(W, kx + half_size + 1)
                ky_start = max(0, ky - half_size)
                ky_end = min(H, ky + half_size + 1)
                image[:, ky_start:ky_end, kx_start:kx_end] = color_tensor  # 填充正方形区域

    return new_images


def extract_landmarks(detections: Tensor) -> Tensor:
    """
    从拼接检测结果中提取 5 个面部关键点坐标。

    Args:
        detections: Tensor[N, 15]，detect() 返回的检测结果（单张图像或多张拼接）。

    Returns:
        Tensor[N, 5, 2]，每行为一个人脸的 5 个关键点 (x, y) 像素坐标，
        顺序为：左眼、右眼、鼻尖、左嘴角、右嘴角。
    """
    return detections[:, -10:].reshape(-1, 5, 2)


if __name__ == "__main__":
    from torchvision import utils

    from misc.utils import ImageDirectory, Timer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    model = RetinaFace(use_mobilenet=True).to(device=device)

    dataset = ImageDirectory("/home/liaohaixun/swap/IDAssets")
    # dataset = ImageDirectory("/opt/share/deepfake/dataset_1/ffhq_1024")

    images_org = [dataset.sample_tensor().float().to(device=device) for _ in range(batch_size)]

    # images_org = torch.stack(images_org)

    for _ in range(5):
        detections, segment_lengths = model.detect(images_org)

    with Timer("model infer"):
        detections, segment_lengths = model.detect(images_org)

    visi = draw_detections(images_org, detections, segment_lengths)

    if isinstance(visi, (list, tuple)):
        visi, _ = _resize_and_pad_images(visi, 640, 640)

    utils.save_image(visi, "visi.png", normalize=True, value_range=(0, 255))
