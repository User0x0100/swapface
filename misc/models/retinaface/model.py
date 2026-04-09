import itertools
from math import ceil
from collections import OrderedDict

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision import models, ops
from torchvision.models import _utils

from huggingface_hub import hf_hub_download
from ...models import REPO_ID
from .net import FPN, SSH, MobileNetV1, ClassHead, BboxHead, LandmarkHead

CFG_MNET0_25G = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 32,
    "ngpu": 1,
    "epoch": 250,
    "decay1": 190,
    "decay2": 220,
    "image_size": 640,
    "pretrain": True,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
    "in_channel": 32,
    "out_channel": 64,
}
CFG_R50 = {
    "name": "Resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 24,
    "ngpu": 4,
    "epoch": 100,
    "decay1": 70,
    "decay2": 90,
    "image_size": 840,
    "pretrain": True,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
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
        use_mobile_net:
            True 使用 MobileNetV1-0.25 backbone（更快，精度略低），
            False 使用 ResNet-50 backbone（默认，精度更高）。
        from_normalized:
            输入图像值域是否为 [-1, 1]（例如来自 tanh 输出或标准化预处理）。
        from_unit_range:
            输入图像值域是否为 [0, 1]（例如 torchvision 默认 ToTensor）。
        from_rgb:
            输入图像通道顺序是否为 RGB。True 时内部自动转换为 BGR。

    Raises:
        ValueError: 若 from_normalized 与 from_unit_range 同时为 True。

    Note:
        from_normalized 与 from_unit_range 互斥，均为 False 时表示
        输入已为 [0, 255] BGR 像素值（无需任何转换）。


    https://github.com/biubug6/Pytorch_Retinaface

    """

    def __init__(self, use_mobile_net: bool = False, from_normalized: bool = False, from_unit_range: bool = False, from_rgb: bool = True) -> None:

        super().__init__()
        if from_normalized and from_unit_range:
            raise ValueError("from_normalized 与 from_unit_range 不能同时为 True")

        self.from_normalized = from_normalized
        self.from_unit_range = from_unit_range
        self.from_rgb = from_rgb

        self.register_buffer("mean", torch.tensor([104.0, 117.0, 123.0], dtype=torch.float).view(1, 3, 1, 1), persistent=False)  # bgr order

        cfg = CFG_MNET0_25G if use_mobile_net else CFG_R50

        match cfg["name"]:
            case "mobilenet0.25":
                pretrain_ckpt = hf_hub_download(repo_id=REPO_ID, filename="mobilenet0.25_Final.pth")
                backbone = MobileNetV1()
                if cfg["pretrain"]:
                    checkpoint = hf_hub_download(repo_id=REPO_ID, filename="mobilenetV1X0.25_pretrain.tar")
                    state_dict: dict[str, Tensor] = torch.load(checkpoint, map_location=torch.device("cpu"))["state_dict"]
                    state_dict = {cleaned: v for k, v in state_dict.items() if (cleaned := k.removeprefix("module."))}  # 去掉 "module."
                    backbone.load_state_dict(state_dict)

            case "Resnet50":
                pretrain_ckpt = hf_hub_download(repo_id=REPO_ID, filename="Resnet50_Final.pth")
                backbone = models.resnet50(weights=None)

        self.body = _utils.IntermediateLayerGetter(backbone, cfg["return_layers"])
        in_channels_stage2 = cfg["in_channel"]
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg["out_channel"]
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg["out_channel"])

        pretrain_state_dict: dict[str, Tensor] = torch.load(pretrain_ckpt, map_location=torch.device("cpu"))
        pretrain_state_dict = {cleaned: v for k, v in pretrain_state_dict.items() if (cleaned := k.removeprefix("module."))}  # 去掉 "module."
        self.load_state_dict(pretrain_state_dict)

        self.cfg = cfg
        self._prior_cache: OrderedDict[tuple[int, int], Tensor] = OrderedDict()

    def _make_class_head(self, fpn_num: int = 3, inchannels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        classhead = nn.ModuleList()
        for _ in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num: int = 3, inchannels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        bboxhead = nn.ModuleList()
        for _ in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num: int = 3, inchannels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        landmarkhead = nn.ModuleList()
        for _ in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    @torch.inference_mode()
    def detector(
        self, images: list[Tensor] | tuple[Tensor] | Tensor, conf_thresh: float = 0.9, iou_thresh: float = 0.5, min_box_size: tuple[int, int] = (0, 0)
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
                传入 list/tuple 时自动等比缩放并 padding 到 cfg["image_size"]。
            conf_thresh:
                人脸置信度阈值，低于该值的候选框会被过滤。
            iou_thresh:
                NMS IoU 阈值。
            min_box_size:
                (min_w, min_h)，宽或高小于该值的检测框会被丢弃。

        Returns:
            detections: Tensor[M, 15]
                拼接 batch 后的检测结果，列布局为：
                [score, x1, y1, x2, y2, lx0, ly0, lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4]
                其中 score 为人脸置信度，(x1,y1,x2,y2) 为 bbox 像素坐标，
                (lx_i, ly_i) 为 5 个面部关键点（左眼、右眼、鼻尖、左嘴角、右嘴角）。

            offsets: list[int]
                长度等于 batch size，第 i 个元素表示第 i 张图像的检测框数量。
                可用于将 detections 反拆回 per-image 结构：

                    det, offsets = detector(images)
                    per_image = torch.split(det, offsets)   # list[Tensor[Ni, 15]]

        Raises:
            TypeError: 若 images 既非 Tensor 也非 Tensor 的 list/tuple。
        """

        if isinstance(images, Tensor):
            images, scales = images.clone(), None
        elif isinstance(images, (list, tuple)) and all(isinstance(x, Tensor) for x in images):
            images, scales = batch_resize_and_pad_varsize(images, self.cfg["image_size"], self.cfg["image_size"])
        else:
            raise TypeError("images 必须为 Tensor 或 tuple/list[Tensor]")

        if self.from_normalized:
            images.add_(1.0).mul_(127.5)
        elif self.from_unit_range:
            images.mul_(255.0)

        if self.from_rgb:
            images = images[:, [2, 1, 0], :, :]

        images = images.sub_(self.mean)

        out = self.body(images)
        fpn = self.fpn(out)

        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        loc = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        conf = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        landms = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        H, W = images.shape[-2:]
        detected = self._retinaface_postprocess(loc, conf, landms, (H, W), resize=scales, conf_thresh=conf_thresh, iou_thresh=iou_thresh, min_box_size=min_box_size)

        return detected

    def _retinaface_postprocess(
        self, loc: Tensor, conf: Tensor, landms: Tensor, image_shape: tuple[int, int], conf_thresh: float, iou_thresh: float, resize: Tensor | None, min_box_size: tuple[int, int]
    ) -> tuple[Tensor, list[int]]:
        """
        RetinaFace 后处理（decode → threshold → size filter → NMS）。

        Args:
            loc:     Tensor[B, N, 4]   bbox 回归偏移 (dx, dy, dw, dh)，编码格式
                     与 prior anchor 对应，需通过 decode_boxes 还原为像素坐标。
            conf:    Tensor[B, N, 2]   分类 logit，[:,0] 为背景，[:,1] 为人脸。
            landms:  Tensor[B, N, 10]  5 关键点回归偏移（x,y 交替）。
            image_shape: (H, W)        送入网络的图像尺寸（用于生成 prior grid）。
            conf_thresh: float         人脸置信度阈值。
            iou_thresh: float          NMS IoU 阈值。
            resize: Tensor[B, 1] | None
                list/tuple 输入路径下 batch_resize_and_pad_varsize 返回的缩放
                因子（将 resize 后坐标映射回原始图像空间）。None 表示未缩放（scale=1）。
            min_box_size: (min_w, min_h)
                过滤小检测框的最小宽高阈值（像素，原始图像坐标系）。

        Returns:
            detections: Tensor[M, 15]
                拼接 batch 后的检测结果，格式同 detector() 的返回值。
                若所有 batch 均无检测结果，返回 shape=(0, 15) 的空 Tensor。
            offsets: list[int]
                每张图像的检测框数量，sum(offsets) == M。
        """

        B = loc.shape[0]
        device = loc.device

        priors = self._generate_grid(*image_shape, device=device)

        if resize is None:
            resize = torch.ones((B, 1), dtype=torch.float, device=device)

        H, W = image_shape
        inv_resize = 1.0 / resize
        variance = self.cfg["variance"]

        detected: list[Tensor] = []
        offset: list[int] = []

        for b in range(B):
            conf_b = conf[b]  # [N, 2]
            scores = conf_b[:, 1]  # 人脸概率

            mask = scores > conf_thresh
            if mask.sum() == 0:
                offset.append(0)
                continue

            boxes_b = decode_boxes(loc[b][mask], priors[mask], variance)
            # boxes: [x1,y1,x2,y2]
            boxes_b[:, 0::2] *= W * inv_resize[b]  # X方向
            boxes_b[:, 1::2] *= H * inv_resize[b]  # Y方向

            widths = boxes_b[:, 2] - boxes_b[:, 0]
            heights = boxes_b[:, 3] - boxes_b[:, 1]
            size_mask = (widths >= min_box_size[0]) & (heights >= min_box_size[1])

            if size_mask.sum() == 0:
                offset.append(0)
                continue

            boxes_b = boxes_b[size_mask]
            scores_b = scores[mask][size_mask]

            landms_b = decode_landmarks(landms[b][mask][size_mask], priors[mask][size_mask], variance)
            landms_b[:, 0::2] *= W * inv_resize[b]
            landms_b[:, 1::2] *= H * inv_resize[b]

            keep = ops.nms(boxes_b, scores_b, iou_thresh)

            boxes_b = boxes_b[keep].clone()
            landms_b = landms_b[keep].clone()
            scores_b = scores_b[keep].unsqueeze(1).clone()  # [K] -> [K, 1]

            detection = torch.cat([scores_b, boxes_b, landms_b], dim=1)  # [K, 15]
            detected.append(detection)
            offset.append(detection.size(0))

        if len(detected) == 0:
            return torch.zeros((0, 15), device=device), offset

        return torch.cat(detected, dim=0), offset

    def _generate_grid(self, h: int, w: int, device=torch.device("cpu")) -> Tensor:
        """
        生成给定输入分辨率下的 prior anchor grid。

        Anchor 按 FPN 层级、空间位置（row-major）、anchor 尺寸三层顺序遍历，
        坐标以相对图像宽高的归一化值表示（cx, cy, w, h），与 decode_boxes /
        decode_landmarks 的输入约定一致。

        结果以 (h, w) 为 key 缓存在 _prior_cache（LRU，最多 10 种分辨率），
        避免重复计算。

        Args:
            h: 输入图像高度（像素）。
            w: 输入图像宽度（像素）。
            device: 目标设备。

        Returns:
            Tensor[N, 4]，每行为一个 anchor (cx, cy, sw, sh)，
            cx/cy/sw/sh 均为归一化值（相对图像宽/高）。
            N = sum(ceil(h/step) * ceil(w/step) * len(min_sizes[k]) for k in levels)
        """
        key = (h, w)
        # 如果在缓存中，移动到末尾（最近使用）
        if key in self._prior_cache:
            self._prior_cache.move_to_end(key)
            return self._prior_cache[key].to(device)

        min_sizes, steps, clip = self.cfg["min_sizes"], self.cfg["steps"], self.cfg["clip"]

        # 计算特征图尺寸 [height, width]
        feature_maps = [[ceil(h / step), ceil(w / step)] for step in steps]

        anchors = []

        for k, f in enumerate(feature_maps):
            min_sizes_k = min_sizes[k]
            for i, j in itertools.product(range(f[0]), range(f[1])):
                for min_size in min_sizes_k:
                    s_kx = min_size / w  # width方向
                    s_ky = min_size / h  # height方向

                    # 计算中心点
                    dense_cx = (j + 0.5) * steps[k] / w
                    dense_cy = (i + 0.5) * steps[k] / h

                    anchors.extend([dense_cx, dense_cy, s_kx, s_ky])

        grid = torch.tensor(anchors, device=device).view(-1, 4)

        if clip:
            grid.clamp_(max=1, min=0)

        # 添加到缓存
        self._prior_cache[key] = grid
        self._prior_cache.move_to_end(key)

        # 如果超过缓存大小，丢弃最旧的
        if len(self._prior_cache) > 10:
            self._prior_cache.popitem(last=False)

        return grid


def batch_resize_and_pad_varsize(imgs: tuple[Tensor] | list[Tensor], out_h: int, out_w: int) -> tuple[Tensor, Tensor]:
    """
    将一组不同尺寸的图像等比例缩放后 padding 到固定大小，左上角对齐。

    缩放因子取宽高方向的最小值，保证缩放后图像不超出目标尺寸；
    空白区域以 0 填充。

    Args:
        imgs: list 或 tuple，每个元素为 Tensor[C, H, W]，允许不同 H/W。
        out_h: 输出高度（像素）。
        out_w: 输出宽度（像素）。

    Returns:
        out:    Tensor[N, C, out_h, out_w]，缩放 + padding 后的图像 batch。
        scales: Tensor[N, 1]，每张图像的缩放因子 scale = min(out_h/H, out_w/W)。
                用于将检测坐标从缩放后图像空间还原到原始图像空间（除以 scale）。
    """
    N = len(imgs)
    device = imgs[0].device
    dtype = imgs[0].dtype
    C = imgs[0].shape[0]

    # 输出张量初始化
    out = torch.zeros((N, C, out_h, out_w), device=device, dtype=dtype)
    scales = torch.zeros((N, 1), device=device, dtype=dtype)

    for i, img in enumerate(imgs):
        H_i, W_i = img.shape[-2:]
        scale = min(out_h / H_i, out_w / W_i)
        scales[i] = scale

        new_h = int(H_i * scale)
        new_w = int(W_i * scale)

        img_resized = F.interpolate(img.unsqueeze(0), scale_factor=scale, mode="bicubic", align_corners=False).squeeze_(0)
        out[i, :, :new_h, :new_w] = img_resized

    return out, scales


def decode_boxes(loc: Tensor, priors: Tensor, variances: list[float]) -> torch.Tensor:
    """
    将网络输出的 bbox 回归偏移解码为 (x1, y1, x2, y2) 格式的归一化坐标。

    编码约定（与 RetinaFace 训练时一致）：
        dx = (cx_gt - cx_prior) / (variance[0] * w_prior)
        dy = (cy_gt - cy_prior) / (variance[0] * h_prior)
        dw = log(w_gt / w_prior) / variance[1]
        dh = log(h_gt / h_prior) / variance[1]

    Args:
        loc:       Tensor[N, 4]，回归偏移 (dx, dy, dw, dh)。
        priors:    Tensor[N, 4]，prior anchor (cx, cy, w, h)，归一化坐标。
        variances: list[float, float]，[variance_xy, variance_wh]。

    Returns:
        Tensor[N, 4]，解码后的 bbox (x1, y1, x2, y2)，归一化坐标。
        乘以图像宽高后得到像素坐标。
    """
    # 预分配输出
    boxes = loc.new_empty(loc.size(0), 4)

    # 中心点
    cx = priors[:, 0] + loc[:, 0] * variances[0] * priors[:, 2]
    cy = priors[:, 1] + loc[:, 1] * variances[0] * priors[:, 3]

    w = priors[:, 2] * torch.exp(loc[:, 2] * variances[1])
    h = priors[:, 3] * torch.exp(loc[:, 3] * variances[1])

    # 半宽高
    half_w = w * 0.5
    half_h = h * 0.5

    # 左上右下
    boxes[:, 0] = cx - half_w
    boxes[:, 1] = cy - half_h
    boxes[:, 2] = cx + half_w
    boxes[:, 3] = cy + half_h

    return boxes


def decode_landmarks(landms: Tensor, priors: Tensor, variances: list[float]) -> torch.Tensor:
    """
    将网络输出的关键点回归偏移解码为归一化坐标。

    编码约定：
        dx_i = (x_gt_i - cx_prior) / (variance[0] * w_prior)
        dy_i = (y_gt_i - cy_prior) / (variance[0] * h_prior)

    Args:
        landms:    Tensor[N, 10]，5 个关键点的回归偏移，x/y 交替排列。
        priors:    Tensor[N, 4]，prior anchor (cx, cy, w, h)，归一化坐标。
        variances: list[float, float]，仅使用 variances[0]（xy 方向方差）。

    Returns:
        Tensor[N, 10]，解码后的关键点归一化坐标，x/y 交替排列。
        乘以图像宽高后得到像素坐标。
    """
    N = landms.size(0)
    out = landms.new_empty(N, 10)

    cx = priors[:, 0]
    cy = priors[:, 1]
    w = priors[:, 2] * variances[0]
    h = priors[:, 3] * variances[0]

    out[:, 0::2] = cx.unsqueeze(1) + landms[:, 0::2] * w.unsqueeze(1)
    out[:, 1::2] = cy.unsqueeze(1) + landms[:, 1::2] * h.unsqueeze(1)

    return out


@torch.inference_mode()
def draw_boxes_batch(
    images: Tensor | tuple[Tensor] | list[Tensor], detected: Tensor, offset: list[int], color=(255.0, 0.0, 0.0), thickness=3, point_size=7
) -> Tensor | list[Tensor]:
    """
    在图像上绘制检测结果（bbox + 5 关键点）。

    Args:
        images:
            Tensor[B, C, H, W] | list[Tensor[C, H, W]] | tuple[Tensor[C, H, W]]
            输入图像，不会被原地修改（内部 clone）。
        detected:
            Tensor[M, 15]，detector() 返回的拼接检测结果。
        offsets:
            list[int]，detector() 返回的每张图像检测框数量，用于拆分 detected。
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

    for boxes, image in zip(torch.split(detected, offset), new_images):
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


def get_pts(detected: Tensor) -> Tensor:
    """
    从拼接检测结果中提取 5 个面部关键点坐标。

    Args:
        detected: Tensor[N, 15]，detector() 返回的检测结果（单张图像或多张拼接）。

    Returns:
        Tensor[N, 5, 2]，每行为一个人脸的 5 个关键点 (x, y) 像素坐标，
        顺序为：左眼、右眼、鼻尖、左嘴角、右嘴角。
    """
    return detected[:, -10:].reshape(-1, 5, 2)


if __name__ == "__main__":
    import torchvision.utils as utils

    from misc.utils import ImageFolder, Timer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    model = RetinaFace(True).to(device=device)

    dataset = ImageFolder("/home/liaohaixun/swap/IDAssets")
    # dataset = ImageFolder("/opt/share/deepfake/dataset_1/ffhq_1024")

    images_org = [dataset.sample2tensor().float().to(device=device) for _ in range(batch_size)]

    # images_org = torch.stack(images_org)

    for _ in range(5):
        detected, offset = model.detector(images_org)

    with Timer("model infer"):
        detected, offset = model.detector(images_org)

    visi = draw_boxes_batch(images_org, detected, offset)

    if isinstance(visi, (list, tuple)):
        visi, _ = batch_resize_and_pad_varsize(visi, 640, 640)

    utils.save_image(visi, "visi.png", normalize=True, value_range=(0, 255))
