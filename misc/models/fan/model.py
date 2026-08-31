import math
from enum import Enum
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .models import FAN, ResNetDepth
from huggingface_hub import hf_hub_download
from ...models import MODEL_REPOSITORY_ID


class LandmarkMode(Enum):
    L2D = 0
    L3D = 1


def hf_dl(name: str) -> str:
    return hf_hub_download(repo_id=MODEL_REPOSITORY_ID, filename=f"FaceAlignmentFan/{name}")


MODEL_LIST = {
    "2DFAN-4": "2DFAN4-11f355bf06.pth.tar",
    "3DFAN-4": "3DFAN4-7835d9f11d.pth.tar",
    "depth": "depth-2a464da4ea.pth.tar",
}


class FANLandmarkModel(nn.Module):
    """
    输入:
        image: [N, 3, H, W]
        H == W
        H/W 不一定是 256
        image 值域建议为 [0, 1]

    输出:
        mode="2d": [N, 68, 2]
        mode="3d": [N, 68, 3]

    坐标系:
        输出 x/y 坐标与输入 image 的像素坐标一致。
    """

    NUM_LANDMARKS: int = 68
    CROP_SIZE: int = 256
    HEATMAP_SIZE: int = 64
    NETWORK_SIZE: int = 4

    LR_ORDER = [
        16,
        15,
        14,
        13,
        12,
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
        0,
        26,
        25,
        24,
        23,
        22,
        21,
        20,
        19,
        18,
        17,
        27,
        28,
        29,
        30,
        35,
        34,
        33,
        32,
        31,
        45,
        44,
        43,
        42,
        47,
        46,
        39,
        38,
        37,
        36,
        41,
        40,
        54,
        53,
        52,
        51,
        50,
        49,
        48,
        59,
        58,
        57,
        56,
        55,
        64,
        63,
        62,
        61,
        60,
        67,
        66,
        65,
    ]

    def __init__(self, mode: LandmarkMode = LandmarkMode.L2D) -> None:
        super().__init__()

        self.mode = mode

        self.fan = FAN(self.NETWORK_SIZE)
        self.depth_net = None

        fan_weight_name = "2DFAN" if mode is LandmarkMode.L2D else "3DFAN"
        fan_weight = torch.load(hf_dl(MODEL_LIST[f"{fan_weight_name}-{self.NETWORK_SIZE}"]), map_location=torch.device("cpu"), weights_only=True)
        self.fan.load_state_dict(fan_weight)

        if mode is LandmarkMode.L3D:
            self.depth_net = ResNetDepth()
            depth_net_weight = torch.load(hf_dl(MODEL_LIST["depth"]), map_location=torch.device("cpu"), weights_only=False)
            depth_net_weight = {k.removeprefix("module."): v for k, v in depth_net_weight["state_dict"].items()}
            self.depth_net.load_state_dict(depth_net_weight)

        self.register_buffer("lr_order", torch.tensor(self.LR_ORDER, dtype=torch.long), persistent=False)
        self.register_buffer("depth_gaussian_kernel", self._build_depth_gaussian_kernel(), persistent=False)

    @staticmethod
    def _build_depth_gaussian_kernel() -> Tensor:

        size = 13
        sigma = 0.25
        center = 0.5 * size + 0.5
        denom = sigma * size

        ys = torch.arange(1, size + 1, dtype=torch.float)
        xs = torch.arange(1, size + 1, dtype=torch.float)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        kernel = torch.exp(-(((xx - center) / denom).square() / 2.0 + ((yy - center) / denom).square() / 2.0))

        return kernel

    @torch.inference_mode()
    def forward(self, image: Tensor, return_scores: bool = False):
        if image.ndim != 4:
            raise ValueError(f"image must be NCHW, got shape={tuple(image.shape)}")

        n, c, h, w = image.shape

        if c != 3:
            raise ValueError(f"image channel must be 3, got C={c}")

        if h != w:
            raise ValueError(f"input image must be square, got H={h}, W={w}")

        if h == self.CROP_SIZE:
            x = image
        else:
            x = F.interpolate(image, size=(self.CROP_SIZE, self.CROP_SIZE), mode="bilinear", align_corners=False)

        heatmaps = self._forward_fan(x)
        pts_256, scores = self._decode_heatmaps(heatmaps)

        scale = float(h) / float(self.CROP_SIZE)
        xy = pts_256 * scale

        if self.mode is LandmarkMode.L2D:
            out = xy
        else:
            assert self.depth_net is not None

            depth_hm = self._make_depth_heatmaps(pts_256)
            depth_in = torch.cat([x, depth_hm], dim=1)

            z = self.depth_net(depth_in)
            z = self._last_output(z)
            z = z.reshape(n, self.NUM_LANDMARKS, 1)

            # 输入图像本身就是 crop，因此 z 同样缩放到输入图像尺度
            z = z * scale

            out = torch.cat([xy, z], dim=-1)

        if return_scores:
            return out, scores

        return out

    def _forward_fan(self, x: Tensor) -> Tensor:
        out = self.fan(x)
        heatmaps = self._last_output(out)

        flip_out = self.fan(torch.flip(x, dims=(-1,)))
        flip_heatmaps = self._last_output(flip_out)
        flip_heatmaps = self._flip_heatmaps(flip_heatmaps)
        heatmaps = heatmaps + flip_heatmaps

        return heatmaps.to(dtype=torch.float32)

    @staticmethod
    def _last_output(out):
        return out[-1] if isinstance(out, (list, tuple)) else out

    def _flip_heatmaps(self, heatmaps: Tensor) -> Tensor:
        heatmaps = torch.flip(heatmaps, dims=(-1,))
        heatmaps = heatmaps.index_select(dim=1, index=self.lr_order)
        return heatmaps

    def _decode_heatmaps(self, heatmaps: Tensor) -> tuple[Tensor, Tensor]:
        n, k, hm_h, hm_w = heatmaps.shape

        if k != self.NUM_LANDMARKS:
            raise ValueError(f"expected 68 heatmaps, got {k}")

        flat = heatmaps.reshape(n, k, hm_h * hm_w)
        scores, idx = flat.max(dim=-1)

        x = (idx % hm_w).to(dtype=torch.float32)
        y = (idx // hm_w).to(dtype=torch.float32)

        pts = torch.stack([x + 0.5, y + 0.5], dim=-1)

        ix = x.to(dtype=torch.long)
        iy = y.to(dtype=torch.long)

        inside = (ix > 0) & (ix < hm_w - 1) & (iy > 0) & (iy < hm_h - 1)

        ix_c = ix.clamp(1, hm_w - 2)
        iy_c = iy.clamp(1, hm_h - 2)

        b_idx = torch.arange(n, device=heatmaps.device).view(n, 1).expand(n, k)
        k_idx = torch.arange(k, device=heatmaps.device).view(1, k).expand(n, k)

        dx = heatmaps[b_idx, k_idx, iy_c, ix_c + 1] - heatmaps[b_idx, k_idx, iy_c, ix_c - 1]
        dy = heatmaps[b_idx, k_idx, iy_c + 1, ix_c] - heatmaps[b_idx, k_idx, iy_c - 1, ix_c]

        offset = torch.stack([dx, dy], dim=-1).sign() * 0.25
        pts = pts + torch.where(inside.unsqueeze(-1), offset, torch.zeros_like(offset))

        pts_256 = pts * (float(self.CROP_SIZE) / float(self.HEATMAP_SIZE))

        return pts_256, scores

    def _make_depth_heatmaps(self, pts_256: Tensor) -> Tensor:

        device = pts_256.device
        n, k, _ = pts_256.shape
        h = w = self.CROP_SIZE

        heatmaps = torch.zeros(n, k, h, w, device=device, dtype=torch.float)

        kernel = self.depth_gaussian_kernel.to(device=device, dtype=torch.float)
        size = kernel.shape[0]  # 13

        # 官方 draw_gaussian(..., sigma=2)
        sigma = 2
        radius = 3 * sigma  # 6

        pts = pts_256.to(device=device, dtype=torch.float)

        x = pts[..., 0]
        y = pts[..., 1]

        ul_x = torch.floor(x - radius).to(torch.long)
        ul_y = torch.floor(y - radius).to(torch.long)
        br_x = torch.floor(x + radius).to(torch.long)
        br_y = torch.floor(y + radius).to(torch.long)

        # 官方外层逻辑:
        # if pts[j, 0] > 0 and pts[j, 1] > 0:
        valid_point = (x > 0.0) & (y > 0.0)

        # 官方 draw_gaussian 的越界判断
        valid_box = valid_point & (ul_x <= w) & (ul_y <= h) & (br_x >= 1) & (br_y >= 1)

        one_x = torch.ones_like(ul_x)
        one_y = torch.ones_like(ul_y)

        w_t = torch.full_like(ul_x, w)
        h_t = torch.full_like(ul_y, h)

        # 对齐官方:
        # g_x = [max(1, -ul_x),
        #        min(br_x, W) - max(1, ul_x) + max(1, -ul_x)]
        # img_x = [max(1, ul_x), min(br_x, W)]
        g_x0 = torch.maximum(one_x, -ul_x)
        g_y0 = torch.maximum(one_y, -ul_y)

        g_x1 = torch.minimum(br_x, w_t) - torch.maximum(one_x, ul_x) + g_x0
        g_y1 = torch.minimum(br_y, h_t) - torch.maximum(one_y, ul_y) + g_y0

        img_x0 = torch.maximum(one_x, ul_x)
        img_y0 = torch.maximum(one_y, ul_y)

        # 13x13 kernel 的 zero-based index
        gx = torch.arange(size, device=device, dtype=torch.long).view(1, 1, 1, size)
        gy = torch.arange(size, device=device, dtype=torch.long).view(1, 1, size, 1)

        # 官方切片是:
        # kernel[g_y0 - 1:g_y1, g_x0 - 1:g_x1]
        g_x_start = (g_x0 - 1).view(n, k, 1, 1)
        g_y_start = (g_y0 - 1).view(n, k, 1, 1)
        g_x_end = g_x1.view(n, k, 1, 1)
        g_y_end = g_y1.view(n, k, 1, 1)

        valid_x = (gx >= g_x_start) & (gx < g_x_end)  # [N, K, 1, 13]
        valid_y = (gy >= g_y_start) & (gy < g_y_end)  # [N, K, 13, 1]

        # 对应官方:
        # image[img_y0 - 1:img_y1, img_x0 - 1:img_x1]
        img_x_start = (img_x0 - 1).view(n, k, 1, 1)
        img_y_start = (img_y0 - 1).view(n, k, 1, 1)

        x_idx = img_x_start + (gx - g_x_start)  # [N, K, 1, 13]
        y_idx = img_y_start + (gy - g_y_start)  # [N, K, 13, 1]

        x_idx = x_idx.expand(n, k, size, size)
        y_idx = y_idx.expand(n, k, size, size)

        valid = valid_box.view(n, k, 1, 1) & valid_x & valid_y

        if not valid.any():
            return heatmaps

        b_idx = torch.arange(n, device=device).view(n, 1, 1, 1).expand(n, k, size, size)
        k_idx = torch.arange(k, device=device).view(1, k, 1, 1).expand(n, k, size, size)

        values = kernel.view(1, 1, size, size).expand(n, k, size, size)

        heatmaps[b_idx[valid], k_idx[valid], y_idx[valid], x_idx[valid]] = values[valid]

        # 官方有 image[image > 1] = 1
        # 这里每个 landmark 独占一个 channel，通常不会超过 1，但保留以保持语义一致。
        heatmaps.clamp_(0.0, 1.0)

        return heatmaps


def draw_landmarks_on_tensor(image: Tensor, landmarks: Tensor, *, color: tuple[float, ...] = (1.0, 0.0, 0.0), radius: int = 2, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """
    在 NCHW tensor 图像上绘制 landmark 点。

    Args:
        image:
            [N, C, H, W]，通常值域为 [0, 1]。
        landmarks:
            [N, K, 2] 或 [N, K, 3]。
            坐标为输入 image 的像素坐标。
        color:
            绘制颜色。
            C=3 时通常为 RGB，例如红色 (1, 0, 0)。
            C=1 时会使用 color[0]。
        radius:
            点半径，单位为像素。
            radius=0 表示单像素点。
        alpha:
            颜色混合系数。
            alpha=1.0 表示直接覆盖。
        inplace:
            是否原地修改 image。

    Returns:
        drawn:
            [N, C, H, W]
    """
    if image.ndim != 4:
        raise ValueError(f"image must be NCHW, got shape={tuple(image.shape)}")

    if landmarks.ndim != 3:
        raise ValueError(f"landmarks must be [N, K, 2/3], got shape={tuple(landmarks.shape)}")

    n, c, h, w = image.shape

    if landmarks.shape[0] != n:
        raise ValueError(f"batch size mismatch: image N={n}, landmarks N={landmarks.shape[0]}")

    if landmarks.shape[-1] < 2:
        raise ValueError(f"landmarks last dim must be >= 2, got {landmarks.shape[-1]}")

    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    out = image if inplace else image.clone()

    device = image.device
    dtype = image.dtype

    xy = landmarks[..., :2].to(device=device, dtype=torch.float)
    finite = torch.isfinite(xy).all(dim=-1)

    # round 到最近像素
    xy = xy.round().to(dtype=torch.long)
    base_x = xy[..., 0]
    base_y = xy[..., 1]

    # 生成圆形 brush offsets
    if radius == 0:
        dx = torch.zeros(1, device=device, dtype=torch.long)
        dy = torch.zeros(1, device=device, dtype=torch.long)
    else:
        r = torch.arange(-radius, radius + 1, device=device, dtype=torch.long)
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        mask = xx.square() + yy.square() <= radius * radius
        dx = xx[mask]
        dy = yy[mask]

    # [N, K, M]
    px = base_x.unsqueeze(-1) + dx.view(1, 1, -1)
    py = base_y.unsqueeze(-1) + dy.view(1, 1, -1)

    valid = finite.unsqueeze(-1) & (px >= 0) & (px < w) & (py >= 0) & (py < h)

    if not valid.any():
        return out

    # batch index: [N, K, M]
    b_idx = torch.arange(n, device=device).view(n, 1, 1).expand_as(px)

    # spatial flat index inside each image
    spatial_idx = py * w + px

    b = b_idx[valid]
    s = spatial_idx[valid]

    color_t = torch.as_tensor(color, device=device, dtype=dtype)

    if color_t.numel() < c:
        # 例如 C=3 但只传了 (1.0,)
        color_t = color_t[0].expand(c)
    else:
        color_t = color_t[:c]

    flat = out.reshape(n, c, h * w)

    if alpha >= 1.0:
        for ch in range(c):
            flat[b, ch, s] = color_t[ch]
    else:
        a = torch.as_tensor(alpha, device=device, dtype=dtype)
        for ch in range(c):
            old = flat[b, ch, s]
            flat[b, ch, s] = old * (1.0 - a) + color_t[ch] * a

    return out


if __name__ == "__main__":
    import torch.nn.functional as F
    import torchvision.utils as utils
    from misc.utils import ImageDirectory

    # from misc.face_alignment import center_crop_and_resize
    from torchvision.transforms import functional as TF

    sample_dir = ImageDirectory("/opt/share/deepfake/dataset_1/vggface2_hq512/align_result")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    print(f"使用设备: {device}")

    images = torch.stack([sample_dir.sample_tensor() for _ in range(batch_size)]).to(device=device)
    H, C, H, W = images.size()
    images = TF.affine(images, angle=0, translate=(0, -(H * 0.075)), scale=0.81, shear=0)

    images.div_(255.0)

    net = FANLandmarkModel(LandmarkMode.L3D).to(device)
    net.eval()

    with torch.no_grad():
        lmks = net(images)
        print(f"关键点形状: {lmks.shape}")

    # 使用PyTorch绘制关键点
    images_with_landmarks = draw_landmarks_on_tensor(images, lmks, color=(0, 1, 0), radius=2)

    # 使用torchvision创建网格并保存
    grid = utils.make_grid(images_with_landmarks, nrow=4, padding=2, normalize=True, value_range=(0.0, 1.0))
    utils.save_image(grid, "landmarks_grid_pytorch.png")

    print(f"处理了 {batch_size} 张图片")
    print(f"每张图片检测到 {lmks.shape[2]} 个关键点")
    print("关键点可视化结果已保存到 landmarks_grid_pytorch.png")
