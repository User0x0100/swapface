import torch
import torch.nn.functional as F
from torch import Tensor, nn, autocast
from typing import Literal, Callable, Mapping

from .vgg import VGGFeatureExtractor
from misc.models.idencoder import PROVIDER, IDEncoder

EPS = 1e-8

LossFn = Callable[[Tensor, Tensor], Tensor]


def create_weighted_loss(loss_fn, weight=1.0, reduction="mean") -> LossFn:
    """将任意损失函数包装为带权重的版本。

    Args:
        loss_fn: 签名为 ``(pred, target, reduction=...) -> Tensor`` 的损失函数。
        weight (float): 标量权重，结果乘以该值。默认 ``1.0``。
        reduction (str): 传递给 ``loss_fn`` 的 reduction 模式。默认 ``"mean"``。

    Returns:
        LossFn: 签名为 ``(pred, target) -> Tensor`` 的包装函数。
    """

    def wrapper(pred, target):
        return weight * loss_fn(pred, target, reduction=reduction)

    return wrapper


def charbonnier_loss(pred: Tensor, target: Tensor, reduction: Literal["none", "mean", "sum"] = "mean") -> Tensor:
    """Charbonnier 损失（pseudo-Huber / L1 平滑近似）。

    计算公式：``loss = sqrt((pred - target)^2 + eps)``，
    相比 L1 在零点附近可微，相比 MSE 对离群值更鲁棒。

    Args:
        pred (Tensor):   预测张量，任意形状。
        target (Tensor): 目标张量，与 ``pred`` 形状相同。
        reduction (str): ``"none"`` / ``"mean"`` / ``"sum"``。默认 ``"mean"``。

    Returns:
        Tensor: 标量（``mean``/``sum``）或与输入同形张量（``none``）。
    """

    loss = torch.sqrt((pred - target).pow(2).add_(EPS))

    match reduction:
        case "none":
            return loss
        case "mean":
            return loss.mean()
        case "sum":
            return loss.sum()
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def orthogonal_loss(x: Tensor, y: Tensor, reduction: Literal["none", "mean", "sum"] = "mean") -> Tensor:
    """正交损失：惩罚两组特征向量之间的余弦相似度。

    先对输入沿 ``dim=1`` 做 L2 归一化，再计算逐样本点积绝对值，
    鼓励 ``x`` 与 ``y`` 在特征空间中相互正交（解耦）。

    Args:
        x (Tensor): 形状 ``(N, C)`` 的特征张量。
        y (Tensor): 形状 ``(N, C)`` 的特征张量。
        reduction (str): ``"none"`` / ``"mean"`` / ``"sum"``。默认 ``"mean"``。

    Returns:
        Tensor: 标量或形状 ``(N,)`` 的逐样本损失。
    """

    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)

    loss = torch.abs(torch.sum(x * y, dim=1))

    match reduction:
        case "none":
            return loss
        case "mean":
            return loss.mean()
        case "sum":
            return loss.sum()


def l1_loss_fn(weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean") -> LossFn:
    """工厂函数：返回带权重的 L1 损失。"""
    return create_weighted_loss(F.l1_loss, weight, reduction)


def mse_loss_fn(weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean") -> LossFn:
    """工厂函数：返回带权重的 MSE（L2）损失。"""
    return create_weighted_loss(F.mse_loss, weight, reduction)


def charbonnier_loss_fn(weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean") -> LossFn:
    """工厂函数：返回带权重的 Charbonnier 损失。"""
    return create_weighted_loss(charbonnier_loss, weight, reduction)


def bce_loss_fn(weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean") -> LossFn:
    """工厂函数：返回带权重的 BCE 损失（强制禁用 AMP）。

    ``binary_cross_entropy`` 要求输入已经过 sigmoid，数值范围 ``[0, 1]``。
    内部通过 ``autocast(enabled=False)`` 规避混合精度溢出。
    """
    f = create_weighted_loss(F.binary_cross_entropy, weight, reduction)

    def disable_amp(*args, **kwargs):
        with autocast(device_type="cuda", enabled=False):
            return f(*args, **kwargs)

    return disable_amp


def bce_with_logits_loss_fn(weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean") -> LossFn:
    """工厂函数：返回带权重的 BCE-with-logits 损失（强制禁用 AMP）。

    接受未经 sigmoid 的 logits，内部数值更稳定。
    同样通过 ``autocast(enabled=False)`` 禁用混合精度。
    """
    f = create_weighted_loss(F.binary_cross_entropy_with_logits, weight, reduction)

    def disable_amp(*args, **kwargs):
        with autocast(device_type="cuda", enabled=False):
            return f(*args, **kwargs)

    return disable_amp


def orthogonal_loss_fn(weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean") -> LossFn:
    """工厂函数：返回带权重的正交损失。"""
    return create_weighted_loss(orthogonal_loss, weight, reduction)


class WFMLoss(nn.Module):
    def __init__(
        self,
        layer_weights: Mapping[int, float],
        criterion: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()

        self.criterion = {
            "l1": F.l1_loss,
            "mse": F.mse_loss,
            "charbonnier": charbonnier_loss,
        }[criterion]

        self.reduction = reduction
        self.layer_weights = layer_weights

    def forward(self, x_feats: list[Tensor], y_feats: list[Tensor]) -> Tensor:

        loss = 0.0
        for idx, weight in self.layer_weights.items():
            match self.reduction:
                case "mean" | "sum":
                    loss += self.criterion(x_feats[idx], y_feats[idx], reduction=self.reduction) * weight
                case "none":
                    diff = self.criterion(x_feats[idx], y_feats[idx], reduction="none")
                    loss += diff.mean(dim=tuple(range(1, diff.dim()))) * weight

        return loss


class DINOv2PerceptualLoss(nn.Module):
    def __init__(
        self,
        layer_weights: Mapping[int, float],
        criterion: Literal["l1", "mse", "charbonnier", "cosine"] = "cosine",
        reduction: Literal["mean", "sum", "none"] = "mean",
        dino_type="dinov2_vitb14_reg",
        use_input_norm: bool = True,
        range_norm: bool = True,
    ):
        """
        Args:
            layer_weights (Mapping): The weight for each layer of vgg feature.
            use_input_norm (bool):  If True, normalize the input image.
                Default: True.
            range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
                Default: False.
        """
        super().__init__()

        self.criterion = {
            "l1": F.l1_loss,
            "mse": F.mse_loss,
            "charbonnier": charbonnier_loss,
            "cosine": self._cosine_distance,
        }[criterion]

        self.reduction = reduction
        self.layer_weights = layer_weights

        self.dino = torch.hub.load("facebookresearch/dinov2", dino_type)
        self.dino.eval()
        self.dino.requires_grad_(False)

        self.n_blocks = max(self.layer_weights.keys()) + 1

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.register_buffer("mean", Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.range_norm = range_norm

    @staticmethod
    def _cosine_distance(x: Tensor, y: Tensor, reduction: Literal["mean", "sum", "none"] = "mean") -> Tensor:

        x_norm, y_norm = F.normalize(x, p=2, dim=-1), F.normalize(y, p=2, dim=-1)
        loss = 1.0 - F.cosine_similarity(x_norm, y_norm, dim=-1)
        match reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case "none":
                return loss

    # @torch.compile(fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})
    def forward(self, x: Tensor, y: Tensor) -> Tensor:

        x = F.interpolate(x, [224, 224], mode="bilinear", align_corners=False)
        y = F.interpolate(y, [224, 224], mode="bilinear", align_corners=False)

        if self.range_norm:
            x = (x + 1) * 0.5
            y = (y + 1) * 0.5
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        x_patch = self.dino.get_intermediate_layers(x, n=self.n_blocks)
        y_patch = self.dino.get_intermediate_layers(y, n=self.n_blocks)

        loss = 0.0
        for idx, weight in self.layer_weights.items():
            x_feat, y_feat = x_patch[idx], y_patch[idx]
            match self.reduction:
                case "mean" | "sum":
                    loss += self.criterion(x_feat, y_feat, reduction=self.reduction) * weight
                case "none":
                    diff = self.criterion(x_feat, y_feat, reduction="none")
                    loss += diff.mean(dim=tuple(range(1, diff.dim()))) * weight

        return loss


class VGGPerceptualLoss(nn.Module):
    def __init__(
        self,
        layer_weights: Mapping[str, float],
        criterion: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: Literal["mean", "sum", "none"] = "mean",
        vgg_type="vgg19",
        use_input_norm: bool = True,
        range_norm: bool = True,
    ):
        """
        参数：
            layer_weights (Mapping[str, float]):
                指定需要提取的 VGG 特征层以及对应的损失权重。
                例如：{'conv5_4': 1.0} 表示提取 VGG 的 conv5_4（relu5_4 之前）
                特征，并在计算感知损失时乘以权重 1.0。

            criterion (str):
                用于计算特征图差异的损失函数类型，可选：
                    - 'l1'：L1 损失
                    - 'mse'：均方误差损失
                    - 'charbonnier'：Charbonnier 损失（L1 的平滑版本）

            reduction (str):
                指定每一层特征损失的聚合方式，可选：

                    - 'mean'：
                        对特征差异的所有元素求平均值，
                        最终返回一个标量 loss。

                    - 'sum'：
                        对特征差异的所有元素求和，
                        最终返回一个标量 loss。

                    - 'none'：
                        不对 batch 维度进行聚合。内部仍会对
                        (C, H, W) 维度求平均，因此返回 shape
                        为 (N,) 的逐样本 perceptual loss。

            vgg_type (str):
                使用的 VGG 网络类型，例如 'vgg16' 或 'vgg19'。

            use_input_norm (bool):
                若为 True，则在输入 VGG 前使用 ImageNet
                的均值和方差对输入图像进行归一化。

            range_norm (bool):
                若为 True，则会先将输入图像从 [-1, 1] 范围
                映射到 [0, 1]，再进行 VGG 归一化。
        """
        super().__init__()

        self.vgg = VGGFeatureExtractor(layer_names=list(layer_weights.keys()), vgg_type=vgg_type).eval().requires_grad_(False)

        self.criterion = {
            "l1": F.l1_loss,
            "mse": F.mse_loss,
            "charbonnier": charbonnier_loss,
        }[criterion]

        self.weights = list(layer_weights.values())
        self.reduction = reduction

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.range_norm = range_norm

    @torch.compile(fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})
    def forward(self, x: Tensor, y: Tensor) -> Tensor:

        if self.range_norm:
            x = (x + 1) * 0.5
            y = (y + 1) * 0.5
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        fx: list[Tensor] = self.vgg(x)
        fy: list[Tensor] = self.vgg(y)

        loss = 0.0
        for weight, a, b in zip(self.weights, fx, fy):
            match self.reduction:
                case "mean" | "sum":
                    loss += self.criterion(a, b, reduction=self.reduction) * weight
                case "none":
                    diff = self.criterion(a, b, reduction="none")
                    loss += diff.mean(dim=tuple(range(1, diff.dim()))) * weight

        return loss


class DSSIMLoss(nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
        reduction: Literal["none", "mean", "sum"] = "none",
        range_norm: bool = True,
    ):
        super().__init__()

        assert window_size % 2 == 1, "window_size must be odd"
        self.range_norm = range_norm
        self.C = 3

        self.weight = weight
        self.window_size = window_size
        self.sigma = sigma
        self.reduction = reduction

        self.register_buffer("window", self._create_window(window_size, sigma).expand(self.C, 1, self.window_size, self.window_size))

    @staticmethod
    def _gaussian_1d(window_size: int, sigma: float) -> Tensor:
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g

    def _create_window(self, window_size: int, sigma: float) -> Tensor:
        g1d = self._gaussian_1d(window_size, sigma)
        g2d = torch.outer(g1d, g1d)
        window = g2d[None, None, :, :]  # (1,1,ks,ks)
        return window

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:
        """
        返回 SSIM map: (N, 1, H, W)
        """

        C = self.C
        window = self.window

        # 使用 group conv，逐通道独立计算
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=C)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=C)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=self.window_size // 2, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=self.window_size // 2, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=C) - mu_xy

        sigma_x2 = torch.clamp(sigma_x2, min=0.0)
        sigma_y2 = torch.clamp(sigma_y2, min=0.0)

        # SSIM 常量（假设输入范围 [0,1]）
        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map: Tensor = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + EPS)

        # RGB → mean over channel
        return ssim_map.mean(dim=1, keepdim=True)

    @torch.compile(fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})
    def forward(self, x: Tensor, y: Tensor) -> Tensor:

        if self.range_norm:
            x = x.add(1.0).mul(0.5)
            y = y.add(1.0).mul(0.5)

        ssim = self._ssim(x, y)  # (N,1,H,W)
        dssim = (1.0 - ssim) * 0.5 * self.weight  # (N,1,H,W)

        match self.reduction:
            case "none":
                return dssim
            case "mean":
                return dssim.mean()
            case "sum":
                return dssim.sum()


def rgb_to_linear_rgb(image: Tensor) -> Tensor:
    r"""Convert an sRGB image to linear RGB. Used in colorspace conversions.

    .. image:: _static/img/rgb_to_linear_rgb.png

    Args:
        image: sRGB Image to be converted to linear RGB of shape :math:`(*,3,H,W)`.

    Returns:
        linear RGB version of the image with shape of :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_linear_rgb(input) # 2x3x4x5

    """

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    lin_rgb: Tensor = torch.where(image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92)

    return lin_rgb


def rgb_to_xyz(image: Tensor) -> Tensor:
    r"""Convert a RGB image to XYZ.

    .. image:: _static/img/rgb_to_xyz.png

    Args:
        image: RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
         XYZ version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # 2x3x4x5

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0, :, :]
    g: Tensor = image[..., 1, :, :]
    b: Tensor = image[..., 2, :, :]

    x: Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: Tensor = torch.stack([x, y, z], -3)

    return out


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to Lab.

    .. image:: _static/img/rgb_to_lab.png

    The input RGB image is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image: RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.
        The L channel values are in the range 0..100. a and b are in the range -128..127.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116.0 * y) - 16.0
    a: torch.Tensor = 500.0 * (x - y)
    _b: torch.Tensor = 200.0 * (y - z)

    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

    return out


class StyleLossLabChroma(nn.Module):
    def __init__(self, weight: float = 1.0, range_norm: bool = True):
        """
        Args:
            range_norm (bool): 如果输入值域为 [-1, 1] 则转换为 [0, 1].
                Default: True.
        """
        super().__init__()

        self.range_norm = range_norm
        self.weight = weight

    def _mean_std(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var + EPS)
        return mean, std

    def _style_loss_mean_std(self, pred: Tensor, target: Tensor, weight: float = 1.0) -> Tensor:

        m_p, s_p = self._mean_std(pred)
        m_s, s_s = self._mean_std(target)

        loss = F.l1_loss(m_p, m_s) + F.l1_loss(s_p, s_s)
        return loss * weight

    @torch.compile(fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})
    def forward(self, pred_rgb: Tensor, target_rgb: Tensor) -> Tensor:

        if self.range_norm:
            pred_rgb = pred_rgb.add(1.0).mul(0.5)
            target_rgb = target_rgb.add(1.0).mul(0.5)

        pred_rgb = pred_rgb.clamp(0.0, 1.0)
        target_rgb = target_rgb.clamp(0.0, 1.0)

        pred_lab = rgb_to_lab(pred_rgb)
        target_lab = rgb_to_lab(target_rgb)

        return self._style_loss_mean_std(pred_lab, target_lab, self.weight)


def r1_reg_loss(real_score: Tensor, real_img: Tensor, gamma: float = 10.0) -> Tensor:
    if not real_img.requires_grad:
        raise ValueError("real_img must have requires_grad=True for R1 regularization. Call real_img.requires_grad_(True) before the discriminator forward pass.")
    r1_grads = torch.autograd.grad(outputs=[real_score.sum()], inputs=[real_img], create_graph=True, only_inputs=True)[0]
    r1_penalty = r1_grads.square().sum([1, 2, 3])
    r1_loss = r1_penalty * (gamma / 2)

    return r1_loss.mean()


class DLoss(nn.Module):
    def __init__(
        self,
        loss_type: Literal["hinge", "wgan", "ls", "bce"] = "hinge",
        weight: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "none",
    ):
        super().__init__()

        self.reduction = reduction
        self.weight = weight

        self.loss_fn = {"hinge": self._hinge, "wgan": self._wgan, "ls": self._ls, "bce": self._bce}.get(loss_type)
        if self.loss_fn is None:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Only 'hinge' 'wgan' 'ls' 'bce' are supported.")

    def _hinge(self, fake_score: Tensor, real_score: Tensor) -> Tensor:
        loss_real = torch.relu(1.0 - real_score)
        loss_fake = torch.relu(1.0 + fake_score)
        loss = loss_real + loss_fake
        return loss

    def _wgan(self, fake_score: Tensor, real_score: Tensor) -> Tensor:
        loss = fake_score - real_score
        return loss

    def _ls(self, fake_score: Tensor, real_score: Tensor) -> Tensor:
        loss_real = (real_score - 1) ** 2
        loss_fake = fake_score**2
        loss = loss_real + loss_fake
        return loss

    @torch.autocast(device_type="cuda", enabled=False)
    def _bce(self, fake_score: Tensor, real_score: Tensor) -> Tensor:
        real_label = torch.ones_like(real_score)
        fake_label = torch.zeros_like(fake_score)
        loss_real = F.binary_cross_entropy_with_logits(real_score, real_label, reduction="none")
        loss_fake = F.binary_cross_entropy_with_logits(fake_score, fake_label, reduction="none")
        loss = loss_real + loss_fake
        return loss

    def forward(self, fake_score: Tensor, real_score: Tensor) -> Tensor:

        loss = self.weight * self.loss_fn(fake_score, real_score)

        match self.reduction:
            case "none":
                return loss
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()


class GANLoss(nn.Module):
    def __init__(self, loss_type: Literal["hinge", "wgan", "ls", "bce"] = "hinge", weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "none"):
        super().__init__()

        self.reduction = reduction
        self.weight = weight

        self.loss_fn = {"hinge": self._hinge_and_wgan, "wgan": self._hinge_and_wgan, "ls": self._ls, "bce": self._bce}.get(loss_type)
        if self.loss_fn is None:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Only 'hinge' 'wgan' 'ls' 'bce' are supported.")

    def _hinge_and_wgan(self, pred_score: Tensor) -> Tensor:
        return -pred_score

    def _ls(self, pred_score: Tensor) -> Tensor:
        return (pred_score - 1).pow(2)

    @torch.autocast(device_type="cuda", enabled=False)
    def _bce(self, pred_score: Tensor) -> Tensor:
        real_label = torch.ones_like(pred_score)
        return F.binary_cross_entropy_with_logits(pred_score, real_label, reduction="none")

    def forward(self, pred_score: Tensor) -> Tensor:

        loss = self.weight * self.loss_fn(pred_score)

        match self.reduction:
            case "none":
                return loss
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()


class IDLoss(nn.Module):
    Provider = PROVIDER

    def __init__(self, provider: Provider = Provider.MS1MV3_ARCFACE_R50_FP16, weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.idencoder = IDEncoder(provider=provider)

        self.eval()
        self.requires_grad_(False)

    @torch.compile(fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})
    def get_id_feats(self, face: Tensor) -> Tensor:
        return self.idencoder(face)

    def forward(self, fake_id: Tensor, real_id: Tensor) -> Tensor:
        loss = (1.0 - F.cosine_similarity(fake_id, real_id)) * self.weight

        match self.reduction:
            case "none":
                return loss
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()


class IFSRLoss(nn.Module):
    def __init__(self, ifsr_scale: float, ifsr_weight: dict[str, tuple[float, float]], idencoder_provider: PROVIDER = PROVIDER.MS1MV3_ARCFACE_R50_FP16):
        super().__init__()

        idencoder = IDEncoder(provider=idencoder_provider)
        self.feature_layer_indices: dict[int, str] = {}

        self.ifsr_weight = {k: (m * ifsr_scale, w) for k, (m, w) in ifsr_weight.items()}

        net = nn.ModuleList()

        ifsr_layer_name = list(ifsr_weight.keys())
        max_layer_idx = 0
        idx = 0
        for module_name_0, module_0 in idencoder.backbone.named_children():
            if not list(module_0.children()):
                net.append(module_0)
                idx += 1
                continue

            for module_name_1, module_1 in module_0.named_children():
                module_name = f"{module_name_0}.{module_name_1}"

                net.append(module_1)
                if module_name in ifsr_layer_name:
                    self.feature_layer_indices[idx] = module_name
                    max_layer_idx = max(max_layer_idx, idx)

                idx += 1

        self.net = net[: max_layer_idx + 1].eval().requires_grad_(False)

    @torch.compile(
        fullgraph=True,
        dynamic=False,
        options={"epilogue_fusion": True, "max_autotune": True},
    )
    def get_ifsr_feats(self, x: Tensor) -> dict[str, Tensor]:
        feates: dict[str, Tensor] = {}

        for idx, module in enumerate(self.net):
            x = module(x)

            if (layer_name := self.feature_layer_indices.get(idx)) is not None:
                feates[layer_name] = x

        return feates

    def forward(self, src_ifsr_feats: dict[str, Tensor], dst_ifsr_feats: dict[str, Tensor]):
        loss = 0.0
        for layer_name, (margin, weight) in self.ifsr_weight.items():
            feat_true_flat = src_ifsr_feats[layer_name].flatten(1)
            feat_pred_flat = dst_ifsr_feats[layer_name].flatten(1)
            distance = (1.0 - F.cosine_similarity(feat_true_flat, feat_pred_flat, dim=1)).mean()

            d_loss = F.relu(distance - margin) * weight
            loss += d_loss

        return loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    losses = VGGPerceptualLoss(
        layer_weights={
            "relu2_2": 1.0,
            "relu3_3": 1.0,
        },
        range_norm=False,
        reduction="none",
    ).to(device)

    x = torch.randn((batch_size, 3, 256, 256), dtype=torch.float, device=device)
    y = torch.randn((batch_size, 3, 256, 256), dtype=torch.float, device=device)

    loss: Tensor = losses(x, y)

    print(loss.shape)
