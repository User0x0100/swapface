import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .functional import EPS

RGB_CHANNELS = 3


def _validate_rgb_image(image: Tensor) -> None:
    """校验张量最后三个维度是否符合 ``(3, H, W)`` RGB 图像约定。"""
    if not isinstance(image, Tensor):
        raise TypeError(f"image must be a Tensor, got {type(image).__name__}")
    if image.ndim < 3 or image.shape[-3] != RGB_CHANNELS:
        raise ValueError(
            f"image must have shape (*, {RGB_CHANNELS}, H, W), got {tuple(image.shape)}"
        )


def rgb_to_linear_rgb(image: Tensor) -> Tensor:
    """将 sRGB 图像转换为线性 RGB。

    参数:
        image: 形状为 ``(*, 3, H, W)`` 的 sRGB 图像张量。

    返回:
        与输入同形的线性 RGB 图像张量。

    示例:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_linear_rgb(input)
    """

    _validate_rgb_image(image)

    lin_rgb: Tensor = torch.where(
        image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92
    )

    return lin_rgb


def rgb_to_xyz(image: Tensor) -> Tensor:
    """将线性 RGB 图像转换为 CIE XYZ 颜色空间。

    参数:
        image: 形状为 ``(*, 3, H, W)`` 的线性 RGB 图像张量。

    返回:
        与输入同形的 XYZ 图像张量。

    示例:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)
    """
    _validate_rgb_image(image)

    r: Tensor = image[..., 0, :, :]
    g: Tensor = image[..., 1, :, :]
    b: Tensor = image[..., 2, :, :]

    x: Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: Tensor = torch.stack([x, y, z], -3)

    return out


def rgb_to_lab(image: Tensor) -> Tensor:
    """将 RGB 图像转换为 CIE Lab 颜色空间。

    假定输入 RGB 图像值域为 ``[0, 1]``。Lab 颜色使用 D65 标准光源和
    2° 标准观察者计算。

    参数:
        image: 形状为 ``(*, 3, H, W)``、值域为 ``[0, 1]`` 的 RGB 图像张量。

    返回:
        与输入同形的 Lab 图像张量；L 通道通常位于 ``0..100``，a/b 通道通常位于 ``-128..127``。

    示例:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)
    """
    _validate_rgb_image(image)

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im: Tensor = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype
    )[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x: Tensor = xyz_int[..., 0, :, :]
    y: Tensor = xyz_int[..., 1, :, :]
    z: Tensor = xyz_int[..., 2, :, :]

    lightness = (116.0 * y) - 16.0
    a_channel = 500.0 * (x - y)
    b_channel = 200.0 * (y - z)

    return torch.stack((lightness, a_channel, b_channel), dim=-3)


class LabStyleLoss(nn.Module):
    """基于 CIE Lab 一阶统计量计算颜色与光照风格损失。

    输入先转换到 Lab 空间，再匹配每个通道的空间均值和标准差。该损失同时约束
    ``L``、``a``、``b`` 三个通道，因此同时包含亮度/光照和色度差异。
    """

    def __init__(self, weight: float = 1.0, range_norm: bool = True) -> None:
        """初始化 Lab 风格损失。

        参数:
            weight: 最终损失的标量权重。
            range_norm: 若为 True，将输入从 ``[-1, 1]`` 映射到 ``[0, 1]``。
        """
        super().__init__()
        self.range_norm = range_norm
        self.weight = weight

    @staticmethod
    def _mean_std(x: Tensor) -> tuple[Tensor, Tensor]:
        """计算每个样本、每个通道在空间维度上的均值和标准差。"""
        variance, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, unbiased=False)
        return mean, torch.sqrt(variance + EPS)

    def _style_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """计算两组 Lab 特征均值与标准差的 L1 距离。"""
        prediction_mean, prediction_std = self._mean_std(prediction)
        target_mean, target_std = self._mean_std(target)
        return F.l1_loss(prediction_mean, target_mean) + F.l1_loss(
            prediction_std, target_std
        )

    @torch.compile(
        fullgraph=True,
        dynamic=False,
        options={"epilogue_fusion": True, "max_autotune": True},
    )
    def forward(self, prediction_rgb: Tensor, target_rgb: Tensor) -> Tensor:
        """计算两组 RGB 图像之间的 Lab 风格损失。

        参数:
            prediction_rgb: 预测 RGB 张量，形状为 ``(N, 3, H, W)``。
            target_rgb: 目标 RGB 张量，与 ``prediction_rgb`` 同形。

        返回:
            标量损失。

        异常:
            ValueError: 两个输入形状不一致或输入不符合 RGB 图像约定。
        """
        if prediction_rgb.shape != target_rgb.shape:
            raise ValueError(
                f"prediction and target shapes must match: {tuple(prediction_rgb.shape)} != {tuple(target_rgb.shape)}"
            )

        if self.range_norm:
            prediction_rgb = prediction_rgb.add(1.0).mul(0.5)
            target_rgb = target_rgb.add(1.0).mul(0.5)

        prediction_lab = rgb_to_lab(prediction_rgb.clamp(0.0, 1.0))
        target_lab = rgb_to_lab(target_rgb.clamp(0.0, 1.0))
        return self._style_loss(prediction_lab, target_lab) * self.weight
