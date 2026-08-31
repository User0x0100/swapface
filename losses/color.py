import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .functional import EPS

RGB_CHANNELS = 3


def _validate_rgb_image(image: Tensor) -> None:
    if not isinstance(image, Tensor):
        raise TypeError(f"image must be a Tensor, got {type(image).__name__}")
    if image.ndim < 3 or image.shape[-3] != RGB_CHANNELS:
        raise ValueError(
            f"image must have shape (*, {RGB_CHANNELS}, H, W), got {tuple(image.shape)}"
        )


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

    _validate_rgb_image(image)

    lin_rgb: Tensor = torch.where(
        image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92
    )

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
    def __init__(self, weight: float = 1.0, range_norm: bool = True) -> None:
        super().__init__()
        self.range_norm = range_norm
        self.weight = weight

    @staticmethod
    def _mean_std(x: Tensor) -> tuple[Tensor, Tensor]:
        variance, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, unbiased=False)
        return mean, torch.sqrt(variance + EPS)

    def _style_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
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
