# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom PyTorch ops for efficient resampling of 2D images."""

import torch
import numpy as np
from torch import Tensor, nn


def _assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}")
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            torch._assert(torch.equal(torch.as_tensor(size), ref_size), f"Wrong size for dimension {idx}")
        elif isinstance(size, torch.Tensor):
            torch._assert(torch.equal(size, torch.as_tensor(ref_size)), f"Wrong size for dimension {idx}: expected {ref_size}")
        elif size != ref_size:
            raise AssertionError(f"Wrong size for dimension {idx}: got {size}, expected {ref_size}")


def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _get_filter_size(f: Tensor):

    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    _assert_shape(f, [fh, fw][: f.ndim])
    assert fw >= 1 and fh >= 1
    return fw, fh


def setup_filter(f, normalize: bool = True, flip_filter: bool = False, gain: int = 1, separable=None) -> Tensor:
    r"""Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = f.ndim == 1 and f.numel() >= 8
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    return f


def upfirdn2d(
    x: Tensor,
    f: Tensor,
    upx: int,
    upy: int,
    downx: int,
    downy: int,
    padx0: int,
    padx1: int,
    pady0: int,
    pady1: int,
    flip_filter: bool,
    gain: float,
) -> Tensor:
    r"""
    Perform fused upsample → FIR filter → downsample (UpFirDn) on 2D images.

    This is a high-performance resampling primitive widely used in GANs
    (e.g., StyleGAN2/3) to ensure alias-free scaling.

    Pipeline:
        1. Upsample (insert zeros between pixels)
        2. Apply FIR filter (low-pass filtering)
        3. Downsample (strided subsampling)

    This function dispatches to a custom CUDA kernel:
        torch.ops.upfirdn2d.upfirdn2d

    Args:
        x:
            Input tensor of shape [N, C, H, W]

        f:
            FIR filter:
                - [kh, kw]&#58; 2D filter
                - [k]&#58; separable 1D filter

        upx, upy:
            Upsampling factors (>=1)

        downx, downy:
            Downsampling factors (>=1)

        padx0, padx1, pady0, pady1:
            Padding applied AFTER upsampling, BEFORE filtering.
            Negative values indicate cropping.

        flip_filter:
            False → convolution (filter flipped)
            True  → correlation (no flip)

        gain:
            Output scaling factor

    Returns:
        Tensor:
            Output shape:
                H_out = floor((H * upy + pady0 + pady1 - fh) / downy) + 1
                W_out = floor((W * upx + padx0 + padx1 - fw) / downx) + 1

    Implementation details:
        - If f is 2D → single fused CUDA call
        - If f is 1D → separable filtering:
            * horizontal pass
            * vertical pass
        - gain is split as sqrt(gain) per pass for separable case

    Notes:
        - Padding is NOT standard conv padding; it is applied in signal space
        - This operator is fully differentiable (custom backward registered)
        - Critical for preventing aliasing in downsampling
    """

    if f.ndim == 2:
        x = torch.ops.upfirdn2d.upfirdn2d(x, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
    else:
        x = torch.ops.upfirdn2d.upfirdn2d(x, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, np.sqrt(gain))
        x = torch.ops.upfirdn2d.upfirdn2d(x, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, np.sqrt(gain))
    return x


class UpFIRDn2d(nn.Module):
    def __init__(self, filt_size: int = 4, scale_factor: int = 2, padding: int = 0, flip_filter: bool = False, gain: int = 1) -> None:
        r"""Anti-aliased upsampling layer using FIR filtering.

        Performs:
            upsample → FIR filter → optional padding adjustment

        This layer is commonly used in GAN generators to replace naive
        interpolation (e.g., nearest/bilinear) with alias-free upsampling.

        Filter:
            Uses binomial (Pascal) filters:
                filt_size=4 → [1, 3, 3, 1]
                filt_size=5 → [1, 4, 6, 4, 1]
            These approximate Gaussian low-pass filters.

        Args:
            filt_size:
                Size of FIR filter kernel (1–7 supported)

            scale_factor:
                Upsampling factor (int or [x, y])

            padding:
                Additional padding applied to output

            flip_filter:
                Convolution vs correlation mode

            gain:
                Output scaling factor

        Behavior:
            - Inserts zeros between pixels (upsampling)
            - Applies low-pass filter to remove spectral images
            - Adjusts padding to maintain alignment

        Notes:
            - Prevents checkerboard artifacts
            - Essential for high-quality GAN image synthesis
            - Matches StyleGAN2 implementation
        """
        super().__init__()

        resample_filter = {
            1: [1.0],
            2: [1.0, 1.0],
            3: [1.0, 2.0, 1.0],
            4: [1.0, 3.0, 3.0, 1.0],
            5: [1.0, 4.0, 6.0, 4.0, 1.0],
            6: [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
            7: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        }[filt_size]

        resample_filter = setup_filter(resample_filter, flip_filter=flip_filter)
        self.register_buffer("f", resample_filter, persistent=False)
        fw, fh = _get_filter_size(resample_filter)

        self.upx, self.upy = _parse_scaling(scale_factor)
        self.downx, self.downy = _parse_scaling(1)
        padx0, padx1, pady0, pady1 = _parse_padding(padding)

        self.padx0 = padx0 + (fw + self.upx - 1) // 2
        self.padx1 = padx1 + (fw - self.upx) // 2
        self.pady0 = pady0 + (fh + self.upy - 1) // 2
        self.pady1 = pady1 + (fh - self.upy) // 2

        self.flip_filter = flip_filter
        self.gain = gain * self.upx * self.upy

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x:           Float32/float64/float16/bfloat16 input tensor of the shape
                        `[batch_size, num_channels, in_height, in_width]`.
        Returns:
            Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
        """

        return upfirdn2d(x, self.f, self.upx, self.upy, self.downx, self.downy, self.padx0, self.padx1, self.pady0, self.pady1, self.flip_filter, self.gain)


class DownFIRDn2d(nn.Module):
    def __init__(self, filt_size: int = 4, scale_factor: int = 2, padding: int = 0, flip_filter: bool = False, gain: int = 1) -> None:
        r"""Anti-aliased downsampling layer using FIR filtering.

        Performs:
            FIR filter → downsample

        This layer ensures proper low-pass filtering before subsampling,
        preventing aliasing artifacts.

        Args:
            filt_size:
                FIR filter size (binomial kernel)

            scale_factor:
                Downsampling factor

            padding:
                Input-side padding

            flip_filter:
                Convolution vs correlation

            gain:
                Output scaling

        Behavior:
            - Applies low-pass filter BEFORE decimation
            - Ensures signal is band-limited
            - Avoids aliasing during downsampling

        Notes:
            - Critical for discriminator stability
            - Often paired with UpFIRDn2d
            - Equivalent to anti-aliased pooling
        """
        super().__init__()

        resample_filter = {
            1: [1.0],
            2: [1.0, 1.0],
            3: [1.0, 2.0, 1.0],
            4: [1.0, 3.0, 3.0, 1.0],
            5: [1.0, 4.0, 6.0, 4.0, 1.0],
            6: [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
            7: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        }[filt_size]

        resample_filter = setup_filter(resample_filter, flip_filter=flip_filter)
        self.register_buffer("f", resample_filter, persistent=False)
        fw, fh = _get_filter_size(resample_filter)
        self.scale_factor = scale_factor

        self.upx, self.upy = _parse_scaling(1)
        self.downx, self.downy = _parse_scaling(scale_factor)
        padx0, padx1, pady0, pady1 = _parse_padding(padding)

        self.padx0 = padx0 + (fw - self.downx + 1) // 2
        self.padx1 = padx1 + (fw - self.downx) // 2
        self.pady0 = pady0 + (fh - self.downy + 1) // 2
        self.pady1 = pady1 + (fh - self.downy) // 2

        self.flip_filter = flip_filter
        self.gain = gain

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x:           Float32/float64/float16/bfloat16 input tensor of the shape
                    `[batch_size, num_channels, in_height, in_width]`.
        Returns:
            Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
        """
        return upfirdn2d(x, self.f, self.upx, self.upy, self.downx, self.downy, self.padx0, self.padx1, self.pady0, self.pady1, self.flip_filter, self.gain)


def setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, int, int, int, int, int, int, int, int, bool, float],
    output: Tensor,
):
    x, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain = inputs
    ctx.save_for_backward(f)
    ctx.x_shape = x.shape

    ctx.padx0, ctx.pady0 = padx0, pady0
    ctx.upx, ctx.upy, ctx.downx, ctx.downy = upx, upy, downx, downy
    ctx.flip_filter = flip_filter
    ctx.gain = gain


def backward(ctx, dy: Tensor):
    f = ctx.saved_tensors[0]
    _, _, ih, iw = ctx.x_shape
    _, _, oh, ow = dy.shape
    fw, fh = f.shape
    padx0, pady0 = ctx.padx0, ctx.pady0
    upx, upy, downx, downy = ctx.upx, ctx.upy, ctx.downx, ctx.downy
    flip_filter = ctx.flip_filter
    gain = ctx.gain

    px0 = fw - padx0 - 1
    px1 = iw * upx - ow * downx + padx0 - upx + 1
    py0 = fh - pady0 - 1
    py1 = ih * upy - oh * downy + pady0 - upy + 1

    dx = None
    df = None

    if ctx.needs_input_grad[0]:
        dx = upfirdn2d(dy, f, downx, downy, upx, upy, px0, px1, py0, py1, not flip_filter, gain)

    assert not ctx.needs_input_grad[1]
    return dx, df, None, None, None, None, None, None, None, None, None, None


def load_kernel_fn():

    from pathlib import Path
    from torch.utils.cpp_extension import load
    from torch.library import register_autograd

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)

    cuda_arch_flags = f"-arch=sm_{major}{minor}"

    BASE_DIR = Path(__file__).resolve().parent
    load(
        "upfirdn2d",
        sources=[BASE_DIR / "upfirdn2d.cpp", BASE_DIR / "upfirdn2d.cu"],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=["-O3", "-use_fast_math", "--expt-relaxed-constexpr", cuda_arch_flags],
        extra_cflags=["-O3"],
    )

    """
    Autograd support:

    The backward pass of upfirdn2d is implemented by applying the same operator
    with swapped up/down factors and adjusted padding.

    This ensures:
        - Exact gradient propagation
        - No numerical approximation
        - Full compatibility with PyTorch autograd

    Key idea:
        Forward:  up → filter → down
        Backward: down → filter → up
    """
    register_autograd("upfirdn2d::upfirdn2d", backward, setup_context=setup_context)


load_kernel_fn()
