"""StyleGAN2-order FIR blur + stride-2 convolutions in the global discriminator."""

import math

import pytest
import torch
import torch.nn.functional as F

from models.discriminator import FIRMinibatchStdDiscriminator
from models.discriminator.network import FIRDownBlock
from models.discriminator.upfirdn2d import DownFIRDn2d
from models.discriminator.upfirdn2d.upfirdn2d import setup_filter


def _reference_blur(x: torch.Tensor, padding: int) -> torch.Tensor:
    """Independent depthwise FIR reference, without the project's UpFirDn operator."""
    kernel = setup_filter([1, 3, 3, 1]).to(x).view(1, 1, 4, 4)
    x = F.pad(x, (padding, padding, padding, padding))
    return F.conv2d(x, kernel.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])


@pytest.mark.parametrize("kernel_size,padding,expected_offset", [(1, 1, -1), (3, 2, 1)])
def test_downfirdn2d_prefilter_is_not_a_second_downsample(kernel_size: int, padding: int, expected_offset: int) -> None:
    layer = DownFIRDn2d(conv_kernel_size=kernel_size)
    x = torch.randn(2, 4, 32, 32)
    actual = layer(x)
    assert layer.downx == layer.downy == 1
    assert actual.shape == (2, 4, 32 + expected_offset, 32 + expected_offset)
    torch.testing.assert_close(actual, _reference_blur(x, padding), rtol=1e-5, atol=1e-6)


def test_downfirdn2d_rejects_invalid_prefilter() -> None:
    with pytest.raises(ValueError, match="positive odd"):
        DownFIRDn2d(conv_kernel_size=2)
    with pytest.raises(ValueError, match="scale_factor"):
        DownFIRDn2d(scale_factor=1, conv_kernel_size=3)


@pytest.mark.parametrize("resolution,in_ch,out_ch", [(16, 4, 8), (32, 8, 8), (128, 8, 16)])
def test_downblock_matches_stylegan2_reference(resolution: int, in_ch: int, out_ch: int) -> None:
    block = FIRDownBlock(in_ch, out_ch)
    assert block.conv1.stride == (2, 2) and block.conv1.padding == (0, 0)
    assert block.shortcut_conv.stride == (2, 2) and block.shortcut_conv.padding == (0, 0)
    assert (block.residual_filter.padx0, block.residual_filter.padx1) == (2, 2)
    assert (block.shortcut_filter.padx0, block.shortcut_filter.padx1) == (1, 1)
    assert block.residual_filter.downx == block.shortcut_filter.downx == 1

    x = torch.randn(2, in_ch, resolution, resolution, requires_grad=True)
    actual = block(x)
    shortcut = block.shortcut_conv(_reference_blur(x, 1))
    residual = F.leaky_relu(block.conv0(x), 0.2)
    residual = F.leaky_relu(block.conv1(_reference_blur(residual, 2)), 0.2)
    expected = (shortcut + residual) / math.sqrt(2)
    assert actual.shape == (2, out_ch, resolution // 2, resolution // 2)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    actual.square().mean().backward()
    for tensor in (x.grad, block.conv0.weight.grad, block.conv1.weight.grad, block.shortcut_conv.weight.grad):
        assert tensor is not None and torch.isfinite(tensor).all()


def test_discriminator_features_and_r1_double_backward() -> None:
    model = FIRMinibatchStdDiscriminator(img_resolution=32, base_ch=8, max_ch=32, minibatch_std_group_size=2)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    score, features = model(x, return_feats=True)
    assert score.shape == (2, 1)
    assert [t.shape for t in features] == [(2, 16, 16, 16), (2, 32, 8, 8), (2, 32, 4, 4)]
    gradient = torch.autograd.grad(score.sum(), x, create_graph=True)[0]
    gradient.square().mean().backward()
    assert model.down_blocks[0].conv1.weight.grad is not None
    assert torch.isfinite(model.down_blocks[0].conv1.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm GPU")
def test_cuda_downblock_forward_backward() -> None:
    block = FIRDownBlock(8, 16).cuda()
    x = torch.randn(2, 8, 32, 32, device="cuda", requires_grad=True)
    output = block(x)
    assert output.shape == (2, 16, 16, 16)
    output.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
