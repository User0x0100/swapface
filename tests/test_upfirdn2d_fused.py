"""GPU fused upfirdn2d correctness and differentiability, including ROCm."""

import pytest
import torch

from models.discriminator import FIRMinibatchStdDiscriminator
from models.discriminator.upfirdn2d.upfirdn2d import _upfirdn2d_native, upfirdn2d


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize(
    ("filter_1d", "scale", "padding", "flip"),
    [
        (False, (1, 1, 2, 2), (1, 1, 1, 1), False),
        (False, (2, 2, 1, 1), (1, 1, 1, 1), False),
        (True, (1, 1, 2, 2), (1, 1, 1, 1), False),
        (False, (1, 1, 1, 1), (-1, 2, -1, 2), True),
        (False, (1, 1, 2, 2), (3, 3, 3, 3), False),
    ],
)
def test_fused_matches_float32_reference(dtype, channels_last, filter_1d, scale, padding, flip):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unsupported")

    torch.manual_seed(42)
    x = torch.randn(2, 5, 16, 20, device="cuda", dtype=torch.float32).to(dtype)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    f = torch.randn(4 if filter_1d else (4, 4), device="cuda")
    args = (*scale, *padding, flip, 1.5)

    result = upfirdn2d(x, f, *args)
    # The custom kernel accumulates with float32 filter coefficients even for BF16 inputs.
    # Native BF16 conv2d instead rounds filter coefficients to BF16.
    reference = _upfirdn2d_native(x.float(), f, *args).to(dtype)
    torch.testing.assert_close(result, reference, rtol=0.03 if dtype is torch.bfloat16 else 0.003,
                               atol=0.06 if dtype is torch.bfloat16 else 0.01)


def test_fused_first_and_second_order_gradients():
    torch.manual_seed(43)
    f = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device="cuda") / 16
    args = (1, 1, 2, 2, 1, 1, 1, 1, False, 1.)

    x = torch.randn(2, 5, 16, 20, device="cuda", requires_grad=True)
    y = upfirdn2d(x, f, *args)
    first = torch.autograd.grad(y.square().mean(), x, create_graph=True)[0]
    r1 = first.square().sum()
    second = torch.autograd.grad(r1, x)[0]
    assert first.isfinite().all()
    assert second.isfinite().all()
    assert second.abs().max() > 0

    ref_x = x.detach().clone().requires_grad_(True)
    ref_y = _upfirdn2d_native(ref_x, f, *args)
    ref_first = torch.autograd.grad(ref_y.square().mean(), ref_x)[0]
    torch.testing.assert_close(first, ref_first, atol=1e-6, rtol=1e-5)


def test_fused_discriminator_r1_on_gpu():
    torch.manual_seed(44)
    model = FIRMinibatchStdDiscriminator(
        img_resolution=32, base_ch=8, max_ch=32, minibatch_std_group_size=2
    ).cuda()
    real = torch.randn(2, 3, 32, 32, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", enabled=False):
        real_score = model(real)
        input_gradient = torch.autograd.grad(real_score.sum(), real, create_graph=True)[0]
        r1 = input_gradient.square().mean()
    r1.backward()
    grad = model.down_blocks[0].conv1.weight.grad
    assert grad is not None and grad.isfinite().all()
