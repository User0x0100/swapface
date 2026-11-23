"""R1 matmul forward must share discriminator weights and preserve R1 gradients."""

import pytest
import torch

from losses import r1_reg_loss
from models.discriminator import build_discriminator
from swapface.r1_matmul import make_r1_matmul_discriminator


def test_r1_matmul_shares_live_parameters_and_matches_loss_gradients() -> None:
    torch.manual_seed(11)
    model = build_discriminator({
        "type": "FIR_MINIBATCH_STD",
        "img_resolution": 16,
        "base_ch": 4,
        "max_ch": 16,
        "minibatch_std_group_size": 2,
    })
    r1_model = make_r1_matmul_discriminator(model)

    original_params = dict(model.named_parameters())
    for name, parameter in r1_model.named_parameters():
        assert parameter is original_params[name]

    real = torch.randn(2, 3, 16, 16)

    def calculate(module):
        model.zero_grad(set_to_none=True)
        image = real.clone().requires_grad_(True)
        score = module(image)
        penalty = r1_reg_loss(score, image, gamma=1.6384)
        penalty.backward()
        grads = [param.grad.clone() if param.grad is not None else None for param in model.parameters()]
        return score.detach(), penalty.detach(), grads

    native_score, native_loss, native_grads = calculate(model)
    matmul_score, matmul_loss, matmul_grads = calculate(r1_model)

    torch.testing.assert_close(matmul_score, native_score, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(matmul_loss, native_loss, rtol=1e-3, atol=1e-6)
    for original, matmul in zip(native_grads, matmul_grads, strict=True):
        assert (original is None) == (matmul is None)
        if original is not None:
            torch.testing.assert_close(matmul, original, rtol=5e-3, atol=1e-5)

    # The copy must always see updated optimizer parameters, not stale initial values.
    with torch.no_grad():
        model.from_rgb.weight.add_(0.1)
    torch.testing.assert_close(r1_model.from_rgb.weight, model.from_rgb.weight)


@pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm required")
def test_rocm_bf16_r1_matmul_backward() -> None:
    model = build_discriminator({
        "type": "FIR_MINIBATCH_STD",
        "img_resolution": 16,
        "base_ch": 4,
        "max_ch": 16,
        "minibatch_std_group_size": 2,
    }).to("cuda")
    r1_model = make_r1_matmul_discriminator(model)
    real = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    with torch.backends.cudnn.flags(enabled=False), torch.autocast("cuda", dtype=torch.bfloat16):
        score = r1_model(real)
        penalty = r1_reg_loss(score, real, gamma=2.0)
    penalty.backward()
    assert torch.isfinite(penalty)
    assert real.grad is not None and torch.isfinite(real.grad).all()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
