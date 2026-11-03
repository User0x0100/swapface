"""SMIRK frozen teacher loss: I/O, per-attribute gradients and fail-closed loading."""

import hashlib
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from losses.smirk import SMIRKMotionLoss
from misc.models.smirk import SMIRKEncoder
from misc.models.smirk import model as smirk_model


class _ToyEncoder(nn.Module):
    def forward(self, images):
        x = torch.nn.functional.adaptive_avg_pool2d(images, (2, 2)).flatten(1)
        return {
            "pose_params": x[:, :3],
            "expression_params": x.repeat(1, 5)[:, :50],
            "jaw_params": x[:, 3:6],
            "eyelid_params": x[:, 6:8],
        }


def test_smirk_separate_attribute_gradients():
    loss = SMIRKMotionLoss(
        encoder=_ToyEncoder(),
        pose_weight=2.0,
        expression_weight=3.0,
        jaw_weight=4.0,
        eyelid_weight=5.0,
    )
    generated = torch.randn(2, 3, 64, 64, requires_grad=True)
    reference = torch.randn_like(generated, requires_grad=True)
    components = loss.forward_components(generated, reference)
    assert set(components) == {"pose", "expression", "jaw", "eyelid"}
    for value in components.values():
        grad = torch.autograd.grad(value, generated, retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    loss(generated, reference).backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()
    assert reference.grad is None


def test_smirk_disabled_terms_and_bad_inputs():
    loss = SMIRKMotionLoss(encoder=_ToyEncoder(), pose_weight=0, expression_weight=0, jaw_weight=0, eyelid_weight=0)
    generated = torch.randn(1, 3, 32, 32, requires_grad=True)
    assert loss.forward_components(generated, generated.detach()) == {}
    loss(generated, generated.detach()).backward()
    assert generated.grad is not None and generated.grad.abs().sum() == 0
    with pytest.raises(ValueError, match="shape 不匹配"):
        loss(generated, torch.rand(2, 3, 32, 32))
    with pytest.raises(ValueError, match="有限非负"):
        SMIRKMotionLoss(encoder=_ToyEncoder(), pose_weight=math.nan, expression_weight=1, jaw_weight=1, eyelid_weight=1)


def test_smirk_download_failure_fails_closed():
    with patch.object(smirk_model, "hf_hub_download", side_effect=OSError("connection failed")) as download, pytest.raises(OSError, match="connection failed"):
        SMIRKEncoder()
    download.assert_called_once_with(
        repo_id=smirk_model.SMIRK_REPOSITORY_ID,
        filename=smirk_model.SMIRK_FILENAME,
        revision=smirk_model.SMIRK_REVISION,
    )


def test_smirk_download_hash_mismatch_fails_closed(tmp_path: Path):
    bogus = tmp_path / "bad.pt"
    bogus.write_bytes(b"corrupt checkpoint")
    with patch.object(smirk_model, "hf_hub_download", return_value=str(bogus)), pytest.raises(ValueError, match="SHA-256 校验失败"):
        SMIRKEncoder()


def test_smirk_download_checksum_and_reuse(tmp_path: Path):
    fake = tmp_path / "weights.pt"
    fake.write_bytes(b"cached smirk checkpoint")
    digest = hashlib.sha256(fake.read_bytes()).hexdigest()
    with (
        patch.object(smirk_model, "hf_hub_download", return_value=str(fake)) as download,
        patch.object(smirk_model, "SMIRK_SHA256", digest),
    ):
        assert smirk_model._download_smirk_checkpoint() == fake
        assert smirk_model._download_smirk_checkpoint() == fake
    assert download.call_count == 2
    assert download.call_args.kwargs == {
        "repo_id": "Skywork/SkyReels-A1",
        "filename": "extra_models/smirk/SMIRK_em1.pt",
        "revision": "6cee256d5ea38dbbe87ff026566c95b6401d3124",
    }


def test_removed_loss_tables_rejected():
    from swapface.config import DEFAULT_LOSS_CONFIG

    for legacy in ("hrffa", "facs"):
        # Canonical schema must not silently accept obsolete loss tables.
        assert legacy not in DEFAULT_LOSS_CONFIG["coarse"]
        assert legacy not in DEFAULT_LOSS_CONFIG["hq"]


def test_smirk_checkpoint_integration_when_available():
    # The user-local official checkpoint provides a network-free integration
    # fixture; production always downloads through the Hugging Face cache.
    checkpoint = Path.home() / ".cache/swap/smirk/SMIRK_em1.pt"
    if not checkpoint.is_file():
        pytest.skip("official SMIRK local integration fixture is not installed")
    with patch.object(smirk_model, "hf_hub_download", return_value=str(checkpoint)) as download:
        loss = SMIRKMotionLoss(pose_weight=1, expression_weight=1, jaw_weight=1, eyelid_weight=1)
    download.assert_called_once_with(
        repo_id=smirk_model.SMIRK_REPOSITORY_ID,
        filename=smirk_model.SMIRK_FILENAME,
        revision=smirk_model.SMIRK_REVISION,
    )
    assert not any(param.requires_grad for param in loss.parameters())
    generated = torch.rand(1, 3, 128, 128, requires_grad=True) * 2 - 1
    reference = torch.rand_like(generated) * 2 - 1
    components = loss.forward_components(generated, reference)
    assert set(components) == {"pose", "expression", "jaw", "eyelid"}
    grad = torch.autograd.grad(sum(components.values()), generated)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_smirk_pose_matrix_identity_and_gradients():
    near_zero = torch.zeros(2, 3, requires_grad=True)
    matrix = SMIRKMotionLoss._pose_matrix(near_zero)
    assert torch.allclose(matrix, torch.eye(3).expand(2, -1, -1), atol=1e-7)
    matrix.sum().backward()
    assert near_zero.grad is not None and torch.isfinite(near_zero.grad).all()
    x = torch.tensor([[0.8, -0.4, 0.2]], requires_grad=True)
    matrix = SMIRKMotionLoss._pose_matrix(x)
    assert torch.allclose(matrix.transpose(1, 2) @ matrix, torch.eye(3).unsqueeze(0), atol=1e-5)
    matrix.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_smirk_straight_through_limits_retain_forward_and_gradient():
    from misc.models.smirk.model import _straight_through_clamp

    for raw, low, high, target in (
        (-0.5, 0.0, 1.0, 0.65),
        (1.4, 0.0, 1.0, 0.2),
        (-0.2, 0.0, float("inf"), 0.3),
        (-0.5, -0.2, 0.2, 0.15),
    ):
        value = torch.tensor([raw], requires_grad=True)
        prediction = _straight_through_clamp(value, low, high)
        assert prediction.detach().item() == pytest.approx(max(low, min(raw, high)))
        torch.nn.functional.smooth_l1_loss(prediction, torch.tensor([target])).backward()
        assert value.grad is not None and abs(float(value.grad)) > 0
        assert float(value.grad) < 0 if raw < target else float(value.grad) > 0


def test_smirk_pose_loss_fp32_under_cuda_autocast():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda")
    loss = SMIRKMotionLoss(encoder=_ToyEncoder(), pose_weight=1, expression_weight=0, jaw_weight=0, eyelid_weight=0).to(device)
    generated = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
    reference = torch.randn_like(generated)
    reference_loss = loss(generated, reference)
    for dtype in (torch.float16, torch.bfloat16):
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        with torch.autocast("cuda", dtype=dtype):
            value = loss(generated, reference)
        assert value.dtype is torch.float32
        torch.testing.assert_close(value, reference_loss, rtol=1e-6, atol=1e-7)
        grad = torch.autograd.grad(value, generated, retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_shared_smirk_reference_only_encoded_once_for_joint_stages():
    from swapface.train import Trainer

    class CounterEncoder(_ToyEncoder):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            return super().forward(images)

    encoder = CounterEncoder()
    kwargs = {"pose_weight": 1, "expression_weight": 1, "jaw_weight": 1, "eyelid_weight": 1}
    coarse = SMIRKMotionLoss(encoder=encoder, **kwargs)
    hq = SMIRKMotionLoss(encoder=encoder, **kwargs)
    trainer = Trainer.__new__(Trainer)
    trainer.enable_coarse_smirk_loss = True
    trainer.enable_hq_smirk_loss = True
    trainer.coarse_smirk_loss = coarse
    trainer.hq_smirk_loss = hq

    reference = torch.rand(2, 3, 64, 64)
    generated_coarse = torch.rand_like(reference, requires_grad=True)
    generated_hq = torch.rand_like(reference, requires_grad=True)
    coarse_reference, hq_reference = trainer._prepare_smirk_references(reference)
    assert coarse_reference is hq_reference
    assert encoder.calls == 1
    coarse_loss = sum(coarse.forward_components(generated_coarse, reference, reference_features=coarse_reference).values())
    hq_loss = sum(hq.forward_components(generated_hq, reference, reference_features=hq_reference).values())
    assert encoder.calls == 3  # 1 target + 1 generated per stage
    (coarse_grad,) = torch.autograd.grad(coarse_loss, generated_coarse)
    (hq_grad,) = torch.autograd.grad(hq_loss, generated_hq)
    assert torch.isfinite(coarse_grad).all() and coarse_grad.abs().sum() > 0
    assert torch.isfinite(hq_grad).all() and hq_grad.abs().sum() > 0
    # Simulate Generator overflow retry. Target embedding must not be rerun.
    generated_retry = generated_coarse.detach().requires_grad_(True)
    sum(coarse.forward_components(generated_retry, reference, reference_features=coarse_reference).values()).backward()
    assert encoder.calls == 4
    torch.testing.assert_close(generated_retry.grad, coarse_grad)


def test_smirk_nonfinite_target_exclusion_is_per_sample():
    loss = SMIRKMotionLoss(encoder=_ToyEncoder(), pose_weight=1, expression_weight=1, jaw_weight=1, eyelid_weight=1)
    reference = torch.rand(2, 3, 64, 64)
    reference[1] = float("nan")
    target_features = loss.encode_reference(reference)
    assert target_features["valid"].tolist() == [True, False]
    assert all(torch.isfinite(v).all() for v in target_features.values())
    generated = torch.rand(2, 3, 64, 64, requires_grad=True)
    values = loss.forward_components(generated, reference, reference_features=target_features)
    loss_sum = torch.stack(tuple(values.values())).sum()
    assert torch.isfinite(loss_sum)
    loss_sum.backward()
    assert generated.grad[0].abs().sum() > 0
    assert generated.grad[1].abs().sum() == 0


def test_smirk_occlusion_reduces_only_affected_sample_gradient():
    loss = SMIRKMotionLoss(encoder=_ToyEncoder(), pose_weight=1, expression_weight=1, jaw_weight=1, eyelid_weight=1)
    generated = torch.rand(2, 3, 64, 64, requires_grad=True)
    reference = torch.zeros_like(generated)
    features = loss.encode_reference(reference)
    values = loss.forward_components(generated, reference, reference_features=features, sample_weights=torch.tensor([1.0, 0.0]))
    sum(values.values()).backward()
    assert generated.grad[0].abs().sum() > 0
    assert generated.grad[1].abs().sum() == 0
    with pytest.raises(ValueError, match="shape 不匹配"):
        loss.forward_components(generated, reference, reference_features=features, sample_weights=torch.tensor([1.0]))
