"""Single-stage face coordinate restoration without redundant sampling."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from misc.face_alignment import make_arcface128_to_arcface112_grid, make_ffhq_to_arcface_112_grid
from swapface.dataloader_common import is_identity_geometry
from swapface.train import Trainer


def test_identity_geometry_requires_all_ranges():
    ranges = ((0., 0.), (1., 1.), (0., 0.), (0., 0.))
    assert is_identity_geometry(*ranges)
    for i, changed in enumerate(((0., .01), (1., 1.01), (0., .01), (0., .01))):
        other = list(ranges)
        other[i] = changed
        assert not is_identity_geometry(*other)


@pytest.mark.parametrize("alignment", ["ffhq", "arcface"])
@pytest.mark.parametrize("resolution", [64, 128])
def test_identity_encoder_crop_without_geometry_restore(alignment: str, resolution: int):
    trainer = Trainer.__new__(Trainer)
    trainer.img_resolution = resolution
    trainer.face_alignment = alignment
    trainer.identity_encoder_grid = make_ffhq_to_arcface_112_grid(resolution, 2, torch.device("cpu"))
    trainer.arcface_identity_grid = make_arcface128_to_arcface112_grid(resolution, 2, torch.device("cpu"))
    x = torch.rand(2, 3, resolution, resolution, requires_grad=True)
    theta = torch.eye(3)[:2].expand(2, -1, -1).clone()
    expected = trainer.prepare_identity_encoder_faces(x, theta)
    with patch("swapface.train.transform_sampling_grid", side_effect=AssertionError("no affine for identity")):
        actual = trainer.prepare_identity_encoder_faces(x)
    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)
    assert actual.shape == (2, 3, 112, 112)
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError, match="resolution"):
        trainer.prepare_identity_encoder_faces(torch.rand(2, 3, resolution // 2, resolution // 2))


def test_geometry_restoration_skips_identity_resampling():
    trainer = Trainer.__new__(Trainer)
    trainer.geometry_augmentation_enabled = False
    image = torch.randn(2, 3, 16, 16, requires_grad=True)
    with patch("swapface.train.NF.grid_sample", side_effect=AssertionError("no extra sample")):
        result = trainer._restore_for_geometry_loss(image, (16, 16), None)
    assert result.data_ptr() == image.data_ptr()
    result.square().mean().backward()
    assert torch.isfinite(image.grad).all()
    trainer.geometry_augmentation_enabled = True
    theta = torch.tensor([[[1., 0., .125], [0., 1., 0.]]]).expand(2, -1, -1)
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    torch.testing.assert_close(trainer._restore_for_geometry_loss(image, (16, 16), grid),
                               F.grid_sample(image, grid, mode="bilinear", padding_mode="reflection", align_corners=False))
    with pytest.raises(RuntimeError, match="grid"):
        trainer._restore_for_geometry_loss(image, (16, 16), None)
