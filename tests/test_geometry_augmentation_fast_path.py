"""无随机几何增强时，训练损失与样本梯度图不应执行恒等恢复重采样。"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from misc.face_alignment import make_arcface128_to_arcface112_grid, make_ffhq_to_arcface_112_grid
from swapface.dataloader_common import is_identity_geometry
from swapface.train import Trainer


def test_identity_geometry_requires_all_four_ranges() -> None:
    ranges = ((0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    assert is_identity_geometry(*ranges)
    for idx, changed in enumerate(((0.0, 0.01), (1.0, 1.01), (0.0, 0.01), (0.0, 0.01))):
        non_identity = list(ranges)
        non_identity[idx] = changed
        assert not is_identity_geometry(*non_identity)


@pytest.mark.parametrize('alignment', ['ffhq', 'arcface'])
@pytest.mark.parametrize('resolution', [64, 128])
def test_teacher_crop_remains_while_identity_restore_is_skipped(alignment: str, resolution: int) -> None:
    """禁用恢复不等于禁用身份裁剪；合法的采样网格全在边界内。"""
    trainer = Trainer.__new__(Trainer)
    trainer.face_alignment = alignment
    trainer.img_resolution = resolution
    trainer.coarse_resolution = resolution // 2
    trainer.identity_encoder_grid = make_ffhq_to_arcface_112_grid(resolution, 2, torch.device('cpu'))
    trainer.coarse_identity_encoder_grid = make_ffhq_to_arcface_112_grid(resolution // 2, 2, torch.device('cpu'))
    trainer.arcface_identity_grid = make_arcface128_to_arcface112_grid(resolution, 2, torch.device('cpu'))
    trainer.coarse_arcface_identity_grid = make_arcface128_to_arcface112_grid(resolution // 2, 2, torch.device('cpu'))

    theta = torch.eye(3, dtype=torch.float32)[:2].expand(2, -1, -1).clone()
    for size in (resolution, resolution // 2):
        generated = torch.rand(2, 3, size, size, requires_grad=True)
        original = trainer.prepare_identity_encoder_faces(generated, theta)
        with patch('swapface.train.transform_sampling_grid', side_effect=AssertionError('unexpected affine composition')):
            fast = trainer.prepare_identity_encoder_faces(generated, None)
        assert fast.shape == (2, 3, 112, 112)
        torch.testing.assert_close(fast, original, atol=1e-5, rtol=1e-5)
        grad_fast, = torch.autograd.grad(fast.sum(), generated, retain_graph=True)
        grad_original, = torch.autograd.grad(original.sum(), generated)
        torch.testing.assert_close(grad_fast, grad_original, atol=2e-4, rtol=1e-5)


def test_geometry_restore_fast_path_preserves_hq_and_coarse_gradients() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.geometry_augmentation_enabled = False
    hq = torch.randn(2, 3, 16, 16, requires_grad=True)
    coarse = torch.randn(2, 3, 8, 8, requires_grad=True)
    with patch('swapface.train.NF.grid_sample', side_effect=AssertionError('unexpected resampling')):
        hq_result = trainer._restore_for_geometry_loss(hq, (16, 16), None)
        coarse_result = trainer._restore_for_geometry_loss(coarse, (16, 16), None)
    assert hq_result.data_ptr() == hq.data_ptr()
    torch.testing.assert_close(coarse_result, F.interpolate(coarse, size=(16, 16), mode='bilinear', align_corners=False))
    (hq_result.square().mean() + coarse_result.square().mean()).backward()
    assert torch.isfinite(hq.grad).all() and hq.grad.abs().sum() > 0
    assert torch.isfinite(coarse.grad).all() and coarse.grad.abs().sum() > 0

    trainer.geometry_augmentation_enabled = True
    theta = torch.tensor([[[1.0, 0.0, 0.125], [0.0, 1.0, 0.0]]]).expand(2, -1, -1)
    restore_grid = F.affine_grid(theta, hq.shape, align_corners=False)
    restored = trainer._restore_for_geometry_loss(hq, (16, 16), restore_grid)
    expected = F.grid_sample(hq, restore_grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match='restore_grid'):
        trainer._restore_for_geometry_loss(hq, (16, 16), None)


def _stub_trainer_for_losses() -> tuple[Trainer, list[torch.Tensor | None]]:
    trainer = Trainer.__new__(Trainer)
    trainer.geometry_augmentation_enabled = False
    trainer.train_d_coarse = lambda x, *args: x.square().mean()
    trainer.train_d = lambda x, *args: x.square().mean()
    trainer.coarse_gan_loss = lambda score: score
    trainer.hq_gan_loss = lambda score: score
    trainer.coarse_identity_embeddings_forward = lambda x: x
    trainer.hq_identity_embeddings_forward = lambda x: x
    received_theta: list[torch.Tensor | None] = []

    def identity_faces(x, theta):
        received_theta.append(theta)
        return x

    trainer.prepare_identity_encoder_faces = identity_faces
    trainer.coarse_id_loss = lambda gen, ref: gen.square().mean()
    trainer.hq_id_loss = lambda gen, ref: gen.square().mean()
    trainer.enable_coarse_wfm_loss = trainer.enable_hq_wfm_loss = False
    trainer.enable_coarse_gaze_loss = trainer.enable_hq_gaze_loss = True
    trainer.enable_coarse_smirk_loss = trainer.enable_hq_smirk_loss = True
    trainer.enable_coarse_vgg_loss = trainer.enable_hq_vgg_loss = False
    trainer.enable_coarse_l1_loss = trainer.enable_hq_l1_loss = False
    trainer.coarse_gaze_loss_forward = trainer.hq_gaze_loss_forward = lambda a, b: (a - b).square().mean()
    smirk = SimpleNamespace(forward_components=lambda a, b, **kwargs: {'pose': (a - b).square().mean()})
    trainer.coarse_smirk_loss = trainer.hq_smirk_loss = smirk
    trainer.log = lambda *args, **kwargs: None
    return trainer, received_theta


def test_generator_stages_keep_gaze_smirk_and_id_gradients_without_restore() -> None:
    trainer, received_theta = _stub_trainer_for_losses()
    canonical = torch.randn(2, 3, 16, 16)
    theta = torch.eye(3)[:2].expand(2, -1, -1).clone()
    coarse = torch.randn(2, 3, 8, 8, requires_grad=True)
    hq = torch.randn(2, 3, 16, 16, requires_grad=True)
    with patch('swapface.train.NF.grid_sample', side_effect=AssertionError('disabled geometry cannot resample')):
        coarse_loss = trainer._coarse_generator_stage_loss(coarse, coarse.detach(), canonical, theta, None, None, None, None)
        hq_loss = trainer._hq_generator_stage_loss(hq, hq.detach(), canonical, theta, None, None, None, None)
    (coarse_loss + hq_loss).backward()
    assert received_theta == [None, None]
    assert torch.isfinite(coarse.grad).all() and coarse.grad.abs().sum() > 0
    assert torch.isfinite(hq.grad).all() and hq.grad.abs().sum() > 0


def test_sample_gradients_skip_restore_but_keep_motion_loss() -> None:
    trainer, received_theta = _stub_trainer_for_losses()
    discriminator = torch.nn.Conv2d(3, 1, 1)
    trainer.net_d = discriminator
    trainer.net_d_coarse = discriminator
    trainer.coarse_gan_loss = lambda x: x.square().mean()
    trainer.hq_gan_loss = lambda x: x.square().mean()
    trainer.coarse_id_loss = lambda gen, ref: gen.square().mean()
    trainer.hq_id_loss = lambda gen, ref: gen.square().mean()
    trainer.coarse_reconstruction_scope = trainer.hq_reconstruction_scope = 'all'
    trainer.amp_enabled = False
    trainer.amp_dtype = None
    canonical = torch.randn(2, 3, 16, 16)
    with patch('swapface.train.NF.grid_sample', side_effect=AssertionError('disabled geometry cannot resample')):
        for stage, resolution in [('coarse', 8), ('hq', 16)]:
            image = torch.randn(2, 3, resolution, resolution)
            rows = trainer._sample_stage_grad_maps(stage, image, image, canonical, None, None, torch.zeros_like(image), None, (16, 16), None)
            assert any('GAZE GRAD' in label for label, _ in rows)
            assert any('SMIRK GRAD' in label for label, _ in rows)
    assert received_theta == [None, None]


def test_save_sample_skips_all_identity_restore_grids(tmp_path) -> None:
    """样本预览可同时使用参考/动态样本，但无随机仿射时不构造恢复网格。"""
    trainer = Trainer.__new__(Trainer)
    trainer.geometry_augmentation_enabled = False
    trainer.device = torch.device('cpu')
    trainer.batch_size = 2
    trainer.coarse_resolution = 8
    trainer.amp_dtype = None
    trainer.amp_enabled = False
    trainer.coarse_stage_active = trainer.hq_stage_active = True
    trainer.reuse_generator_identity_for_coarse_source = True
    trainer.reuse_generator_identity_for_hq_source = True
    trainer.enable_coarse_gaze_loss = trainer.enable_hq_gaze_loss = True
    trainer.enable_coarse_smirk_loss = trainer.enable_hq_smirk_loss = True
    trainer.generator_id_encoder_forward = lambda x: x.mean(dim=(2, 3))
    trainer.net_g_ema = lambda dst, identity, return_coarse: (dst * 0.9, F.interpolate(dst, size=(8, 8), mode='bilinear', align_corners=False))
    trainer._prepare_smirk_references = lambda x: (None, None)
    trainer._completed_step = 0
    trainer.run_paths = SimpleNamespace(samples=tmp_path)
    seen: list[str] = []

    def identity_faces(x, theta):
        assert theta is None
        seen.append('identity')
        return x

    def stage_gradients(stage, generated, reconstruction_target, canonical, theta, same_mask, source, restore_grid, *args):
        assert theta is None and restore_grid is None
        seen.append(stage)
        return []

    trainer.prepare_identity_encoder_faces = identity_faces
    trainer._sample_stage_grad_maps = stage_gradients

    def batch(count):
        face = torch.rand(count, 3, 16, 16)
        theta = torch.eye(3)[:2].expand(count, -1, -1).clone()
        return face, face, face, theta, torch.zeros(count, dtype=torch.bool), torch.rand(count, 3, 112, 112)

    with patch('swapface.train.NF.affine_grid', side_effect=AssertionError('unexpected identity restore grid')):
        trainer._save_sample(batch(1), batch(2))
    assert seen == ['identity', 'coarse', 'hq']
    assert (tmp_path / 'step_000000000.png').is_file()
