"""全局 FFHQ/ArcFace 训练坐标系的 CPU 回归：python -m swapface.test_generation_alignment。"""

import copy
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from misc.face_alignment import (
    FFHQ_TO_ARCFACE_112_AFFINE_512, ffhq_to_arcface_affine,
    make_ffhq_to_arcface_grid, make_ffhq_to_arcface_112_grid, ffhq_to_arcface_112,
    make_arcface112_to_arcface128_grid, make_arcface128_to_arcface112_grid,
)
from misc.models.id_encoder import get_alignment_template
from .config import resolve_train_config
from .dataloader import TrainingDataLoader
from .dataloader_native import make_affine_thetas
from .experiment import config_sha256


def main():
    import tomllib
    config = tomllib.loads((Path(__file__).parents[1] / "experiments/train.toml").read_text())
    ffhq_config = resolve_train_config(config)
    assert "alignment" not in ffhq_config["data"]
    explicit = copy.deepcopy(config)
    explicit["data"]["alignment"] = "ffhq"
    assert resolve_train_config(explicit) == ffhq_config
    assert config_sha256(resolve_train_config(explicit)) == config_sha256(ffhq_config)
    arcface_config = copy.deepcopy(config)
    arcface_config["data"]["alignment"] = "arcface"
    assert resolve_train_config(arcface_config)["data"]["alignment"] == "arcface"
    for invalid in ("arcface112", "other", None, 5):
        bad = copy.deepcopy(config)
        bad["data"]["alignment"] = invalid
        try:
            resolve_train_config(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"接受了无效对齐标准：{invalid!r}")

    for size in (128, 224, 256, 512):
        affine = torch.tensor(ffhq_to_arcface_affine(size, size))
        ref = torch.tensor(FFHQ_TO_ARCFACE_112_AFFINE_512)
        ffhq_five = torch.linalg.solve(ref[:, :2], (torch.tensor(get_alignment_template(112)) - ref[:, 2]).T).T
        mapped = (ffhq_five * size / 512) @ affine[:, :2].T + affine[:, 2]
        target = torch.tensor(get_alignment_template(128)) * (size / 128.0)
        torch.testing.assert_close(mapped, target, atol=1e-4, rtol=0)
        assert not torch.allclose(mapped, torch.tensor(get_alignment_template(112)) * (size / 112.0), atol=1e-3, rtol=0)
        grid = make_ffhq_to_arcface_grid(size, size, 2, torch.device("cpu"))
        assert grid.shape == (2, size, size, 2)
        assert torch.isfinite(grid).all()
        old = make_ffhq_to_arcface_112_grid(size, 2, torch.device("cpu"))
        assert old.shape == (2, 112, 112, 2)
        inv = make_arcface128_to_arcface112_grid(size, 2, torch.device("cpu"))
        assert inv.shape == old.shape and torch.isfinite(inv).all()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ffhq_dir, arc_dir, dst_dir = (root / "ffhq", root / "arc", root / "dst")
        for p in (ffhq_dir, arc_dir, dst_dir):
            p.mkdir()
        y, x = np.mgrid[:112, :112]
        rgba = np.stack((x * 2, y * 2, (x + y) % 256), axis=-1).astype(np.uint8)
        for p in (ffhq_dir, arc_dir, dst_dir):
            Image.fromarray(rgba).save(p / "face.png")
        native = dict(
            num_threads=1, prefetch_queue_depth=1, py_num_workers=0, py_start_method="spawn",
            reader_prefetch_queue_depth=1, decoder_backend=__import__("swapface.dataloader_common", fromlist=["ImageDecoderBackend"]).ImageDecoderBackend.CPU,
            decoder_hw_load=0.0,
            brightness=0.0, contrast=0.0, saturation=0.0, flip_prob=0.0,
            same_prob=1.0, rotation_range=(0.0, 0.0), scale_factor_range=(1.0, 1.0),
            tx_range=(0.0, 0.0), ty_range=(0.0, 0.0), alignment="arcface",
        )
        device = torch.device("cpu")
        loader = TrainingDataLoader(batch_size=2, device=device, img_resolution=128,
                                    src=[(arc_dir, 0.0, "arcface")], dst=[dst_dir], **native)
        src, dst, canonical, theta, same, identity = loader.next()
        assert same.all()
        assert src.shape == dst.shape == canonical.shape == (2, 3, 128, 128)
        torch.testing.assert_close(src, canonical, atol=3e-6, rtol=0)
        torch.testing.assert_close(canonical, dst, atol=3e-6, rtol=0)
        orig = torch.from_numpy(np.asarray(Image.open(dst_dir / "face.png").resize((128, 128), Image.Resampling.LANCZOS)).copy()).permute(2,0,1).float().unsqueeze(0).expand(2,-1,-1,-1).div(127.5).sub(1)
        grid = make_ffhq_to_arcface_grid(128, 128, 2, device)
        expected = F.grid_sample(orig, grid, padding_mode="border", align_corners=False)
        torch.testing.assert_close(src, expected, atol=3e-6, rtol=0)
        expected_identity = ffhq_to_arcface_112(orig, make_ffhq_to_arcface_112_grid(128, 2, device))
        torch.testing.assert_close(identity, expected_identity, atol=0, rtol=0)

        native["same_prob"] = 0.0
        loader = TrainingDataLoader(batch_size=16, device=device, img_resolution=128,
                                    src=[ffhq_dir, (arc_dir, 0.0, "arcface")], dst=[dst_dir], **native)
        src, dst, canonical, theta, same, identity = loader.next()
        assert not same.any()
        arc_source_grid = make_arcface112_to_arcface128_grid(128, 128, 16, device)
        direct = F.grid_sample(orig[:1].expand(16,-1,-1,-1), arc_source_grid, padding_mode="border", align_corners=False)
        direct_arc_identity = torch.from_numpy(rgba.copy()).permute(2,0,1).float().div(127.5).sub(1).unsqueeze(0).expand(16,-1,-1,-1)
        ffhq_identity = expected_identity[:1].expand(16,-1,-1,-1)
        src_ffhq = expected[:1].expand(16,-1,-1,-1)
        # 所有样本分别是 FFHQ->ArcFace128 或 ArcFace112 SRC->ArcFace128。
        is_native_arc = (identity == direct_arc_identity).flatten(1).all(1)
        assert is_native_arc.any() and (~is_native_arc).any()
        torch.testing.assert_close(src[~is_native_arc], src_ffhq[~is_native_arc], atol=0, rtol=0)
        torch.testing.assert_close(src[is_native_arc], direct[is_native_arc], atol=0, rtol=0)
        torch.testing.assert_close(identity[~is_native_arc], ffhq_identity[~is_native_arc], atol=0, rtol=0)

    # 输出网格的增强恢复遵循 ArcFace canonical 坐标；不额外应用 FFHQ 身份 crop。
    theta_aug, theta_restore = make_affine_thetas(
        batch_size=2, img_resolution=128, rotation_range=(8.0, 8.0),
        scale_factor_range=(1.1, 1.1), tx_range=(0.02, 0.02), ty_range=(-0.04, -0.04),
        device=torch.device("cpu"),
    )
    eye = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    eye[:, :2, :] = theta_aug
    restore = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    restore[:, :2, :] = theta_restore
    torch.testing.assert_close(torch.bmm(eye, restore), torch.eye(3).expand(2,-1,-1), atol=2e-6, rtol=0)
    # 训练损失 ID teacher 在 ArcFace 模式不做第二次 FFHQ crop，且保持梯度链路。
    from .train import Trainer
    fake_trainer = object.__new__(Trainer)
    fake_trainer.face_alignment = "arcface"
    fake_trainer.img_resolution = 128
    fake_trainer.coarse_resolution = 64
    fake_trainer.arcface_identity_grid = make_arcface128_to_arcface112_grid(128, 2, torch.device("cpu"))
    fake_trainer.coarse_arcface_identity_grid = make_arcface128_to_arcface112_grid(64, 2, torch.device("cpu"))
    generated = torch.rand((2, 3, 128, 128), requires_grad=True)
    teacher_input = Trainer.prepare_identity_encoder_faces(fake_trainer, generated)
    assert teacher_input.shape == (2, 3, 112, 112)
    torch.testing.assert_close(
        teacher_input, generated[:, :, :112, 8:120], atol=2e-5, rtol=0,
    )
    coarse_generated = torch.rand((2, 3, 64, 64), requires_grad=True)
    coarse_teacher = Trainer.prepare_identity_encoder_faces(fake_trainer, coarse_generated)
    torch.testing.assert_close(
        coarse_teacher,
        F.grid_sample(coarse_generated, fake_trainer.coarse_arcface_identity_grid, padding_mode="border", align_corners=False),
        atol=0, rtol=0,
    )
    restored_input = Trainer.prepare_identity_encoder_faces(fake_trainer, generated, theta_restore)
    from misc.face_alignment import transform_sampling_grid
    teacher_grid = transform_sampling_grid(fake_trainer.arcface_identity_grid, theta_restore)
    torch.testing.assert_close(
        restored_input,
        F.grid_sample(generated, teacher_grid, padding_mode="reflection", align_corners=False),
        atol=0, rtol=0,
    )
    (teacher_input.mean() + restored_input.mean()).backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()

    # 模型保存/导出：旧 checkpoint 视作 FFHQ，新 checkpoint 随元数据选择 ArcFace。
    from unittest.mock import patch
    from models.networks import Generator
    from misc.models.id_encoder import IDEncoderProvider
    from .contracts import CHECKPOINT_VERSION
    from .inference import get_arcface_alignment_template, load_generator
    from . import export
    model = Generator(
        img_resolution=16, coarse_resolution=8, coarse_latent_resolution=4,
        coarse_base_ch=2, coarse_max_ch=8, coarse_num_latent=1,
        hq_bottleneck_resolution=4, hq_base_ch=2, hq_max_ch=8,
    ).eval()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = Path(tmp) / "test.pth"
        ckpt = {
            "version": CHECKPOINT_VERSION,
            "step": 123,
            "identity_encoders": {"generator": IDEncoderProvider.BLENDFACE.name},
            "net_g": {"network_cfg": model.network_cfg, "state_dict": model.state_dict()},
        }
        torch.save(ckpt, checkpoint_path)
        legacy, _, _ = load_generator(checkpoint_path)
        assert legacy.face_alignment == "ffhq"
        ckpt["face_alignment"] = "arcface"
        torch.save(ckpt, checkpoint_path)
        modern, _, _ = load_generator(checkpoint_path)
        assert modern.face_alignment == "arcface"
        template = get_arcface_alignment_template(128, torch.device("cpu"))
        torch.testing.assert_close(template, torch.tensor(get_alignment_template(128)))
        torch.testing.assert_close(
            get_arcface_alignment_template(256, torch.device("cpu")),
            torch.tensor(get_alignment_template(128)) * 2,
        )
        with patch.object(export, "export_to_onnx", return_value=Path(tmp) / "fake.onnx") as mocked:
            export.export_swapface(str(checkpoint_path), batch_size=1, nhwc=True, device="cpu")
        assert mocked.call_args.kwargs["metadata"]["swapface.face_alignment"] == "arcface"
        ckpt["face_alignment"] = "invalid"
        torch.save(ckpt, checkpoint_path)
        try:
            load_generator(checkpoint_path)
        except ValueError:
            pass
        else:
            raise AssertionError("未拒绝无效对齐 checkpoint")

    print("PASS: ArcFace128 template family 128/224/256/512, 112 identity crop, mixed/same src, restored gradients, checkpoint & ONNX metadata")


if __name__ == "__main__":
    main()
