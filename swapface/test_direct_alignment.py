"""直接从原图生成 ArcFace 训练图像：python -m swapface.test_direct_alignment。"""

import tempfile
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from misc.face_alignment import FFHQ_TO_ARCFACE_112_AFFINE_512, ffhq_to_arcface_affine

from .dataloader import TrainingDataLoader
from .dataloader_common import FFHQ_TO_ARCFACE112_ROI, ImageDecoderBackend, generation_alignment_rois
from .dataloader_native import _resize_roi


def _normalized(image: Image.Image, batch_size: int) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).expand(batch_size, -1, -1, -1)


def _config() -> dict:
    return {
        "img_resolution": 128, "num_threads": 2, "prefetch_queue_depth": 1,
        "py_start_method": "spawn", "reader_prefetch_queue_depth": 1,
        "decoder_backend": ImageDecoderBackend.CPU, "decoder_hw_load": 0.0,
        "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "flip_prob": 0.0,
        "same_prob": 0.0, "rotation_range": (0.0, 0.0), "scale_factor_range": (1.0, 1.0),
        "tx_range": (0.0, 0.0), "ty_range": (0.0, 0.0), "alignment": "arcface",
    }


def main() -> None:
    # 每个输出分辨率均与原有固定仿射的像素中心一致，允许小角度近似（约 0.01px/128）。
    for output_size in (32, 128, 224, 256, 512):
        ffhq_roi, arcface_roi = generation_alignment_rois(output_size)
        for roi, matrix, source_size, size in (
            (ffhq_roi, ffhq_to_arcface_affine(512, output_size), 512, output_size),
            (arcface_roi, ((output_size / 128, 0.0, 8 * output_size / 128), (0.0, output_size / 128, 0.0)), 112, output_size),
            (FFHQ_TO_ARCFACE112_ROI, FFHQ_TO_ARCFACE_112_AFFINE_512, 512, 112),
        ):
            affine = np.array(matrix)
            for x in (0, size // 2, size - 1):
                for y in (0, size // 2, size - 1):
                    sampled = np.array([
                        (roi[0] + (x + 0.5) / size * (roi[2] - roi[0])) * source_size - 0.5,
                        (roi[1] + (y + 0.5) / size * (roi[3] - roi[1])) * source_size - 0.5,
                    ])
                    transformed = affine[:, :2] @ sampled + affine[:, 2]
                    assert np.max(np.abs(transformed - (x, y))) < 0.01 * size / 128 + 0.001

    with tempfile.TemporaryDirectory(prefix="swapface-direct-alignment-") as tmp:
        root = Path(tmp)
        ffhq_dir, arc_dir, dst_dir = root / "ffhq", root / "arcface", root / "dst"
        for folder in (ffhq_dir, arc_dir, dst_dir):
            folder.mkdir()
        y, x = np.mgrid[:512, :512]
        ffhq_pixels = np.stack(((3 * x + y) % 256, (x // 3 + 2 * y) % 256, ((x // 4 + y // 4) % 2) * 255), axis=-1).astype(np.uint8)
        ffhq_image = Image.fromarray(ffhq_pixels)
        ffhq_image.save(ffhq_dir / "face.png")
        ffhq_image.save(dst_dir / "face.png")
        y, x = np.mgrid[:112, :112]
        arc_pixels = np.stack((x * 2, y * 2, ((x // 3 + y // 3) % 2) * 255), axis=-1).astype(np.uint8)
        arc_image = Image.fromarray(arc_pixels)
        arc_image.save(arc_dir / "face.png")

        ffhq_roi, arcface_roi = generation_alignment_rois(128)
        expected_ffhq = _normalized(_resize_roi(ffhq_image, 128, ffhq_roi), 2)
        expected_id = _normalized(_resize_roi(ffhq_image, 112, FFHQ_TO_ARCFACE112_ROI), 2)
        expected_arc = _normalized(_resize_roi(arc_image, 128, arcface_roi), 2)
        expected_arc_id = _normalized(arc_image, 2)
        config = _config()

        for source, expected_src, expected_identity in (
            ([ffhq_dir], expected_ffhq, expected_id),
            ([(arc_dir, 0.0, "arcface")], expected_arc, expected_arc_id),
        ):
            native = TrainingDataLoader(batch_size=2, device=torch.device("cpu"), src=source, dst=[dst_dir], py_num_workers=0, **config)
            # 禁用随机几何时，不允许偷偷对已缩小的图像执行第二次 grid_sample。
            with patch("swapface.dataloader_native.F.grid_sample", side_effect=AssertionError("重复空间采样")):
                src, dst, canonical, theta, same, identity = native.next()
            torch.testing.assert_close(src, expected_src, atol=0, rtol=0)
            torch.testing.assert_close(dst, expected_ffhq, atol=0, rtol=0)
            torch.testing.assert_close(canonical, expected_ffhq, atol=0, rtol=0)
            torch.testing.assert_close(identity, expected_identity, atol=0, rtol=0)
            assert not same.any()
            torch.testing.assert_close(theta, torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]).expand(2, -1, -1), atol=0, rtol=0)

            flipped = TrainingDataLoader(batch_size=2, device=torch.device("cpu"), src=source, dst=[dst_dir], py_num_workers=0, **(config | {"flip_prob": 1.0}))
            flip_src, flip_dst, _, _, _, flip_id = flipped.next()
            torch.testing.assert_close(flip_src, expected_src.flip(-1), atol=0, rtol=0)
            torch.testing.assert_close(flip_dst, expected_ffhq.flip(-1), atol=0, rtol=0)
            torch.testing.assert_close(flip_id, expected_identity.flip(-1), atol=0, rtol=0)

            if torch.cuda.is_available() and torch.version.hip is None:
                try:
                    import_module("nvidia.dali")
                except ImportError:
                    continue
                dali = TrainingDataLoader(batch_size=2, device=torch.device("cuda"), src=source, dst=[dst_dir], py_num_workers=1, **config)
                dali_batch = dali.next()
                # 各实现 Lanczos 核的数值处理略有差异。检查几何与画质尺度，而非逐像素 bitwise 一致。
                for name, reference, gpu in zip(("src", "dst", "canonical", "theta", "same", "identity"), (src, dst, canonical, theta, same, identity), dali_batch, strict=True):
                    gpu = gpu.cpu()
                    if reference.dtype == torch.bool:
                        assert torch.equal(reference, gpu), name
                    else:
                        error = (reference - gpu).abs()
                        assert float(error.mean()) < 0.01 and float(error.max()) < 0.08, (name, float(error.mean()), float(error.max()))
                del dali

        if torch.cuda.is_available() and torch.version.hip is None:
            try:
                import_module("nvidia.dali")
            except ImportError:
                pass
            else:
                # 混合池必须逐样本选择 ROI，不能因 batch 中第一个元素的对齐方式而串用模板。
                mixed = TrainingDataLoader(batch_size=16, device=torch.device("cuda"),
                                           src=[ffhq_dir, (arc_dir, 0.0, "arcface")], dst=[dst_dir], py_num_workers=1, **config)
                mixed_src, mixed_dst, _, _, mixed_same, mixed_identity = mixed.next()
                mixed_src, mixed_dst, mixed_same, mixed_identity = (item.cpu() for item in (mixed_src, mixed_dst, mixed_same, mixed_identity))
                assert not mixed_same.any()
                torch.testing.assert_close(mixed_dst, expected_ffhq[:1].expand(16, -1, -1, -1), atol=0.08, rtol=0)
                id_ffhq_error = (mixed_identity - expected_id[:1]).abs().mean((1, 2, 3))
                id_arc_error = (mixed_identity - expected_arc_id[:1]).abs().mean((1, 2, 3))
                is_arc = id_arc_error < id_ffhq_error
                assert is_arc.any() and (~is_arc).any()
                assert (torch.where(is_arc, id_arc_error, id_ffhq_error) < 0.01).all()
                src_ffhq_error = (mixed_src - expected_ffhq[:1]).abs().mean((1, 2, 3))
                src_arc_error = (mixed_src - expected_arc[:1]).abs().mean((1, 2, 3))
                assert (torch.where(is_arc, src_arc_error, src_ffhq_error) < 0.01).all()
                del mixed

        # 高分辨率路径的真实像素采样回归。测试实际 loader，而非仅比对 ROI 公式。
        for size in (224, 256, 512):
            roi_ffhq, roi_arcface = generation_alignment_rois(size)
            for source, expected_source in (
                ([ffhq_dir], ffhq_image),
                ([(arc_dir, 0.0, "arcface")], arc_image),
            ):
                source_roi = roi_ffhq if source[0] == ffhq_dir else roi_arcface
                expected_src = _normalized(_resize_roi(expected_source, size, source_roi), 2)
                expected_dst = _normalized(_resize_roi(ffhq_image, size, roi_ffhq), 2)
                native = TrainingDataLoader(batch_size=2, device=torch.device("cpu"), src=source, dst=[dst_dir], py_num_workers=0, **(config | {"img_resolution": size}))
                src_batch, dst_batch, canonical, _, same_batch, identity = native.next()
                torch.testing.assert_close(src_batch, expected_src, atol=0, rtol=0)
                torch.testing.assert_close(dst_batch, expected_dst, atol=0, rtol=0)
                torch.testing.assert_close(canonical, expected_dst, atol=0, rtol=0)
                assert not same_batch.any()
                if torch.cuda.is_available() and torch.version.hip is None:
                    dali = TrainingDataLoader(batch_size=2, device=torch.device("cuda"), src=source, dst=[dst_dir], py_num_workers=1, **(config | {"img_resolution": size}))
                    dsrc, ddst, dcanonical, _, dsame, didentity = dali.next()
                    for ref, actual in ((expected_src, dsrc), (expected_dst, ddst), (expected_dst, dcanonical), (identity, didentity)):
                        error = (ref - actual.cpu()).abs()
                        # 224/256/512 的高频棋盘纹理暴露不同 Lanczos 实现的振铃差异；
                        # 使用宽松的像素误差界限，几何坐标由上面的独立仿射检验严格约束。
                        assert float(error.mean()) < 0.015 and float(error.max()) < 0.26, (size, float(error.mean()), float(error.max()))
                    assert not dsame.any()
                    del dali

        if torch.cuda.is_available() and torch.version.hip is None:
            # 之前未覆盖的 P1：SRC 全为 ArcFace，但 same pair 必须从 FFHQ dst 获取。
            for probability, batch_size in ((1.0, 2), (0.5, 32)):
                dali = TrainingDataLoader(batch_size=batch_size, device=torch.device("cuda"),
                                          src=[(arc_dir, 0.0, "arcface")], dst=[dst_dir], py_num_workers=1,
                                          **(config | {"same_prob": probability}))
                dsrc, ddst, _, _, dsame, didentity = (item.cpu() for item in dali.next())
                if probability >= 1.0:
                    assert dsame.all()
                else:
                    assert dsame.any() and (~dsame).any()
                expected_src = torch.where(
                    dsame[:, None, None, None],
                    expected_ffhq[:1], expected_arc[:1],
                )
                expected_identity = torch.where(
                    dsame[:, None, None, None],
                    expected_id[:1], expected_arc_id[:1],
                )
                for ref, actual in ((expected_src, dsrc), (expected_identity, didentity), (expected_ffhq[:1].expand(batch_size, -1, -1, -1), ddst)):
                    error = (ref - actual).abs()
                    assert float(error.mean()) < 0.01 and float(error.max()) < 0.08, (probability, float(error.mean()), float(error.max()))
                del dali

        if torch.cuda.is_available():
            # 原生 ROCm/CUDA 数据加载器：512×512 原图不经过 CPU resize，直接在 GPU
            # 对同尺寸 FFHQ 执行一次 bicubic；112 身份分支仍独立 Lanczos 采样。
            from misc.face_alignment import make_ffhq_to_arcface_grid

            from .dataloader_native import _NativeTrainingDataLoader

            native_gpu = _NativeTrainingDataLoader(batch_size=2, device=torch.device("cuda"),
                                                   src=[ffhq_dir], dst=[dst_dir], py_num_workers=0,
                                                   **(config | {"img_resolution": 512}))
            src_gpu, dst_gpu, canonical_gpu, _, same_gpu, identity_gpu = native_gpu.next()
            raw = torch.from_numpy(ffhq_pixels.copy()).permute(2, 0, 1).to("cuda", dtype=torch.float32).unsqueeze(0).expand(2, -1, -1, -1)
            grid = make_ffhq_to_arcface_grid(512, 512, 2, torch.device("cuda"))
            expected_gpu = torch.nn.functional.grid_sample(raw, grid, mode="bicubic", padding_mode="border", align_corners=False).clamp(0, 255).div(127.5).sub(1)
            for output in (src_gpu, dst_gpu, canonical_gpu):
                torch.testing.assert_close(output, expected_gpu, atol=1e-6, rtol=0)
            assert not same_gpu.any()
            torch.testing.assert_close(identity_gpu.cpu(), expected_id, atol=2e-7, rtol=0)
            del native_gpu

            # 混合 SRC：只有原生 512 FFHQ 样本走 GPU，再对齐后的 ArcFace112 样本不能重复变换。
            mixed_native = _NativeTrainingDataLoader(batch_size=16, device=torch.device("cuda"),
                                                     src=[ffhq_dir, (arc_dir, 0.0, "arcface")], dst=[dst_dir],
                                                     py_num_workers=0, **(config | {"img_resolution": 512}))
            mixed_src, _, _, _, mixed_same, mixed_id = mixed_native.next()
            assert not mixed_same.any()
            expected_ffhq_512 = expected_gpu[:1]
            expected_arc_512 = _normalized(_resize_roi(arc_image, 512, generation_alignment_rois(512)[1]), 1).to("cuda")
            matches_ffhq_id = ((mixed_id - expected_id[:1].to("cuda")).abs().flatten(1).max(1).values < 1e-5)
            assert matches_ffhq_id.any() and (~matches_ffhq_id).any()
            expected_mixed = torch.where(matches_ffhq_id[:, None, None, None], expected_ffhq_512, expected_arc_512)
            torch.testing.assert_close(mixed_src, expected_mixed, atol=2e-5, rtol=0)
            del mixed_native

        same_loader = TrainingDataLoader(batch_size=2, device=torch.device("cpu"), src=[(arc_dir, 0.0, "arcface")], dst=[dst_dir], py_num_workers=0, **(config | {"same_prob": 1.0}))
        same_src, same_dst, _, _, same_mask, same_identity = same_loader.next()
        assert same_mask.all()
        torch.testing.assert_close(same_src, same_dst, atol=0, rtol=0)
        torch.testing.assert_close(same_identity, expected_id, atol=0, rtol=0)

    print("PASS: direct ArcFace generation, multi-resolution ROIs, high-res GPU path, 112 teacher, padding, same pairs and DALI comparison")


if __name__ == "__main__":
    main()
