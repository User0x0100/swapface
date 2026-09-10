"""无需下载权重的回归检查：.venv/bin/python -m faceswap.test_inference"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import onnx
import torch
import torch.nn.functional as NF
from torch import nn
from torchvision.io import write_png

from faceswap import export, swapper
from faceswap.contracts import CHECKPOINT_VERSION
from faceswap.inference import get_ffhq_alignment_template, load_generator
from misc.face_alignment import FFHQ_TO_ARCFACE_112_AFFINE_512, make_alignment_grid_theta, restore_faces_to_original, transform_sampling_grid
from misc.models.id_encoder import IDEncoderProvider, get_alignment_template
from models.networks import Generator


class RecordingEncoder(nn.Module):
    def __init__(self, provider):
        super().__init__()
        self.provider = provider

    def forward(self, faces):
        self.last_input = faces.clone()
        return torch.nn.functional.normalize(faces.mean((2, 3)).repeat(1, 171)[:, :512], dim=1)


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {"image_size": 840}

    def detect(self, images):
        return None, [1]


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    provider = IDEncoderProvider.MS1MV3_ARCFACE_R50_FP16  # 非默认编码器，防止静默回退 BlendFace。
    model = Generator(img_resolution=16, num_depth=1, num_latent=1, base_ch=4, max_ch=8, aad_skip_layers=[0]).eval()
    faces = torch.rand(3, 3, 16, 16) * 2 - 1
    identity = torch.nn.functional.normalize(torch.randn(1, 512), dim=1)
    with tempfile.TemporaryDirectory(prefix="faceswap-check-") as temporary:
        directory = Path(temporary)
        checkpoint_path = directory / "current.pth"
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "iter": 123,
            "identity_encoders": {"generator": provider.name, "identity_loss": "BLENDFACE"},
            "net_g": {"network_cfg": model.network_cfg, "state_dict": model.state_dict()},
            "training_state": {"net_g": {key: torch.zeros_like(value) for key, value in model.state_dict().items()}},
        }
        torch.save(checkpoint, checkpoint_path)
        loaded, saved_provider, iteration = load_generator(checkpoint_path)
        assert saved_provider is provider and iteration == 123
        for key, value in loaded.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key])

        checkpoint_with_step = directory / "with-step.pth"
        torch.save({**checkpoint, "step": 124}, checkpoint_with_step)
        _, _, completed_step = load_generator(checkpoint_with_step)
        assert completed_step == 124

        # 五点目标经训练端 affine 映射后必须回到原 ArcFace 112 坐标。
        affine = torch.tensor(FFHQ_TO_ARCFACE_112_AFFINE_512)
        for resolution in (256, 512, 1024):
            template = get_ffhq_alignment_template(resolution, device)
            mapped = (template * (512.0 / resolution)) @ affine[:, :2].T + affine[:, 2]
            torch.testing.assert_close(mapped, torch.tensor(get_alignment_template(112)), atol=2e-5, rtol=0)

        with patch.object(swapper, "RetinaFace", return_value=Detector()), patch.object(swapper, "IDEncoder", RecordingEncoder), patch.object(swapper, "FaceMasker", return_value=nn.Identity()):
            native = swapper.FaceSwapper(str(checkpoint_path), device="cpu")
            assert native.id_encoder.provider is provider
            with torch.inference_mode():
                expected = model(faces, identity.expand(3, -1))
            torch.testing.assert_close(native.swap_faces(faces, identity), expected)
            assert native.swap_faces(faces[:0], identity).shape == (0, 3, 16, 16)

            # 源图身份提取直接组合 raw->FFHQ->ArcFace 网格，只允许一次图像重采样。
            image = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
            image_path = directory / "source.png"
            write_png(image, str(image_path))
            landmarks = get_ffhq_alignment_template(32, device).unsqueeze(0)
            with (
                patch.object(swapper, "extract_landmarks", return_value=landmarks),
                patch.object(swapper.NF, "grid_sample", wraps=swapper.NF.grid_sample) as grid_sample,
            ):
                embedding = native.extract_identity_embedding(image_path)
            assert grid_sample.call_count == 1

            normalized = image.float().unsqueeze(0) / 127.5 - 1.0
            raw_to_ffhq_theta = make_alignment_grid_theta(landmarks, native.alignment_template, (32, 32), 16)
            direct_grid = transform_sampling_grid(native.identity_encoder_grid, raw_to_ffhq_theta)
            expected_encoder_input = NF.grid_sample(normalized, direct_grid, mode="bilinear", padding_mode="border", align_corners=False)
            torch.testing.assert_close(native.id_encoder.last_input, expected_encoder_input, atol=0, rtol=0)
            assert embedding.shape == (1, 512)
            torch.testing.assert_close(embedding.norm(dim=1), torch.ones(1))

            for nhwc, batch_size in ((True, 2), (False, 2)):
                export_device = "cuda" if nhwc and torch.cuda.is_available() else "cpu"
                exported_path, exported_provider = export.export_face_swap(str(checkpoint_path), batch_size=batch_size, nhwc=nhwc, device=export_device, output_dir=temporary, file_prefix=f"swap-{nhwc}")
                assert exported_provider is provider
                onnx_model = onnx.load(exported_path)
                onnx.checker.check_model(onnx_model)
                runtime = swapper.FaceSwapper(str(exported_path), device="cpu")
                assert runtime.id_encoder.provider is provider
                actual = runtime.swap_faces(faces, identity)
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
                assert runtime.swap_faces(faces[:0], identity).shape == (0, 3, 16, 16)
                print(f"PASS: {'NHWC' if nhwc else 'NCHW'}, fixed batch={batch_size}, 3 faces, max error={(actual - expected).abs().max().item():.3g}")
                if nhwc and torch.cuda.is_available():
                    gpu_native = swapper.FaceSwapper(str(checkpoint_path), device="cuda")
                    assert gpu_native.bf16 == torch.cuda.is_bf16_supported(including_emulation=False)
                    gpu_output = gpu_native.swap_faces(faces, identity).cpu()
                    torch.testing.assert_close(gpu_output, expected, atol=5e-3, rtol=5e-3)
                    gpu_onnx = swapper.FaceSwapper(str(exported_path), device="cuda")
                    assert "CUDAExecutionProvider" in gpu_onnx.ort_session.get_providers()
                    assert gpu_onnx.ort_session.get_session_options().get_session_config_entry("session.disable_cpu_ep_fallback") == "1"
                    # CUDA ORT 热路径不得经过旧的 NumPy batch 复制。
                    with patch.object(np, "repeat", side_effect=AssertionError("CUDA ONNX path copied through NumPy")):
                        gpu_onnx_output = gpu_onnx.swap_faces(faces, identity).cpu()
                    torch.testing.assert_close(gpu_onnx_output, expected, atol=2e-4, rtol=2e-4)
                    print(f"PASS: CUDA PyTorch (BF16={gpu_native.bf16}) and ONNX IOBinding")

            # PyTorch/ONNX 都经过遮罩和原帧还原；检测缩放可关闭，小帧无需放大。
            for engine, full_resolution, detection_size in ((native, False, 840), (native, False, 16), (runtime, True, 16)):
                original = torch.full((1, 3, 32, 32), 127.5)
                crop = (faces[:1] + 1) * 127.5
                theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
                decoder = Mock()
                decoder.__len__ = Mock(return_value=2)
                decoder.__getitem__ = Mock(side_effect=[Mock(data=original.clone()), Mock(data=original.clone())])
                context = Mock()
                context.Process.return_value.is_alive.return_value = False
                engine.face_detector.config = {"image_size": detection_size}
                with (
                    patch.object(engine, "extract_identity_embedding", return_value=identity),
                    patch.object(swapper, "VideoDecoder", return_value=decoder),
                    patch.object(engine.face_detector, "detect", side_effect=[(None, [1]), (None, [0])]) as detect,
                    patch.object(swapper, "extract_landmarks", return_value=torch.empty(1, 5, 2)),
                    patch.object(swapper, "align_faces", side_effect=[(crop.clone(), theta), (crop[:0], theta[:0])]) as align,
                    patch.object(swapper.mp, "get_context", return_value=context),
                    patch.object(swapper, "restore_faces_to_original", wraps=swapper.restore_faces_to_original) as restore,
                ):
                    engine.swap_video("unused.mp4", "unused.png", batch_size=1, full_resolution_detection=full_resolution)
                    assert restore.call_count == 1
                    assert isinstance(detect.call_args_list[0].args[0], list) == (not full_resolution and detection_size < 32)
                    assert detect.call_args_list[0].kwargs == {"confidence_threshold": 0.99, "iou_threshold": 0.2, "min_face_size": (256, 256)}
                    torch.testing.assert_close(align.call_args_list[0].args[3], engine.alignment_template)
                queue = context.Queue.return_value
                assert queue.put.call_count == 3  # 两批视频帧 + 结束标记。
                assert queue.put.call_args_list[0].args[0].shape == (1, 8, 16, 3)
                assert queue.put.call_args_list[-1].args == (None,)

            # 预览减复制不能改变像素：覆盖非 4 整除宽度、重叠人脸、无脸帧。
            for width in (32, 33, 34, 35):
                original = torch.rand(3, 3, 31, width) * 2 - 1
                theta = torch.tensor([[[0.8, 0.1, -0.1], [-0.1, 0.8, 0.1]], [[0.6, -0.1, 0.2], [0.1, 0.6, 0.0]], [[0.9, 0.1, 0.8], [-0.1, 0.9, 0.7]]])
                for test_faces, test_theta, lengths in ((faces, theta, [2, 0, 1]), (faces[:0], theta[:0], [0, 0, 0])):
                    restored = restore_faces_to_original(original.clone(), test_faces, test_theta, lengths)
                    expected_preview = NF.interpolate(torch.cat((original, restored), dim=3), scale_factor=0.25, mode="bilinear", align_corners=False)
                    actual_preview = swapper.FaceSwapper._make_preview(original.clone(), test_faces, test_theta, lengths)
                    torch.testing.assert_close(actual_preview, expected_preview, atol=0, rtol=0)
            try:
                native.swap_video("unused.mp4", "unused.png", batch_size=0)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid video batch size was accepted")

            # 缺少新约定的 ONNX 必须明确拒绝，不能猜 provider 或 alignment。
            del onnx_model.metadata_props[:]
            old_path = directory / "missing-metadata.onnx"
            onnx.save(onnx_model, old_path)
            try:
                swapper.FaceSwapper(str(old_path), device="cpu")
            except ValueError as error:
                assert "重新导出" in str(error)
            else:
                raise AssertionError("ONNX without metadata was accepted")

        # all 必须使用 checkpoint 的生成器编码器；不能误用身份损失编码器或默认 provider。
        with patch.object(export, "export_face_swap", return_value=(directory / "model.onnx", provider)), patch.object(export, "export_id_encoder") as export_encoder:
            with patch("sys.argv", ["export", "all", "--checkpoint", str(checkpoint_path), "--device", "cpu"]):
                export.main()
            assert export_encoder.call_args.kwargs["provider_name"] == provider.name

        # 编码器 ONNX 明确接收已对齐的 112 RGB 图像，验证 NHWC/NCHW 两个适配入口。
        import onnxruntime as ort

        with patch.object(export, "IDEncoder", RecordingEncoder):
            for nhwc in (True, False):
                encoded_path = export.export_id_encoder(provider.name, nhwc=nhwc, device="cpu", output_dir=temporary, file_prefix=f"encoder-{nhwc}")
                session = ort.InferenceSession(str(encoded_path), providers=["CPUExecutionProvider"])
                metadata = session.get_modelmeta().custom_metadata_map
                assert metadata["faceswap.face_alignment"] == "arcface112"
                assert metadata["faceswap.provider"] == provider.name
                inputs = torch.rand(1, 3, 112, 112) * 2 - 1
                expected_identity = RecordingEncoder(provider)(inputs).numpy()
                inputs_np = inputs.permute(0, 2, 3, 1).numpy() if nhwc else inputs.numpy()
                actual_identity = session.run(["identity"], {"faces": inputs_np})[0]
                np.testing.assert_allclose(actual_identity, expected_identity, atol=1e-5, rtol=1e-5)

        for invalid in ({**checkpoint, "version": CHECKPOINT_VERSION - 1}, {key: value for key, value in checkpoint.items() if key != "identity_encoders"}):
            torch.save(invalid, directory / "invalid.pth")
            try:
                load_generator(directory / "invalid.pth")
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid checkpoint was accepted")
    print("PASS: EMA loading, provider selection, single-sample identity preprocessing, ONNX round-trips and invalid metadata/checkpoint rejection")


if __name__ == "__main__":
    main()
