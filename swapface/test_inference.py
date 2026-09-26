"""无需下载权重的回归检查：.venv/bin/python -m swapface.test_inference"""

import random
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import onnx
import torch
import torch.nn.functional as NF
from torch import nn
from torchvision.io import write_png

from misc.face_alignment import FFHQ_TO_ARCFACE_112_AFFINE_512, make_alignment_grid_theta, restore_faces_to_original, transform_sampling_grid
from misc.models.id_encoder import IDEncoderProvider, get_alignment_template
from models.networks import Generator
from swapface.contracts import CHECKPOINT_VERSION
from swapface.inference import get_ffhq_alignment_template, load_generator


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
    backend_before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    with (
        patch.object(random, "seed", side_effect=AssertionError("推理库导入不应修改 Python 随机种子")),
        patch.object(torch, "manual_seed", side_effect=AssertionError("推理库导入不应修改 Torch 随机种子")),
        patch.object(torch, "set_float32_matmul_precision", side_effect=AssertionError("推理库导入不应修改矩阵乘精度")),
    ):
        from swapface import export, swapper

    parser = export.build_parser()
    default_args = parser.parse_args(["swapface", "--checkpoint", "model.pth"])
    assert default_args.optimize is False and default_args.opset_version is None
    override_args = parser.parse_args(["swapface", "--checkpoint", "model.pth", "--optimize", "--opset-version", "20"])
    assert override_args.optimize is True and override_args.opset_version == 20
    with tempfile.TemporaryDirectory(prefix="swapface-export-failure-") as temporary:
        failed_output_dir = Path(temporary) / "onnx_export"
        with patch.object(torch.export, "export", side_effect=RuntimeError("capture failed")):
            try:
                export.export_to_onnx(
                    nn.Identity(),
                    (torch.zeros(1, 1),),
                    output_dir=failed_output_dir,
                    metadata={},
                    input_names=["input"],
                    output_name="output",
                )
            except RuntimeError as error:
                assert str(error) == "capture failed"
            else:
                raise AssertionError("torch.export failure was swallowed")
        assert not failed_output_dir.exists()
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) == backend_before

    torch.manual_seed(7)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    provider = IDEncoderProvider.MS1MV3_ARCFACE_R50_FP16  # 非默认编码器，防止静默回退 BlendFace。
    model = Generator(
        img_resolution=16,
        coarse_resolution=8,
        coarse_bottleneck_resolution=4,
        coarse_base_ch=2,
        coarse_max_ch=8,
        num_style_blocks=1,
        hq_bottleneck_resolution=4,
        hq_base_ch=2,
        hq_max_ch=8,
        hq_channel_hold_level=1,
    ).eval()
    faces = torch.rand(3, 3, 16, 16) * 2 - 1
    identity = torch.nn.functional.normalize(torch.randn(1, 512), dim=1)
    with tempfile.TemporaryDirectory(prefix="swapface-check-") as temporary:
        directory = Path(temporary)
        checkpoint_path = directory / "current.pth"
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "step": 123,
            "identity_encoders": {"generator": provider.name, "identity_loss": "BLENDFACE"},
            "net_g": {"network_cfg": model.network_cfg, "state_dict": model.state_dict()},
            "training_state": {"net_g": {key: torch.zeros_like(value) for key, value in model.state_dict().items()}},
        }
        torch.save(checkpoint, checkpoint_path)
        loaded, saved_provider, completed_step = load_generator(checkpoint_path)
        assert saved_provider is provider and completed_step == 123
        for key, value in loaded.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key])

        # 五点目标经训练端 affine 映射后必须回到原 ArcFace 112 坐标。
        affine = torch.tensor(FFHQ_TO_ARCFACE_112_AFFINE_512)
        for resolution in (128, 256, 512, 1024):
            template = get_ffhq_alignment_template(resolution, device)
            mapped = (template * (512.0 / resolution)) @ affine[:, :2].T + affine[:, 2]
            torch.testing.assert_close(mapped, torch.tensor(get_alignment_template(112)), atol=2e-5, rtol=0)

        with patch.object(swapper, "RetinaFace", return_value=Detector()), patch.object(swapper, "IDEncoder", RecordingEncoder), patch.object(swapper, "FaceMasker", return_value=nn.Identity()):
            native = swapper.SwapFace(str(checkpoint_path), device="cpu")
            assert native.id_encoder.provider is provider
            with torch.inference_mode():
                expected = model(faces, identity.expand(3, -1))
                expected_with_coarse, coarse = model(faces, identity.expand(3, -1), return_coarse=True)
            torch.testing.assert_close(expected_with_coarse, expected)
            assert coarse.shape == (3, 3, model.network_cfg["coarse_resolution"], model.network_cfg["coarse_resolution"])
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
                exported_path, exported_provider = export.export_swapface(str(checkpoint_path), batch_size=batch_size, nhwc=nhwc, device=export_device, output_dir=temporary, file_prefix=f"swap-{nhwc}")
                assert exported_provider is provider
                assert exported_path.parent.parent == directory
                assert any(exported_path.parent.glob("*.md"))
                onnx_model = onnx.load(exported_path)
                onnx.checker.check_model(onnx_model)
                exported_metadata = {item.key: item.value for item in onnx_model.metadata_props}
                assert exported_metadata["swapface.format"] == "2"
                assert exported_metadata["swapface.step"] == "123"
                assert exported_metadata["swapface.export.strict"] == "true"
                assert exported_metadata["swapface.export.dynamo"] == "true"
                assert exported_metadata["swapface.export.optimize"] == "false"
                assert exported_metadata["swapface.export.verify"] == "false"
                assert exported_metadata["swapface.export.report"] == "true"
                assert exported_metadata["swapface.export.opset_version"] == "auto"
                assert int(exported_metadata["swapface.export.opset"]) > 0
                assert "swapface.iter" not in exported_metadata
                graph_inputs = {value.name: value for value in onnx_model.graph.input}
                graph_outputs = {value.name: value for value in onnx_model.graph.output}
                assert set(graph_inputs) == {"faces", "identity"}
                assert set(graph_outputs) == {"swapped_faces"}
                face_shape = [dim.dim_value for dim in graph_inputs["faces"].type.tensor_type.shape.dim]
                identity_shape = [dim.dim_value for dim in graph_inputs["identity"].type.tensor_type.shape.dim]
                output_shape = [dim.dim_value for dim in graph_outputs["swapped_faces"].type.tensor_type.shape.dim]
                expected_shape = [batch_size, 16, 16, 3] if nhwc else [batch_size, 3, 16, 16]
                assert face_shape == expected_shape
                assert identity_shape == [batch_size, 512]
                assert output_shape == expected_shape
                assert exported_metadata["swapface.layout"] == ("NHWC" if nhwc else "NCHW")
                print(f"PASS: {'NHWC' if nhwc else 'NCHW'}, fixed batch={batch_size}, ONNX structure")
                if nhwc and torch.cuda.is_available():
                    gpu_native = swapper.SwapFace(str(checkpoint_path), device="cuda")
                    assert gpu_native.bf16 == torch.cuda.is_bf16_supported(including_emulation=False)
                    gpu_output = gpu_native.swap_faces(faces, identity).cpu()
                    torch.testing.assert_close(gpu_output, expected, atol=5e-3, rtol=5e-3)
                    print(f"PASS: CUDA PyTorch (BF16={gpu_native.bf16})")

            # 视频路径经过遮罩和原帧还原；检测缩放可关闭，小帧无需放大。
            for engine, full_resolution, detection_size in ((native, False, 840), (native, False, 16), (native, True, 16)):
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
                    actual_preview = swapper.SwapFace._make_preview(original.clone(), test_faces, test_theta, lengths)
                    torch.testing.assert_close(actual_preview, expected_preview, atol=0, rtol=0)
            try:
                native.swap_video("unused.mp4", "unused.png", batch_size=0)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid video batch size was accepted")

        # all 必须使用 checkpoint 的生成器编码器，并透传导出选项。
        with patch.object(export, "export_swapface", return_value=(directory / "model.onnx", provider)) as export_swapface, patch.object(export, "export_id_encoder") as export_encoder:
            with patch("sys.argv", ["export", "all", "--checkpoint", str(checkpoint_path), "--device", "cpu", "--optimize", "--opset-version", "20"]):
                export.main()
            options = export_swapface.call_args.kwargs["export_options"]
            assert options.optimize is True and options.opset_version == 20
            assert export_encoder.call_args.kwargs["export_options"] == options
            assert export_encoder.call_args.kwargs["export_dir"] == directory
            assert export_encoder.call_args.kwargs["provider_name"] == provider.name

        # 编码器 ONNX 明确接收已对齐的 112 RGB 图像，验证 NHWC/NCHW 两个导出接口。
        with patch.object(export, "IDEncoder", RecordingEncoder):
            for nhwc in (True, False):
                encoded_path = export.export_id_encoder(provider.name, nhwc=nhwc, device="cpu", output_dir=temporary, file_prefix=f"encoder-{nhwc}")
                assert encoded_path.parent.parent == directory
                assert any(encoded_path.parent.glob("*.md"))
                encoded_model = onnx.load(encoded_path)
                onnx.checker.check_model(encoded_model)
                metadata = {item.key: item.value for item in encoded_model.metadata_props}
                assert metadata["swapface.format"] == "2"
                assert metadata["swapface.face_alignment"] == "arcface112"
                assert metadata["swapface.provider"] == provider.name
                assert metadata["swapface.layout"] == ("NHWC" if nhwc else "NCHW")
                inputs = {value.name: value for value in encoded_model.graph.input}
                outputs = {value.name: value for value in encoded_model.graph.output}
                assert set(inputs) == {"faces"}
                assert set(outputs) == {"identity"}
                input_shape = [dim.dim_value for dim in inputs["faces"].type.tensor_type.shape.dim]
                output_shape = [dim.dim_value for dim in outputs["identity"].type.tensor_type.shape.dim]
                assert input_shape == ([1, 112, 112, 3] if nhwc else [1, 3, 112, 112])
                assert output_shape == [1, 512]

        for invalid in (
            {**checkpoint, "version": CHECKPOINT_VERSION - 1},
            {key: value for key, value in checkpoint.items() if key != "identity_encoders"},
            {key: value for key, value in checkpoint.items() if key != "step"},
        ):
            torch.save(invalid, directory / "invalid.pth")
            try:
                load_generator(directory / "invalid.pth")
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid checkpoint was accepted")
    print("PASS: EMA loading, provider selection, single-sample identity preprocessing, ONNX structure checks and invalid checkpoint rejection")


if __name__ == "__main__":
    main()
