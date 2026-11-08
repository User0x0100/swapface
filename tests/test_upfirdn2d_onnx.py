"""Regression checks for CUDA -> ONNX export of fused FIR resampling."""

import numpy as np
import pytest
import torch
from torch import nn

from models.discriminator.upfirdn2d import DownFIRDn2d, UpFIRDn2d, upfirdn2d
from models.discriminator.upfirdn2d.upfirdn2d import setup_filter


class _Filter(nn.Module):
    def __init__(self, separable: bool) -> None:
        super().__init__()
        self.register_buffer("kernel", setup_filter([1, 3, 3, 1], separable=separable))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upfirdn2d(x, self.kernel, 2, 2, 1, 1, 2, 1, 2, 1, False, 4.0)


def _export_and_compare(module: nn.Module, image: torch.Tensor, path: str) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    module = module.to(image.device).eval()
    with torch.inference_mode():
        expected = module(image).cpu().numpy()

    torch.onnx.export(
        module,
        (image,),
        path,
        dynamo=True,
        opset_version=18,
        external_data=False,
        input_names=["image"],
        output_names=["output"],
    )

    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert all(node.domain != "upfirdn2d" for node in model.graph.node)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": image.cpu().numpy()})[0]

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("separable", [False, True])
def test_cuda_fused_fir_exports_as_standard_onnx(tmp_path, separable: bool) -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("requires NVIDIA CUDA for fused-kernel export regression")

    image = torch.randn(1, 3, 9, 9, device="cuda")
    _export_and_compare(_Filter(separable), image, str(tmp_path / f"fir_{separable}.onnx"))


@pytest.mark.parametrize("layer", [UpFIRDn2d, DownFIRDn2d])
def test_cuda_fir_modules_export_as_standard_onnx(tmp_path, layer: type[nn.Module]) -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("requires NVIDIA CUDA for fused-kernel export regression")

    image = torch.randn(1, 3, 16, 16, device="cuda")
    _export_and_compare(layer(), image, str(tmp_path / f"{layer.__name__}.onnx"))


def test_stylegan2_cuda_generator_exports_without_custom_fir(tmp_path) -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("requires NVIDIA CUDA for fused-kernel export regression")

    from models.stylegan2 import Generator

    torch.manual_seed(2026)
    generator = Generator(img_resolution=32, mapping_layers=1, channel_multiplier=1).cuda().eval()
    target = torch.randn(1, 3, 32, 32, device="cuda")
    identity = torch.randn(1, 512, device="cuda")

    with torch.inference_mode():
        reference = generator(target, identity).cpu().numpy()

    path = str(tmp_path / "stylegan2_cuda.onnx")
    torch.onnx.export(
        generator,
        (target, identity),
        path,
        dynamo=True,
        opset_version=18,
        external_data=False,
        input_names=["target", "id_feat"],
        output_names=["output"],
    )

    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    exported = onnx.load(path)
    onnx.checker.check_model(exported)
    assert all(node.domain in ("", "ai.onnx") for node in exported.graph.node)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"target": target.cpu().numpy(), "id_feat": identity.cpu().numpy()})[0]

    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-3)


def test_cpu_fir_export_fallback(tmp_path) -> None:
    image = torch.randn(1, 3, 9, 9)
    _export_and_compare(_Filter(separable=False), image, str(tmp_path / "fir_cpu.onnx"))


def test_normal_cuda_dispatch_keeps_fused_kernel(monkeypatch) -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("requires NVIDIA CUDA fused kernel")

    import importlib

    module = importlib.import_module("models.discriminator.upfirdn2d.upfirdn2d")

    # Normal execution must not silently switch to the export-only native path.
    def _reject_native(*_args, **_kwargs):
        raise AssertionError("native FIR path unexpectedly selected outside ONNX export")

    monkeypatch.setattr(module, "_upfirdn2d_native", _reject_native)
    filt = setup_filter([1, 3, 3, 1]).cuda()
    image = torch.randn(1, 3, 9, 9, device="cuda")
    result = module.upfirdn2d(image, filt, 2, 2, 1, 1, 2, 1, 2, 1, False, 4.0)
    assert result.shape == (1, 3, 18, 18)
