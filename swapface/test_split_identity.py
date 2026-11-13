"""Split identity export regression: .venv/bin/python -m swapface.test_split_identity.

No model downloads: a tiny fake encoder exercises the real checkpoint/export
pair and ONNX Runtime evaluation for both Coarse architectures.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from misc.models.id_encoder import IDEncoderProvider
from models.networks import Generator
from swapface.contracts import CHECKPOINT_VERSION
from swapface.export import OnnxExportOptions, build_parser, export_to_onnx
from swapface.split_export import _pair_identity, export_split
from swapface.split_identity import build_split_models


class FakeEncoder(nn.Module):
    def __init__(self, provider: IDEncoderProvider | None = None) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 512)

    def forward(self, image: Tensor) -> Tensor:
        return F.normalize(self.backbone(image.mean((2, 3))), dim=1)


def _generator(coarse_type: str) -> Generator:
    return Generator(
        img_resolution=16, img_channels=3, coarse_resolution=8,
        coarse_latent_resolution=4, coarse_num_latent=2,
        coarse_base_ch=4, coarse_max_ch=8,
        hq_bottleneck_resolution=4, hq_base_ch=2, hq_max_ch=8,
        coarse_type=coarse_type,
    ).eval()


def _run_pair(coarse_type: str, root: Path) -> None:
    model = _generator(coarse_type)
    if coarse_type == "stylegan2":
        # Exercise Target feature concatenation, not only the zero-gain initial state.
        with torch.no_grad():
            for module in model.coarse.modules():
                if hasattr(module, "inject_gain"):
                    module.inject_gain.fill_(0.25)
    encoder = FakeEncoder().eval()
    server, client = build_split_models(model, encoder)
    assert server.projection.coarse_type == client.coarse_type == coarse_type
    assert server.projection.condition_width == client.condition_width
    assert not any(isinstance(module, nn.Linear) for module in client.modules())
    if coarse_type == "inswapper":
        assert len(server.projection.mapping.linears) == len(model.coarse.latent_space.layers) * 2
    else:
        assert len(server.projection.mapping.modulations) == 2 + len(model.coarse.generator.stages) * 3
    assert not any(
        any(part in key for part in ("linear", "modulation", "id_mapping", ".style."))
        for key in client.state_dict()
    ), "client contains private identity mapping weights"

    # Source and target are independently variable. Identity parameters can be
    # computed once and reused for multiple targets, with no output regression.
    with torch.inference_mode():
        for batch in (1, 2):
            images = torch.randn(batch, 3, 16, 16)
            identity_faces = torch.randn(batch, 3, 112, 112)
            identity = encoder(identity_faces)
            condition = server(identity_faces)
            torch.testing.assert_close(condition, server.projection(identity), rtol=0, atol=0)
            for target in (images, images * 0.3):
                torch.testing.assert_close(
                    model(target, identity), client(target, condition), rtol=1e-6, atol=1e-6,
                )

    # Exercise the actual CLI export path, including independent directories,
    # metadata contract, pair ID, and CPU ONNX Runtime numeric compatibility.
    path = root / f"{coarse_type}.pth"
    provider = IDEncoderProvider.MS1MV3_ARCFACE_R50_FP16
    torch.save({
        "version": CHECKPOINT_VERSION,
        "step": 12,
        "identity_encoders": {"generator": provider.name},
        "net_g": {"network_cfg": model.network_cfg, "state_dict": model.state_dict()},
    }, path)
    # Patch the freshly instantiated encoder so ONNX input/output uses
    # the same weights as the encoder used in the torch reference.
    torch.manual_seed(999)
    sample_encoder = FakeEncoder()
    with patch("swapface.split_export.IDEncoder", return_value=sample_encoder):
        client_path, server_path, actual_provider = export_split(
            checkpoint_path=str(path), batch_size=1,
            device="cpu", output_dir=str(root / "onnx"),
            file_prefix=coarse_type,
            export_options=OnnxExportOptions(opset_version=18),
        )
    assert actual_provider is provider
    assert client_path.parent == server_path.parent
    assert client_path.parent.parent == (root / "onnx").resolve()
    manifest = json.loads((client_path.parent / "manifest.json").read_text())
    assert manifest["condition_width"] == client.condition_width
    assert len(manifest["pair_id"]) == 64
    assert manifest["generator_id"] and manifest["encoder_id"]
    assert manifest["coarse_type"] == coarse_type
    assert manifest["batch_size"] == 1 and manifest["version"] == 2
    assert manifest["artifacts"]["client"]["onnx"] == client_path.name
    assert manifest["artifacts"]["server"]["onnx"] == server_path.name
    assert {item.name for item in client_path.parent.iterdir()} == {client_path.name, server_path.name, "manifest.json"}

    onnx_client = onnx.load(str(client_path))
    onnx_server = onnx.load(str(server_path))
    onnx.checker.check_model(onnx_client)
    onnx.checker.check_model(onnx_server)
    metadata = lambda proto: {field.key: field.value for field in proto.metadata_props}
    assert metadata(onnx_client)["swapface.split.pair_id"] == metadata(onnx_server)["swapface.split.pair_id"] == manifest["pair_id"]
    assert metadata(onnx_client)["swapface.split.role"] == "client"
    assert metadata(onnx_server)["swapface.split.role"] == "server"
    assert metadata(onnx_client)["swapface.split.batch_size"] == "1"
    assert metadata(onnx_server)["swapface.split.encoder_id"] == manifest["encoder_id"]
    assert metadata(onnx_client)["swapface.split.generator_id"] == manifest["generator_id"]
    assert not any(node.op_type in ("SplitToSequence", "SequenceAt") for node in onnx_client.graph.node)
    assert [item.name for item in onnx_client.graph.input] == ["faces", "identity_condition"]
    assert [item.name for item in onnx_server.graph.input] == ["faces"]

    # Parser-only validation: no need to build a GPU engine. The small test
    # exports must work with default optimize=False, not only --optimize.
    try:
        import tensorrt as trt
    except ImportError:
        pass
    else:
        if torch.cuda.is_available():
            logger = trt.Logger(trt.Logger.ERROR)
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            assert parser.parse_from_file(str(client_path)), [str(parser.get_error(i)) for i in range(parser.num_errors)]
            if os.environ.get("SWAPFACE_TEST_TRT_ENGINE") == "1":
                config = builder.create_builder_config()
                config.builder_optimization_level = 0
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
                assert builder.build_serialized_network(network, config) is not None

    ort_server = ort.InferenceSession(str(server_path), providers=["CPUExecutionProvider"])
    ort_client = ort.InferenceSession(str(client_path), providers=["CPUExecutionProvider"])
    source = torch.randn(1, 3, 112, 112).clamp(-1, 1)
    target = torch.randn(1, 3, 16, 16).clamp(-1, 1)
    with torch.inference_mode():
        source_condition = sample_encoder(source)
        expected_condition = server.projection(source_condition)
        expected_output = model(target, source_condition)
    actual_condition = ort_server.run(None, {"faces": source.permute(0, 2, 3, 1).numpy()})[0]
    np.testing.assert_allclose(actual_condition, expected_condition.numpy(), atol=1e-4, rtol=1e-4)
    actual_output = ort_client.run(None, {
        "faces": target.permute(0, 2, 3, 1).numpy(),
        "identity_condition": actual_condition,
    })[0]
    np.testing.assert_allclose(actual_output, expected_output.permute(0, 2, 3, 1).numpy(), atol=5e-4, rtol=5e-4)



def _test_nchw_pair(root: Path, coarse_type: str) -> None:
    model = _generator(coarse_type)
    checkpoint = root / f"{coarse_type}-nchw.pth"
    provider = IDEncoderProvider.BLENDFACE
    torch.save({
        "version": CHECKPOINT_VERSION,
        "step": 31,
        "identity_encoders": {"generator": provider.name},
        "net_g": {"network_cfg": model.network_cfg, "state_dict": model.state_dict()},
    }, checkpoint)
    with patch("swapface.split_export.IDEncoder", FakeEncoder):
        client_path, server_path, _ = export_split(
            str(checkpoint), batch_size=2, nhwc=False, device="cpu",
            output_dir=str(root / "nchw-public"),
            export_options=OnnxExportOptions(opset_version=18),
        )
    assert client_path.parent == server_path.parent
    client_meta = json.loads((client_path.parent / "manifest.json").read_text())
    assert client_meta["artifacts"]["client"]["onnx"] == client_path.name
    assert client_meta["artifacts"]["server"]["onnx"] == server_path.name
    assert client_meta["layout"] == "NCHW" and client_meta["batch_size"] == 2
    ort_server = ort.InferenceSession(str(server_path), providers=["CPUExecutionProvider"])
    ort_client = ort.InferenceSession(str(client_path), providers=["CPUExecutionProvider"])
    condition = ort_server.run(None, {"faces": torch.rand(2, 3, 112, 112).numpy()})[0]
    result = ort_client.run(None, {"faces": torch.rand(2, 3, 16, 16).numpy(), "identity_condition": condition})[0]
    assert result.shape == (2, 3, 16, 16)


def _test_pair_identity() -> None:
    model = _generator("inswapper")
    encoder = FakeEncoder()
    provider = IDEncoderProvider.BLENDFACE

    def pair(gen: Generator = model, enc: nn.Module = encoder, batch: int = 1, layout: str = "NHWC") -> dict:
        return _pair_identity(gen, enc, provider=provider, coarse_type="inswapper", condition_width=64, batch_size=batch, layout=layout)

    initial = pair()
    assert initial == pair(deepcopy(model), deepcopy(encoder)), "pair id must follow effective weights, not checkpoint packaging"
    assert initial["pair_id"] != pair(batch=2)["pair_id"]
    assert initial["pair_id"] != pair(layout="NCHW")["pair_id"]
    changed = deepcopy(model)
    with torch.no_grad():
        changed.coarse.latent_space.layers[0].adain0.linear.weight[0, 0] += 0.125
    assert initial["generator_id"] != pair(gen=changed)["generator_id"]
    assert initial["encoder_id"] == pair(gen=changed)["encoder_id"]
    changed_encoder = deepcopy(encoder)
    with torch.no_grad():
        changed_encoder.backbone.weight[0, 0] += 0.125
    assert initial["encoder_id"] != pair(enc=changed_encoder)["encoder_id"]


def _test_failure_cleanup(root: Path) -> None:
    model = _generator("inswapper")
    checkpoint_path = root / "failure.pth"
    torch.save({
        "version": CHECKPOINT_VERSION,
        "step": 123,
        "identity_encoders": {"generator": IDEncoderProvider.BLENDFACE.name},
        "net_g": {"network_cfg": model.network_cfg, "state_dict": model.state_dict()},
    }, checkpoint_path)
    output_root = root / "export"

    # Failure after the first ONNX must never leave a published release.
    def fail_server(*args, **kwargs):
        if kwargs["metadata"]["swapface.split.role"] == "server":
            raise RuntimeError("injected server export failure")
        return export_to_onnx(*args, **kwargs)

    with patch("swapface.split_export.IDEncoder", FakeEncoder), patch("swapface.split_export.export_to_onnx", side_effect=fail_server):
        try:
            export_split(str(checkpoint_path), device="cpu", output_dir=str(output_root),
                         export_options=OnnxExportOptions(opset_version=18))
        except RuntimeError as error:
            assert "injected" in str(error)
        else:
            raise AssertionError("server export failure must propagate")
    assert not list(output_root.iterdir()), "failed export leaked staged models"

    # A publish failure must also remove the complete staged directory.
    original_rename = Path.rename
    def fail_publish(self: Path, target: Path):
        if self.parent == output_root and self.name.startswith(".split-stage-"):
            raise OSError("injected publish failure")
        return original_rename(self, target)

    with patch("swapface.split_export.IDEncoder", FakeEncoder), patch.object(Path, "rename", fail_publish):
        try:
            export_split(str(checkpoint_path), device="cpu", output_dir=str(output_root),
                         export_options=OnnxExportOptions(opset_version=18))
        except OSError as error:
            assert "injected" in str(error)
        else:
            raise AssertionError("publish failure must propagate")
    assert not list(output_root.iterdir()), "failed publish left staged files"


def main() -> None:
    torch.manual_seed(29)
    torch.set_num_threads(2)
    parser = build_parser()
    split_args = parser.parse_args(["split", "--checkpoint", "model.pth"])
    assert split_args.command == "split" and split_args.opset_version == 18
    assert not hasattr(split_args, "private_output_dir")
    _test_pair_identity()
    with TemporaryDirectory(prefix="swapface-split-test-") as tmp:
        _test_failure_cleanup(Path(tmp))
        for coarse_type in ("inswapper", "stylegan2"):
            print(f"Testing split export: {coarse_type}", flush=True)
            _run_pair(coarse_type, Path(tmp))
            _test_nchw_pair(Path(tmp), coarse_type)
    print("PASS: split InSwapper/StyleGAN2 model parity, ONNX export, and ORT execution")


if __name__ == "__main__":
    main()
