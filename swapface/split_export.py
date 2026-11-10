"""Version-bound ONNX exports for server identity and client inference."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import onnx
import torch
from torch import Tensor, nn

from misc.models.id_encoder import IDEncoder, IDEncoderProvider

from .contracts import ONNX_CONTRACT
from .export import OnnxExportOptions, export_to_onnx
from .inference import load_generator
from .split_identity import SplitGenerator, SplitIdentityEncoder, build_split_models

SPLIT_PROTOCOL_VERSION = 2
_UNSUPPORTED_TRT_OPS = frozenset({"SplitToSequence", "SequenceAt"})


class ClientNHWCAdapter(nn.Module):
    def __init__(self, runtime: SplitGenerator) -> None:
        super().__init__()
        self.runtime = runtime

    def forward(self, faces: Tensor, identity_condition: Tensor) -> Tensor:
        images = torch.clamp(faces, -1.0, 1.0).permute(0, 3, 1, 2).contiguous()
        output = self.runtime(images, identity_condition)
        return output.permute(0, 2, 3, 1)


class ServerNHWCAdapter(nn.Module):
    def __init__(self, server: SplitIdentityEncoder) -> None:
        super().__init__()
        self.server = server

    def forward(self, faces: Tensor) -> Tensor:
        images = torch.clamp(faces, -1.0, 1.0).permute(0, 3, 1, 2).contiguous()
        return self.server(images)


def _stable_digest(model: nn.Module, *, metadata: dict[str, Any]) -> str:
    """Hash effective model parameters, not training history/checkpoint packaging.

    Streaming tensor bytes avoids copying the entire generator/encoder into an
    additional Python bytes object. Handles BF16 and scalar tensors as well.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        flat = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(flat))
    return digest.hexdigest()


def _pair_identity(
    generator: nn.Module,
    encoder: nn.Module,
    *,
    provider: IDEncoderProvider,
    coarse_type: str,
    condition_width: int,
    batch_size: int,
    layout: str,
) -> dict[str, Any]:
    generator_id = _stable_digest(generator, metadata={
        "network_cfg": generator.network_cfg,
        "face_alignment": getattr(generator, "face_alignment", "ffhq"),
    })
    encoder_metadata: dict[str, Any] = {"provider": provider.name, "encoder_type": type(encoder).__qualname__}
    for name in ("input_mean", "input_std"):
        value = getattr(encoder, name, None)
        if isinstance(value, Tensor):
            encoder_metadata[name] = value.detach().cpu().tolist()
    encoder_id = _stable_digest(encoder, metadata=encoder_metadata)
    contract = {
        "version": SPLIT_PROTOCOL_VERSION,
        "generator_id": generator_id,
        "encoder_id": encoder_id,
        "provider": provider.name,
        "coarse_type": coarse_type,
        "condition_width": condition_width,
        "condition_dtype": "float32",
        "batch_size": batch_size,
        "layout": layout,
    }
    contract["pair_id"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return contract


def _validate_onnx(path: Path, *, client: bool) -> None:
    onnx.checker.check_model(str(path))
    if client:
        model = onnx.load(str(path), load_external_data=False)
        unsupported = sorted(set(node.op_type for node in model.graph.node) & _UNSUPPORTED_TRT_OPS)
        if unsupported:
            raise RuntimeError(f"客户端 ONNX 存在 TensorRT 不支持的序列算子: {unsupported}")
        if [item.name for item in model.graph.input] != ["faces", "identity_condition"]:
            raise RuntimeError("客户端 ONNX 输入协议不匹配")


def _prune_export_reports(stage: Path, keep: set[Path]) -> None:
    """Only publish the two ONNX models and manifest, not export reports."""
    for entry in stage.iterdir():
        if entry in keep:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_split(
    checkpoint_path: str,
    batch_size: int = 1,
    nhwc: bool = True,
    device: str = "cuda",
    output_dir: str = "onnx_export",
    file_prefix: str = "swapface",
    export_options: OnnxExportOptions | None = None,
) -> tuple[Path, Path, IDEncoderProvider]:
    """Export both models and one manifest into a single release folder.

    The output directory is a *build artifact*, not a client distribution
    package. Only the client ONNX may be deployed to end-user machines.
    Publishing is atomic for the complete release folder: neither ONNX file
    becomes visible as a final release before both pass validation.
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    export_options = export_options or OnnxExportOptions(opset_version=18)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", file_prefix):
        raise ValueError("file_prefix 必须是安全的文件名，不能包含路径分隔符")

    generator, provider, step = load_generator(checkpoint_path)
    encoder = IDEncoder(provider)
    server, runtime = build_split_models(generator, encoder)
    coarse_type = server.projection.coarse_type
    width = server.projection.condition_width
    layout = "NHWC" if nhwc else "NCHW"
    contract = _pair_identity(
        generator, encoder, provider=provider, coarse_type=coarse_type,
        condition_width=width, batch_size=batch_size, layout=layout,
    )
    pair_id = contract["pair_id"]
    release_name = f"{file_prefix}-split-{step}-{pair_id[:12]}-{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')}"
    client_name = f"{file_prefix}-{step}-client.onnx"
    server_name = f"{file_prefix}-{step}-server.onnx"
    output_root = Path(output_dir).resolve()
    release_dir = output_root / release_name

    metadata = ONNX_CONTRACT | {
        "swapface.split.version": str(SPLIT_PROTOCOL_VERSION),
        "swapface.split.pair_id": pair_id,
        "swapface.split.generator_id": contract["generator_id"],
        "swapface.split.encoder_id": contract["encoder_id"],
        "swapface.split.coarse_type": coarse_type,
        "swapface.split.condition_width": str(width),
        "swapface.split.condition_dtype": "float32",
        "swapface.split.batch_size": str(batch_size),
        "swapface.step": str(step),
        "swapface.provider": provider.name,
        "swapface.layout": layout,
        "swapface.face_alignment": getattr(generator, "face_alignment", "ffhq"),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    # Stage everything on the final filesystem, then publish with one rename.
    with TemporaryDirectory(prefix=".split-stage-", dir=output_root) as stage_dir:
        stage = Path(stage_dir)

        client = (ClientNHWCAdapter(runtime) if nhwc else runtime).cpu().eval()
        target_shape = (batch_size, generator.img_resolution, generator.img_resolution, generator.img_channels) if nhwc else (
            batch_size, generator.img_channels, generator.img_resolution, generator.img_resolution,
        )
        client_path = export_to_onnx(
            client, (torch.randn(target_shape), torch.randn(batch_size, width)),
            output_dir=output_dir, file_prefix=client_name.removesuffix(".onnx"),
            metadata=metadata | {"swapface.kind": "generator_split", "swapface.split.role": "client"},
            input_names=["faces", "identity_condition"], output_name="swapped_faces",
            options=export_options, export_dir=stage,
        )
        _validate_onnx(client_path, client=True)

        server_device = torch.device(device)
        server_model = (ServerNHWCAdapter(server) if nhwc else server).to(server_device).eval()
        identity_shape = (batch_size, 112, 112, 3) if nhwc else (batch_size, 3, 112, 112)
        server_path = export_to_onnx(
            server_model, (torch.randn(identity_shape, device=server_device),),
            output_dir=output_dir, file_prefix=server_name.removesuffix(".onnx"),
            metadata=metadata | {"swapface.kind": "identity_conditioner", "swapface.split.role": "server", "swapface.face_alignment": "arcface112"},
            input_names=["faces"], output_name="identity_condition",
            options=export_options, export_dir=stage,
        )
        _validate_onnx(server_path, client=False)
        _prune_export_reports(stage, {client_path, server_path})

        manifest = contract | {
            "step": step,
            "artifacts": {
                "client": {
                    "onnx": client_name,
                    "onnx_sha256": _file_sha256(client_path),
                    "face_alignment": getattr(generator, "face_alignment", "ffhq"),
                    "input_names": ["faces", "identity_condition"],
                    "output_names": ["swapped_faces"],
                },
                "server": {
                    "onnx": server_name,
                    "onnx_sha256": _file_sha256(server_path),
                    "face_alignment": "arcface112",
                    "input_names": ["faces"],
                    "output_names": ["identity_condition"],
                },
            },
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if release_dir.exists():
            raise FileExistsError("导出目标目录已存在，拒绝覆盖现有发行")
        stage.rename(release_dir)

    print(f"拆分导出完成: {release_dir}")
    print("注意：服务端 ONNX 含私有身份映射权重，客户端发布时仅分发 client.onnx。")
    return release_dir / client_name, release_dir / server_name, provider
