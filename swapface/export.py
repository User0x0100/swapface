import argparse
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from misc.models.id_encoder import IDEncoder, IDEncoderProvider
from models.networks import Generator

from .contracts import ONNX_CONTRACT
from .inference import load_generator

DEFAULT_ID_ENCODER_PROVIDER_NAME = IDEncoderProvider.BLENDFACE.name


@dataclass(frozen=True, slots=True)
class OnnxExportOptions:
    optimize: bool = False
    opset_version: int | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
def _configure_export_runtime() -> None:
    random.seed(42)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers
# ──────────────────────────────────────────────────────────────────────────────
class SwapFaceNHWCAdapter(nn.Module):
    def __init__(self, model: Generator) -> None:
        super().__init__()
        self.model = model

    def forward(self, images_nhwc: Tensor, identity_embedding: Tensor) -> Tensor:
        images_nhwc = torch.clamp(images_nhwc, -1.0, 1.0)
        images_nchw = images_nhwc.permute(0, 3, 1, 2).contiguous()
        output = self.model(images_nchw, identity_embedding)
        return output.permute(0, 2, 3, 1)


class IDEncoderNHWCAdapter(nn.Module):
    def __init__(self, model: IDEncoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, images_nhwc: Tensor) -> Tensor:
        images_nhwc = torch.clamp(images_nhwc, -1.0, 1.0)
        return self.model(images_nhwc.permute(0, 3, 1, 2).contiguous())


# ──────────────────────────────────────────────────────────────────────────────
# Core export helper
# ──────────────────────────────────────────────────────────────────────────────
def export_to_onnx(
    model: nn.Module,
    example_inputs: tuple[Any, ...],
    output_dir: str | os.PathLike = "onnx_export",
    file_prefix: str | None = None,
    *,
    metadata: dict[str, str],
    input_names: list[str],
    output_name: str,
    options: OnnxExportOptions | None = None,
    export_dir: str | os.PathLike | None = None,
) -> Path:
    options = options or OnnxExportOptions()
    output_dir_path = Path(output_dir)

    prefix = file_prefix or "model"
    export_dir_path = Path(export_dir) if export_dir is not None else output_dir_path / f"{prefix}-{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')}"

    with torch.inference_mode():
        exported_program = torch.export.export(model.eval(), example_inputs, strict=True)
        export_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = export_dir_path / f"{prefix}.onnx"
        onnx_program = torch.onnx.export(
            model=exported_program,
            input_names=input_names,
            output_names=[output_name],
            opset_version=options.opset_version,
            dynamo=True,
            optimize=options.optimize,
            verify=False,
            report=True,
            profile=False,
            dump_exported_program=False,
            external_data=False,
            artifacts_dir=export_dir_path,
            verbose=False,
        )

    actual_opset = onnx_program.model.opset_imports.get("")
    export_metadata = {
        "swapface.export.torch": str(torch.__version__),
        "swapface.export.opset": str(actual_opset),
        "swapface.export.strict": "true",
        "swapface.export.dynamo": "true",
        "swapface.export.optimize": str(options.optimize).lower(),
        "swapface.export.verify": "false",
        "swapface.export.report": "true",
        "swapface.export.profile": "false",
        "swapface.export.dump_exported_program": "false",
        "swapface.export.external_data": "false",
        "swapface.export.opset_version": "auto" if options.opset_version is None else str(options.opset_version),
    }
    onnx_program.model.metadata_props.update(metadata | export_metadata)
    onnx_program.save(output_path, external_data=False)

    print(f"模型成功导出到: {output_path}")
    print(f"导出目录: {export_dir_path}")
    print(f"ONNX: opset={actual_opset}, optimize={options.optimize}, verify=False, report=True")
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Export functions
# ──────────────────────────────────────────────────────────────────────────────
def export_swapface(
    checkpoint_path: str,
    batch_size: int = 1,
    nhwc: bool = True,
    device: str = "cuda",
    output_dir: str = "onnx_export",
    file_prefix: str = "swapface",
    export_options: OnnxExportOptions | None = None,
    export_dir: str | os.PathLike | None = None,
) -> tuple[Path, IDEncoderProvider]:
    target_device = torch.device(device)

    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    model, provider, training_step = load_generator(checkpoint_path)
    print(f"Loading EMA checkpoint: {checkpoint_path}, step={training_step}, provider={provider.name}")

    network_config = model.network_cfg
    image_resolution, image_channels, identity_dim = network_config["img_resolution"], network_config["img_channels"], network_config["id_dim"]

    if nhwc:
        wrapped = SwapFaceNHWCAdapter(model).to(target_device)
        x = torch.randn((batch_size, image_resolution, image_resolution, image_channels), device=target_device, dtype=torch.float32)
    else:
        wrapped = model.to(target_device)
        x = torch.randn((batch_size, image_channels, image_resolution, image_resolution), device=target_device, dtype=torch.float32)

    identity_embedding = torch.randn((batch_size, identity_dim), device=target_device, dtype=torch.float32)
    output_path = export_to_onnx(
        wrapped,
        (x, identity_embedding),
        output_dir=output_dir,
        file_prefix=f"{file_prefix}-{training_step}",
        metadata=ONNX_CONTRACT | {"swapface.kind": "generator", "swapface.provider": provider.name, "swapface.layout": "NHWC" if nhwc else "NCHW", "swapface.step": str(training_step)},
        input_names=["faces", "identity"],
        output_name="swapped_faces",
        options=export_options,
        export_dir=export_dir,
    )
    return output_path, provider


def export_id_encoder(
    provider_name: str = DEFAULT_ID_ENCODER_PROVIDER_NAME,
    batch_size: int = 1,
    nhwc: bool = True,
    device: str = "cuda",
    output_dir: str = "onnx_export",
    file_prefix: str = "IDEncoder",
    export_options: OnnxExportOptions | None = None,
    export_dir: str | os.PathLike | None = None,
) -> Path:
    """导出接收 ArcFace 112 对齐图像的编码器；FFHQ 图像须先使用训练端映射。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    target_device = torch.device(device)
    provider = IDEncoderProvider[provider_name]
    encoder = IDEncoder(provider)

    if nhwc:
        wrapped = IDEncoderNHWCAdapter(encoder).to(target_device).eval()
        x = torch.randn((batch_size, 112, 112, 3), device=target_device, dtype=torch.float32)
    else:
        wrapped = encoder.to(target_device).eval()
        x = torch.randn((batch_size, 3, 112, 112), device=target_device, dtype=torch.float32)

    return export_to_onnx(
        wrapped,
        (x,),
        output_dir=output_dir,
        file_prefix=f"{file_prefix}-id-encoder-{provider_name}",
        metadata=ONNX_CONTRACT | {"swapface.kind": "id_encoder", "swapface.face_alignment": "arcface112", "swapface.provider": provider.name, "swapface.layout": "NHWC" if nhwc else "NCHW"},
        input_names=["faces"],
        output_name="identity",
        options=export_options,
        export_dir=export_dir,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
ID_ENCODER_PROVIDER_CHOICES = [p.name for p in IDEncoderProvider]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 SwapFace / IDEncoder 模型导出为 ONNX 格式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── 公共参数工厂 ──────────────────────────────────────────────────────────
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--batch-size", type=int, default=1, metavar="N", help="导出时的 batch size")
        p.add_argument("--nchw", action="store_true", help="使用 NCHW 布局（默认 NHWC）")
        p.add_argument("--device", default="cuda", help="推理设备，如 cuda / cuda:1 / cpu")
        p.add_argument("--output-dir", default="onnx_export", help="ONNX 文件输出目录")
        p.add_argument("--file-prefix", default="swapface", help="ONNX 文件名")
        p.add_argument("--optimize", action="store_true", help="启用 torch.onnx.export 图优化；SwapFace 默认关闭")
        p.add_argument("--opset-version", type=int, default=None, help="显式指定 ONNX opset；默认由 PyTorch exporter 选择")

    # ── swapface 子命令 ───────────────────────────────────────────────────────
    p_fs = sub.add_parser("swapface", help="导出 Generator（换脸模型）")
    p_fs.add_argument("--checkpoint", required=True, metavar="PATH", help="当前 train.py 保存的检查点路径（.pth）")
    add_common(p_fs)

    # ── id-encoder 子命令 ──────────────────────────────────────────────────────
    p_id = sub.add_parser("id-encoder", help="导出 IDEncoder（人脸识别编码器）")
    p_id.add_argument(
        "--provider",
        default=DEFAULT_ID_ENCODER_PROVIDER_NAME,
        choices=ID_ENCODER_PROVIDER_CHOICES,
        help="IDEncoder 权重/骨干网络选择；输入必须为 ArcFace 112 对齐的 RGB 图像，值域 [-1,1]",
    )
    add_common(p_id)

    # ── all 子命令（两者一起导出）────────────────────────────────────────────
    p_all = sub.add_parser("all", help="同时导出 SwapFace 与 IDEncoder")
    p_all.add_argument("--checkpoint", required=True, metavar="PATH", help="Generator 检查点；自动导出其配套身份编码器")
    add_common(p_all)

    return parser


def main() -> None:
    _configure_export_runtime()
    parser = build_parser()
    args = parser.parse_args()

    nhwc = not args.nchw
    export_options = OnnxExportOptions(optimize=args.optimize, opset_version=args.opset_version)
    kwargs = {
        "batch_size": args.batch_size,
        "nhwc": nhwc,
        "device": args.device,
        "output_dir": args.output_dir,
        "file_prefix": args.file_prefix,
        "export_options": export_options,
    }

    if args.command in ("swapface", "all"):
        output_path, provider = export_swapface(checkpoint_path=args.checkpoint, **kwargs)

    if args.command == "id-encoder":
        export_id_encoder(provider_name=args.provider, **kwargs)
    elif args.command == "all":
        export_id_encoder(provider_name=provider.name, export_dir=output_path.parent, **kwargs)


if __name__ == "__main__":
    main()
