import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import onnxruntime as ort
import torch
from torch import Tensor, nn

from misc.models.id_encoder import IDEncoder, IDEncoderProvider
from models.networks import Generator

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
random.seed(42)
torch.manual_seed(42)
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False


def print_mapping(title: str, mapping: dict, indent: int = 2) -> None:
    print(f"{title}:")
    for key, value in mapping.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_mapping("", value, indent + 2)
        else:
            print(" " * indent + f"{key:25}: {value}")


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers
# ──────────────────────────────────────────────────────────────────────────────
class FaceSwapNHWCAdapter(nn.Module):
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
):

    model = model.eval()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    with torch.inference_mode():
        exported_program = torch.export.export(model, example_inputs, strict=True)
        onnx_program = torch.onnx.export(
            model=exported_program,
            # opset_version=21,
            dynamo=True,
            optimize=True,
            verify=True,
            report=True,
            external_data=False,
            artifacts_dir=output_dir_path / "onnx_artifacts",
            verbose=True,
            profile=True,
        )

    file_prefix = "" if file_prefix is None else file_prefix
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir_path / f"{file_prefix}-{timestamp}.onnx"

    onnx_program.save(output_path)
    print(f"模型成功导出到: {output_path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print(f"ONNX Runtime: {ort.__version__}")
    print(f"ORT providers: {providers}")

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0
    ort.InferenceSession(output_path, providers=providers, sess_options=sess_options)


# ──────────────────────────────────────────────────────────────────────────────
# Export functions
# ──────────────────────────────────────────────────────────────────────────────
def export_face_swap(
    checkpoint_path: str | None = None,
    batch_size: int = 1,
    nhwc: bool = True,
    device: str = "cuda",
    output_dir: str = "onnx_export",
    file_prefix: str = "faceswap",
) -> None:
    target_device = torch.device(device)

    if checkpoint_path is not None:
        print(f"Loading ckpt from {checkpoint_path}")
        checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
        training_iteration = checkpoint["iter"]
        print(f"ckpt info:\n  {'iter':25}: {training_iteration}")
        print_mapping("net_g", checkpoint["net_g"]["network_cfg"])
        print_mapping("net_d", checkpoint["net_d"]["network_cfg"])
        model = Generator(**checkpoint["net_g"]["network_cfg"])
        model.load_state_dict(checkpoint["net_g"]["state_dict"])
    else:
        training_iteration = 0
        model = Generator()

    network_config = model.network_cfg
    image_resolution, image_channels, identity_dim = network_config["img_resolution"], network_config["img_channels"], network_config["id_dim"]

    if nhwc:
        wrapped = FaceSwapNHWCAdapter(model).to(target_device)
        x = torch.randn((batch_size, image_resolution, image_resolution, image_channels), device=target_device, dtype=torch.float32)
    else:
        wrapped = model.to(target_device)
        x = torch.randn((batch_size, image_channels, image_resolution, image_resolution), device=target_device, dtype=torch.float32)

    identity_embedding = torch.randn((batch_size, identity_dim), device=target_device, dtype=torch.float32)
    export_to_onnx(wrapped, (x, identity_embedding), output_dir=output_dir, file_prefix=f"{file_prefix}-{training_iteration}")


def export_id_encoder(
    provider_name: str = "MS1MV3_ARCFACE_R50_FP16",
    batch_size: int = 1,
    nhwc: bool = True,
    device: str = "cuda",
    output_dir: str = "onnx_export",
    file_prefix: str = "IDEncoder",
) -> None:
    target_device = torch.device(device)
    provider = IDEncoderProvider[provider_name]
    encoder = IDEncoder(provider)

    if nhwc:
        wrapped = IDEncoderNHWCAdapter(encoder).to(target_device).eval()
        x = torch.randn((batch_size, 112, 112, 3), device=target_device, dtype=torch.float32)
    else:
        wrapped = encoder.to(target_device).eval()
        x = torch.randn((batch_size, 3, 112, 112), device=target_device, dtype=torch.float32)

    export_to_onnx(wrapped, (x,), output_dir=output_dir, file_prefix=f"{file_prefix}-{provider_name}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
ID_ENCODER_PROVIDER_CHOICES = [p.name for p in IDEncoderProvider]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 FaceSwap / IDEncoder 模型导出为 ONNX 格式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── 公共参数工厂 ──────────────────────────────────────────────────────────
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--batch-size", type=int, default=1, metavar="N", help="导出时的 batch size")
        p.add_argument("--nchw", action="store_true", help="使用 NCHW 布局（默认 NHWC）")
        p.add_argument("--device", default="cuda", help="推理设备，如 cuda / cuda:1 / cpu")
        p.add_argument("--output-dir", default="onnx_export", help="ONNX 文件输出目录")
        p.add_argument("--file-prefix", default="faceswap", help="ONNX 文件名")

    # ── faceswap 子命令 ───────────────────────────────────────────────────────
    p_fs = sub.add_parser("faceswap", help="导出 Generator（换脸模型）")
    p_fs.add_argument("--checkpoint", default=None, metavar="PATH", help="检查点路径（.pth）；不传则使用默认权重")
    add_common(p_fs)

    # ── id-encoder 子命令 ──────────────────────────────────────────────────────
    p_id = sub.add_parser("id-encoder", help="导出 IDEncoder（人脸识别编码器）")
    p_id.add_argument(
        "--provider",
        default="BLENDFACE",
        choices=ID_ENCODER_PROVIDER_CHOICES,
        help="IDEncoder 权重/骨干网络选择",
    )
    add_common(p_id)

    # ── all 子命令（两者一起导出）────────────────────────────────────────────
    p_all = sub.add_parser("all", help="同时导出 FaceSwap 与 IDEncoder")
    p_all.add_argument("--checkpoint", default=None, metavar="PATH", help="Generator 检查点路径")
    p_all.add_argument(
        "--provider",
        default="BLENDFACE",
        choices=ID_ENCODER_PROVIDER_CHOICES,
        help="IDEncoder provider",
    )
    add_common(p_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    nhwc = not args.nchw
    kwargs = {
        "batch_size": args.batch_size,
        "nhwc": nhwc,
        "device": args.device,
        "output_dir": args.output_dir,
        "file_prefix": args.file_prefix,
    }

    if args.command in ("faceswap", "all"):
        export_face_swap(checkpoint_path=args.checkpoint, **kwargs)

    if args.command in ("id-encoder", "all"):
        export_id_encoder(provider_name=args.provider, **kwargs)


if __name__ == "__main__":
    main()
