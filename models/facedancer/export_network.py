import os
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

import torch
from torch import nn, Tensor
from .networks import Generator
from misc.models.idencoder import IDEncoder, PROVIDER
import onnxruntime as ort

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
random.seed(42)
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True


def print_dict(title: str, d: dict, indent: int = 2):
    print(f"{title}:")
    for k, v in d.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_dict("", v, indent + 2)
        else:
            print(" " * indent + f"{k:25}: {v}")


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers
# ──────────────────────────────────────────────────────────────────────────────
class FaceSwapNHWCWrapper(nn.Module):
    def __init__(self, model: Generator) -> None:
        super().__init__()
        self.model = model

    def forward(self, nhwc: Tensor, id_feat: Tensor) -> Tensor:
        nhwc = torch.clamp(nhwc, -1.0, 1.0)
        nchw = nhwc.permute(0, 3, 1, 2)
        x = self.model(nchw, id_feat)
        return x.permute(0, 2, 3, 1)


class IDEncoderNHWCWrapper(nn.Module):
    def __init__(self, model: IDEncoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, nhwc: Tensor) -> Tensor:
        nhwc = torch.clamp(nhwc, -1.0, 1.0)
        return self.model(nhwc.permute(0, 3, 1, 2))


# ──────────────────────────────────────────────────────────────────────────────
# Core export helper
# ──────────────────────────────────────────────────────────────────────────────
def torch2onnx(
    model: nn.Module,
    args: tuple[Any, ...],
    output_dir: str | os.PathLike = "onnx_export",
    file_prefix: str | None = None,
):

    model = model.eval()
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with torch.inference_mode():
        exported_program = torch.export.export(model, args, strict=False)
        onnx_program = torch.onnx.export(
            model=exported_program,
            opset_version=21,
            dynamo=True,
            optimize=False,
            verify=True,
            report=True,
            external_data=False,
            artifacts_dir=out_dir / "onnx_artifacts",
            verbose=True,
            profile=True,
        )

    file_prefix = "" if file_prefix is None else file_prefix
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{file_prefix}-{time_str}.onnx"

    onnx_program.save(out_path)
    print(f"模型成功导出到: {out_path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print(f"ONNX Runtime: {ort.__version__}")
    print(f"ORT providers: {providers}")

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0
    ort.InferenceSession(out_path, providers=providers, sess_options=sess_options)


# ──────────────────────────────────────────────────────────────────────────────
# Export functions
# ──────────────────────────────────────────────────────────────────────────────
def export_FaceSwap(
    ckpt: str | None = None,
    batch_size: int = 1,
    nhwc: bool = True,
    device: str = "cuda",
    output_dir: str = "onnx_export",
    file_prefix: str = "faceswap",
) -> None:
    dev = torch.device(device)

    if ckpt is not None:
        print(f"Loading ckpt from {ckpt}")
        state: dict[str, Any] = torch.load(ckpt, map_location=dev, weights_only=False)
        iter = state["iter"]
        print(f"ckpt info:\n  {'iter':25}: {iter}")
        print_dict("net_g", state["net_g"]["network_cfg"])
        print_dict("net_d", state["net_d"]["network_cfg"])
        model = Generator(**state["net_g"]["network_cfg"])
        model.load_state_dict(state["net_g"]["state_dict"])
    else:
        iter = 0
        model = Generator()

    cfg = model.network_cfg
    res, ch, id_dim = cfg["img_resolution"], cfg["img_channels"], cfg["id_dim"]

    if nhwc:
        wrapped = FaceSwapNHWCWrapper(model).to(dev)
        x = torch.randn((batch_size, res, res, ch), device=dev, dtype=torch.float32)
    else:
        wrapped = model.to(dev)
        x = torch.randn((batch_size, ch, res, res), device=dev, dtype=torch.float32)

    id_feat = torch.randn((batch_size, id_dim), device=dev, dtype=torch.float32)
    torch2onnx(wrapped, (x, id_feat), output_dir=output_dir, file_prefix=f"{file_prefix}-{iter}")


def export_IDEncoder(
    provider_name: str = "MS1MV3_ARCFACE_R50_FP16",
    batch_size: int = 1,
    nhwc: bool = True,
    device: str = "cuda",
    output_dir: str = "onnx_export",
    file_prefix: str = "IDEncoder",
) -> None:
    _ = file_prefix
    dev = torch.device(device)
    provider = PROVIDER[provider_name]
    encoder = IDEncoder(provider)

    if nhwc:
        wrapped = IDEncoderNHWCWrapper(encoder).to(dev).eval()
        x = torch.randn((batch_size, 112, 112, 3), device=dev, dtype=torch.float32)
    else:
        wrapped = encoder.to(dev).eval()
        x = torch.randn((batch_size, 3, 112, 112), device=dev, dtype=torch.float32)

    torch2onnx(wrapped, (x,), output_dir=output_dir, file_prefix=f"IDEncoder-{provider_name}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
PROVIDER_CHOICES = [p.name for p in PROVIDER]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 FaceSwap / IDEncoder 模型导出为 ONNX 格式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── 公共参数工厂 ──────────────────────────────────────────────────────────
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--batch-size", type=int, default=1, metavar="N", help="导出时的 batch size")
        p.add_argument("--nchw", action="store_true", help="使用 NCHW 布局（默认 NHWC）")
        p.add_argument("--device", default="cuda", help="推理设备，如 cuda / cuda:1 / cpu")
        p.add_argument("--output-dir", default="onnx_export", help="ONNX 文件输出目录")
        p.add_argument("--file-prefix", default="faceswap", help="ONNX 文件名")

    # ── faceswap 子命令 ───────────────────────────────────────────────────────
    p_fs = sub.add_parser("faceswap", help="导出 Generator（换脸模型）")
    p_fs.add_argument("--ckpt", default=None, metavar="PATH", help="检查点路径（.pth）；不传则使用默认权重")
    add_common(p_fs)

    # ── idencoder 子命令 ──────────────────────────────────────────────────────
    p_id = sub.add_parser("idencoder", help="导出 IDEncoder（人脸识别编码器）")
    p_id.add_argument(
        "--provider",
        default="BLENDFACE",
        choices=PROVIDER_CHOICES,
        help="IDEncoder 权重/骨干网络选择",
    )
    add_common(p_id)

    # ── all 子命令（两者一起导出）────────────────────────────────────────────
    p_all = sub.add_parser("all", help="同时导出 FaceSwap 与 IDEncoder")
    p_all.add_argument("--ckpt", default=None, metavar="PATH", help="Generator 检查点路径")
    p_all.add_argument(
        "--provider",
        default="BLENDFACE",
        choices=PROVIDER_CHOICES,
        help="IDEncoder provider",
    )
    add_common(p_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    nhwc = not args.nchw
    kwargs = dict(
        batch_size=args.batch_size,
        nhwc=nhwc,
        device=args.device,
        output_dir=args.output_dir,
        file_prefix=args.file_prefix,
    )

    if args.cmd in ("faceswap", "all"):
        export_FaceSwap(ckpt=args.ckpt, **kwargs)

    if args.cmd in ("idencoder", "all"):
        export_IDEncoder(provider_name=args.provider, **kwargs)


if __name__ == "__main__":
    main()
