import io
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

import torch
from torch import nn, Tensor
import onnx
from onnx.checker import check_model
from .networks import Generator
from misc.models.idencoder import IDEncoder, PROVIDER


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
random.seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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
    model: torch.nn.Module,
    args: tuple[Any, ...],
    export_folder: str = "onnx_export",
    f_prefix: str | None = None,
    optimize: bool = False,
) -> Path:
    f = io.BytesIO()
    torch.onnx.export(
        model,
        args,
        f,
        export_params=True,
        dynamo=True,
        optimize=optimize,
        verify=True,
        external_data=False,
        keep_initializers_as_inputs=True,
        verbose=True,
        profile=True,
    )
    f.seek(0)

    onnx_model = onnx.load(f)
    check_model(onnx_model, full_check=True)

    out_dir = Path(export_folder)
    out_dir.mkdir(exist_ok=True, parents=True)

    prefix = "" if f_prefix is None else (f_prefix if f_prefix.endswith("_") else f_prefix + "_")
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{prefix}{time_str}.onnx"

    onnx.save(onnx_model, out_path)
    print(f"模型成功导出到: {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Export functions
# ──────────────────────────────────────────────────────────────────────────────
def export_FaceSwap(
    ckpt: str | None = None,
    batch_size: int = 1,
    nhwc: bool = True,
    device: str = "cuda",
    export_folder: str = "onnx_export",
    optimize: bool = False,
) -> None:
    dev = torch.device(device)

    if ckpt is not None:
        print(f"Loading ckpt from {ckpt}")
        state: dict[str, Any] = torch.load(ckpt, map_location=dev, weights_only=False)
        print(f"ckpt Info:\n  {'iter':25}: {state['iter']}")
        print("net_g:")
        for k, v in state["net_g"]["network_cfg"].items():
            print(f"  {k:25}: {v}")
        print("net_d:")
        for k, v in state["net_d"]["network_cfg"].items():
            print(f"  {k:25}: {v}")
        model = Generator(**state["net_g"]["network_cfg"])
        model.load_state_dict(state["net_g"]["state_dict"])
    else:
        model = Generator()

    cfg = model.network_cfg
    res, ch, id_dim = cfg["img_resolution"], cfg["img_channels"], cfg["id_dim"]

    if nhwc:
        wrapped = FaceSwapNHWCWrapper(model).to(dev).eval()
        x = torch.randn((batch_size, res, res, ch), device=dev, dtype=torch.float32)
    else:
        wrapped = model.to(dev).eval()
        x = torch.randn((batch_size, ch, res, res), device=dev, dtype=torch.float32)

    id_feat = torch.randn((batch_size, id_dim), device=dev, dtype=torch.float32)
    torch2onnx(wrapped, (x, id_feat), export_folder=export_folder, f_prefix="faceswap", optimize=optimize)


def export_IDEncoder(
    provider_name: str = "BLENDFACE",
    batch_size: int = 1,
    nhwc: bool = True,
    device: str = "cuda",
    export_folder: str = "onnx_export",
    optimize: bool = False,
) -> None:
    dev = torch.device(device)
    provider = PROVIDER[provider_name]
    encoder = IDEncoder(provider)

    if nhwc:
        wrapped = IDEncoderNHWCWrapper(encoder).to(dev).eval()
        x = torch.randn((batch_size, 112, 112, 3), device=dev, dtype=torch.float32)
    else:
        wrapped = encoder.to(dev).eval()
        x = torch.randn((batch_size, 3, 112, 112), device=dev, dtype=torch.float32)

    torch2onnx(wrapped, (x,), export_folder=export_folder, f_prefix="idencoder", optimize=optimize)


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
        p.add_argument("--optimize", action="store_true", help="启用 torch.onnx 导出时的 optimize 选项")

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
        export_folder=args.output_dir,
        optimize=args.optimize,
    )

    if args.cmd in ("faceswap", "all"):
        export_FaceSwap(ckpt=args.ckpt, **kwargs)

    if args.cmd in ("idencoder", "all"):
        export_IDEncoder(provider_name=args.provider, **kwargs)


if __name__ == "__main__":
    main()
