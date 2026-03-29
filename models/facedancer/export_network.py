import io
from pathlib import Path
from datetime import datetime
from typing import Any
import torch
from torch import nn, Tensor
import onnx
from onnx.checker import check_model
from .networks import Generator
from misc.models.idencoder import IDEncoder, PROVIDER


class FaceSwapNHWCWrapper(nn.Module):
    def __init__(self, model: Generator) -> None:
        super().__init__()
        self.model = model

    def forward(self, nhwc: Tensor, id_feat: Tensor) -> Tensor:
        nchw = nhwc.permute(0, 3, 1, 2)
        x = self.model(nchw, id_feat)
        x = x.permute(0, 2, 3, 1)

        return x


class IDEncoderNHWCWrapper(nn.Module):
    def __init__(self, model: IDEncoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, nhwc: Tensor) -> Tensor:
        nchw = nhwc.permute(0, 3, 1, 2)

        return self.model(nchw)


def torch2onnx(
    model: torch.nn.Module | torch.export.ExportedProgram | torch.jit.ScriptModule | torch.jit.ScriptFunction,
    args: tuple[Any, ...] = (),
    export_folder: str = "onnx_export",
    f_prefix: str | None = None,
):

    f = io.BytesIO()
    torch.onnx.export(
        model,
        args,
        f,
        export_params=True,
        dynamo=True,
        fallback=False,
        optimize=True,
        verify=True,
        external_data=False,
        keep_initializers_as_inputs=False,
        verbose=True,
        profile=True,
    )
    f.seek(0)

    onnx_model = onnx.load(f)
    check_model(onnx_model, full_check=True)

    fp = Path(export_folder)
    fp.mkdir(exist_ok=True, parents=True)

    f_prefix = "" if f_prefix is None else (f_prefix if f_prefix.endswith("_") else f_prefix + "_")
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    fp = fp / f"{f_prefix}{time_str}.onnx"
    onnx.save(onnx_model, fp)
    print(f"模型成功导出到: {fp}")


def export_FaceSwap(ckpt: str | None = None, batch_size=1, nhwc: bool = True, device: str = "cuda"):

    device = torch.device(device)

    if ckpt is not None:
        print(f"Loading ckpt from {ckpt}")
        ckpt: dict[str, Any] = torch.load(ckpt, map_location=device, weights_only=False)

        print(f"ckpt Info:\n  {'iter':25}: {ckpt['iter']}")
        print("net_g:")
        for k, v in ckpt["net_g"]["network_cfg"].items():
            print(f"  {k:25}: {v}")
        print("net_d:")
        for k, v in ckpt["net_d"]["network_cfg"].items():
            print(f"  {k:25}: {v}")
        model = Generator(**ckpt["net_g"]["network_cfg"])
        model.load_state_dict(ckpt["net_g"]["state_dict"])
    else:
        model = Generator()

    img_resolution = model.network_cfg["img_resolution"]
    img_channels = model.network_cfg["img_channels"]
    id_feat_dim = model.network_cfg["id_dim"]

    if nhwc:
        model = FaceSwapNHWCWrapper(model).to(device=device).eval()
        x = torch.randn((batch_size, img_resolution, img_resolution, img_channels), device=device, dtype=torch.float32)
    else:
        model = model.to(device=device).eval()
        x = torch.randn((batch_size, img_channels, img_resolution, img_resolution), device=device, dtype=torch.float32)

    id_feat = torch.randn((batch_size, id_feat_dim), device=device, dtype=torch.float32)

    torch2onnx(model, (x, id_feat), f_prefix="faceswap")


def export_IDEncoder(provider: PROVIDER = PROVIDER.BLENDFACE, batch_size=1, nhwc: bool = True, device: str = "cuda"):

    device = torch.device(device)
    ID_Encoder = IDEncoder(provider)

    if nhwc:
        model = IDEncoderNHWCWrapper(ID_Encoder).to(device=device).eval()
        x = torch.randn((batch_size, 112, 112, 3), device=device, dtype=torch.float32)
    else:
        model = ID_Encoder.to(device=device).eval()
        x = torch.randn((batch_size, 3, 112, 112), device=device, dtype=torch.float32)

    torch2onnx(model, (x,), f_prefix="idencoder")


if __name__ == "__main__":
    export_FaceSwap("train_log/256_BLENDFACE_AlphaDise_New_ID_5_Injection_2_Same0.2/ckpt/1943120.pth")
    # export_IDEncoder(PROVIDER.BLENDFACE)
