import io
from pathlib import Path
from datetime import datetime
from typing import Any
import torch
from torch import nn, Tensor
import onnx
from onnx.checker import check_model
from .networks import Generator


class NHWCWrapper(nn.Module):
    def __init__(self, model: Generator) -> None:
        super().__init__()
        self.model = model

    def forward(self, nhwc: Tensor, id_feat: Tensor) -> Tensor:
        nchw = nhwc.permute(0, 3, 1, 2)
        x = self.model(nchw, id_feat)
        x = x.permute(0, 2, 3, 1)

        return x


def exp(ckpt: str | None = None, batch_size=1, opset_version: int | None = None, nhwc: bool = True, export_folder: str = "onnx_export", device: str = "cuda"):

    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    device = torch.device(device)
    dtype = torch.float32

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

    # def stat(t: torch.Tensor):
    #     return f"shape={tuple(t.shape)}, mean={t.mean().item():.4f}, std={t.std().item():.4f}"

    # for name, module in model.named_modules():
    #     if module.__class__.__name__ == "AdaIN":
    #         print(f"\n[AdaIN] {name}")
    #         print("  gamma.weight:", stat(module.fc_gamma.weight))
    #         print("  gamma.bias  :", stat(module.fc_gamma.bias))
    #         print("  beta.weight :", stat(module.fc_beta.weight))
    #         print("  beta.bias   :", stat(module.fc_beta.bias))

    img_resolution = model.network_cfg["img_resolution"]
    img_channels = model.network_cfg["img_channels"]
    id_feat_dim = model.network_cfg["id_dim"]

    if nhwc:
        model = NHWCWrapper(model).to(device=device, dtype=dtype).eval()
        x = torch.randn((batch_size, img_resolution, img_resolution, img_channels), device=device, dtype=dtype)
    else:
        model = model.to(device=device).eval()
        x = torch.randn((batch_size, img_channels, img_resolution, img_resolution), device=device, dtype=dtype)

    id_feat = torch.randn((batch_size, id_feat_dim), device=device, dtype=dtype)

    f = io.BytesIO()
    torch.onnx.export(
        model,
        (x, id_feat),
        f,
        opset_version=opset_version,
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
    fp = fp / f"model_{time_str}.onnx"
    onnx.save(onnx_model, fp)
    print(f"模型成功导出到: {fp}")


if __name__ == "__main__":
    exp("train_log/256_BLENDFACE_ADAIN_WFM_Same0.0/ckpt/1490169.pth")
