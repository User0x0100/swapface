import torch
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

from ...models import MODEL_REPOSITORY_ID
from .arch import CodeFormer


def load_codeformer(device: torch.device | str = "cuda") -> CodeFormer:
    net = CodeFormer()
    weight_path = hf_hub_download(repo_id=MODEL_REPOSITORY_ID, filename="codeformer.pth")
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)["params_ema"]
    net.load_state_dict(state_dict)
    return net.to(device).eval().requires_grad_(False)


class _CodeFormerONNX(nn.Module):
    def __init__(self, net: CodeFormer, nhwc: bool):
        super().__init__()
        self.net = net
        self.nhwc = nhwc

    def forward(self, image: Tensor, w: Tensor) -> Tensor:
        if self.nhwc:
            image = image.permute(0, 3, 1, 2)
        output = self.net(image, w)
        if self.nhwc:
            output = output.permute(0, 2, 3, 1)
        return output


def export_codeformer_onnx(
    output_path: str,
    *,
    nhwc: bool = False,
    device: torch.device | str = "cuda",
) -> None:
    """Export CodeFormer to ONNX with NCHW or NHWC image I/O.

    Image I/O stays RGB float32 in approximately [-1, 1]. The exported model
    uses a fixed batch size of 1 and 512x512 resolution. Fidelity ``w`` remains
    a runtime float32 tensor input shaped [1].
    """
    net = load_codeformer(device=device)
    model = _CodeFormerONNX(net, nhwc).eval()

    image_shape = (1, 512, 512, 3) if nhwc else (1, 3, 512, 512)
    image = torch.zeros(image_shape, device=device, dtype=torch.float32)
    w = torch.tensor([0.5], device=device, dtype=torch.float32)

    torch.onnx.export(
        model,
        (image, w),
        output_path,
        input_names=["input", "w"],
        output_names=["output"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )


if __name__ == "__main__":
    from pathlib import Path

    import cv2

    from misc.utils import ImageDirectory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = load_codeformer(device=device)

    sample_dir = ImageDirectory("codeformer_sample")
    save_dir = Path("codeformer_result")

    save_dir.mkdir(exist_ok=True, parents=True)

    sample_nb = 5

    for idx in range(sample_nb):
        image, fn = sample_dir.sample_tensor(return_stem=True, device=device)  # image: 0.0 ~ 255.0 RGB CHW

        image.div_(127.5).sub_(1.0).unsqueeze_(0)  # -1.0 ~ 1.0 1CHW RGB

        image_after: Tensor = net(image, torch.tensor([0.5], device=device, dtype=image.dtype))

        image_after = image_after.add_(1.0).mul_(127.5).clamp_(0.0, 255.0).squeeze_(0)[[2, 1, 0], :, :].permute(1, 2, 0)  # 0.0 ~ 255.0 HWC

        image_after_cpu = image_after.to(device="cpu", dtype=torch.uint8).numpy()

        cv2.imwrite(
            save_dir / f"{idx}.png",
            image_after_cpu,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
