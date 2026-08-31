import torch
from huggingface_hub import hf_hub_download
from torch import Tensor

from ...models import MODEL_REPOSITORY_ID
from .arch import CodeFormer


def load_codeformer(device: torch.device | str = "cuda") -> CodeFormer:
    net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"])
    weight_path = hf_hub_download(repo_id=MODEL_REPOSITORY_ID, filename="codeformer.pth")
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)["params_ema"]
    net.load_state_dict(state_dict)
    return net.to(device).eval().requires_grad_(False)


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

    with torch.inference_mode():
        for idx in range(sample_nb):
            image, fn = sample_dir.sample_tensor(return_stem=True, device=device)  # image: 0.0 ~ 255.0 RGB CHW

            image.div_(127.5).sub_(1.0).unsqueeze_(0)  # -1.0 ~ 1.0 1CHW RGB

            image_after: Tensor = net(image, 0.5)

            image_after = image_after.add_(1.0).mul_(127.5).clamp_(0.0, 255.0).squeeze_(0)[[2, 1, 0], :, :].permute(1, 2, 0)  # 0.0 ~ 255.0 HWC

            image_after_cpu = image_after.to(device="cpu", dtype=torch.uint8).numpy()

            cv2.imwrite(save_dir / f"{idx}.png", image_after_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3])
