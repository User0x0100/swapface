from typing import Literal

import torch
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from torch.nn import functional as F

from ...models import MODEL_REPOSITORY_ID


class FaceMasker(nn.Module):
    def __init__(
        self,
        input_range: Literal["minus_one_to_one", "zero_to_one"] = "minus_one_to_one",
        mode: Literal["parsing", "occlusion"] = "parsing",
    ) -> None:
        """
        Args:
            input_range: 输入张量值域。
            mode: 使用 face parsing 或 occlusion 模型生成二值 mask。
        """
        super().__init__()
        if input_range not in {"minus_one_to_one", "zero_to_one"}:
            raise ValueError(f"unsupported input_range: {input_range}")
        if mode not in {"parsing", "occlusion"}:
            raise ValueError(f"unsupported mode: {mode}")

        self.input_range = input_range
        self.mode = mode

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "face_part_indices",
            torch.tensor([1, 2, 3, 4, 5, 6, 10, 11, 12, 13], dtype=torch.long),
            persistent=False,
        )

        if self.mode == "occlusion":
            # https://github.com/face3d0725/FaceExtraction
            from segmentation_models_pytorch import Unet

            self.model = Unet(
                encoder_name="resnet18",
                encoder_weights=None,
                decoder_attention_type=None,
                classes=1,
                activation=None,
            )

            checkpoint_path = hf_hub_download(
                repo_id=MODEL_REPOSITORY_ID, filename="epoch_16_best.ckpt"
            )
            state_dict: dict[str, Tensor] = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            state_dict = {
                key.removeprefix("module."): value for key, value in state_dict.items()
            }
            self.model.load_state_dict(state_dict)

        else:
            # https://github.com/yakhyo/face-parsing
            from .bisenet import BiSeNet

            self.model = BiSeNet(n_classes=19)
            weight_path = hf_hub_download(
                repo_id=MODEL_REPOSITORY_ID, filename="79999_iter.pth"
            )
            self.model.load_state_dict(
                torch.load(
                    weight_path, map_location=torch.device("cpu"), weights_only=True
                )
            )

    def forward(self, images: Tensor) -> Tensor:

        height, width = images.shape[-2:]

        if self.mode == "occlusion":
            images = F.interpolate(images, [224, 224], mode="bicubic")
        else:
            images = F.interpolate(images, [512, 512], mode="bicubic")

        if self.input_range == "minus_one_to_one":
            images = images.add(1.0).mul(0.5)

        mean = self.get_buffer("mean")
        std = self.get_buffer("std")
        images = images.sub(mean).div(std)

        mask = self.model(images)
        mask: Tensor = F.interpolate(mask, (height, width), mode="bicubic")

        if self.mode == "occlusion":
            mask = (mask > 0).float()
        else:
            parsing = mask.argmax(dim=1)
            mask = (
                torch.isin(parsing, self.get_buffer("face_part_indices"))
                .float()
                .unsqueeze(1)
            )

        return mask


if __name__ == "__main__":
    import random
    import time

    from torchvision import utils
    from torchvision.io import read_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    masker = FaceMasker(input_range="minus_one_to_one", mode="occlusion")
    masker.to(device)

    images = [
        read_image(
            f"/opt/share/deepfake/dataset_1/ffhq_1024/{random.randint(0, 69999):05d}.png"
        )
        for _ in range(batch_size)
    ]

    # 0~1
    # images = [img.float().div(255.0) for img in images]

    # -1~1
    images = [img.float() / 127.5 - 1.0 for img in images]
    images = torch.stack(images, dim=0).to(device=device)

    for i in range(10):
        mask = masker(images)

    with torch.inference_mode():
        # 性能测试
        print("执行性能测试...")
        times = []
        for i in range(100):
            start_time = time.time()
            mask = masker(images)
            times.append((time.time() - start_time) * 1000)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    fps = 1000 / avg_time

    print(f"平均处理时间: {avg_time:.2f} ms")
    print(f"最快处理时间: {min_time:.2f} ms")
    print(f"处理帧率: {fps:.1f} FPS")

    print("保存对比图像...")

    # mask = mask.repeat(1, 3, 1, 1)
    mask = (mask.repeat(1, 3, 1, 1) - 0.5) / 0.5

    grid = torch.cat((mask, images), dim=0)
    utils.save_image(
        grid, "mask.png", nrow=batch_size, normalize=True, value_range=(-1, 1)
    )
