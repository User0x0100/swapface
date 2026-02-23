import torch
from torch import nn, Tensor
from torch.nn import functional as F

from huggingface_hub import hf_hub_download
from ...models import REPO_ID


class FaceParsing(nn.Module):
    def __init__(self, range_norm: bool = True, occ: bool = False) -> None:
        """
        Args:
            range_norm (bool): 如果输入值域为 [-1, 1] 则转换为 [0, 1].
                Default: True.
            occ (bool): 使用面部遮挡检测模型
        """
        super().__init__()

        self.range_norm = range_norm
        self.occ = occ

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("parts_idx", torch.tensor([1, 2, 3, 4, 5, 6, 10, 11, 12, 13], dtype=torch.float), persistent=False)

        if self.occ:
            # https://github.com/face3d0725/FaceExtraction
            from segmentation_models_pytorch import Unet

            self.net = Unet(encoder_name="resnet18", encoder_weights=None, decoder_attention_type=None, classes=1, activation=None)

            encoder_state_dict = hf_hub_download(repo_id=REPO_ID, filename="resnet18-5c106cde.pth")
            encoder_state_dict = torch.load(encoder_state_dict, map_location=torch.device("cpu"), weights_only=False)
            self.net.encoder.load_state_dict(encoder_state_dict)

            state_dict = hf_hub_download(repo_id=REPO_ID, filename="epoch_16_best.ckpt")
            state_dict: dict[str, Tensor] = torch.load(state_dict, map_location=torch.device("cpu"))
            state_dict = {cleaned: v for k, v in state_dict.items() if (cleaned := k.removeprefix("module."))}
            self.net.load_state_dict(state_dict)

        else:
            # https://github.com/yakhyo/face-parsing
            from .bisenet import BiSeNet

            self.net = BiSeNet(n_classes=19)
            model_path = hf_hub_download(repo_id=REPO_ID, filename="79999_iter.pth")
            self.net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))

    def forward(self, image: Tensor) -> Tensor:

        H, W = image.shape[-2:]

        if self.occ:
            image = F.interpolate(image, [224, 224], mode="bicubic")
        else:
            image = F.interpolate(image, [512, 512], mode="bicubic")

        image.sub_(self.mean).div_(self.std)

        if self.range_norm:
            image.add_(1.0).div_(2.0)

        mask = self.net(image)
        mask: Tensor = F.interpolate(mask, (H, W), mode="bicubic")

        if self.occ:
            mask = (mask > 0).float()
        else:
            parsing = mask.argmax(dim=1)
            mask = torch.isin(parsing, self.parts_idx).float().unsqueeze(1)

        return mask


if __name__ == "__main__":
    import random
    import time

    from torchvision import utils
    from torchvision.io import read_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    parser = FaceParsing(range_norm=True, occ=True)
    parser.to(device)

    images = [read_image(f"/opt/share/deepfake/dataset_1/ffhq_1024/{random.randint(0,69999):05d}.png") for _ in range(batch_size)]

    # 0~1
    # images = [img.float().div(255.0) for img in images]

    # -1~1
    images = [img.float() / 127.5 - 1.0 for img in images]
    images = torch.stack(images, dim=0).to(device=device)

    for i in range(10):
        mask = parser(images)

    with torch.inference_mode():
        # 性能测试
        print("执行性能测试...")
        times = []
        for i in range(100):
            start_time = time.time()
            mask = parser(images)
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
    utils.save_image(grid, "mask.png", nrow=batch_size, normalize=True, value_range=(-1, 1))
