"""L2CS-Net Gaze360 的冻结 PyTorch 推理封装。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from torchvision.models.resnet import Bottleneck

from .. import ImageInputRange

L2CS_REPOSITORY_ID = "tianfxc/l2cs"
L2CS_REVISION = "424a499580d1217b57f8f1f3c2a5582c30cd6610"
L2CS_WEIGHT_FILENAME = "L2CSNet_gaze360.pkl"


class L2CSNet(nn.Module):
    """使用官方 ``L2CSNet_gaze360.pkl`` 权重的 ResNet-50 gaze estimator。

    模型输入采用官方训练/测试路径的 448×448 RGB + ImageNet normalization。
    权重中的 ``fc_yaw_gaze`` / ``fc_pitch_gaze`` 名称按原 checkpoint 保留；forward
    的返回顺序也与上游训练/测试实际使用保持一致。
    """

    INPUT_SIZE = 448
    NUM_BINS = 90
    BIN_WIDTH_DEGREES = 4.0
    ANGLE_OFFSET_DEGREES = -180.0

    def __init__(self, input_range: ImageInputRange = ImageInputRange.MINUS_ONE_TO_ONE) -> None:
        super().__init__()
        if not isinstance(input_range, ImageInputRange):
            raise TypeError(f"input_range 必须为 ImageInputRange，实际为 {type(input_range).__name__}")

        self.input_range = input_range
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feature_dim = 512 * Bottleneck.expansion
        self.fc_yaw_gaze = nn.Linear(feature_dim, self.NUM_BINS)
        self.fc_pitch_gaze = nn.Linear(feature_dim, self.NUM_BINS)
        # 官方 checkpoint 中保留的历史层；推理不使用，但需要它才能 strict load。
        self.fc_finetune = nn.Linear(feature_dim + 3, 3)

        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        checkpoint_path = hf_hub_download(repo_id=L2CS_REPOSITORY_ID, filename=L2CS_WEIGHT_FILENAME, revision=L2CS_REVISION)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            raise TypeError(f"L2CS checkpoint 必须为 state_dict，实际为 {type(state_dict).__name__}")
        self.load_state_dict(state_dict, strict=True)
        self.eval().requires_grad_(False)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample: nn.Module | None = None
        out_channels = planes * Bottleneck.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers: list[nn.Module] = [Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = out_channels
        layers.extend(Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _prepare_input(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(f"images 必须为 NCHW，实际 shape={tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"images 必须为 RGB 三通道，实际 C={images.shape[1]}")

        x = images.float()
        match self.input_range:
            case ImageInputRange.ZERO_TO_255:
                x = x.div(255.0)
            case ImageInputRange.ZERO_TO_ONE:
                pass
            case ImageInputRange.MINUS_ONE_TO_ONE:
                x = x.add(1.0).mul(0.5)

        if x.shape[-2:] != (self.INPUT_SIZE, self.INPUT_SIZE):
            x = F.interpolate(x, size=(self.INPUT_SIZE, self.INPUT_SIZE), mode="bilinear", align_corners=False, antialias=True)
        return (x - self.get_buffer("image_mean")) / self.get_buffer("image_std")

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """返回上游训练/测试语义顺序的两个 90-bin gaze logits。"""
        x = self._prepare_input(images)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)

        # 上游 train.py/test.py 将第一个 head 当作第 0 个 gaze angle、第二个 head 当作第 1 个。
        # checkpoint 的历史属性名与其调用变量名不一致，因此这里不按属性名交换顺序。
        angle0_logits = self.fc_yaw_gaze(x)
        angle1_logits = self.fc_pitch_gaze(x)
        return angle0_logits, angle1_logits
