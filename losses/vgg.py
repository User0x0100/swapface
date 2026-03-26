import torch
from torch import Tensor, nn
from torchvision.models import vgg

VGG_Layer_Name = {
    "vgg11": [
        "conv1_1",
        "relu1_1",
        "pool1",
        "conv2_1",
        "relu2_1",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "pool5",
    ],
    "vgg13": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "pool5",
    ],
    "vgg16": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "conv3_3",
        "relu3_3",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "conv4_3",
        "relu4_3",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "conv5_3",
        "relu5_3",
        "pool5",
    ],
    "vgg19": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "conv3_3",
        "relu3_3",
        "conv3_4",
        "relu3_4",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "conv4_3",
        "relu4_3",
        "conv4_4",
        "relu4_4",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "conv5_3",
        "relu5_3",
        "conv5_4",
        "relu5_4",
        "pool5",
    ],
}


def get_vgg_support_vgg_type() -> list[str]:
    return VGG_Layer_Name.keys()


def get_vgg_layer_name(vgg_type: str):
    return VGG_Layer_Name[vgg_type]


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_names: list[str], vgg_type: str):
        super().__init__()

        create_vgg_fn = getattr(vgg, vgg_type)
        weights = getattr(vgg, f"{vgg_type.upper()}_Weights").DEFAULT

        vgg_features: nn.Module = create_vgg_fn(weights=weights).features

        vgg_features.eval()
        vgg_features.requires_grad_(False)

        Layer_Name = VGG_Layer_Name[vgg_type]
        self.selected_layers = [Layer_Name.index(i) for i in layer_names]
        self.selected_layers.sort()

        max_idx = self.selected_layers[-1] + 1
        self.features = vgg_features[: max_idx + 1]

    def forward(self, x: Tensor) -> list[Tensor]:

        features: list[Tensor] = []

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.selected_layers:
                features.append(x)

        return features
