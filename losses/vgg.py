from torch import Tensor, nn
from torchvision.models import vgg

VGG_LAYER_NAMES = {
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


def get_supported_vgg_types() -> tuple[str, ...]:
    return tuple(VGG_LAYER_NAMES)


def get_vgg_layer_names(vgg_type: str) -> tuple[str, ...]:
    try:
        return tuple(VGG_LAYER_NAMES[vgg_type])
    except KeyError as exc:
        supported = ", ".join(VGG_LAYER_NAMES)
        raise ValueError(
            f"Unsupported VGG type: {vgg_type!r}. Supported types: {supported}"
        ) from exc


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_names: list[str], vgg_type: str):
        super().__init__()

        if not layer_names:
            raise ValueError("layer_names must not be empty")

        try:
            available_layer_names = VGG_LAYER_NAMES[vgg_type]
        except KeyError as exc:
            supported = ", ".join(VGG_LAYER_NAMES)
            raise ValueError(
                f"Unsupported VGG type: {vgg_type!r}. Supported types: {supported}"
            ) from exc

        unknown_layers = [
            name for name in layer_names if name not in available_layer_names
        ]
        if unknown_layers:
            raise ValueError(f"Unknown {vgg_type} layer names: {unknown_layers}")

        create_vgg_fn = getattr(vgg, vgg_type)
        weights = getattr(vgg, f"{vgg_type.upper()}_Weights").DEFAULT
        vgg_features = create_vgg_fn(weights=weights).features
        if not isinstance(vgg_features, nn.Sequential):
            raise TypeError(
                f"Expected VGG features to be nn.Sequential, got {type(vgg_features).__name__}"
            )

        vgg_features.eval().requires_grad_(False)
        for layer in vgg_features:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

        self.selected_layers = tuple(
            sorted(available_layer_names.index(name) for name in layer_names)
        )
        self._selected_layer_set = frozenset(self.selected_layers)

        # Slice is exclusive: include the last requested layer, but do not execute one extra layer.
        stop_idx = self.selected_layers[-1] + 1
        self.features = vgg_features[:stop_idx]

    def forward(self, x: Tensor) -> list[Tensor]:

        features: list[Tensor] = []

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self._selected_layer_set:
                features.append(x)

        return features
