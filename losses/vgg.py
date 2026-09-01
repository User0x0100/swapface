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
    """返回当前特征提取器支持的 torchvision VGG 型号。"""
    return tuple(VGG_LAYER_NAMES)


def get_vgg_layer_names(vgg_type: str) -> tuple[str, ...]:
    """返回指定 VGG 型号的可选特征层名称，顺序与网络执行顺序一致。

    异常:
        ValueError: ``vgg_type`` 不受支持。"""
    try:
        return tuple(VGG_LAYER_NAMES[vgg_type])
    except KeyError as exc:
        supported = ", ".join(VGG_LAYER_NAMES)
        raise ValueError(f"Unsupported VGG type: {vgg_type!r}. Supported types: {supported}") from exc


class VGGFeatureExtractor(nn.Module):
    """从冻结 torchvision VGG 中提取指定中间层特征。

    只保留到最深请求层为止的特征子网络；所有 ReLU 均关闭原地模式，防止后续层修改
    已经捕获的 ReLU 前卷积特征。"""

    def __init__(self, layer_names: list[str], vgg_type: str):
        """初始化 VGG 特征提取器。

        参数:
            layer_names: 要返回的层名列表。输出始终按 VGG 执行顺序排列。
            vgg_type: torchvision VGG 型号。

        异常:
            ValueError: 层列表为空、VGG 型号无效或层名不存在。
            TypeError: torchvision 模型未提供 ``nn.Sequential`` features。"""
        super().__init__()

        if not layer_names:
            raise ValueError("layer_names must not be empty")

        try:
            available_layer_names = VGG_LAYER_NAMES[vgg_type]
        except KeyError as exc:
            supported = ", ".join(VGG_LAYER_NAMES)
            raise ValueError(f"Unsupported VGG type: {vgg_type!r}. Supported types: {supported}") from exc

        unknown_layers = [name for name in layer_names if name not in available_layer_names]
        if unknown_layers:
            raise ValueError(f"Unknown {vgg_type} layer names: {unknown_layers}")

        create_vgg_fn = getattr(vgg, vgg_type)
        weights = getattr(vgg, f"{vgg_type.upper()}_Weights").DEFAULT
        vgg_features = create_vgg_fn(weights=weights).features
        if not isinstance(vgg_features, nn.Sequential):
            raise TypeError(f"Expected VGG features to be nn.Sequential, got {type(vgg_features).__name__}")

        vgg_features.eval().requires_grad_(False)
        for layer in vgg_features:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

        self.selected_layers = tuple(sorted(available_layer_names.index(name) for name in layer_names))
        self._selected_layer_set = frozenset(self.selected_layers)

        # Slice is exclusive: include the last requested layer, but do not execute one extra layer.
        stop_idx = self.selected_layers[-1] + 1
        self.features = vgg_features[:stop_idx]

    def forward(self, x: Tensor) -> list[Tensor]:
        """顺序执行 VGG 特征层并返回选定中间特征。

        参数:
            x: 已按调用方要求预处理的 NCHW RGB 张量。

        返回:
            按 VGG 执行顺序排列的特征张量列表。"""

        features: list[Tensor] = []

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self._selected_layer_set:
                features.append(x)

        return features
