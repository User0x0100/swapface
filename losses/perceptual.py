from collections.abc import Mapping
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from misc.models import ImageInputRange

from .functional import (
    CriterionFn,
    Reduction,
    _reduce_loss,
    _weighted_feature_loss,
    charbonnier_loss,
)
from .vgg import VGGFeatureExtractor, get_vgg_layer_names


class WeightedFeatureMatchingLoss(nn.Module):
    """对判别器或其他网络的多层中间特征进行加权匹配。"""

    def __init__(
        self,
        layer_weights: Mapping[int, float],
        criterion: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: Reduction = "mean",
    ) -> None:
        """初始化多层特征匹配损失。

        参数:
            layer_weights: ``特征列表索引 -> 权重`` 映射。
            criterion: 单层差异函数，可选 L1、MSE、Charbonnier。
            reduction: ``none`` 返回逐样本损失；``mean``/``sum`` 返回标量。

        异常:
            ValueError: 未配置任何层或存在负索引。"""
        super().__init__()

        if not layer_weights:
            raise ValueError("layer_weights must not be empty")
        if any(index < 0 for index in layer_weights):
            raise ValueError("layer indices must be non-negative")

        self.criterion: CriterionFn = {
            "l1": F.l1_loss,
            "mse": F.mse_loss,
            "charbonnier": charbonnier_loss,
        }[criterion]
        self.reduction = reduction
        self.layer_weights = dict(layer_weights)

    def forward(self, predicted_features: list[Tensor], target_features: list[Tensor]) -> Tensor:
        """计算两组 特征列表 的加权匹配损失。

        参数:
            predicted_features: 预测特征列表。
            target_features: 参考特征列表，与预测列表长度一致。

        返回:
            所有配置层损失之和。

        异常:
            IndexError: 配置层索引超出任一特征列表范围。"""
        max_index = max(self.layer_weights)
        if max_index >= len(predicted_features) or max_index >= len(target_features):
            raise IndexError(f"feature layer index {max_index} is out of range: predicted={len(predicted_features)}, target={len(target_features)}")

        total: Tensor | None = None
        for index, weight in self.layer_weights.items():
            term = _weighted_feature_loss(
                self.criterion,
                predicted_features[index],
                target_features[index],
                weight,
                self.reduction,
            )
            total = term if total is None else total + term

        assert total is not None
        return total


class DINOv2PerceptualLoss(nn.Module):
    """基于冻结 DINOv2 Transformer 块特征的感知损失。

    ``layer_weights`` 的整数键表示 DINOv2 的**真实 Transformer 块索引**，不是返回列表
    中的相对索引。只请求配置的块，避免计算/返回无关中间层。"""

    def __init__(
        self,
        layer_weights: Mapping[int, float],
        criterion: Literal["l1", "mse", "charbonnier", "cosine"] = "cosine",
        reduction: Reduction = "mean",
        dino_type: str = "dinov2_vitb14_reg",
        use_input_norm: bool = True,
        input_range: ImageInputRange = ImageInputRange.MINUS_ONE_TO_ONE,
    ) -> None:
        """初始化 DINOv2 感知损失。

        参数:
            layer_weights: ``块索引 -> 权重`` 映射。
            criterion: L1、MSE、Charbonnier 或 ``cosine``（余弦距离）。
            reduction: 单层损失的聚合方式。
            dino_type: ``torch.hub`` 中的 DINOv2 模型入口名称。
            use_input_norm: 是否应用 ImageNet 均值/标准差。
            input_range: 输入 RGB 张量的值域。

        异常:
            ValueError: 未配置块、块索引为负或超出模型深度。
            TypeError: 加载的模型没有公开 ``blocks``。"""
        super().__init__()

        if not isinstance(input_range, ImageInputRange):
            raise TypeError(f"input_range 必须为 ImageInputRange，实际为 {type(input_range).__name__}")
        if not layer_weights:
            raise ValueError("layer_weights must not be empty")
        if any(index < 0 for index in layer_weights):
            raise ValueError("layer indices must be non-negative")

        self.criterion: CriterionFn = {
            "l1": F.l1_loss,
            "mse": F.mse_loss,
            "charbonnier": charbonnier_loss,
            "cosine": self._cosine_distance,
        }[criterion]
        self.reduction = reduction
        self.layer_weights = dict(layer_weights)

        self.dino = torch.hub.load("facebookresearch/dinov2", dino_type)
        self.dino.eval().requires_grad_(False)

        blocks = getattr(self.dino, "blocks", None)
        if blocks is None:
            raise TypeError(f"{dino_type} does not expose transformer blocks")
        block_count = len(blocks)
        invalid_indices = sorted(index for index in self.layer_weights if index >= block_count)
        if invalid_indices:
            raise ValueError(f"DINOv2 block indices out of range for {dino_type} with {block_count} blocks: {invalid_indices}")
        self.layer_indices = tuple(sorted(self.layer_weights))

        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )

        self.input_range = input_range

    @staticmethod
    def _cosine_distance(x: Tensor, y: Tensor, reduction: Reduction = "mean") -> Tensor:
        """计算 令牌特征最后一维上的余弦距离。"""
        loss = 1.0 - F.cosine_similarity(x, y, dim=-1)
        return _reduce_loss(loss, reduction)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """计算预测图像与目标图像的 DINOv2 多块感知损失。

        输入会缩放为 224×224。冻结 DINOv2 参数不会阻断对 ``prediction`` 的梯度。

        返回:
            各配置块的加权损失之和。"""
        prediction = F.interpolate(prediction, size=(224, 224), mode="bilinear", align_corners=False)
        target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)

        match self.input_range:
            case ImageInputRange.MINUS_ONE_TO_ONE:
                prediction = prediction.add(1.0).mul(0.5)
                target = target.add(1.0).mul(0.5)
            case ImageInputRange.ZERO_TO_255:
                prediction = prediction.div(255.0)
                target = target.div(255.0)
            case ImageInputRange.ZERO_TO_ONE:
                pass
        if self.use_input_norm:
            mean = self.get_buffer("mean")
            std = self.get_buffer("std")
            prediction = (prediction - mean) / std
            target = (target - mean) / std

        requested_blocks = list(self.layer_indices)
        predicted_features = self.dino.get_intermediate_layers(prediction, n=requested_blocks)
        target_features = self.dino.get_intermediate_layers(target, n=requested_blocks)

        total: Tensor | None = None
        for block_index, predicted_feature, target_feature in zip(self.layer_indices, predicted_features, target_features):
            term = _weighted_feature_loss(
                self.criterion,
                predicted_feature,
                target_feature,
                self.layer_weights[block_index],
                self.reduction,
            )
            total = term if total is None else total + term

        assert total is not None
        return total


class VGGPerceptualLoss(nn.Module):
    """基于冻结 torchvision VGG 中间特征的感知损失。

    层名对应真实 VGG 执行位置；VGG 的 ReLU 被设置为非原地，确保 ``conv*`` 层保存的是
    稳定的 ReLU 前特征。层权重按网络执行顺序与输出严格对齐。"""

    def __init__(
        self,
        layer_weights: Mapping[str, float],
        criterion: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: Reduction = "mean",
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        input_range: ImageInputRange = ImageInputRange.MINUS_ONE_TO_ONE,
    ) -> None:
        """初始化 VGG 感知损失。

        参数:
            layer_weights: ``VGG 层名 -> 权重`` 映射。
            criterion: L1、MSE 或 Charbonnier。
            reduction: ``none`` 返回逐样本损失；``mean``/``sum`` 返回标量。
            vgg_type: torchvision VGG 型号，例如 ``vgg16`` 或 ``vgg19``。
            use_input_norm: 是否应用 ImageNet 均值/标准差。
            input_range: 输入 RGB 张量的值域。

        异常:
            ValueError: 配置为空、VGG 型号或层名无效。"""
        super().__init__()

        if not isinstance(input_range, ImageInputRange):
            raise TypeError(f"input_range 必须为 ImageInputRange，实际为 {type(input_range).__name__}")
        if not layer_weights:
            raise ValueError("layer_weights must not be empty")

        available_layers = get_vgg_layer_names(vgg_type)
        unknown_layers = [name for name in layer_weights if name not in available_layers]
        if unknown_layers:
            raise ValueError(f"Unknown {vgg_type} layer names: {unknown_layers}")

        layer_order = {name: index for index, name in enumerate(available_layers)}
        ordered_layers = sorted(layer_weights, key=layer_order.__getitem__)
        self.layer_weights = tuple(layer_weights[name] for name in ordered_layers)
        self.vgg = VGGFeatureExtractor(layer_names=ordered_layers, vgg_type=vgg_type).eval().requires_grad_(False)

        self.criterion: CriterionFn = {
            "l1": F.l1_loss,
            "mse": F.mse_loss,
            "charbonnier": charbonnier_loss,
        }[criterion]
        self.reduction = reduction

        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )

        self.input_range = input_range

    @torch.compile(fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """计算预测图像和目标图像的多层 VGG 感知损失。

        参数:
            prediction: 预测 RGB 图像张量。
            target: 目标 RGB 图像张量。

        返回:
            各配置 VGG 层的加权损失之和。"""
        match self.input_range:
            case ImageInputRange.MINUS_ONE_TO_ONE:
                prediction = prediction.add(1.0).mul(0.5)
                target = target.add(1.0).mul(0.5)
            case ImageInputRange.ZERO_TO_255:
                prediction = prediction.div(255.0)
                target = target.div(255.0)
            case ImageInputRange.ZERO_TO_ONE:
                pass
        if self.use_input_norm:
            mean = self.get_buffer("mean")
            std = self.get_buffer("std")
            prediction = (prediction - mean) / std
            target = (target - mean) / std

        predicted_features = self.vgg(prediction)
        target_features = self.vgg(target)

        total: Tensor | None = None
        for weight, predicted_feature, target_feature in zip(self.layer_weights, predicted_features, target_features):
            term = _weighted_feature_loss(
                self.criterion,
                predicted_feature,
                target_feature,
                weight,
                self.reduction,
            )
            total = term if total is None else total + term

        assert total is not None
        return total
