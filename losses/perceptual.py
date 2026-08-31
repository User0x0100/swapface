from collections.abc import Mapping
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .functional import (
    CriterionFn,
    Reduction,
    _reduce_loss,
    _weighted_feature_loss,
    charbonnier_loss,
)
from .vgg import VGGFeatureExtractor, get_vgg_layer_names


class WeightedFeatureMatchingLoss(nn.Module):
    def __init__(
        self,
        layer_weights: Mapping[int, float],
        criterion: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: Reduction = "mean",
    ) -> None:
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

    def forward(
        self, predicted_features: list[Tensor], target_features: list[Tensor]
    ) -> Tensor:
        if len(predicted_features) != len(target_features):
            raise ValueError(
                f"feature list length mismatch: {len(predicted_features)} != {len(target_features)}"
            )

        max_index = max(self.layer_weights)
        if max_index >= len(predicted_features):
            raise IndexError(
                f"feature layer index {max_index} is out of range for {len(predicted_features)} features"
            )

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
    def __init__(
        self,
        layer_weights: Mapping[int, float],
        criterion: Literal["l1", "mse", "charbonnier", "cosine"] = "cosine",
        reduction: Reduction = "mean",
        dino_type: str = "dinov2_vitb14_reg",
        use_input_norm: bool = True,
        range_norm: bool = True,
    ) -> None:
        super().__init__()

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
        invalid_indices = sorted(
            index for index in self.layer_weights if index >= block_count
        )
        if invalid_indices:
            raise ValueError(
                f"DINOv2 block indices out of range for {dino_type} "
                f"with {block_count} blocks: {invalid_indices}"
            )
        self.layer_indices = tuple(sorted(self.layer_weights))

        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
                persistent=False,
            )
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
                persistent=False,
            )

        self.range_norm = range_norm

    @staticmethod
    def _cosine_distance(x: Tensor, y: Tensor, reduction: Reduction = "mean") -> Tensor:
        loss = 1.0 - F.cosine_similarity(x, y, dim=-1)
        return _reduce_loss(loss, reduction)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction = F.interpolate(
            prediction, size=(224, 224), mode="bilinear", align_corners=False
        )
        target = F.interpolate(
            target, size=(224, 224), mode="bilinear", align_corners=False
        )

        if self.range_norm:
            prediction = prediction.add(1.0).mul(0.5)
            target = target.add(1.0).mul(0.5)
        if self.use_input_norm:
            mean = self.get_buffer("mean")
            std = self.get_buffer("std")
            prediction = (prediction - mean) / std
            target = (target - mean) / std

        requested_blocks = list(self.layer_indices)
        predicted_features = self.dino.get_intermediate_layers(
            prediction, n=requested_blocks
        )
        target_features = self.dino.get_intermediate_layers(target, n=requested_blocks)

        total: Tensor | None = None
        for block_index, predicted_feature, target_feature in zip(
            self.layer_indices, predicted_features, target_features
        ):
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
    def __init__(
        self,
        layer_weights: Mapping[str, float],
        criterion: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: Reduction = "mean",
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        range_norm: bool = True,
    ) -> None:
        super().__init__()

        if not layer_weights:
            raise ValueError("layer_weights must not be empty")

        available_layers = get_vgg_layer_names(vgg_type)
        unknown_layers = [
            name for name in layer_weights if name not in available_layers
        ]
        if unknown_layers:
            raise ValueError(f"Unknown {vgg_type} layer names: {unknown_layers}")

        layer_order = {name: index for index, name in enumerate(available_layers)}
        ordered_layers = sorted(layer_weights, key=layer_order.__getitem__)
        self.layer_weights = tuple(layer_weights[name] for name in ordered_layers)
        self.vgg = (
            VGGFeatureExtractor(layer_names=ordered_layers, vgg_type=vgg_type)
            .eval()
            .requires_grad_(False)
        )

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
                torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
                persistent=False,
            )
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
                persistent=False,
            )

        self.range_norm = range_norm

    @torch.compile(
        fullgraph=True,
        dynamic=False,
        options={"epilogue_fusion": True, "max_autotune": True},
    )
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.range_norm:
            prediction = prediction.add(1.0).mul(0.5)
            target = target.add(1.0).mul(0.5)
        if self.use_input_norm:
            mean = self.get_buffer("mean")
            std = self.get_buffer("std")
            prediction = (prediction - mean) / std
            target = (target - mean) / std

        predicted_features = self.vgg(prediction)
        target_features = self.vgg(target)

        total: Tensor | None = None
        for weight, predicted_feature, target_feature in zip(
            self.layer_weights, predicted_features, target_features
        ):
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
