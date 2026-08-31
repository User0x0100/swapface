from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from misc.models.id_encoder import IDEncoder, IDEncoderProvider

from .functional import Reduction, _reduce_loss


class IdentityLoss(nn.Module):
    def __init__(
        self,
        provider: IDEncoderProvider = IDEncoderProvider.MS1MV3_ARCFACE_R50_FP16,
        weight: float = 1.0,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.id_encoder = IDEncoder(provider=provider).eval().requires_grad_(False)
        self.eval()

    @torch.compile(
        fullgraph=True,
        dynamic=False,
        options={"epilogue_fusion": True, "max_autotune": True},
    )
    def extract_identity_embeddings(self, faces: Tensor) -> Tensor:
        return self.id_encoder(faces)

    def forward(
        self, generated_embeddings: Tensor, reference_embeddings: Tensor
    ) -> Tensor:
        if generated_embeddings.shape != reference_embeddings.shape:
            raise ValueError(
                "generated and reference embedding shapes must match: "
                f"{tuple(generated_embeddings.shape)} != {tuple(reference_embeddings.shape)}"
            )
        loss = (
            1.0 - F.cosine_similarity(generated_embeddings, reference_embeddings, dim=1)
        ) * self.weight
        return _reduce_loss(loss, self.reduction)


class IFSRLoss(nn.Module):
    def __init__(
        self,
        ifsr_margin_scale: float,
        ifsr_constraints: Mapping[str, tuple[float, float]],
        id_encoder_provider: IDEncoderProvider = IDEncoderProvider.MS1MV3_ARCFACE_R50_FP16,
    ) -> None:
        super().__init__()
        if not ifsr_constraints:
            raise ValueError("ifsr_constraints must not be empty")

        id_encoder = IDEncoder(provider=id_encoder_provider)
        self.feature_layer_indices: dict[int, str] = {}
        self.layer_constraints = {
            layer_name: (margin * ifsr_margin_scale, weight)
            for layer_name, (margin, weight) in ifsr_constraints.items()
        }

        requested_layers = set(ifsr_constraints)
        feature_modules = nn.ModuleList()
        max_layer_index = -1
        index = 0

        for parent_name, parent_module in id_encoder.backbone.named_children():
            children = list(parent_module.named_children())
            if not children:
                feature_modules.append(parent_module)
                if parent_name in requested_layers:
                    self.feature_layer_indices[index] = parent_name
                    max_layer_index = max(max_layer_index, index)
                index += 1
                continue

            for child_name, child_module in children:
                layer_name = f"{parent_name}.{child_name}"
                feature_modules.append(child_module)
                if layer_name in requested_layers:
                    self.feature_layer_indices[index] = layer_name
                    max_layer_index = max(max_layer_index, index)
                index += 1

        found_layers = set(self.feature_layer_indices.values())
        missing_layers = sorted(requested_layers - found_layers)
        if missing_layers:
            raise ValueError(
                f"IFSR layers not found in {id_encoder_provider.name}: {missing_layers}"
            )

        self.net = feature_modules[: max_layer_index + 1].eval().requires_grad_(False)

    @torch.compile(
        fullgraph=True,
        dynamic=False,
        options={"epilogue_fusion": True, "max_autotune": True},
    )
    def extract_features(self, x: Tensor) -> dict[str, Tensor]:
        features: dict[str, Tensor] = {}
        for index, module in enumerate(self.net):
            x = module(x)
            layer_name = self.feature_layer_indices.get(index)
            if layer_name is not None:
                features[layer_name] = x
        return features

    def forward(
        self,
        generated_features: Mapping[str, Tensor],
        reference_features: Mapping[str, Tensor],
    ) -> Tensor:
        total: Tensor | None = None
        for layer_name, (margin, weight) in self.layer_constraints.items():
            try:
                generated = generated_features[layer_name].flatten(1)
                reference = reference_features[layer_name].flatten(1)
            except KeyError as exc:
                raise KeyError(f"Missing IFSR feature: {layer_name}") from exc

            if reference.shape != generated.shape:
                raise ValueError(
                    f"IFSR feature shape mismatch at {layer_name}: "
                    f"{tuple(reference.shape)} != {tuple(generated.shape)}"
                )

            distance = (1.0 - F.cosine_similarity(reference, generated, dim=1)).mean()
            term = F.relu(distance - margin) * weight
            total = term if total is None else total + term

        assert total is not None
        return total
