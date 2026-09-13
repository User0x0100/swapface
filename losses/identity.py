from collections.abc import Mapping

import torch.nn.functional as F
from torch import Tensor, nn

from misc.models.id_encoder import IDEncoder, IDEncoderProvider

from .functional import Reduction, _reduce_loss


class IdentityLoss(nn.Module):
    """基于冻结身份编码器的余弦身份损失。

    编码器参数被冻结，但不会对输入使用 ``no_grad``，因此生成图像经过身份编码器后的梯度仍可
    回传到生成器。"""

    def __init__(
        self,
        provider: IDEncoderProvider = IDEncoderProvider.MS1MV3_ARCFACE_R50_FP16,
        weight: float = 1.0,
        reduction: Reduction = "mean",
    ) -> None:
        """初始化身份损失。

        参数:
            provider: 身份编码器模型配置。
            weight: 最终身份损失权重。
            reduction: 输出聚合方式。"""
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.id_encoder = IDEncoder(provider=provider).eval().requires_grad_(False)
        self.eval()

    def extract_identity_embeddings(self, faces: Tensor) -> Tensor:
        """从人脸图像提取 L2 归一化身份嵌入向量。

        参数:
            faces: RGB 人脸张量，通常为 ``(N, 3, H, W)``、值域 ``[-1, 1]``。编码器内部会按需缩放到 112×112。

        返回:
            形状通常为 ``(N, 512)`` 的身份嵌入向量。"""
        return self.id_encoder(faces)

    def forward(self, generated_embeddings: Tensor, reference_embeddings: Tensor) -> Tensor:
        """计算生成身份与参考身份嵌入向量的余弦距离。

        参数:
            generated_embeddings: 生成图像的身份嵌入向量。
            reference_embeddings: 目标身份嵌入向量，必须与生成嵌入向量同形。

        返回:
            ``1 - cosine_similarity`` 经权重和 reduction 处理后的损失。"""
        if generated_embeddings.shape != reference_embeddings.shape:
            raise ValueError(f"generated and reference embedding shapes must match: {tuple(generated_embeddings.shape)} != {tuple(reference_embeddings.shape)}")
        loss = (1.0 - F.cosine_similarity(generated_embeddings, reference_embeddings, dim=1)) * self.weight
        return _reduce_loss(loss, self.reduction)


class IFSRLoss(nn.Module):
    """基于身份编码器中间特征的间隔约束损失。

    每个配置项为 ``layer_name -> (margin, weight)``。仅构建到最深请求层为止的冻结子网络，
    减少不必要的后续身份编码器计算。"""

    def __init__(
        self,
        ifsr_margin_scale: float,
        ifsr_constraints: Mapping[str, tuple[float, float]],
        id_encoder_provider: IDEncoderProvider = IDEncoderProvider.MS1MV3_ARCFACE_R50_FP16,
    ) -> None:
        """初始化 IFSR 中间特征损失。

        参数:
            ifsr_margin_scale: 对所有配置间隔阈值的统一缩放系数。
            ifsr_constraints: ``layer_name -> (margin, weight)`` 映射。
            id_encoder_provider: 用于提取中间特征的身份编码器。

        异常:
            ValueError: 约束为空，或请求的层在所选身份编码器中不存在。"""
        super().__init__()
        if not ifsr_constraints:
            raise ValueError("ifsr_constraints must not be empty")

        id_encoder = IDEncoder(provider=id_encoder_provider)
        self.feature_layer_indices: dict[int, str] = {}
        self.layer_constraints = {layer_name: (margin * ifsr_margin_scale, weight) for layer_name, (margin, weight) in ifsr_constraints.items()}

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
            raise ValueError(f"IFSR layers not found in {id_encoder_provider.name}: {missing_layers}")

        self.net = feature_modules[: max_layer_index + 1].eval().requires_grad_(False)

    def extract_features(self, x: Tensor) -> dict[str, Tensor]:
        """提取 IFSR 配置要求的中间特征。

        参数:
            x: 输入人脸张量。

        返回:
            以配置层名为键 的特征字典。冻结网络仍允许梯度回传到 ``x``。"""
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
        """计算生成特征与参考特征的间隔损失。

        每层先计算展平特征的余弦距离 ``d``，再累加 ``relu(d - margin) * weight``。

        异常:
            KeyError: 输入特征字典缺少配置层。
            ValueError: 同一层的生成/参考特征形状不一致。"""
        total: Tensor | None = None
        for layer_name, (margin, weight) in self.layer_constraints.items():
            try:
                generated = generated_features[layer_name].flatten(1)
                reference = reference_features[layer_name].flatten(1)
            except KeyError as exc:
                raise KeyError(f"Missing IFSR feature: {layer_name}") from exc

            if reference.shape != generated.shape:
                raise ValueError(f"IFSR feature shape mismatch at {layer_name}: {tuple(reference.shape)} != {tuple(generated.shape)}")

            distance = (1.0 - F.cosine_similarity(reference, generated, dim=1)).mean()
            term = F.relu(distance - margin) * weight
            total = term if total is None else total + term

        assert total is not None
        return total
