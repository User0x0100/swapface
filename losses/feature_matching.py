from collections.abc import Mapping, Sequence

import torch.nn.functional as F
from torch import Tensor, nn


class WeakFeatureMatchingLoss(nn.Module):
    """对同一判别器的 fake/real encoder 中间特征做加权 L1 匹配。"""

    def __init__(self, layer_weights: Mapping[int, float]) -> None:
        super().__init__()
        if not layer_weights:
            raise ValueError("WeakFeatureMatchingLoss 至少需要一个 feature layer")
        normalized = {int(index): float(weight) for index, weight in layer_weights.items()}
        if any(index < 0 for index in normalized):
            raise ValueError("WFM feature index 必须 >= 0")
        if any(weight < 0.0 for weight in normalized.values()):
            raise ValueError("WFM feature weight 必须 >= 0")
        self.layer_weights = normalized

    def forward(self, fake_feats: Sequence[Tensor], real_feats: Sequence[Tensor]) -> Tensor:
        if len(fake_feats) != len(real_feats):
            raise ValueError(f"WFM fake/real feature 数量不一致：{len(fake_feats)} != {len(real_feats)}")
        max_index = max(self.layer_weights)
        if max_index >= len(fake_feats):
            raise ValueError(f"WFM feature index={max_index} 超出判别器 feature 数量={len(fake_feats)}")

        loss = fake_feats[0].new_zeros(())
        for index, weight in self.layer_weights.items():
            if fake_feats[index].shape != real_feats[index].shape:
                raise ValueError(
                    f"WFM feature[{index}] shape 不一致：{tuple(fake_feats[index].shape)} != {tuple(real_feats[index].shape)}"
                )
            loss = loss + F.l1_loss(fake_feats[index], real_feats[index]) * weight
        return loss
