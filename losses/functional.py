from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, autocast

EPS = 1e-8

Reduction = Literal["none", "mean", "sum"]
LossFn = Callable[[Tensor, Tensor], Tensor]
CriterionFn = Callable[..., Tensor]


def _reduce_loss(loss: Tensor, reduction: Reduction) -> Tensor:
    match reduction:
        case "none":
            return loss
        case "mean":
            return loss.mean()
        case "sum":
            return loss.sum()
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def _weighted_feature_loss(
    criterion: CriterionFn,
    prediction: Tensor,
    target: Tensor,
    weight: float,
    reduction: Reduction,
) -> Tensor:
    if reduction == "none":
        difference = criterion(prediction, target, reduction="none")
        feature_dims = tuple(range(1, difference.ndim))
        reduced = difference.mean(dim=feature_dims) if feature_dims else difference
    else:
        reduced = criterion(prediction, target, reduction=reduction)
    return reduced * weight


def _make_weighted_loss(
    loss_fn: CriterionFn, weight: float = 1.0, reduction: Reduction = "mean"
) -> LossFn:
    """将任意损失函数包装为带权重的版本。

    Args:
        loss_fn: 签名为 ``(pred, target, reduction=...) -> Tensor`` 的损失函数。
        weight (float): 标量权重，结果乘以该值。默认 ``1.0``。
        reduction (str): 传递给 ``loss_fn`` 的 reduction 模式。默认 ``"mean"``。

    Returns:
        LossFn: 签名为 ``(pred, target) -> Tensor`` 的包装函数。
    """

    def wrapper(pred: Tensor, target: Tensor) -> Tensor:
        return weight * loss_fn(pred, target, reduction=reduction)

    return wrapper


def charbonnier_loss(
    pred: Tensor, target: Tensor, reduction: Reduction = "mean"
) -> Tensor:
    """Charbonnier 损失（pseudo-Huber / L1 平滑近似）。

    计算公式：``loss = sqrt((pred - target)^2 + eps)``，
    相比 L1 在零点附近可微，相比 MSE 对离群值更鲁棒。

    Args:
        pred (Tensor):   预测张量，任意形状。
        target (Tensor): 目标张量，与 ``pred`` 形状相同。
        reduction (str): ``"none"`` / ``"mean"`` / ``"sum"``。默认 ``"mean"``。

    Returns:
        Tensor: 标量（``mean``/``sum``）或与输入同形张量（``none``）。
    """

    loss = torch.sqrt((pred - target).pow(2).add_(EPS))

    match reduction:
        case "none":
            return loss
        case "mean":
            return loss.mean()
        case "sum":
            return loss.sum()
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def orthogonal_loss(x: Tensor, y: Tensor, reduction: Reduction = "mean") -> Tensor:
    """正交损失：惩罚两组特征向量之间的余弦相似度。

    先对输入沿 ``dim=1`` 做 L2 归一化，再计算逐样本点积绝对值，
    鼓励 ``x`` 与 ``y`` 在特征空间中相互正交（解耦）。

    Args:
        x (Tensor): 形状 ``(N, C)`` 的特征张量。
        y (Tensor): 形状 ``(N, C)`` 的特征张量。
        reduction (str): ``"none"`` / ``"mean"`` / ``"sum"``。默认 ``"mean"``。

    Returns:
        Tensor: 标量或形状 ``(N,)`` 的逐样本损失。
    """

    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)

    loss = torch.abs(torch.sum(x * y, dim=1))

    return _reduce_loss(loss, reduction)


def make_l1_loss(weight: float = 1.0, reduction: Reduction = "mean") -> LossFn:
    """工厂函数：返回带权重的 L1 损失。"""
    return _make_weighted_loss(F.l1_loss, weight, reduction)


def make_mse_loss(weight: float = 1.0, reduction: Reduction = "mean") -> LossFn:
    """工厂函数：返回带权重的 MSE（L2）损失。"""
    return _make_weighted_loss(F.mse_loss, weight, reduction)


def make_charbonnier_loss(weight: float = 1.0, reduction: Reduction = "mean") -> LossFn:
    """工厂函数：返回带权重的 Charbonnier 损失。"""
    return _make_weighted_loss(charbonnier_loss, weight, reduction)


def make_bce_loss(weight: float = 1.0, reduction: Reduction = "mean") -> LossFn:
    """工厂函数：返回带权重的 BCE 损失（强制禁用 AMP）。

    ``binary_cross_entropy`` 要求输入已经过 sigmoid，数值范围 ``[0, 1]``。
    内部通过 ``autocast(enabled=False)`` 规避混合精度溢出。
    """
    f = _make_weighted_loss(F.binary_cross_entropy, weight, reduction)

    def disable_amp(prediction: Tensor, target: Tensor) -> Tensor:
        with autocast(device_type="cuda", enabled=False):
            return f(prediction.float(), target.float())

    return disable_amp


def make_bce_with_logits_loss(
    weight: float = 1.0, reduction: Reduction = "mean"
) -> LossFn:
    """工厂函数：返回带权重的 BCE-with-logits 损失（强制禁用 AMP）。

    接受未经 sigmoid 的 logits，内部数值更稳定。
    同样通过 ``autocast(enabled=False)`` 禁用混合精度。
    """
    f = _make_weighted_loss(F.binary_cross_entropy_with_logits, weight, reduction)

    def disable_amp(prediction: Tensor, target: Tensor) -> Tensor:
        with autocast(device_type="cuda", enabled=False):
            return f(prediction.float(), target.float())

    return disable_amp


def make_orthogonal_loss(weight: float = 1.0, reduction: Reduction = "mean") -> LossFn:
    """工厂函数：返回带权重的正交损失。"""
    return _make_weighted_loss(orthogonal_loss, weight, reduction)


def r1_reg_loss(real_score: Tensor, real_image: Tensor, gamma: float = 10.0) -> Tensor:
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    if not real_image.requires_grad:
        raise ValueError(
            "real_image must have requires_grad=True before the discriminator forward pass"
        )

    gradients = torch.autograd.grad(
        outputs=real_score.sum(),
        inputs=real_image,
        create_graph=True,
        only_inputs=True,
    )[0]
    penalty = gradients.square().flatten(1).sum(dim=1)
    return penalty.mean() * (gamma / 2)
