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
    """按 ``none``、``mean`` 或 ``sum`` 聚合损失张量。"""
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
    """计算单层特征损失，并统一 ``reduction="none"`` 的逐样本语义。"""
    if reduction == "none":
        difference = criterion(prediction, target, reduction="none")
        feature_dims = tuple(range(1, difference.ndim))
        reduced = difference.mean(dim=feature_dims) if feature_dims else difference
    else:
        reduced = criterion(prediction, target, reduction=reduction)
    return reduced * weight


def _make_weighted_loss(loss_fn: CriterionFn, weight: float = 1.0, reduction: Reduction = "mean") -> LossFn:
    """将任意损失函数包装为带权重的版本。

    参数:
        loss_fn: 签名为 ``(pred, target, reduction=...) -> Tensor`` 的损失函数。
        weight (float): 标量权重，结果乘以该值。默认 ``1.0``。
        reduction (str): 传递给 ``loss_fn`` 的 reduction 模式。默认 ``"mean"``。

    返回:
        LossFn: 签名为 ``(pred, target) -> Tensor`` 的包装函数。
    """

    def wrapper(pred: Tensor, target: Tensor) -> Tensor:
        return weight * loss_fn(pred, target, reduction=reduction)

    return wrapper


def charbonnier_loss(pred: Tensor, target: Tensor, reduction: Reduction = "mean") -> Tensor:
    """Charbonnier 损失（pseudo-Huber / L1 平滑近似）。

    计算公式：``loss = sqrt((pred - target)^2 + eps)``，
    相比 L1 在零点附近可微，相比 MSE 对离群值更鲁棒。

    参数:
        pred (Tensor):   预测张量，任意形状。
        target (Tensor): 目标张量，与 ``pred`` 形状相同。
        reduction (str): ``"none"`` / ``"mean"`` / ``"sum"``。默认 ``"mean"``。

    返回:
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

    参数:
        x (Tensor): 形状 ``(N, C)`` 的特征张量。
        y (Tensor): 形状 ``(N, C)`` 的特征张量。
        reduction (str): ``"none"`` / ``"mean"`` / ``"sum"``。默认 ``"mean"``。

    返回:
        Tensor: 标量或形状 ``(N,)`` 的逐样本损失。
    """

    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)

    loss = torch.abs(torch.sum(x * y, dim=1))

    return _reduce_loss(loss, reduction)


def make_l1_loss(weight: float = 1.0, reduction: Reduction = "mean") -> LossFn:
    """创建带权重的 L1 损失函数。

    参数:
        weight: 损失标量权重。
        reduction: ``none``、``mean`` 或 ``sum``。

    返回:
        接受 ``(prediction, target)`` 并返回张量的损失函数。
    """
    return _make_weighted_loss(F.l1_loss, weight, reduction)


def make_mse_loss(weight: float = 1.0, reduction: Reduction = "mean") -> LossFn:
    """创建带权重的均方误差（MSE/L2）损失函数。

    参数:
        weight: 损失标量权重。
        reduction: ``none``、``mean`` 或 ``sum``。

    返回:
        接受 ``(prediction, target)`` 并返回张量的损失函数。
    """
    return _make_weighted_loss(F.mse_loss, weight, reduction)


def make_charbonnier_loss(weight: float = 1.0, reduction: Reduction = "mean") -> LossFn:
    """创建带权重的 Charbonnier 损失函数。

    参数:
        weight: 损失标量权重。
        reduction: ``none``、``mean`` 或 ``sum``。

    返回:
        接受 ``(prediction, target)`` 并返回张量的损失函数。
    """
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


def make_bce_with_logits_loss(weight: float = 1.0, reduction: Reduction = "mean") -> LossFn:
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
    """创建带权重的特征正交损失函数。

    参数:
        weight: 损失标量权重。
        reduction: ``none``、``mean`` 或 ``sum``。

    返回:
        接受两组 ``(N, C)`` 特征并返回张量的损失函数。
    """
    return _make_weighted_loss(orthogonal_loss, weight, reduction)


def r1_reg_loss(real_score: Tensor, real_image: Tensor, gamma: float = 10.0) -> Tensor:
    """计算 R1 判别器梯度惩罚。

    对每个真实样本计算 ``||∇_x D(x)||²``，取批次均值后乘以 ``gamma / 2``。

    参数:
        real_score: 判别器对 ``real_image`` 的输出，必须保留到输入的计算图。
        real_image: 真实图像张量，必须设置 ``requires_grad=True``。
        gamma: R1 正则强度，必须非负。

    返回:
        标量 R1 损失。

    异常:
        ValueError: ``gamma`` 为负数，或 ``real_image`` 未启用梯度。"""
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    if not real_image.requires_grad:
        raise ValueError("real_image must have requires_grad=True before the discriminator forward pass")

    gradients = torch.autograd.grad(
        outputs=real_score.sum(),
        inputs=real_image,
        create_graph=True,
        only_inputs=True,
    )[0]
    penalty = gradients.square().flatten(1).sum(dim=1)
    return penalty.mean() * (gamma / 2)
