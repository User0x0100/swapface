import torch
import torch.nn.functional as F
from torch import Tensor, nn, autocast
from kornia.color import rgb_to_lab
from typing import Literal, Callable, Mapping

from .vgg import VGGFeatureExtractor
from misc.models.idencoder import PROVIDER, IDEncoder

EPS = 1e-8

LossFn = Callable[[Tensor, Tensor], Tensor]


def create_weighted_loss(loss_fn, weight=1.0, reduction="mean"):
    return lambda *args, **kw: weight * loss_fn(*args, reduction=reduction, **kw)


def charbonnier_loss(pred: Tensor, target: Tensor, reduction: Literal["none", "mean", "sum"] = "mean") -> Tensor:
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


def l1_loss_fn(weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean") -> LossFn:
    return create_weighted_loss(F.l1_loss, weight, reduction)


def mse_loss_fn(weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean") -> LossFn:
    return create_weighted_loss(F.mse_loss, weight, reduction)


def charbonnier_loss_fn(weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean") -> LossFn:
    return create_weighted_loss(charbonnier_loss, weight, reduction)


def bce_loss_fn(weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean") -> LossFn:
    f = create_weighted_loss(F.binary_cross_entropy, weight, reduction)

    def disable_amp(*args, **kwargs):
        with autocast(device_type="cuda", enabled=False):
            return f(*args, **kwargs)

    return disable_amp


def bce_with_logits_loss_fn(weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean") -> LossFn:
    f = create_weighted_loss(F.binary_cross_entropy_with_logits, weight, reduction)

    def disable_amp(*args, **kwargs):
        with autocast(device_type="cuda", enabled=False):
            return f(*args, **kwargs)

    return disable_amp


class WFMLoss(nn.Module):
    def __init__(self, layer_weights: Mapping[int, float], criterion: Literal["l1", "mse", "charbonnier"] = "l1"):
        super().__init__()

        self.criterion = {"l1": F.l1_loss, "mse": F.mse_loss, "charbonnier": charbonnier_loss}.get(criterion)
        if self.criterion is None:
            raise NotImplementedError(f"{criterion} criterion has not been supported. Only 'l1' 'mse' 'charbonnier' are supported.")

        self.layer_weights = layer_weights

    def forward(self, x_feats: list[Tensor], y_feats: list[Tensor]) -> Tensor:

        loss = x_feats[0].new_tensor(0.0)
        for idx, weight in self.layer_weights.items():
            loss += self.criterion(x_feats[idx], y_feats[idx]) * weight

        return loss


class DINOv2PerceptualLoss(nn.Module):
    def __init__(
        self,
        layer_weights: Mapping[int, float],
        criterion: Literal["l1", "mse", "charbonnier", "cosine"] = "cosine",
        dino_type="dinov2_vitb14_reg",
        use_input_norm: bool = True,
        range_norm: bool = True,
    ):
        """
        Args:
            layer_weights (Mapping): The weight for each layer of vgg feature.
            use_input_norm (bool):  If True, normalize the input image.
                Default: True.
            range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
                Default: False.
        """
        super().__init__()

        self.criterion = {"l1": F.l1_loss, "mse": F.mse_loss, "charbonnier": charbonnier_loss, "cosine": self._cosine_distance}.get(criterion)
        if self.criterion is None:
            raise NotImplementedError(f"{criterion} criterion has not been supported. Only 'l1' 'mse' 'charbonnier' 'cosine' are supported.")
        self.layer_weights = layer_weights

        self.dino = torch.hub.load("facebookresearch/dinov2", dino_type)
        self.dino.eval()
        self.dino.requires_grad_(False)

        self.n_blocks = max(self.layer_weights.keys()) + 1

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.register_buffer("mean", Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.range_norm = range_norm

    @staticmethod
    def _cosine_distance(x: Tensor, y: Tensor) -> Tensor:

        x_norm, y_norm = F.normalize(x, p=2, dim=-1), F.normalize(y, p=2, dim=-1)
        return 1.0 - F.cosine_similarity(x_norm, y_norm, dim=-1).mean()

    # @torch.compile(fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})
    def forward(self, x: Tensor, y: Tensor) -> Tensor:

        x = F.interpolate(x, [224, 224], mode="bilinear", align_corners=False)
        y = F.interpolate(y, [224, 224], mode="bilinear", align_corners=False)

        if self.range_norm:
            x = (x + 1) * 0.5
            y = (y + 1) * 0.5
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        x_patch = self.dino.get_intermediate_layers(x, n=self.n_blocks)
        y_patch = self.dino.get_intermediate_layers(y, n=self.n_blocks)

        loss = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        for idx, weight in self.layer_weights.items():
            x_feat, y_feat = x_patch[idx], y_patch[idx]
            loss += self.criterion(x_feat, y_feat) * weight

        return loss


class PerceptualLoss(nn.Module):
    def __init__(
        self, layer_weights: Mapping[str, float], criterion: Literal["l1", "mse", "charbonnier"] = "l1", vgg_type="vgg19", use_input_norm: bool = True, range_norm: bool = True
    ):
        """
        Args:
            layer_weights (Mapping): The weight for each layer of vgg feature.
                    Here is an example: {'conv5_4': 1.}, which means the conv5_4
                    feature layer (before relu5_4) will be extracted with weight
                    1.0 in calculting losses.
            use_input_norm (bool):  If True, normalize the input image in vgg.
                Default: True.
            range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
                Default: False.
        """
        super().__init__()

        self.vgg = VGGFeatureExtractor(layer_names=list(layer_weights.keys()), vgg_type=vgg_type, use_input_norm=use_input_norm, range_norm=range_norm)
        self.vgg.eval()
        self.vgg.requires_grad_(False)

        self.criterion = {"l1": F.l1_loss, "mse": F.mse_loss, "charbonnier": charbonnier_loss}.get(criterion)
        if self.criterion is None:
            raise NotImplementedError(f"{criterion} criterion has not been supported. Only 'l1' 'mse' 'charbonnier' are supported.")

        self.weights = list(layer_weights.values())

    @torch.compile(fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        fx: list[Tensor] = self.vgg(x)
        fy: list[Tensor] = self.vgg(y)

        loss = 0.0
        for weight, a, b in zip(self.weights, fx, fy):
            loss += weight * self.criterion(a, b, reduction="mean")

        return loss


class DSSIMLoss(nn.Module):
    def __init__(self, weight: float = 1.0, window_size: int = 11, sigma: float = 1.5, reduction: Literal["none", "mean", "sum"] = "none"):
        super().__init__()

        assert window_size % 2 == 1, "window_size must be odd"

        self.C = 3

        self.weight = weight
        self.window_size = window_size
        self.sigma = sigma
        self.reduction = reduction

        self.register_buffer("window", self._create_window(window_size, sigma).expand(self.C, 1, self.window_size, self.window_size))

    @staticmethod
    def _gaussian_1d(window_size: int, sigma: float) -> Tensor:
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g

    def _create_window(self, window_size: int, sigma: float) -> Tensor:
        g1d = self._gaussian_1d(window_size, sigma)
        g2d = torch.outer(g1d, g1d)
        window = g2d[None, None, :, :]  # (1,1,ks,ks)
        return window

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:
        """
        返回 SSIM map: (N, 1, H, W)
        """

        C = self.C
        window = self.window

        # 使用 group conv，逐通道独立计算
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=C)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=C)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=self.window_size // 2, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=self.window_size // 2, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=C) - mu_xy

        sigma_x2 = torch.clamp(sigma_x2, min=0.0)
        sigma_y2 = torch.clamp(sigma_y2, min=0.0)

        # SSIM 常量（假设输入范围 [0,1]）
        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map: Tensor = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + EPS)

        # RGB → mean over channel
        return ssim_map.mean(dim=1, keepdim=True)

    @torch.compile(fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        x, y: (N,3,H,W), float in [0,1]
        """

        ssim = self._ssim(x, y)  # (N,1,H,W)
        dssim = (1.0 - ssim) * 0.5 * self.weight  # (N,1,H,W)

        match self.reduction:
            case "none":
                return dssim
            case "mean":
                return dssim.mean()
            case "sum":
                return dssim.sum()


class StyleLossLabChroma(nn.Module):
    def __init__(self, weight: float = 1.0, range_norm: bool = True):
        """
        Args:
            range_norm (bool): 如果输入值域为 [-1, 1] 则转换为 [0, 1].
                Default: True.
        """
        super().__init__()

        self.range_norm = range_norm
        self.weight = weight

    def _mean_std(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var + EPS)
        return mean, std

    def _style_loss_mean_std(self, pred: Tensor, target: Tensor, weight: float = 1.0) -> Tensor:

        m_p, s_p = self._mean_std(pred)
        m_s, s_s = self._mean_std(target)

        loss = F.l1_loss(m_p, m_s) + F.l1_loss(s_p, s_s)
        return loss * weight

    @torch.compile(fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})
    def forward(self, pred_rgb: Tensor, target_rgb: Tensor) -> Tensor:

        if self.range_norm:
            pred_rgb = pred_rgb.add(1.0).mul(0.5)
            target_rgb = target_rgb.add(1.0).mul(0.5)

        pred_rgb = pred_rgb.clamp(0.0, 1.0)
        target_rgb = target_rgb.clamp(0.0, 1.0)

        pred_lab = rgb_to_lab(pred_rgb)
        target_lab = rgb_to_lab(target_rgb)

        return self._style_loss_mean_std(pred_lab, target_lab, self.weight)


@autocast(device_type="cuda", enabled=False)
def r1_reg_loss(real_score: Tensor, real_img: Tensor, gamma: float = 10.0) -> Tensor:
    real_grads = torch.autograd.grad(outputs=real_score.sum(), inputs=real_img, create_graph=True, retain_graph=True, only_inputs=True)[0]
    penalty = real_grads.square().sum(dim=(1, 2, 3))
    return 0.5 * gamma * penalty.mean()


class DLoss(nn.Module):
    def __init__(self, loss_type: Literal["hinge", "wgan", "ls", "bce"] = "hinge", weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "none"):
        super().__init__()

        self.reduction = reduction
        self.weight = weight

        self.loss_fn = {"hinge": self._hinge, "wgan": self._wgan, "ls": self._ls, "bce": self._bce}.get(loss_type)
        if self.loss_fn is None:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Only 'hinge' 'wgan' 'ls' 'bce' are supported.")

    def _hinge(self, fake_score: Tensor, real_score: Tensor) -> Tensor:
        loss_real = torch.relu(1.0 - real_score)
        loss_fake = torch.relu(1.0 + fake_score)
        loss = 0.5 * (loss_real + loss_fake)
        return loss

    def _wgan(self, fake_score: Tensor, real_score: Tensor) -> Tensor:
        loss = fake_score - real_score
        return loss

    def _ls(self, fake_score: Tensor, real_score: Tensor) -> Tensor:
        loss_real = (real_score - 1) ** 2
        loss_fake = fake_score**2
        loss = 0.5 * (loss_real + loss_fake)
        return loss

    @torch.autocast(device_type="cuda", enabled=False)
    def _bce(self, fake_score: Tensor, real_score: Tensor) -> Tensor:
        real_label = torch.ones_like(real_score)
        fake_label = torch.zeros_like(fake_score)
        loss_real = F.binary_cross_entropy_with_logits(real_score, real_label)
        loss_fake = F.binary_cross_entropy_with_logits(fake_score, fake_label)
        loss = 0.5 * (loss_real + loss_fake)
        return loss

    def forward(self, fake_score: Tensor, real_score: Tensor) -> Tensor:

        loss = self.weight * self.loss_fn(fake_score, real_score)

        match self.reduction:
            case "none":
                return loss
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()


class GANLoss(nn.Module):
    def __init__(self, loss_type: Literal["hinge", "wgan", "ls", "bce"] = "hinge", weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "none"):
        super().__init__()

        self.reduction = reduction
        self.weight = weight

        self.loss_fn = {"hinge": self._hinge_and_wgan, "wgan": self._hinge_and_wgan, "ls": self._ls, "bce": self._bce}.get(loss_type)
        if self.loss_fn is None:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Only 'hinge' 'wgan' 'ls' 'bce' are supported.")

    def _hinge_and_wgan(self, pred_score: Tensor) -> Tensor:
        return -pred_score

    def _ls(self, pred_score: Tensor) -> Tensor:
        return (pred_score - 1).pow(2)

    @torch.autocast(device_type="cuda", enabled=False)
    def _bce(self, pred_score: Tensor) -> Tensor:
        real_label = torch.ones_like(pred_score)
        return F.binary_cross_entropy_with_logits(pred_score, real_label)

    def forward(self, pred_score: Tensor) -> Tensor:

        loss = self.weight * self.loss_fn(pred_score)

        match self.reduction:
            case "none":
                return loss
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()


class IDLoss(nn.Module):

    Provider = PROVIDER

    def __init__(self, provider: Provider = Provider.MS1MV3_ARCFACE_R50_FP16, weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.idencoder = IDEncoder(provider=provider)

        self.eval()
        self.requires_grad_(False)

    @torch.compile(fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})
    def get_id_feats(self, face: Tensor) -> Tensor:
        return self.idencoder(face)

    def forward(self, fake_id: Tensor, real_id: Tensor) -> Tensor:
        loss = (1.0 - F.cosine_similarity(fake_id, real_id)) * self.weight

        match self.reduction:
            case "none":
                return loss
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()


class IFSRLoss(nn.Module):
    def __init__(self, ifsr_scale: float, ifsr_weight: Mapping[str, tuple[float, float]], idencoder_provider: PROVIDER = PROVIDER.MS1MV3_ARCFACE_R50_FP16):
        super().__init__()

        idencoder = IDEncoder(provider=idencoder_provider)
        self.feature_layer_indices: dict[int, str] = {}
        self.ifsr_weight = ifsr_weight

        for key, (margin, weight) in self.ifsr_weight.items():
            ifsr_weight[key] = (margin * ifsr_scale, weight)

        net = nn.ModuleList()

        ifsr_layer_name = list(ifsr_weight.keys())
        max_layer_idx = 0
        idx = 0
        for module_name_0, module_0 in idencoder.backbone.named_children():

            if not list(module_0.children()):
                net.append(module_0)
                idx += 1
                continue

            for module_name_1, module_1 in module_0.named_children():

                module_name = f"{module_name_0}.{module_name_1}"

                net.append(module_1)
                if module_name in ifsr_layer_name:
                    self.feature_layer_indices[idx] = module_name
                    max_layer_idx = max(max_layer_idx, idx)

                idx += 1

        self.net = net[: max_layer_idx + 1]

    @torch.compile(fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})
    def get_ifsr_feats(self, x: Tensor) -> dict[str, Tensor]:
        feates: dict[str, Tensor] = {}

        for idx, module in enumerate(self.net):
            x = module(x)

            if (layer_name := self.feature_layer_indices.get(idx)) is not None:
                feates[layer_name] = x

        return feates

    def forward(self, src_ifsr_feats: dict[str, Tensor], dst_ifsr_feats: dict[str, Tensor]):
        loss = 0.0
        for layer_name, (margin, weight) in self.ifsr_weight.items():

            feat_true_flat = src_ifsr_feats[layer_name].flatten(1)
            feat_pred_flat = dst_ifsr_feats[layer_name].flatten(1)
            distance = (1.0 - F.cosine_similarity(feat_true_flat, feat_pred_flat, dim=1)).mean()

            d_loss = F.relu(distance - margin) * weight
            loss += d_loss

        return loss


if __name__ == "__main__":
    import random
    from torchvision.io import read_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    losses = DINOv2PerceptualLoss(
        layer_weights={
            11: 1.0,  # 面部组件对齐（眼鼻口位置）
        },
        range_norm=False,
    ).to(device)

    x = read_image(f"/opt/share/deepfake/dataset_1/ffhq_1024/{random.randint(0,69999):05d}.png").to(device=device, dtype=torch.float).div(255.0).unsqueeze(0)
    y = read_image(f"/opt/share/deepfake/dataset_1/ffhq_1024/{random.randint(0,69999):05d}.png").to(device=device, dtype=torch.float).div(255.0).unsqueeze(0)
    # y = x
    loss: torch.Tensor = losses(x, y)

    print(loss.item())
