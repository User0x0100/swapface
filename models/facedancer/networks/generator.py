from enum import Enum
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class RBSampleMode(Enum):
    UP = "Up"
    DOWN = "Down"
    NONE = "None"


class ResBlockBase(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sampling: RBSampleMode, shortcut_bias: bool = False) -> None:
        super().__init__()

        match sampling:
            case RBSampleMode.UP:
                self.residual = nn.Sequential(
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=shortcut_bias),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )

            case RBSampleMode.DOWN:
                self.residual = nn.Sequential(
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.AvgPool2d(2),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=shortcut_bias),
                    nn.AvgPool2d(2),
                )

            case RBSampleMode.NONE:
                self.residual = nn.Sequential(
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                )
                if in_ch == out_ch:
                    self.shortcut = nn.Identity()
                else:
                    self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=shortcut_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, style_dim: int, kernel: int = 3, demod: bool = True, eps: float = 1e-8, rank: int = 4, en_refinement: bool = False) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.demod = demod
        self.eps = eps
        self.en_refinement = en_refinement

        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel))
        self.style = nn.Linear(style_dim, in_ch)
        nn.init.normal_(self.style.weight, mean=0.0, std=1.0)
        nn.init.ones_(self.style.bias)

        if en_refinement:
            self.P = nn.Parameter(torch.randn(rank, out_ch) * 0.01)
            self.Q = nn.Parameter(torch.randn(rank, in_ch * self.kernel * self.kernel) * 0.01)

            self.alpha_proj = nn.Linear(style_dim, rank)
            self.beta_proj = nn.Linear(style_dim, rank)

            nn.init.zeros_(self.alpha_proj.weight)
            nn.init.zeros_(self.alpha_proj.bias)
            nn.init.zeros_(self.beta_proj.weight)
            nn.init.zeros_(self.beta_proj.bias)

    def _compute_low_rank_residual(self, w: Tensor) -> Tensor:

        # a = torch.tanh(self.alpha_proj(w))
        # b = torch.tanh(self.beta_proj(w))
        a = self.alpha_proj(w)
        b = self.beta_proj(w)

        P_s = a.unsqueeze(2) * self.P.unsqueeze(0)
        Q_s = b.unsqueeze(2) * self.Q.unsqueeze(0)

        delta = torch.bmm(P_s.transpose(1, 2), Q_s)
        delta = delta.view(-1, self.out_ch, self.in_ch, self.kernel, self.kernel)

        return delta

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        B, C, H, W = x.shape

        s: Tensor = self.style(w).view(B, 1, C, 1, 1)
        weight = self.weight * s

        if self.en_refinement:
            delta = self._compute_low_rank_residual(w)
            weight = weight + delta

        if self.demod:
            d = torch.rsqrt(weight.pow(2).sum((2, 3, 4)) + self.eps)
            weight = weight * d.view(B, self.out_ch, 1, 1, 1)

        x = x.view(1, B * C, H, W)
        weight = weight.view(B * self.out_ch, self.in_ch, self.kernel, self.kernel)
        out = F.conv2d(x, weight, padding=self.kernel // 2, groups=B).view(B, self.out_ch, H, W)
        return out


class ModulatedConv2dRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, sampling: RBSampleMode, en_refinement: bool = False, shortcut_bias: bool = True) -> None:
        super().__init__()

        self.modconv = ModulatedConv2d(in_ch, out_ch, w_dim, en_refinement=en_refinement)
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.act = nn.SiLU()

        match sampling:
            case RBSampleMode.UP:
                self.resample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=shortcut_bias),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )

            case RBSampleMode.DOWN:
                self.resample = nn.AvgPool2d(2)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=shortcut_bias),
                    nn.AvgPool2d(2),
                )

            case RBSampleMode.NONE:
                self.resample = nn.Identity()
                self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=shortcut_bias)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        skip = self.shortcut(x)

        x = self.modconv(x, w)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.act(x)
        x = self.resample(x)

        return x + skip


class AdaIN(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False)

        self.fc_gamma = nn.Linear(w_dim, channels)
        self.fc_beta = nn.Linear(w_dim, channels)

        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        x = self.norm(x)

        gamma = self.fc_gamma(w).view(w.size(0), -1, 1, 1)
        beta = self.fc_beta(w).view(w.size(0), -1, 1, 1)

        return x.mul(gamma).add(beta)


class AdaINRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, sampling: RBSampleMode) -> None:
        super().__init__()

        self.adain = AdaIN(in_ch, w_dim)
        self.resblock = ResBlockBase(in_ch, out_ch, sampling)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        skip = self.resblock.shortcut(x)

        x = self.adain(x, w)
        x = self.resblock.residual(x)

        return x + skip


class CrossAdaIN(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False)

        self.fc_gamma = nn.Linear(w_dim, channels)
        self.fc_beta = nn.Linear(w_dim, channels)

        self.t_gamma = nn.Conv2d(channels, channels, 1)
        self.t_beta = nn.Conv2d(channels, channels, 1)

        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

        nn.init.zeros_(self.t_gamma.weight)
        nn.init.zeros_(self.t_gamma.bias)
        nn.init.zeros_(self.t_beta.weight)
        nn.init.zeros_(self.t_beta.bias)

    def forward(self, x_target: Tensor, x_source: Tensor, w: Tensor) -> Tensor:

        x_source = self.norm(x_source)

        gamma = self.fc_gamma(w).view(w.size(0), -1, 1, 1)
        beta = self.fc_beta(w).view(w.size(0), -1, 1, 1)

        t_gamma = torch.tanh(self.t_gamma(x_target))
        t_beta = self.t_beta(x_target)

        gamma = gamma * (1.0 + t_gamma)
        beta = beta + t_beta

        return x_source * gamma + beta


class CrossAdaINRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, sampling: RBSampleMode) -> None:
        super().__init__()

        self.id_inject = CrossAdaIN(in_ch, w_dim)
        self.resblock = ResBlockBase(in_ch, out_ch, sampling)

    def forward(self, x_target: Tensor, x_source: Tensor, w: Tensor):

        skip = self.resblock.shortcut(x_source)

        x_source = self.id_inject(x_target, x_source, w)
        x_source = self.resblock.residual(x_source)

        return x_source + skip


class InjectModule(Enum):
    ADAIN = "AdaIN"
    MODCONV = "ModConv"
    CROSSADAIN = "CrossAdaIN"


class NormType(Enum):
    NONE = "None"
    IN = "InstanceNorm2d"
    GN = "GroupNorm"


class NormRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sampling: RBSampleMode, norm: NormType = NormType.IN) -> None:
        super().__init__()

        self.norm = {
            NormType.NONE: nn.Identity,
            NormType.IN: lambda: nn.InstanceNorm2d(in_ch, affine=False),
            NormType.GN: lambda: nn.GroupNorm(max(1, min(32, in_ch // 4)), in_ch),
        }[norm]()

        shortcut_bias = True if norm is NormType.NONE else False

        self.resblock = ResBlockBase(in_ch, out_ch, sampling, shortcut_bias)

    def forward(self, x: Tensor) -> Tensor:

        skip = self.resblock.shortcut(x)

        x = self.norm(x)
        x = self.resblock.residual(x)

        return x + skip


class Add(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x_target: Tensor, x_source: Tensor) -> Tensor:
        w = self.weight.view(1, -1, 1, 1)
        return x_target + w * x_source


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_target: Tensor, x_source: Tensor) -> Tensor:
        return torch.cat((x_target, x_source), dim=1)


class Atten(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.attn_mask_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 3, padding=1),
            nn.InstanceNorm2d(channels // 4, affine=False),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, 1, padding=0),
            nn.Sigmoid(),
        )

        self.last_attn_mask = None

    def forward(self, x_target: Tensor, x_source: Tensor) -> Tensor:

        m = self.attn_mask_proj(torch.cat((x_target, x_source), dim=1))

        self.last_attn_mask = m.detach()

        return x_target + m * (x_source - x_target)

    def get_attention_maps(self) -> Tensor | None:
        return self.last_attn_mask


class SkipFusionModule(Enum):
    ADD = "Add"
    ATTEN = "Atten"
    CONCAT = "Concat"


class SkipFusionModConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, sampling: RBSampleMode, fusion_mode: SkipFusionModule, en_refinement: bool = False) -> None:
        super().__init__()

        self.fusion = {
            SkipFusionModule.CONCAT: Concat,
            SkipFusionModule.ATTEN: lambda: Atten(in_ch),
            SkipFusionModule.ADD: lambda: Add(in_ch),
        }[fusion_mode]()

        if isinstance(self.fusion, Concat):
            in_ch = in_ch * 2

        self.resblock = ModulatedConv2dRB(in_ch, out_ch, w_dim, sampling, en_refinement)

    def forward(self, x_target: Tensor, x_source: Tensor, w: Tensor) -> Tensor:

        x = self.fusion(x_target, x_source)
        x = self.resblock(x, w)

        return x


class SkipFusionAdaIN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample_mode: RBSampleMode, fusion_mode: SkipFusionModule) -> None:
        super().__init__()

        self.fusion = {
            SkipFusionModule.CONCAT: Concat,
            SkipFusionModule.ATTEN: lambda: Atten(in_ch),
            SkipFusionModule.ADD: lambda: Add(in_ch),
        }[fusion_mode]()

        if isinstance(self.fusion, Concat):
            in_ch = in_ch * 2

        self.adain = AdaIN(in_ch, w_dim)
        self.resblock = ResBlockBase(in_ch, out_ch, resample_mode)

    def forward(self, x_target: Tensor, x_source: Tensor, w: Tensor) -> Tensor:

        x = self.fusion(x_target, x_source)
        skip = self.resblock.shortcut(x)

        x = self.adain(x, w)
        x = self.resblock.residual(x)

        return x + skip


class Bottleneck(Enum):
    NormRB = "NormRB"
    AdaINRB = "AdaINRB"
    ModConvRB = "ModConvRB"


class Generator(nn.Module):
    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        num_encoder: int = 5,
        base_ch: int = 64,
        max_ch: int = 512,
        id_dim: int = 512,
        w_dim: int = 512,
        mapping_num: int = 4,
        encode_norm: NormType = NormType.IN,
        skip_index: int = 2,
        id_inject_index: int = 0,
        id_inject_mode: InjectModule = InjectModule.ADAIN,
        bottleneck: Bottleneck = Bottleneck.AdaINRB,
        encode_skip_fusion_mode: SkipFusionModule = SkipFusionModule.ATTEN,
        en_refinement_mask: int = 0,
        to_rgb_skip_fusion_mode: SkipFusionModule = SkipFusionModule.CONCAT,
    ) -> None:
        super().__init__()

        assert 0 < skip_index <= num_encoder, f"skip_conn_start_with must be in (0, {num_encoder}]"

        self.network_cfg = {
            "img_resolution": img_resolution,
            "img_channels": img_channels,
            "num_encoder": num_encoder,
            "base_ch": base_ch,
            "max_ch": max_ch,
            "id_dim": id_dim,
            "w_dim": w_dim,
            "mapping_num": mapping_num,
            "encode_norm": encode_norm,
            "skip_index": skip_index,
            "id_inject_index": id_inject_index,
            "id_inject_mode": id_inject_mode,
            "bottleneck": bottleneck,
            "encode_skip_fusion_mode": encode_skip_fusion_mode,
            "en_refinement_mask": en_refinement_mask,
            "to_rgb_skip_fusion_mode": to_rgb_skip_fusion_mode,
        }

        mapping_layers = []
        for _ in range(mapping_num - 1):
            mapping_layers += [nn.Linear(id_dim, id_dim), nn.SiLU()]
        mapping_layers += [nn.Linear(id_dim, w_dim)]
        self.mapping = nn.Sequential(*mapping_layers)

        self.stem = nn.Conv2d(img_channels, base_ch, 3, padding=1)

        self.encoder = nn.ModuleList()
        ch_pairs = []
        down_in_ch = base_ch
        for _ in range(num_encoder):
            down_out_ch = min(down_in_ch * 2, max_ch)
            ch_pairs.append((down_in_ch, down_out_ch))
            self.encoder.append(NormRB(down_in_ch, down_out_ch, RBSampleMode.DOWN, encode_norm))
            down_in_ch = down_out_ch

        self.bottleneck_encode = NormRB(down_out_ch, down_out_ch, RBSampleMode.NONE, encode_norm)

        match bottleneck:
            case Bottleneck.NormRB:
                self.bottleneck_decode = NormRB(down_out_ch, down_out_ch, RBSampleMode.NONE, encode_norm)
            case Bottleneck.AdaINRB:
                self.bottleneck_decode = AdaINRB(down_out_ch, down_out_ch, w_dim, RBSampleMode.NONE)
            case Bottleneck.ModConvRB:
                self.bottleneck_decode = ModulatedConv2dRB(down_out_ch, down_out_ch, w_dim, RBSampleMode.NONE, bool(en_refinement_mask & (1 << 0)))

        self.decoder = nn.ModuleList()
        for i, (up_out_ch, up_in_ch) in enumerate(reversed(ch_pairs)):
            en_refinement_flag = bool(en_refinement_mask & (1 << (i + 1)))
            if i < id_inject_index:
                decoder_layer = NormRB(up_in_ch, up_out_ch, RBSampleMode.UP, encode_norm)
            elif i < skip_index:
                match id_inject_mode:
                    case InjectModule.ADAIN:
                        decoder_layer = AdaINRB(up_in_ch, up_out_ch, w_dim, RBSampleMode.UP)
                    case InjectModule.MODCONV:
                        decoder_layer = ModulatedConv2dRB(up_in_ch, up_out_ch, w_dim, RBSampleMode.UP, en_refinement_flag)
                    case InjectModule.CROSSADAIN:
                        decoder_layer = NormRB(up_in_ch, up_out_ch, RBSampleMode.UP, encode_norm)
            else:
                match id_inject_mode:
                    case InjectModule.ADAIN:
                        decoder_layer = SkipFusionAdaIN(up_in_ch, up_out_ch, w_dim, RBSampleMode.UP, encode_skip_fusion_mode)
                    case InjectModule.MODCONV:
                        decoder_layer = SkipFusionModConv(up_in_ch, up_out_ch, w_dim, RBSampleMode.UP, encode_skip_fusion_mode, en_refinement_flag)
                    case InjectModule.CROSSADAIN:
                        decoder_layer = CrossAdaINRB(up_in_ch, up_out_ch, w_dim, RBSampleMode.UP)

            self.decoder.append(decoder_layer)

        match id_inject_mode:
            case InjectModule.ADAIN | InjectModule.CROSSADAIN:
                self.to_rgb = SkipFusionAdaIN(base_ch, img_channels, w_dim, RBSampleMode.NONE, to_rgb_skip_fusion_mode)
            case InjectModule.MODCONV:
                self.to_rgb = SkipFusionModConv(base_ch, img_channels, w_dim, RBSampleMode.NONE, to_rgb_skip_fusion_mode, bool(en_refinement_mask & (1 << num_encoder + 1)))

        self.print_refinement_layers()

    def get_attention_maps(self) -> list[Tensor]:
        return [maps for module in self.decoder.modules() if isinstance(module, Atten) if (maps := module.get_attention_maps()) is not None]

    def print_refinement_layers(self):
        print("=== Enabled Refinement Layers ===")
        for module_name, module in self.named_modules():
            if isinstance(module, ModulatedConv2d):
                print(f"{module_name:30} en_refinement = {module.en_refinement}")

    def forward(self, x_target: Tensor, id_feat: Tensor) -> Tensor:

        w = self.mapping(id_feat)

        feats: list[Tensor] = []

        x = self.stem(x_target)
        feats.append(x)

        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            feats.append(x)

        x = self.bottleneck_encode(x)
        x = self.bottleneck_decode(x) if isinstance(self.bottleneck_decode, NormRB) else self.bottleneck_decode(x, w)

        for i, decoder_block in enumerate(self.decoder):
            if isinstance(decoder_block, (SkipFusionAdaIN, SkipFusionModConv, CrossAdaINRB)):
                x = decoder_block(feats[-(i + 1)], x, w)
            elif isinstance(decoder_block, NormRB):
                x = decoder_block(x)
            else:
                x = decoder_block(x, w)

        x = self.to_rgb(feats[0], x, w)

        x = torch.tanh(x)

        return x


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    network_cfg = {
        "img_resolution": 256,
        "img_channels": 3,
        "num_encoder": 5,
        "base_ch": 64,
        "max_ch": 512,
        "id_dim": 512,
        "w_dim": 512,
        "mapping_num": 4,
        "encode_norm": NormType.GN,
        "skip_index": 2,
        "id_inject_index": 0,
        "id_inject_mode": InjectModule.MODCONV,
        "bottleneck": Bottleneck.NormRB,
        "encode_skip_fusion_mode": SkipFusionModule.ATTEN,
        "en_refinement_mask": 0b0000000,
        "to_rgb_skip_fusion_mode": SkipFusionModule.CONCAT,
    }

    model = Generator(**network_cfg).to(device)
    model.eval()

    x_target = torch.randn((batch_size, network_cfg["img_channels"], network_cfg["img_resolution"], network_cfg["img_resolution"]), device=device)
    id_feat = torch.randn((batch_size, network_cfg["id_dim"]), device=device)
    summary(
        model,
        input_data=(x_target, id_feat),
        depth=2,
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ),
        row_settings=("var_names",),
    )

    print("NetWork_Info:")
    for k, v in network_cfg.items():
        print(f"  {k:25}: {v}")
