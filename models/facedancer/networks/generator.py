from enum import Enum
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class RBReSampleMode(Enum):
    NONE = "none"
    UPSAMPLE = "upsample"
    DOWNSAMPLE = "downsample"


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, resample_mode: RBReSampleMode):
        super().__init__()

        match resample_mode:
            case RBReSampleMode.UPSAMPLE:
                self.residual = nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )

            case RBReSampleMode.DOWNSAMPLE:
                self.residual = nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.AvgPool2d(2),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                    nn.AvgPool2d(2),
                )

            case RBReSampleMode.NONE:
                self.residual = nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                )
                if in_ch == out_ch:
                    self.shortcut = nn.Identity()
                else:
                    self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, style_dim: int, kernel: int = 3, demod: bool = True, eps: float = 1e-8, rank: int = 4, use_refinement: bool = False) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.demod = demod
        self.eps = eps
        self.use_refinement = use_refinement

        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel))
        self.style = nn.Linear(style_dim, in_ch)
        nn.init.normal_(self.style.weight, mean=0.0, std=1.0)
        nn.init.ones_(self.style.bias)

        if use_refinement:

            self.P = nn.Parameter(torch.randn(rank, out_ch) * 0.01)
            self.Q = nn.Parameter(torch.randn(rank, in_ch * self.kernel * self.kernel) * 0.001)

            self.alpha_proj = nn.Linear(style_dim, rank)
            self.beta_proj = nn.Linear(style_dim, rank)

            nn.init.zeros_(self.alpha_proj.weight)
            nn.init.zeros_(self.alpha_proj.bias)
            nn.init.zeros_(self.beta_proj.weight)
            nn.init.zeros_(self.beta_proj.bias)

    def _compute_low_rank_residual(self, w: Tensor) -> Tensor:

        # w: (B, style_dim)
        # a = torch.tanh(self.alpha_proj(w))  # (B, rank)
        # b = torch.tanh(self.beta_proj(w))  # (B, rank)
        a = self.alpha_proj(w)  # (B, rank)
        b = self.beta_proj(w)  # (B, rank)

        P_s = a.unsqueeze(2) * self.P.unsqueeze(0)  # (B, rank, out_ch)
        Q_s = b.unsqueeze(2) * self.Q.unsqueeze(0)  # (B, rank, in * k * k)

        delta = torch.bmm(P_s.transpose(1, 2), Q_s)  # (B, out, in * k * k)
        delta = delta.view(-1, self.out_ch, self.in_ch, self.kernel, self.kernel)

        return delta

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        B, C, H, W = x.shape

        s: Tensor = self.style(w).view(B, 1, C, 1, 1)
        weight = self.weight * s  # (B, out_ch, in_ch, k, k)

        if self.use_refinement:
            delta = self._compute_low_rank_residual(w)  # (B, out_ch, in_ch, 1, 1)
            weight = weight + delta

        if self.demod:
            d = torch.rsqrt(weight.pow(2).sum((2, 3, 4)) + self.eps)
            weight = weight * d.view(B, self.out_ch, 1, 1, 1)

        x = x.view(1, B * C, H, W)
        weight = weight.view(B * self.out_ch, self.in_ch, self.kernel, self.kernel)

        out = F.conv2d(x, weight, padding=self.kernel // 2, groups=B).view(B, self.out_ch, H, W)
        return out


class ModConvRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample_mode: RBReSampleMode, use_refinement: bool = False):
        super().__init__()

        self.modconv = ModulatedConv2d(in_ch, out_ch, w_dim, use_refinement=use_refinement)
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.act = nn.LeakyReLU(0.2)

        match resample_mode:
            case RBReSampleMode.UPSAMPLE:
                self.resample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )

            case RBReSampleMode.DOWNSAMPLE:
                self.resample = nn.AvgPool2d(2)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                    nn.AvgPool2d(2),
                )

            case RBReSampleMode.NONE:
                self.resample = nn.Identity()
                self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        skip = self.shortcut(x)

        x = self.modconv(x, w)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.act(x)
        x = self.resample(x)

        return x + skip


class AdaIN(nn.Module):
    def __init__(self, num_channels: int, w_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(num_channels, affine=False)

        self.fc_gamma = nn.Linear(w_dim, num_channels)
        self.fc_beta = nn.Linear(w_dim, num_channels)

        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        x = self.norm(x)

        gamma = self.fc_gamma(w).view(w.size(0), -1, 1, 1)
        beta = self.fc_beta(w).view(w.size(0), -1, 1, 1)

        return x.mul(gamma).add(beta)


class NormRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, resample_mode: RBReSampleMode) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_ch, affine=False)
        self.resblock = ResBlock(in_ch, out_ch, resample_mode)

    def forward(self, x: Tensor) -> Tensor:

        skip = self.resblock.shortcut(x)

        x = self.norm(x)
        x = self.resblock.residual(x)

        return x + skip


class AdaInRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample_mode: RBReSampleMode) -> None:
        super().__init__()

        self.adain = AdaIN(in_ch, w_dim)
        self.resblock = ResBlock(in_ch, out_ch, resample_mode)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        skip = self.resblock.shortcut(x)

        x = self.adain(x, w)
        x = self.resblock.residual(x)

        return x + skip


class FusionMode(Enum):
    CONCAT = "concat"
    ATTN = "attn"


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()

        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class Attn(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.attn_mask_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 3, padding=1),
            nn.InstanceNorm2d(channels // 4, affine=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 4, channels, 1, padding=0),
            nn.Sigmoid(),
        )

        self.last_attn_mask = None

    def forward(self, x_target: Tensor, x_source: Tensor) -> Tensor:

        m = torch.cat([x_target, x_source], dim=1)
        m = self.attn_mask_proj(m)

        self.last_attn_mask = m.detach().requires_grad_(False)

        return x_target + m * (x_source - x_target)


class SkipFusionModConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample_mode: RBReSampleMode, fusion_mode: FusionMode, use_refinement: bool = False) -> None:
        super().__init__()

        self.fusion = {FusionMode.CONCAT: Concat(1), FusionMode.ATTN: Attn(in_ch)}.get(fusion_mode)
        if self.fusion is None:
            raise KeyError(f"Unsupported fusion mode: {fusion_mode}")

        in_ch = in_ch * 2 if fusion_mode in [FusionMode.CONCAT] else in_ch

        self.resblock = ModConvRB(in_ch, out_ch, w_dim, resample_mode, use_refinement)

    def forward(self, x_target: Tensor, x_source: Tensor, w: Tensor) -> Tensor:

        x = self.fusion(x_target, x_source)
        x = self.resblock(x, w)

        return x


class SkipFusionAdaIN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample_mode: RBReSampleMode, fusion_mode: FusionMode) -> None:
        super().__init__()

        self.fusion = {FusionMode.CONCAT: Concat(1), FusionMode.ATTN: Attn(in_ch)}.get(fusion_mode)
        if self.fusion is None:
            raise KeyError(f"Unsupported fusion mode: {fusion_mode}")

        in_ch = in_ch * 2 if fusion_mode in [FusionMode.CONCAT] else in_ch

        self.adain = AdaIN(in_ch, w_dim)
        self.resblock = ResBlock(in_ch, out_ch, resample_mode)

    def forward(self, x_target: Tensor, x_source: Tensor, w: Tensor) -> Tensor:

        x = self.fusion(x_target, x_source)
        skip = self.resblock.shortcut(x)

        x = self.adain(x, w)
        x = self.resblock.residual(x)

        return x + skip


class Generator(nn.Module):
    def __init__(
        self, input_res: int = 256, num_encoder: int = 5, base_ch: int = 64, max_ch: int = 512, mapping_size: int = 512, z_dim: int = 512, skip_conn_start_with: int = 2
    ) -> None:
        super().__init__()

        assert 0 < skip_conn_start_with <= num_encoder, f"skip_conn_start_with must be in (0, {num_encoder}]"

        self.network_cfg = {
            "input_res": input_res,
            "num_encoder": num_encoder,
            "base_ch": base_ch,
            "max_ch": max_ch,
            "mapping_size": mapping_size,
            "z_dim": z_dim,
            "skip_conn_start_with": skip_conn_start_with,
        }

        self.mapping = nn.Sequential(
            nn.Linear(z_dim, mapping_size * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_size * 2, mapping_size * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_size * 2, mapping_size * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_size * 2, mapping_size),
            nn.LeakyReLU(0.2),
        )

        self.stem = nn.Conv2d(3, base_ch, 3, padding=1)

        self.encoder = nn.ModuleList()
        ch_pairs = []
        down_in_ch = base_ch
        for _ in range(num_encoder):
            down_out_ch = min(down_in_ch * 2, max_ch)
            ch_pairs.append((down_in_ch, down_out_ch))
            self.encoder.append(NormRB(down_in_ch, down_out_ch, RBReSampleMode.DOWNSAMPLE))
            down_in_ch = down_out_ch

        self.bottleneck_in = NormRB(down_out_ch, down_out_ch, RBReSampleMode.NONE)
        # self.bottleneck_out = AdaInRB(down_out_ch, down_out_ch, mapping_size, RBReSampleMode.NONE)
        self.bottleneck_out = ModConvRB(down_out_ch, down_out_ch, mapping_size, RBReSampleMode.NONE)

        self.decoder = nn.ModuleList()
        for i, (up_out_ch, up_in_ch) in enumerate(reversed(ch_pairs)):
            if i < skip_conn_start_with:
                # decoder_layer = AdaInRB(up_in_ch, up_out_ch, mapping_size, RBReSampleMode.UPSAMPLE)
                decoder_layer = ModConvRB(up_in_ch, up_out_ch, mapping_size, RBReSampleMode.UPSAMPLE)
            else:
                # decoder_layer = SkipFusionAdaIN(up_in_ch, up_out_ch, mapping_size, RBReSampleMode.UPSAMPLE, FusionMode.ATTN)
                decoder_layer = SkipFusionModConv(up_in_ch, up_out_ch, mapping_size, RBReSampleMode.UPSAMPLE, FusionMode.ATTN, True)
            self.decoder.append(decoder_layer)

        # self.to_rgb = SkipFusionAdaIN(base_ch, 3, mapping_size, RBReSampleMode.NONE, FusionMode.CONCAT)
        self.to_rgb = SkipFusionModConv(base_ch, 3, mapping_size, RBReSampleMode.NONE, FusionMode.CONCAT)

        num_w_layers = len(self.decoder) + 2
        self.w_affines = nn.ModuleList([nn.Linear(mapping_size, mapping_size) for _ in range(num_w_layers)])

        for affine in self.w_affines:
            nn.init.eye_(affine.weight)
            nn.init.zeros_(affine.bias)

    def get_attention_maps(self):
        attention_maps = []
        for module in self.decoder.modules():
            if isinstance(module, Attn):
                attention_maps.append(module.last_attn_mask)
        return attention_maps

    def mapping_z_source(self, z_source: Tensor) -> list[Tensor]:
        w = self.mapping(z_source)
        w_all = [affine(w) for affine in self.w_affines]

        return w_all

    def forward(self, x_target: Tensor, z_source: Tensor, w_all: Tensor | None = None) -> tuple[Tensor, Tensor]:

        if w_all is None:
            w_all = self.mapping_z_source(z_source)

        feats: list[Tensor] = []

        x = self.stem(x_target)
        feats.append(x)

        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            feats.append(x)

        x = self.bottleneck_in(x)
        x = self.bottleneck_out(x, w_all[0])

        for i, up_block in enumerate(self.decoder):
            w_i = w_all[i + 1]
            if isinstance(up_block, (SkipFusionAdaIN, SkipFusionModConv)):
                x = up_block(feats[-(i + 1)], x, w_i)
            else:
                x = up_block(x, w_i)

        x = self.to_rgb(feats[0], x, w_all[-1])

        x = torch.tanh(x)

        return x


if __name__ == "__main__":

    import torch
    from fvcore.nn import FlopCountAnalysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    img_size = 256
    num_encoder = 5
    mapping_size = 512
    z_dim = 512

    model = Generator(input_res=img_size, num_encoder=num_encoder, base_ch=64, max_ch=512, mapping_size=mapping_size, z_dim=z_dim).to(device)

    model.eval()

    x_target = torch.randn(batch_size, 3, img_size, img_size, device=device)
    z_source = torch.randn(batch_size, z_dim, device=device)

    flops = FlopCountAnalysis(model, (x_target, z_source))

    total_flops = flops.total()
    total_params = sum(p.numel() for p in model.parameters())

    print("\n=== Model Profile ===")
    labels = [
        ("Batch size", batch_size),
        ("Network Info", model.network_cfg),
        ("Params", f"{total_params/1e6:.3f} M"),
        ("FLOPs (total)", f"{total_flops/1e9:.3f} GFLOPs"),
    ]
    max_label_len = max(len(label) for label, _ in labels)
    for label, value in labels:
        print(f"{label:<{max_label_len}} : {value}")
    print()
