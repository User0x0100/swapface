import torch
from torch import Tensor
import torch.nn as nn
from .layers import ResBlock, RBResampleMode


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

        gamma: Tensor = self.fc_gamma(w)
        beta: Tensor = self.fc_beta(w)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x * gamma + beta


class NormRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, resample_mode: RBResampleMode) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_ch, affine=False)
        self.resblock = ResBlock(in_ch, out_ch, resample_mode)

    def forward(self, x: Tensor) -> Tensor:

        skip = self.resblock.shortcut(x)

        x = self.norm(x)
        x = self.resblock.residual(x)

        return x + skip


class AdaInRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample_mode: RBResampleMode) -> None:
        super().__init__()

        self.adain = AdaIN(in_ch, w_dim)
        self.resblock = ResBlock(in_ch, out_ch, resample_mode)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        skip = self.resblock.shortcut(x)

        x = self.adain(x, w)
        x = self.resblock.residual(x)

        return x + skip


class AdaptiveAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.attn_mask_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 3, padding=1),
            nn.SiLU(),
            nn.InstanceNorm2d(channels // 4, affine=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

        self.last_attn_mask = None

    def forward(self, x_t: Tensor, x_s: Tensor) -> Tensor:

        m: Tensor
        m = torch.cat([x_t, x_s], dim=1)
        m = self.attn_mask_proj(m)

        self.last_attn_mask = m.detach().requires_grad_(False)

        return (1.0 - m) * x_t + m * x_s


class AFFARB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample_mode: RBResampleMode) -> None:
        super().__init__()

        self.adain = AdaIN(in_ch, w_dim)
        self.resblock = ResBlock(in_ch, out_ch, resample_mode)
        self.attn = AdaptiveAttentionBlock(in_ch)

    def forward(self, x_t: Tensor, x_s: Tensor, w: Tensor) -> Tensor:

        x = self.attn(x_t, x_s)
        skip = self.resblock.shortcut(x)

        x = self.adain(x, w)
        x = self.resblock.residual(x)

        return x + skip


class ConcatRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample_mode: RBResampleMode) -> None:
        super().__init__()

        in_ch = in_ch * 2

        self.adain = AdaIN(in_ch, w_dim)
        self.resblock = ResBlock(in_ch, out_ch, resample_mode)

    def forward(self, x_t: Tensor, x_s: Tensor, w: Tensor) -> Tensor:

        x = torch.cat([x_t, x_s], dim=1)
        skip = self.resblock.shortcut(x)

        x = self.adain(x, w)
        x = self.resblock.residual(x)

        return x + skip


class Generator(nn.Module):
    def __init__(
        self, input_res: int = 256, bottleneck_res: int = 8, base_ch: int = 64, max_ch: int = 512, mapping_depth: int = 4, mapping_size: int = 256, z_dim: int = 512
    ) -> None:
        super().__init__()

        num_encoder = (input_res // bottleneck_res).bit_length() - 1

        self.network_cfg = {
            "input_res": input_res,
            "bottleneck_res": bottleneck_res,
            "base_ch": base_ch,
            "max_ch": max_ch,
            "mapping_depth": mapping_depth,
            "mapping_size": mapping_size,
            "z_dim": z_dim,
        }

        mapping_layers = []
        for _ in range(mapping_depth - 1):
            mapping_layers += [nn.Linear(z_dim, z_dim), nn.SiLU()]
        mapping_layers.append(nn.Linear(z_dim, mapping_size))
        self.mapping = nn.Sequential(*mapping_layers)

        self.conv_in = nn.Conv2d(3, base_ch, 3, padding=1)

        self.encoder = nn.ModuleList()
        ch_pairs = []
        down_in_ch = base_ch
        for _ in range(num_encoder):
            down_out_ch = min(down_in_ch * 2, max_ch)
            ch_pairs.append((down_in_ch, down_out_ch))
            self.encoder.append(NormRB(down_in_ch, down_out_ch, RBResampleMode.DOWNSAMPLE))
            down_in_ch = down_out_ch

        self.encoder.append(NormRB(down_out_ch, down_out_ch, RBResampleMode.NONE))

        self.decoder_first = AdaInRB(down_out_ch, down_out_ch, mapping_size, RBResampleMode.NONE)
        reversed_ch_pairs = [(out_ch, in_ch) for in_ch, out_ch in reversed(ch_pairs)]

        self.decoder = nn.ModuleList()
        for i, (up_in_ch, up_out_ch) in enumerate(reversed_ch_pairs):
            if i < 2:
                self.decoder.append(AdaInRB(up_in_ch, up_out_ch, mapping_size, RBResampleMode.UPSAMPLE))
            else:
                self.decoder.append(AFFARB(up_in_ch, up_out_ch, mapping_size, RBResampleMode.UPSAMPLE))

        _, up_in_ch = reversed_ch_pairs[-1]
        self.decoder.append(ConcatRB(up_in_ch, 3, mapping_size, RBResampleMode.NONE))

    def get_attention_maps(self):
        attention_maps = []
        for module in self.decoder:
            if isinstance(module, AFFARB):
                attention_maps.append(module.attn.last_attn_mask)
        return attention_maps

    def forward(self, x_target: Tensor, z_source: Tensor) -> Tensor:

        w = self.mapping(z_source)

        feats = []
        x = self.conv_in(x_target)
        feats.append(x)

        for down_block in self.encoder:
            x = down_block(x)
            feats.append(x)

        x = self.decoder_first(feats[-1], w)

        for i, up_block in enumerate(self.decoder):
            if isinstance(up_block, AFFARB) or isinstance(up_block, ConcatRB):
                encoder_feat = feats[-(i + 2)]
                x = up_block(encoder_feat, x, w)
            else:
                x = up_block(x, w)

        return x


if __name__ == "__main__":

    import torch
    from fvcore.nn import FlopCountAnalysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    img_size = 256
    z_dim = 512

    model = Generator(img_size).to(device)

    model.eval()

    x_target = torch.randn(batch_size, 3, img_size, img_size, device=device)
    z_source = torch.randn(batch_size, z_dim, device=device)

    flops = FlopCountAnalysis(model, (x_target, z_source))

    total_flops = flops.total()
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n=== Model Profile ===")
    print(f"Device        : {device}")
    print(f"Batch size    : {batch_size}")
    print(f"Image size    : {img_size}x{img_size}")
    print(f"Latent dim    : {z_dim}")
    print(f"Params        : {total_params/1e6:.3f} M")
    print(f"FLOPs (total)  : {total_flops/1e9:.3f} GFLOPs\n")
