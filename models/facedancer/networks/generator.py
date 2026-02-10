import torch
from torch import Tensor
import torch.nn as nn


def get_act() -> nn.Module:
    # act = nn.LeakyReLU(0.2)
    act = nn.SiLU()
    return act


BASE_CH = 64
MAX_CH = 512


class AdaIN(nn.Module):
    def __init__(self, num_channels: int, w_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(num_channels, affine=False)

        self.fc_gamma = nn.Linear(w_dim, num_channels)
        self.fc_beta = nn.Linear(w_dim, num_channels)

        nn.init.xavier_uniform_(self.fc_gamma.weight)
        nn.init.xavier_uniform_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        x = self.norm(x)

        gamma: Tensor = self.fc_gamma(w)
        beta: Tensor = self.fc_beta(w)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        

        return x * (gamma + 1.0) + beta


class ResidualDownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, resample: bool = True) -> None:
        super().__init__()

        self.residual = nn.Sequential(
            nn.InstanceNorm2d(in_ch, affine=True),
            get_act(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            *([nn.AvgPool2d(2)] if resample else []),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            *([nn.AvgPool2d(2)] if resample else []),
        )

    def forward(self, x: Tensor):
        return self.residual(x) + self.shortcut(x)


class ResidualUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample: bool = True) -> None:
        super().__init__()

        self.adain = AdaIN(in_ch, w_dim)
        self.residual = nn.Sequential(
            get_act(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            *([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)] if resample else []),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            *([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)] if resample else []),
        )

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        skip = self.shortcut(x)

        x = self.adain(x, w)
        x = self.residual(x)

        return x + skip


class AdaptiveAttention(nn.Module):
    def forward(self, m: Tensor, a: Tensor, i: Tensor) -> Tensor:
        return (1.0 - m) * a + m * i


class AdaptiveAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.attn_mask_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 3, padding=1),
            get_act(),
            nn.InstanceNorm2d(channels // 4, affine=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

        self.attn = AdaptiveAttention()
        self.last_attn_mask = None

    def forward(self, x_t: Tensor, x_s: Tensor) -> Tensor:

        m: Tensor

        m = torch.cat([x_t, x_s], dim=1)

        m = self.attn_mask_proj(m)

        self.last_attn_mask = m.detach().requires_grad_(False)

        m = self.attn(m, x_t, x_s)

        return m


class AdaptiveFeatureFusionUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample: bool = True) -> None:
        super().__init__()

        self.adain = AdaIN(in_ch, w_dim)
        self.residual = nn.Sequential(
            get_act(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            *([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)] if resample else []),
        )

        self.attn = AdaptiveAttentionBlock(in_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            *([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)] if resample else []),
        )

    def forward(self, x_t: Tensor, x_s: Tensor, w: Tensor) -> Tensor:
        x = self.attn(x_t, x_s)

        skip = self.shortcut(x)

        x = self.adain(x, w)
        x = self.residual(x)

        return x + skip


class ConcatUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample: bool = True) -> None:
        super().__init__()

        self.resample = resample

        self.adain = AdaIN(in_ch * 2, w_dim)
        self.residual = nn.Sequential(
            get_act(),
            nn.Conv2d(in_ch * 2, out_ch, 3, padding=1),
            *([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)] if resample else []),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1),
            *([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)] if resample else []),
        )

    def forward(self, x_t: Tensor, x_s: Tensor, w: Tensor) -> Tensor:
        x = torch.cat([x_t, x_s], dim=1)

        skip = self.shortcut(x)

        x = self.adain(x, w)
        x = self.residual(x)

        return x + skip


class Generator(nn.Module):
    def __init__(self, input_res: int = 256, bottleneck_res: int = 8, mapping_depth: int = 4, mapping_size: int = 512, z_dim: int = 512) -> None:
        super().__init__()

        self.mapping = self._build_mapping(z_dim, mapping_depth, mapping_size)

        self.conv_in = nn.Conv2d(3, BASE_CH, 3, padding=1)

        num_down = int(torch.log2(torch.tensor(input_res // bottleneck_res)))
        self.down = nn.ModuleList()
        ch_pairs = []
        down_in_ch = BASE_CH
        for _ in range(num_down):
            down_out_ch = min(down_in_ch * 2, MAX_CH)
            ch_pairs.append((down_in_ch, down_out_ch))
            self.down.append(ResidualDownBlock(down_in_ch, down_out_ch, resample=True))
            down_in_ch = down_out_ch

        self.down.append(ResidualDownBlock(down_out_ch, down_out_ch, resample=False))

        self.up_first = ResidualUpBlock(down_out_ch, down_out_ch, mapping_size, resample=False)
        reversed_ch_pairs = [(out_ch, in_ch) for in_ch, out_ch in reversed(ch_pairs)]

        self.up = nn.ModuleList()
        for i, (up_in_ch, up_out_ch) in enumerate(reversed_ch_pairs):
            if i < 2:
                self.up.append(ResidualUpBlock(up_in_ch, up_out_ch, mapping_size, True))
            else:
                self.up.append(AdaptiveFeatureFusionUpBlock(up_in_ch, up_out_ch, mapping_size, True))

        _, up_in_ch = reversed_ch_pairs[-1]
        self.up.append(ConcatUpBlock(up_in_ch, 3, mapping_size, False))

    def _build_mapping(self, z_dim: int, depth: int, output_dim: int) -> nn.Module:

        layers = []
        current_dim = z_dim
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(current_dim, output_dim), get_act()]
            current_dim = output_dim

        if depth >= 1:
            layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def get_attention_maps(self):
        attention_maps = []
        for module in self.up:
            if isinstance(module, AdaptiveFeatureFusionUpBlock):
                attention_maps.append(module.attn.last_attn_mask)
        return attention_maps

    def forward(self, x_target: Tensor, z_source: Tensor) -> Tensor:
        w = self.mapping(z_source)

        feats = []
        x = self.conv_in(x_target)
        feats.append(x)

        for down_block in self.down:
            x = down_block(x)
            feats.append(x)

        x = self.up_first(feats[-1], w)

        for i, up_block in enumerate(self.up):
            if isinstance(up_block, AdaptiveFeatureFusionUpBlock) or isinstance(up_block, ConcatUpBlock):
                encoder_feat = feats[-(i + 2)]
                x = up_block(encoder_feat, x, w)
            else:
                x = up_block(x, w)

        return x


if __name__ == "__main__":
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    batch_size = 2
    img_size = 512
    z_dim = 512

    model = Generator(mapping_depth=4, mapping_size=256, z_dim=z_dim).to(device)

    model.eval()

    x_target = torch.randn(batch_size, 3, img_size, img_size, device=device)
    z_source = torch.randn(batch_size, z_dim, device=device)

    with torch.no_grad():
        out = model(x_target, z_source)

    print("Input target shape :", x_target.shape)
    print("Input z_source shape:", z_source.shape)
    print("Output shape       :", out.shape)

    assert out.shape == (batch_size, 3, img_size, img_size), "Output shape mismatch!"

    print("Forward pass successful.")
