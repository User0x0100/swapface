import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    def __init__(self, num_channels: int, w_dim: int) -> None:
        super().__init__()

        self.fc_gamma = nn.Linear(w_dim, num_channels)
        self.fc_beta = nn.Linear(w_dim, num_channels)

        nn.init.xavier_uniform_(self.fc_gamma.weight)
        nn.init.xavier_uniform_(self.fc_beta.weight)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        gamma: Tensor = self.fc_gamma(w)
        beta: Tensor = self.fc_beta(w)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class AdaptiveAttention(nn.Module):
    def forward(self, m: Tensor, a: Tensor, i: Tensor) -> Tensor:
        return (1.0 - m) * a + m * i


class ResidualDownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, resample: bool = True) -> None:
        super().__init__()

        self.conv_r = nn.Conv2d(in_ch, out_ch, 1)
        if resample:
            self.conv_r = nn.Sequential(self.conv_r, nn.AvgPool2d(2))

        self.main_path = nn.Sequential(
            nn.InstanceNorm2d(in_ch, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

        if resample:
            self.main_path.add_module("avgpool", nn.AvgPool2d(2))

    def forward(self, x: Tensor):
        r = self.conv_r(x)
        x = self.main_path(x)
        return x + r


class ResidualUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample: bool = True) -> None:
        super().__init__()

        self.resample = resample

        self.conv_r = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm = nn.InstanceNorm2d(in_ch, affine=True)
        self.adain = AdaIN(in_ch, w_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        r = self.conv_r(x)
        if self.resample:
            r = F.interpolate(r, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.norm(x)
        x = self.adain(x, w)
        x = self.act(x)
        x = self.conv(x)

        if self.resample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return x + r


class AdaptiveAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels // 4, 3, padding=1)
        self.conv2 = nn.Conv2d(channels // 4, channels, 1)
        self.norm = nn.InstanceNorm2d(channels // 4, affine=True)
        self.act = nn.LeakyReLU(0.2)
        self.attn = AdaptiveAttention()
        self.last_attention_mask = None

    def forward(self, x_t: Tensor, x_s: Tensor) -> Tensor:
        m = torch.cat([x_t, x_s], dim=1)
        m = self.conv1(m)
        m = self.act(m)
        m = self.norm(m)
        m = torch.sigmoid(self.conv2(m))
        self.last_attention_mask = m.detach().requires_grad_(False)
        return self.attn(m, x_t, x_s)


class AdaptiveFeatureFusionUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample: bool = True) -> None:
        super().__init__()

        self.resample = resample

        self.attn = AdaptiveAttentionBlock(in_ch)
        self.conv_r = nn.Conv2d(in_ch, out_ch, 1)

        self.norm = nn.InstanceNorm2d(in_ch, affine=True)
        self.adain = AdaIN(in_ch, w_dim)
        self.act = nn.LeakyReLU(0.2)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x_t: Tensor, x_s: Tensor, w: Tensor) -> Tensor:
        x = self.attn(x_t, x_s)

        r = self.conv_r(x)
        if self.resample:
            r = F.interpolate(r, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.norm(x)
        x = self.adain(x, w)
        x = self.act(x)
        x = self.conv(x)

        if self.resample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return x + r


class ConcatUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample: bool = True) -> None:
        super().__init__()

        self.resample = resample

        self.conv_r = nn.Conv2d(in_ch * 2, out_ch, 1)
        self.norm = nn.InstanceNorm2d(in_ch * 2, affine=True)
        self.adain = AdaIN(in_ch * 2, w_dim)
        self.act = nn.LeakyReLU(0.2)
        self.conv = nn.Conv2d(in_ch * 2, out_ch, 3, padding=1)

    def forward(self, x_t: Tensor, x_s: Tensor, w: Tensor) -> Tensor:
        x = torch.cat([x_t, x_s], dim=1)

        r = self.conv_r(x)
        if self.resample:
            r = F.interpolate(r, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.norm(x)
        x = self.adain(x, w)
        x = self.act(x)
        x = self.conv(x)

        if self.resample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return x + r


class Generator(nn.Module):
    def __init__(self, input_res: int = 256, bottleneck_res: int = 8, mapping_depth: int = 4, mapping_size: int = 512, z_dim: int = 512) -> None:
        super().__init__()

        base_ch = 64
        max_ch = 512

        self.mapping = self._build_mapping(z_dim, mapping_depth, mapping_size)

        self.conv_in = nn.Conv2d(3, base_ch, 3, padding=1)

        num_down = int(torch.log2(torch.tensor(input_res // bottleneck_res)))
        self.down = nn.ModuleList()
        ch_pairs = []
        down_in_ch = base_ch
        for _ in range(num_down):
            down_out_ch = min(down_in_ch * 2, max_ch)
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
            layers += [nn.Linear(current_dim, output_dim), nn.LeakyReLU(0.2)]
            current_dim = output_dim

        if depth >= 1:
            layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def get_attention_maps(self):
        attention_maps = []
        for module in self.up:
            if isinstance(module, AdaptiveFeatureFusionUpBlock):
                if hasattr(module.attn, "last_attention_mask") and module.attn.last_attention_mask is not None:
                    attention_maps.append(module.attn.last_attention_mask)
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
