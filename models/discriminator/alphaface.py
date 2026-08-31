import math
import torch
from torch import nn, Tensor
from .upfirdn2d import DownFIRDn2d


class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size: int = 5, num_channels: int = 1) -> None:
        super().__init__()

        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:

        N, C, H, W = x.shape
        G = min(self.group_size, N)
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)
        y = y - y.mean(dim=0)
        y = y.square().mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2, 3, 4])
        y = y.reshape(-1, F, 1, 1)
        y = y.repeat(G, 1, H, W)
        x = torch.cat([x, y], dim=1)
        return x


class DownRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.shortcut = nn.Sequential(
            DownFIRDn2d(),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
        )

        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            DownFIRDn2d(),
        )

        self.scale = 1.0 / math.sqrt(2)

    def forward(self, x: Tensor) -> Tensor:
        return (self.shortcut(x) + self.residual(x)) * self.scale


class AlphaFaceDiscriminator(nn.Module):
    def __init__(self, img_resolution: int = 256, img_channels: int = 3, base_ch: int = 64, max_ch: int = 512, group_size: int = 4):
        super().__init__()

        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}

        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_channels, base_ch, 1),
            nn.LeakyReLU(0.2),
        )

        features = [min(max_ch, base_ch * (2**i)) for i in range(int(math.log2(img_resolution)) - 1)]
        n_blocks = len(features) - 1

        self.down_blocks = nn.ModuleList([DownRB(features[i], features[i + 1]) for i in range(n_blocks)])

        final_features = features[-1] + 1
        self.final_conv = nn.Sequential(
            MinibatchStdLayer(group_size),
            nn.Conv2d(final_features, final_features, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4 * 4 * final_features, final_features),
            nn.LeakyReLU(0.2),
            nn.Linear(final_features, 1),
        )

    def get_feats(self, x: Tensor) -> list[Tensor]:

        x = self.from_rgb(x)

        feats = []
        for down_block in self.down_blocks:
            x = down_block(x)
            feats.append(x)

        return feats

    def forward(self, x: Tensor, return_feats: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:

        x = self.from_rgb(x)

        feats = [] if return_feats else None
        for down_block in self.down_blocks:
            x = down_block(x)
            if feats is not None:
                feats.append(x)

        x = self.final_conv(x)

        return (x, feats) if feats is not None else x
