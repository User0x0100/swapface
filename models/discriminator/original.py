from torch import nn, Tensor


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(2),
        )
        self.residual = nn.Sequential(
            nn.InstanceNorm2d(in_ch, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(2),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, img_resolution: int = 256, img_channels: int = 3, num_encoder: int = 6, base_ch: int = 64, max_ch: int = 512) -> None:
        super().__init__()

        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}

        self.from_rgb = nn.Conv2d(img_channels, base_ch, 3, 1, padding=1)

        features = [min(max_ch, base_ch * (2**i)) for i in range(num_encoder + 1)]

        self.down_blocks = nn.ModuleList([DownBlock(features[i], features[i + 1]) for i in range(num_encoder)])

        final_ch = features[-1]

        self.final_conv = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, img_resolution // (2**num_encoder)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(final_ch, 1, 1),
            nn.Flatten(),
        )

    def forward(self, x: Tensor, return_feats: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:

        x = self.from_rgb(x)

        feats = [] if return_feats else None

        for down_block in self.down_blocks:
            x = down_block(x)
            if feats is not None:
                feats.append(x)

        x = self.final_conv(x)

        return (x, feats) if feats is not None else x
