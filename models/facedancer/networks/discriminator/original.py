from torch import nn, Tensor


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(2),
        )
        self.residual = nn.Sequential(
            # nn.InstanceNorm2d(in_ch, affine=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(2),
            # nn.InstanceNorm2d(out_ch, affine=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, input_res: int = 256, num_encoder: int = 5, base_ch: int = 64, max_ch: int = 512) -> None:
        super().__init__()

        self.network_cfg = {
            "input_res": input_res,
            "num_encoder": num_encoder,
            "base_ch": base_ch,
            "max_ch": max_ch,
        }

        self.conv_first = nn.Conv2d(3, base_ch, 3, 1, 1)
        self.blocks = nn.ModuleList()

        channels = base_ch
        for _ in range(num_encoder):
            next_ch = min(channels * 2, max_ch)
            self.blocks.append(DownBlock(channels, next_ch))
            channels = next_ch

        self.final_conv = nn.Sequential(
            nn.Conv2d(next_ch, next_ch, input_res // (2**num_encoder)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(next_ch, 1, 1),
            nn.Flatten(),
        )

    def forward(self, x: Tensor, return_feats: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:

        x = self.conv_first(x)

        feats = [] if return_feats else None

        for block in self.blocks:
            x = block(x)
            if return_feats:
                feats.append(x)

        x = self.final_conv(x)

        return (x, feats) if feats is not None else x
