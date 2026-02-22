import torch
from torch import nn, Tensor


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2),
        )
        self.residual = nn.Sequential(
            nn.InstanceNorm2d(in_ch, affine=False),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2),
            nn.InstanceNorm2d(out_ch, affine=False),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, input_res: int = 256, bottleneck_res: int = 4, base_ch: int = 64, max_ch: int = 512) -> None:
        super().__init__()

        self.network_cfg = {
            "input_res": input_res,
            "bottleneck_res": bottleneck_res,
            "base_ch": base_ch,
            "max_ch": max_ch,
        }

        num_downsamples = (input_res // bottleneck_res).bit_length() - 1

        self.conv_first = nn.Conv2d(3, base_ch, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential()

        down_in_ch = base_ch
        for _ in range(num_downsamples):
            down_out_ch = min(down_in_ch * 2, max_ch)
            self.blocks.append(DownBlock(down_in_ch, down_out_ch))
            down_in_ch = down_out_ch

        self.final_conv = nn.Sequential(
            nn.Conv2d(down_in_ch, down_in_ch, kernel_size=bottleneck_res, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(down_in_ch, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv_first(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)

        return x


if __name__ == "__main__":
    size = 512
    model = Discriminator(size)
    x = torch.randn(2, 3, size, size)
    out = model(x)
    print(out.shape)
