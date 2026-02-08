import torch

from torch import nn, Tensor


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, resample=True) -> None:
        super().__init__()

        self.residual_path = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if resample:
            self.residual_path = nn.Sequential(self.residual_path, nn.AvgPool2d(kernel_size=2, stride=2))

        self.main_path = nn.Sequential(
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if resample:
            self.main_path.add_module("avgpool", nn.AvgPool2d(kernel_size=2, stride=2))

        self.main_path.add_module("norm2", nn.InstanceNorm2d(out_channels, affine=True))
        self.main_path.add_module("lrelu2", nn.LeakyReLU(0.2, inplace=True))
        self.main_path.add_module("conv3", nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual_path(x)
        main = self.main_path(x)
        return main + residual


class Discriminator(nn.Module):
    def __init__(self, input_res: int = 256, bottleneck_res: int = 4) -> None:
        super().__init__()

        base_ch = 64
        max_ch = 512

        num_down = int(torch.log2(torch.tensor(input_res // bottleneck_res)))

        self.conv_first = nn.Conv2d(3, base_ch, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential()

        down_in_ch = base_ch
        for i in range(num_down):
            down_out_ch = min(down_in_ch * 2, max_ch)
            self.blocks.add_module(f"block{i}", ResidualDownBlock(down_in_ch, down_out_ch, resample=True))
            down_in_ch = down_out_ch

        self.final_conv = nn.Sequential(
            nn.Conv2d(down_in_ch, down_in_ch, kernel_size=bottleneck_res, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
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
