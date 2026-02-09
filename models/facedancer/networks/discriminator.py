import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


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


class UNetDiscriminatorSN(nn.Module):
    """
    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, skip_connection: bool = True) -> None:
        super().__init__()

        self.skip_connection = skip_connection

        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def get_feats(self, x: Tensor) -> Tensor:
        inplace = True

        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=inplace)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=inplace)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=inplace)

        return x2

    def forward(self, x: Tensor, return_features: bool = False) -> Tensor | tuple[Tensor, Tensor]:

        inplace = True

        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=inplace)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=inplace)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=inplace)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=inplace)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=inplace)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=inplace)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=inplace)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=inplace)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=inplace)
        out = self.conv9(out)

        if return_features:
            return out, x2
        else:
            return out


if __name__ == "__main__":
    size = 512
    model = Discriminator(size)
    x = torch.randn(2, 3, size, size)
    out = model(x)
    print(out.shape)
