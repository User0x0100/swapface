import math
from torch import nn, Tensor


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, normalize=False, downsample=False):
        super().__init__()

        shortcut_layers = []
        if dim_in != dim_out:
            shortcut_layers.append(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

        if downsample:
            shortcut_layers.append(nn.AvgPool2d(2))

        self.shortcut = nn.Sequential(*shortcut_layers) if shortcut_layers else nn.Identity()

        residual_layers = []
        if normalize:
            residual_layers.append(nn.InstanceNorm2d(dim_in, affine=True))

        residual_layers.append(nn.LeakyReLU(0.2))
        residual_layers.append(nn.Conv2d(dim_in, dim_in, 3, 1, 1))

        if downsample:
            residual_layers.append(nn.AvgPool2d(2))

        if normalize:
            residual_layers.append(nn.InstanceNorm2d(dim_in, affine=True))

        residual_layers.append(nn.LeakyReLU(0.2))
        residual_layers.append(nn.Conv2d(dim_in, dim_out, 3, 1, 1))

        self.residual = nn.Sequential(*residual_layers)

    def forward(self, x):
        return (self.shortcut(x) + self.residual(x)) / math.sqrt(2)


class AlphaFaceDiscriminator(nn.Module):
    def __init__(self, input_res: int = 256, max_ch: int = 512):
        super().__init__()

        self.network_cfg = {
            "input_res": input_res,
            "max_ch": max_ch,
        }

        dim_in = 2**14 // input_res
        num_domains = 1
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(math.log2(input_res)) - 1
        for i in range(repeat_num):
            dim_out = min(dim_in * 2, max_ch)
            if i % 2 == 0:
                blocks += [ResBlk(dim_in, dim_out, downsample=False)]
            else:
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 1, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.main = nn.Sequential(*blocks)

    def get_feats(self, x: Tensor) -> list[Tensor]:

        feats = []
        for depth, block in enumerate(self.main):
            x = block(x)
            if 7 < depth < 10:
                feats.append(x)

        return feats

    def forward(self, x: Tensor, return_feats: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:

        feats = [] if return_feats else None
        for depth, block in enumerate(self.main):
            x = block(x)
            if return_feats and 7 < depth < 10:
                feats.append(x)

        x = x.view(x.size(0), -1)

        return (x, feats) if feats is not None else x
