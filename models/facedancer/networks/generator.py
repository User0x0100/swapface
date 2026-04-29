from enum import Enum
import torch
from torch import Tensor
import torch.nn as nn


class RBSampleMode(Enum):
    UP = "Up"
    DOWN = "Down"
    NONE = "None"


class ResBlockBase(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sampling: RBSampleMode, shortcut_bias: bool = False) -> None:
        super().__init__()

        match sampling:
            case RBSampleMode.UP:
                self.residual = nn.Sequential(
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, stride=1, padding=0),
                    nn.SiLU(),
                    nn.PixelShuffle(2),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=shortcut_bias),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )

            case RBSampleMode.DOWN:
                self.residual = nn.Sequential(
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                    nn.SiLU(),
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(out_ch * 4, out_ch, kernel_size=3, stride=1, padding=1),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=shortcut_bias),
                    nn.AvgPool2d(2),
                )

            case RBSampleMode.NONE:
                self.residual = nn.Sequential(
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                )
                self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=shortcut_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class AdaIN(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False)

        self.fc_gamma = nn.Linear(w_dim, channels)
        self.fc_beta = nn.Linear(w_dim, channels)

        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        x_norm = self.norm(x)

        gamma = self.fc_gamma(w).view(w.size(0), -1, 1, 1)
        beta = self.fc_beta(w).view(w.size(0), -1, 1, 1)

        mod = x_norm * (1.0 + gamma) + beta

        return x + self.alpha * (mod - x)


class IDInjection(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample_mode: RBSampleMode) -> None:
        super().__init__()

        self.adain = AdaIN(in_ch, w_dim)
        self.resblock = ResBlockBase(in_ch, out_ch, resample_mode)

    def forward(self, x_decoder: Tensor, w: Tensor) -> Tensor:

        x = self.adain(x_decoder, w)
        skip = self.resblock.shortcut(x)
        x = self.resblock.residual(x)
        return x + skip


class Generator(nn.Module):
    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        num_encoder: int = 5,
        base_ch: int = 64,
        max_ch: int = 512,
        id_dim: int = 512,
        w_dim: int = 256,
        mapping_num: int = 4,
    ) -> None:
        super().__init__()

        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}

        mapping_layers = [nn.Linear(id_dim, w_dim), nn.SiLU()]
        for _ in range(mapping_num - 2):
            mapping_layers += [nn.Linear(w_dim, w_dim), nn.SiLU()]
        mapping_layers += [nn.Linear(w_dim, w_dim)]
        self.mapping = nn.Sequential(*mapping_layers)

        self.from_rgb = nn.Conv2d(img_channels, base_ch, 3, padding=1)

        features = [min(max_ch, base_ch * (2**i)) for i in range(num_encoder + 1)]

        self.encoder = nn.Sequential(*[ResBlockBase(features[i], features[i + 1], RBSampleMode.DOWN) for i in range(num_encoder)])

        final_ch = features[-1]

        self.bottleneck_encode = ResBlockBase(final_ch, final_ch, RBSampleMode.NONE)
        self.bottleneck_decode = IDInjection(final_ch, final_ch, w_dim, RBSampleMode.NONE)

        self.decoder = nn.ModuleList([IDInjection(features[-(i + 1)], features[-(i + 2)], w_dim, RBSampleMode.UP) for i in range(num_encoder)])

        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_ch, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x_target: Tensor, id_feat: Tensor) -> Tensor:

        w = self.mapping(id_feat)

        x = self.from_rgb(x_target)

        x = self.encoder(x)

        x = self.bottleneck_encode(x)
        x = self.bottleneck_decode(x, w)

        for decoder_block in self.decoder:
            x = decoder_block(x, w)

        x = self.to_rgb(x)

        return x


if __name__ == "__main__":
    import torch
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    network_cfg = {
        "img_resolution": 256,
        "img_channels": 3,
        "num_encoder": 5,
        "base_ch": 64,
        "max_ch": 512,
        "id_dim": 512,
        "w_dim": 256,
        "mapping_num": 4,
    }

    model = Generator(**network_cfg).to(device)
    model.eval()

    x_target = torch.randn((batch_size, network_cfg["img_channels"], network_cfg["img_resolution"], network_cfg["img_resolution"]), device=device)
    id_feat = torch.randn((batch_size, network_cfg["id_dim"]), device=device)
    summary(model, input_data=(x_target, id_feat), depth=2, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), row_settings=("var_names",))

    print("NetWork_Info:")
    for k, v in network_cfg.items():
        print(f"  {k:25}: {v}")

    flops = FlopCountAnalysis(model, (x_target, id_feat))
    print(f"\n模型总FLOPs: {flops.total() / 1e9:.4f} GFLOPs")
