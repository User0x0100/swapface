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
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=shortcut_bias),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )

            case RBSampleMode.DOWN:
                self.residual = nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.AvgPool2d(2),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=shortcut_bias),
                    nn.AvgPool2d(2),
                )

            case RBSampleMode.NONE:
                self.residual = nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                )
                self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=shortcut_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class NormRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sampling: RBSampleMode) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_ch, affine=False)
        self.resblock = ResBlockBase(in_ch, out_ch, sampling)

    def forward(self, x: Tensor) -> Tensor:

        skip = self.resblock.shortcut(x)

        x = self.norm(x)
        x = self.resblock.residual(x)

        return x + skip


class AdaIN(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False)

        self.fc_gamma = nn.Linear(w_dim, channels)
        self.fc_beta = nn.Linear(w_dim, channels)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        x = self.norm(x)

        gamma = self.fc_gamma(w).view(w.size(0), -1, 1, 1)
        beta = self.fc_beta(w).view(w.size(0), -1, 1, 1)

        return x * gamma + beta


class AdaINRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, sampling: RBSampleMode) -> None:
        super().__init__()

        self.adain = AdaIN(in_ch, w_dim)
        self.resblock = ResBlockBase(in_ch, out_ch, sampling)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        skip = self.resblock.shortcut(x)

        x = self.adain(x, w)
        x = self.resblock.residual(x)

        return x + skip


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_encode: Tensor, x_decode: Tensor) -> Tensor:
        return torch.cat((x_encode, x_decode), dim=1)


class Atten(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.attn_mask_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(channels // 4, affine=False),
            nn.Conv2d(channels // 4, channels, 1, padding=0),
            nn.Sigmoid(),
        )

        self.last_attn_mask = None

    def forward(self, x_encode: Tensor, x_decode: Tensor) -> Tensor:

        m = self.attn_mask_proj(torch.cat((x_encode, x_decode), dim=1))

        self.last_attn_mask = m.detach()

        return (1.0 - m) * x_encode + m * x_decode

    def get_attention_maps(self) -> Tensor | None:
        return self.last_attn_mask


class SkipFusionModule(Enum):
    ATTEN = "Atten"
    CONCAT = "Concat"


class SkipFusion(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resample_mode: RBSampleMode, fusion_mode: SkipFusionModule) -> None:
        super().__init__()

        self.fusion = {
            SkipFusionModule.CONCAT: Concat,
            SkipFusionModule.ATTEN: lambda: Atten(in_ch),
        }[fusion_mode]()

        if fusion_mode is SkipFusionModule.CONCAT:
            in_ch *= 2

        self.adain = AdaIN(in_ch, w_dim)
        self.resblock = ResBlockBase(in_ch, out_ch, resample_mode)

    def forward(self, x_encoder: Tensor, x_decoder: Tensor, w: Tensor) -> Tensor:
        x = self.fusion(x_encoder, x_decoder)

        skip = self.resblock.shortcut(x)
        x = self.adain(x, w)
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

        self.mapping = nn.Sequential(nn.Linear(id_dim, id_dim), nn.LeakyReLU(0.2))
        for _ in range(mapping_num - 2):
            self.mapping.append(nn.Linear(id_dim, id_dim))
            self.mapping.append(nn.LeakyReLU(0.2))
        self.mapping.append(nn.Linear(id_dim, w_dim))

        self.from_rgb = nn.Conv2d(img_channels, base_ch, 3, padding=1)

        features = [min(max_ch, base_ch * (2**i)) for i in range(num_encoder + 1)]

        self.encoder = nn.Sequential(*[NormRB(features[i], features[i + 1], RBSampleMode.DOWN) for i in range(num_encoder)])

        final_ch = features[-1]

        self.bottleneck_encode = NormRB(final_ch, final_ch, RBSampleMode.NONE)
        self.bottleneck_decode = AdaINRB(final_ch, final_ch, w_dim, RBSampleMode.NONE)

        self.decoder = nn.Sequential()
        for i in range(num_encoder):
            in_ch, out_ch = features[-(i + 1)], features[-(i + 2)]
            if i < 2:
                self.decoder.append(AdaINRB(in_ch, out_ch, w_dim, RBSampleMode.UP))
            else:
                self.decoder.append(SkipFusion(features[-(i + 1)], features[-(i + 2)], w_dim, RBSampleMode.UP, SkipFusionModule.ATTEN))

        self.to_rgb = SkipFusion(base_ch, img_channels, w_dim, RBSampleMode.NONE, SkipFusionModule.CONCAT)

    def get_attention_maps(self) -> list[Tensor]:
        return [maps for module in self.decoder.modules() if isinstance(module, Atten) if (maps := module.get_attention_maps()) is not None]

    def forward(self, x_target: Tensor, id_feat: Tensor) -> Tensor:

        w = self.mapping(id_feat)

        from_rgb = self.from_rgb(x_target)

        x = from_rgb
        encode_feats = []
        for encode_layer in self.encoder:
            x = encode_layer(x)
            encode_feats.append(x)

        x = self.bottleneck_encode(x)
        x = self.bottleneck_decode(x, w)

        for i, decode_layer in enumerate(self.decoder):
            if isinstance(decode_layer, SkipFusion):
                x = decode_layer(encode_feats[-(i + 1)], x, w)
            else:
                x = decode_layer(x, w)

        x = self.to_rgb(from_rgb, x, w)

        return x


if __name__ == "__main__":
    import torch
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    network_cfg = {
        "img_resolution": 128,
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
