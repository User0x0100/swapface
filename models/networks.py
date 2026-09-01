import torch
from torch import Tensor, nn


class AdaIN(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.gamma_fc = nn.Linear(w_dim, channels)
        self.beta_fc = nn.Linear(w_dim, channels)

        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.ones_(self.gamma_fc.bias)

        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        gamma = self.gamma_fc(w)[:, :, None, None]
        beta = self.beta_fc(w)[:, :, None, None]

        return self.norm(x) * gamma + beta


class IDInject(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()

        self.act = nn.SiLU()

        self.adain0 = AdaIN(channels, w_dim)
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.adain1 = AdaIN(channels, w_dim)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:

        residual = self.conv0(x)
        residual = self.adain0(residual, w)
        residual = self.act(residual)

        residual = self.conv1(residual)
        residual = self.adain1(residual, w)
        residual = self.act(residual)

        return x + residual


class FromRGB(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=7, stride=1, padding=3),
            nn.SiLU(),
            nn.Conv2d(out_ch // 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )


class ToRGB(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_ch // 2, out_ch, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )


class AAD(nn.Module):
    """Adaptive Attentional Denormalization。

    将 decoder 特征分别按 encoder 属性特征和身份向量反归一化，并通过空间注意力掩码
    自适应融合两条分支。
    """

    def __init__(self, channels: int, attribute_channels: int, id_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.mask = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), nn.Sigmoid())

        self.attribute_gamma = nn.Conv2d(attribute_channels, channels, kernel_size=3, stride=1, padding=1)
        self.attribute_beta = nn.Conv2d(attribute_channels, channels, kernel_size=3, stride=1, padding=1)
        self.identity_gamma = nn.Linear(id_dim, channels)
        self.identity_beta = nn.Linear(id_dim, channels)

    def forward(self, x: Tensor, attribute: Tensor, identity: Tensor) -> Tensor:
        normalized = self.norm(x)

        attribute_feature = self.attribute_gamma(attribute) * normalized + self.attribute_beta(attribute)
        identity_gamma = self.identity_gamma(identity)[:, :, None, None]
        identity_beta = self.identity_beta(identity)[:, :, None, None]
        identity_feature = identity_gamma * normalized + identity_beta

        mask = self.mask(normalized)
        return (1.0 - mask) * attribute_feature + mask * identity_feature


class AADResBlock(nn.Module):
    """使用 encoder skip feature 和身份向量调制 decoder 特征的 AAD 残差块。"""

    def __init__(self, channels: int, attribute_channels: int, id_dim: int) -> None:
        super().__init__()

        self.act = nn.SiLU()
        self.aad0 = AAD(channels, attribute_channels, id_dim)
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.aad1 = AAD(channels, attribute_channels, id_dim)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor, attribute: Tensor, identity: Tensor) -> Tensor:
        residual = self.conv0(self.act(self.aad0(x, attribute, identity)))
        residual = self.conv1(self.act(self.aad1(residual, attribute, identity)))
        return x + residual


class UpSample(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )


class DownSample(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )


class WSpaceMap(nn.Module):
    def __init__(self, id_dim: int, num: int, num_share_layers: int = 4, num_w_p_layers: int = 2) -> None:
        super().__init__()

        self.shared_delta = nn.Sequential()
        for _ in range(num_share_layers - 1):
            self.shared_delta.append(nn.Linear(id_dim, id_dim))
            self.shared_delta.append(nn.SiLU())
        self.shared_delta.append(nn.Linear(id_dim, id_dim))

        self.private_delta = nn.ModuleList()
        for _ in range(num):
            layers = nn.Sequential()
            for _ in range(num_w_p_layers):
                layers.append(nn.SiLU())
                layers.append(nn.Linear(id_dim, id_dim))
            self.private_delta.append(layers)

        self.num = num
        self.id_dim = id_dim

        shared_output = self.shared_delta[-1]
        assert isinstance(shared_output, nn.Linear)
        nn.init.zeros_(shared_output.weight)
        nn.init.zeros_(shared_output.bias)

        for head in self.private_delta:
            assert isinstance(head, nn.Sequential)
            private_output = head[-1]
            assert isinstance(private_output, nn.Linear)
            nn.init.zeros_(private_output.weight)
            nn.init.zeros_(private_output.bias)

    def forward(self, id_feat: Tensor) -> tuple[Tensor, ...]:
        w = id_feat + self.shared_delta(id_feat)
        outputs: list[Tensor] = []
        for head in self.private_delta:
            outputs.append(w + head(w))
        return tuple(outputs)


class LatentBlock(nn.Module):
    def __init__(self, channels: int, w_dim: int, num_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([IDInject(channels, w_dim) for _ in range(num_layers)])

    def forward(self, x: Tensor, w_space: tuple[Tensor, ...]) -> Tensor:

        assert len(w_space) == len(self.layers), f"w_p_all length {len(w_space)} != layers length {len(self.layers)}"

        for layer_index, layer in enumerate(self.layers):
            x = layer(x, w_space[layer_index])

        return x


class Generator(nn.Module):
    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        num_depth: int = 3,
        num_latent: int = 6,
        base_ch: int = 256,
        max_ch: int = 1024,
        id_dim: int = 512,
        aad_skip_layers: tuple[int, ...] | list[int] = (),
    ) -> None:
        super().__init__()

        assert max_ch >= base_ch, f"max_ch={max_ch} must be >= base_ch={base_ch}"
        if num_depth <= 0:
            raise ValueError(f"num_depth must be greater than 0, got {num_depth}")

        skip_layers = tuple(int(index) for index in aad_skip_layers)
        if len(set(skip_layers)) != len(skip_layers):
            raise ValueError(f"aad_skip_layers contains duplicate indices: {skip_layers}")
        invalid_skip_layers = sorted(index for index in skip_layers if index < 0 or index >= num_depth)
        if invalid_skip_layers:
            raise ValueError(f"aad_skip_layers must be within 0~{num_depth - 1}, got {invalid_skip_layers}")
        skip_layers = tuple(sorted(skip_layers))

        self.network_cfg = {
            "img_resolution": img_resolution,
            "img_channels": img_channels,
            "num_depth": num_depth,
            "num_latent": num_latent,
            "base_ch": base_ch,
            "max_ch": max_ch,
            "id_dim": id_dim,
            "aad_skip_layers": list(skip_layers),
        }
        self.aad_skip_layers = skip_layers

        self.w_space_map = WSpaceMap(id_dim, num_latent)
        features = [min(max_ch, base_ch * (2**i)) for i in range(num_depth + 1)]

        self.from_rgb = FromRGB(img_channels, base_ch)
        self.encoder = nn.ModuleList([DownSample(features[i], features[i + 1]) for i in range(num_depth)])
        self.latent_space = LatentBlock(features[-1], id_dim, num_latent)
        self.decoder = nn.ModuleList([UpSample(features[-(i + 1)], features[-(i + 2)]) for i in range(num_depth)])
        self.aad_skip = nn.ModuleDict({str(layer_index): AADResBlock(features[-(layer_index + 2)], features[-(layer_index + 2)], id_dim) for layer_index in skip_layers})
        self.to_rgb = ToRGB(base_ch, img_channels)

    def forward(self, x: Tensor, id_feat: Tensor) -> Tensor:
        w_space = self.w_space_map(id_feat)

        feat = self.from_rgb(x)
        encoder_features = [feat]
        for downsample in self.encoder:
            feat = downsample(feat)
            encoder_features.append(feat)

        feat = self.latent_space(feat, w_space)
        for layer_index, upsample in enumerate(self.decoder):
            feat = upsample(feat)
            if layer_index in self.aad_skip_layers:
                attribute = encoder_features[-(layer_index + 2)]
                feat = self.aad_skip[str(layer_index)](feat, attribute, id_feat)

        return self.to_rgb(feat)


if __name__ == "__main__":
    import torch
    from fvcore.nn import FlopCountAnalysis
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    network_cfg = {
        "img_resolution": 512,
        "img_channels": 3,
        "num_depth": 5,
        "num_latent": 6,
        "base_ch": 16,
        "max_ch": 2048,
        "id_dim": 512,
        "aad_skip_layers": [0, 1, 2, 3, 4],
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
