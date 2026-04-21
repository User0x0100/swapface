import torch


class IdentityEncoderBlock(torch.nn.Module):
    def __init__(self, num_channels, id_dim):
        super().__init__()
        self.normalize = torch.nn.BatchNorm2d(num_channels, affine=False, track_running_stats=True)
        self.fc_mean = torch.nn.Linear(id_dim, num_channels)
        self.fc_var = torch.nn.Linear(id_dim, num_channels)
        self.nonlinear1 = torch.nn.LeakyReLU()
        self.conv1x1 = torch.nn.Conv2d(num_channels, num_channels, 1, 1, 0)
        self.nonlinear2 = torch.nn.LeakyReLU()
        self.conv3x3 = torch.nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False)

    def forward(self, x, feat):
        centered_x = self.normalize(x)
        mean = self.fc_mean(feat).unsqueeze(2).unsqueeze(3)
        var = self.fc_var(feat).unsqueeze(2).unsqueeze(3)
        modulated_x = centered_x * var + mean
        x = self.nonlinear1(modulated_x)
        x = self.conv1x1(x)
        x = self.nonlinear2(x)
        x = self.conv3x3(x)
        return x


class AttributeEncoderBlock(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.normalize = torch.nn.BatchNorm2d(num_channels)
        self.conv1x1 = torch.nn.Conv2d(num_channels, num_channels, 1, 1, 0)
        self.nonlinear = torch.nn.LeakyReLU()
        self.conv3x3 = torch.nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.normalize(x)
        x = self.conv1x1(x)
        x = self.nonlinear(x)
        x = self.conv3x3(x)
        return x


class EncoderHead(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv_image = torch.nn.Conv2d(3, num_channels, 3, 1, 1)
        self.nonlinear1 = torch.nn.LeakyReLU()
        self.conv1x1 = torch.nn.Conv2d(num_channels, num_channels, 1, 1, 0)
        self.nonlinear2 = torch.nn.LeakyReLU()
        self.conv3x3 = torch.nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False)

    def forward(self, images):
        x1 = self.conv_image(images)
        x2 = self.nonlinear1(x1)
        x3 = self.conv1x1(x2)
        x4 = self.nonlinear2(x3)
        x5 = self.conv3x3(x4)
        return x5, x2


class Upsample(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.normalize = torch.nn.BatchNorm2d(num_channels)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv = torch.nn.Conv2d(num_channels, num_channels, 3, 1, 0, bias=False)
        self.nonlinear = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.normalize(x)
        x = self.upsample(x)
        x = self.pad(x)
        x = self.conv(x)
        x = self.nonlinear(x)
        return x


class Downsample(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.normalize = torch.nn.BatchNorm2d(num_channels)
        self.conv = torch.nn.Conv2d(num_channels, num_channels, 4, 2, 1, bias=False)
        self.nonlinear = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.normalize(x)
        x = self.conv(x)
        x = self.nonlinear(x)
        return x


class AttributeEncoderBody(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # 256x256
        self.b256_1 = AttributeEncoderBlock(num_channels)
        self.b256_2 = AttributeEncoderBlock(num_channels)
        # 128x128
        self.downsample128 = Downsample(num_channels)
        self.b128_1 = AttributeEncoderBlock(num_channels)
        self.b128_2 = AttributeEncoderBlock(num_channels)
        self.b128_3 = AttributeEncoderBlock(num_channels)
        # 64x64
        self.downsample64 = Downsample(num_channels)
        self.b64_1 = AttributeEncoderBlock(num_channels)
        self.b64_2 = AttributeEncoderBlock(num_channels)
        self.b64_3 = AttributeEncoderBlock(num_channels)
        self.b64_4 = AttributeEncoderBlock(num_channels)

    def forward(self, x256_0):
        # 256x256
        x256_1 = x256_0 + self.b256_1(x256_0)
        x256_2 = x256_1 + self.b256_2(x256_1)
        # 128x128
        x128_0 = self.downsample128(x256_2)
        x128_1 = x128_0 + self.b128_1(x128_0)
        x128_2 = x128_1 + self.b128_2(x128_1)
        x128_3 = x128_2 + self.b128_3(x128_2)
        # 64x64
        x64_0 = self.downsample64(x128_3)
        x64_1 = x64_0 + self.b64_1(x64_0)
        x64_2 = x64_1 + self.b64_2(x64_1)
        x64_3 = x64_2 + self.b64_3(x64_2)
        x64_4 = x64_3 + self.b64_4(x64_3)
        # attributes
        attributes = [x256_0, x256_1, x256_2, x128_0, x128_1, x128_2, x128_3, x64_0, x64_1, x64_2, x64_3, x64_4]
        return attributes


class IdentityEncoderBody(torch.nn.Module):
    def __init__(self, num_channels, id_dim):
        super().__init__()
        """down"""
        # 256x256
        self.b256_down = IdentityEncoderBlock(num_channels, id_dim)
        # 128x128
        self.down128 = Downsample(num_channels)
        self.b128_down = IdentityEncoderBlock(num_channels, id_dim)
        # 64x64
        self.down64 = Downsample(num_channels)
        self.b64_down = IdentityEncoderBlock(num_channels, id_dim)
        # 32x32
        self.down32 = Downsample(num_channels)
        self.b32_down = IdentityEncoderBlock(num_channels, id_dim)
        # 16x16
        self.down16 = Downsample(num_channels)
        self.b16_down = IdentityEncoderBlock(num_channels, id_dim)
        # 8x8
        self.down8 = Downsample(num_channels)
        self.b8_down = IdentityEncoderBlock(num_channels, id_dim)
        """up"""
        # 16x16
        self.up16 = Upsample(num_channels)
        self.b16_up = IdentityEncoderBlock(num_channels, id_dim)
        # 32x32
        self.up32 = Upsample(num_channels)
        self.b32_up = IdentityEncoderBlock(num_channels, id_dim)
        # 64x64
        self.up64 = Upsample(num_channels)
        self.b64_up = IdentityEncoderBlock(num_channels, id_dim)

    def forward(self, x256_0, embeddings):
        # down
        x256_1 = x256_0 + self.b256_down(x256_0, embeddings)
        x128_0 = self.down128(x256_1)
        x128_1 = x128_0 + self.b128_down(x128_0, embeddings)
        x64_0 = self.down64(x128_1)
        x64_1 = x64_0 + self.b64_down(x64_0, embeddings)
        x32_0 = self.down32(x64_1)
        x32_1 = x32_0 + self.b32_down(x32_0, embeddings)
        x16_0 = self.down16(x32_1)
        x16_1 = x16_0 + self.b16_down(x16_0, embeddings)
        x8_0 = self.down8(x16_1)
        x8_1 = x8_0 + self.b8_down(x8_0, embeddings)
        # up
        y16_0 = self.up16(x8_1) + x16_1
        y16_1 = self.b16_up(y16_0, embeddings) + y16_0
        y32_0 = self.up32(y16_1) + x32_1
        y32_1 = self.b32_up(y32_0, embeddings) + y32_0
        y64_0 = self.up64(y32_1) + x64_1
        y64_1 = self.b64_up(y64_0, embeddings) + y64_0
        return y64_1


class DecoderBlock(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv_attention = torch.nn.Conv2d(num_channels, 1, 1, 1, 0)
        self.basic_block = AttributeEncoderBlock(num_channels)

    def forward(self, x, attribute):
        attention_map = self.conv_attention(x)
        x = attention_map * x + (1.0 - attention_map) * attribute
        x = self.basic_block(x)
        return x


class DecoderImageOutput(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(num_channels, 3, 3, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.b64_4 = DecoderBlock(num_channels)
        self.b64_3 = DecoderBlock(num_channels)
        self.b64_2 = DecoderBlock(num_channels)
        self.b64_1 = DecoderBlock(num_channels)
        self.b64_0 = DecoderBlock(num_channels)
        self.up128 = Upsample(num_channels)
        self.b128_3 = DecoderBlock(num_channels)
        self.b128_2 = DecoderBlock(num_channels)
        self.b128_1 = DecoderBlock(num_channels)
        self.b128_0 = DecoderBlock(num_channels)
        self.up256 = Upsample(num_channels)
        self.b256_2 = DecoderBlock(num_channels)
        self.b256_1 = DecoderBlock(num_channels)
        self.b256_0 = DecoderBlock(num_channels)
        self.image256_output = DecoderImageOutput(num_channels)

    def forward(self, x, attributes):
        """
        attributes = [x256_0, x256_1, x256_2,
              x128_0, x128_1, x128_2, x128_3,
              x64_0,  x64_1,  x64_2,  x64_3,  x64_4]
        """
        x256_0, x256_1, x256_2, x128_0, x128_1, x128_2, x128_3, x64_0, x64_1, x64_2, x64_3, x64_4 = attributes
        x = self.b64_4(x, x64_4) + x
        x = self.b64_3(x, x64_3) + x
        x = self.b64_2(x, x64_2) + x
        x = self.b64_1(x, x64_1) + x
        x = self.b64_0(x, x64_0) + x

        x = self.up128(x)
        x = self.b128_3(x, x128_3) + x
        x = self.b128_2(x, x128_2) + x
        x = self.b128_1(x, x128_1) + x
        x = self.b128_0(x, x128_0) + x

        x = self.up256(x)
        x = self.b256_2(x, x256_2) + x
        x = self.b256_1(x, x256_1) + x
        x = self.b256_0(x, x256_0) + x
        images256 = self.image256_output(x)
        return images256


class Generator(torch.nn.Module):
    def __init__(self, img_resolution: int = 256, num_channels: int = 64, id_dim: int = 512):
        super().__init__()
        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.encoder_head = EncoderHead(num_channels)
        self.identity_encoder = IdentityEncoderBody(num_channels, id_dim)
        self.attribute_encoder = AttributeEncoderBody(num_channels)
        self.decoder = Decoder(num_channels)

    def forward(self, targets, id_feat):
        identity_x, attribute_x = self.encoder_head(targets)
        attributes = self.attribute_encoder(attribute_x)
        identity = self.identity_encoder(identity_x, id_feat)
        images256 = self.decoder(identity, attributes)
        return images256

    def get_attention_maps(self) -> list[torch.Tensor]:
        return []


if __name__ == "__main__":
    import torch
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    img_resolution = 256
    id_dim = 512

    model = Generator().to(device)
    model.eval()

    x_target = torch.randn((batch_size, 3, img_resolution, img_resolution), device=device)
    id_feat = torch.randn((batch_size, id_dim), device=device)
    summary(model, input_data=(x_target, id_feat), depth=3, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), row_settings=("var_names",))

    flops = FlopCountAnalysis(model, (x_target, id_feat))
    print(f"\n模型总FLOPs: {flops.total() / 1e9:.4f} GFLOPs")
