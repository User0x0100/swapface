import torch
from torch import nn, Tensor
from torch.nn import functional as F


EPS = 1e-8


class Downscale(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=5) -> None:
        """
        Downscale component.

        Args:
            in_ch (int): The number of input channels e.g. 3 for a standard RGB image
            out_ch (int): The number of output channels
            kernel_size (int): The kernel size used during the convolution operation
        """
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=kernel_size, stride=2, padding=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        return x

    def get_out_ch(self) -> int:
        return self.out_ch


class Encoder(nn.Module):
    def __init__(self, in_ch: int, e_ch: int) -> None:
        super().__init__()

        self.e_ch = e_ch

        self.down1 = Downscale(in_ch, e_ch, kernel_size=5)
        self.res1 = ResidualBlock(e_ch)

        self.down2 = Downscale(e_ch, e_ch * 2, kernel_size=5)
        self.down3 = Downscale(e_ch * 2, e_ch * 4, kernel_size=5)
        self.down4 = Downscale(e_ch * 4, e_ch * 8, kernel_size=5)
        self.down5 = Downscale(e_ch * 8, e_ch * 8, kernel_size=5)

        self.res5 = ResidualBlock(e_ch * 8)
        self.flatten = nn.Flatten()

    @staticmethod
    def pixel_norm(x: Tensor) -> Tensor:
        return x * torch.rsqrt(torch.mean(torch.square(x), dim=1, keepdim=True) + EPS)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.res5(x)
        x = self.flatten(x)
        x = self.pixel_norm(x)
        return x

    def get_output_length(self, input_resolution: int) -> int:
        res = input_resolution // 32
        return (self.e_ch * 8) * res * res


class Upscale(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        """
        Deconvolution block used for upscaling the input tensor

        Args:
            in_ch (int): The number of input channels
            out_ch (int): The number of output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding="same")
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.pixel_shuffle(x)
        return x


class Inter(nn.Module):
    def __init__(self, in_ch, ae_ch, ae_out_ch, lowest_dense_res) -> None:
        super().__init__()

        self.ae_out_ch = ae_out_ch
        self.lowest_dense_res = lowest_dense_res

        self.dense1 = nn.Linear(in_ch, ae_ch)
        self.dense2 = nn.Linear(ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense1(x)
        x = self.dense2(x)
        return x.view(x.shape[0], self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res)

    def get_out_ch(self) -> int:
        return self.ae_out_ch


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3) -> None:
        """
        Residule convolutional block

        Args:
            ch (int): The number of input and output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")

    def forward(self, inp: Tensor) -> Tensor:
        x = self.conv1(inp)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(inp + x, 0.2)
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch: int, d_ch: int, o_ch: int = 3, w_dim: int = 512) -> None:
        """
        The docoder block

        Args:
            in_ch (int): The number of input channels
            d_ch (int): Parameter to control the number of output channels in each of the Upscale blocks
            d_mask_ch (int): Parameter to control the number of output channels in the mask Upscale block
            image_output_channels (int): The number of channels in the output image
        """
        super().__init__()

        self.upscale0 = Upscale(in_ch, d_ch * 8, kernel_size=3)
        self.upscale1 = Upscale(d_ch * 8, d_ch * 8, kernel_size=3)
        self.upscale2 = Upscale(d_ch * 8, d_ch * 4, kernel_size=3)
        self.upscale3 = Upscale(d_ch * 4, d_ch * 2, kernel_size=3)

        self.res0 = ResidualBlock(d_ch * 8, kernel_size=3)
        self.res1 = ResidualBlock(d_ch * 8, kernel_size=3)
        self.res2 = ResidualBlock(d_ch * 4, kernel_size=3)
        self.res3 = ResidualBlock(d_ch * 2, kernel_size=3)

        self.syle_delta0 = StyleDeltaBlock(d_ch * 8, w_dim)
        self.syle_delta1 = StyleDeltaBlock(d_ch * 8, w_dim)
        self.syle_delta2 = StyleDeltaBlock(d_ch * 4, w_dim)
        self.syle_delta3 = StyleDeltaBlock(d_ch * 2, w_dim)

        self.out_conv = nn.Conv2d(d_ch * 2, o_ch, kernel_size=1, padding="same")
        self.out_conv1 = nn.Conv2d(d_ch * 2, o_ch, kernel_size=3, padding="same")
        self.out_conv2 = nn.Conv2d(d_ch * 2, o_ch, kernel_size=3, padding="same")
        self.out_conv3 = nn.Conv2d(d_ch * 2, o_ch, kernel_size=3, padding="same")
        self.pixel_shuffle_out = nn.PixelShuffle(2)

    def forward(self, inp: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:

        x = self.upscale0(inp)
        x = self.res0(x)
        x = self.syle_delta0(x, w)

        x = self.upscale1(x)
        x = self.res1(x)
        x = self.syle_delta1(x, w)

        x = self.upscale2(x)
        x = self.res2(x)
        x = self.syle_delta2(x, w)

        x = self.upscale3(x)
        x = self.res3(x)
        x = self.syle_delta3(x, w)

        x0 = self.out_conv(x)
        x1 = self.out_conv1(x)
        x2 = self.out_conv2(x)
        x3 = self.out_conv3(x)

        x_concat = torch.cat([x0, x1, x2, x3], dim=1)

        x = torch.sigmoid(self.pixel_shuffle_out(x_concat))

        return x


class StyleDeltaBlock(nn.Module):
    def __init__(self, ch: int, w_dim: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

        self.to_gamma = nn.Linear(w_dim, ch)
        self.to_beta = nn.Linear(w_dim, ch)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        h = self.conv1(F.silu(x))

        gamma = self.to_gamma(w).view(w.size(0), -1, 1, 1)
        beta = self.to_beta(w).view(w.size(0), -1, 1, 1)

        h = h * (1.0 + gamma) + beta
        h = self.conv2(F.silu(h))

        return x + self.alpha * h


class Generator(nn.Module):
    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        e_dims: int = 64,
        ae_dims: int = 256,
        d_dims: int = 64,
        id_dim: int = 512,
        w_dim: int = 256,
    ) -> None:
        super().__init__()

        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}

        mapping_layers = [nn.Linear(id_dim, w_dim), nn.SiLU()]
        for _ in range(2):
            mapping_layers += [nn.Linear(w_dim, w_dim), nn.SiLU()]
        mapping_layers += [nn.Linear(w_dim, w_dim)]
        self.mapping = nn.Sequential(*mapping_layers)

        self.encoder = Encoder(in_ch=3, e_ch=e_dims)
        encoder_out_ch = self.encoder.get_output_length(input_resolution=img_resolution)

        lowest_dense_res = img_resolution // 32
        self.inter_B = Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims * 2, lowest_dense_res=lowest_dense_res)
        # self.inter_AB = Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims * 2, lowest_dense_res=lowest_dense_res)

        # inter_AB_out_ch = self.inter_AB.get_out_ch()
        inter_B_out_ch = self.inter_B.get_out_ch()
        # inters_out_ch = inter_AB_out_ch + inter_B_out_ch

        self.decoder = Decoder(in_ch=inter_B_out_ch * 2, d_ch=d_dims, o_ch=img_channels, w_dim=w_dim)

    def forward(self, x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:

        w = self.mapping(w)

        code = self.encoder(x)
        inter_B = self.inter_B(code)
        inter_concat = torch.cat((inter_B, inter_B), dim=1)
        x = self.decoder(inter_concat, w)

        return x

    def get_attention_maps(self) -> list[Tensor]:
        return []


if __name__ == "__main__":
    import torch
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    network_cfg = {
        "img_resolution": 256,
        "img_channels": 3,
        "e_dims": 64,
        "ae_dims": 256,
        "d_dims": 64,
        "id_dim": 512,
        "w_dim": 256,
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
