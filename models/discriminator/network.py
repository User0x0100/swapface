import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    """U-Net dense discriminator with spectral normalization.

    The encoder builds a large receptive field while the decoder restores spatial
    resolution, producing one realism logit per output location instead of a
    single image-level score.
    """

    feature_count = 4

    def __init__(self, img_resolution: int = 256, img_channels: int = 3, base_ch: int = 64, max_ch: int = 512) -> None:
        super().__init__()

        self.network_cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}

        c0 = base_ch
        c1 = min(max_ch, base_ch * 2)
        c2 = min(max_ch, base_ch * 4)
        c3 = min(max_ch, base_ch * 8)

        self.conv0 = nn.Conv2d(img_channels, c0, kernel_size=3, stride=1, padding=1)

        self.conv1 = spectral_norm(nn.Conv2d(c0, c1, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False))

        self.conv4 = spectral_norm(nn.Conv2d(c3, c2, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(c1, c0, kernel_size=3, stride=1, padding=1, bias=False))

        self.conv7 = spectral_norm(nn.Conv2d(c0, c0, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(c0, c0, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv9 = nn.Conv2d(c0, 1, kernel_size=3, stride=1, padding=1)

    def _encode(self, x: Tensor, max_layer: int | None = None) -> list[Tensor]:
        feats = [F.leaky_relu(self.conv0(x), negative_slope=0.2)]
        if max_layer == 0:
            return feats

        feats.append(F.leaky_relu(self.conv1(feats[-1]), negative_slope=0.2))
        if max_layer == 1:
            return feats

        feats.append(F.leaky_relu(self.conv2(feats[-1]), negative_slope=0.2))
        if max_layer == 2:
            return feats

        feats.append(F.leaky_relu(self.conv3(feats[-1]), negative_slope=0.2))
        return feats

    def get_feats(self, x: Tensor, max_layer: int | None = None) -> list[Tensor]:
        if max_layer is not None and not 0 <= max_layer <= 3:
            raise ValueError(f"max_layer 必须在 [0, 3]，实际为 {max_layer}")
        return self._encode(x, max_layer)

    def forward(self, x: Tensor, return_feats: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        x0, x1, x2, x3 = self._encode(x)

        x4 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x4), negative_slope=0.2)
        x4 = x4 + x2

        x5 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x5), negative_slope=0.2)
        x5 = x5 + x1

        x6 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x6), negative_slope=0.2)
        x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2)
        out = self.conv9(out)

        return (out, [x0, x1, x2, x3]) if return_feats else out
