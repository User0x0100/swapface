from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.scale = in_channels**-0.5

        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        h_ = x
        h_: Tensor = self.norm(h_)
        q: Tensor = self.q(h_)
        k: Tensor = self.k(h_)
        v: Tensor = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * self.scale
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels: int | None = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in: Tensor) -> Tensor:
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        nf: int,
        emb_dim: int,
        ch_mult: list[int] | tuple[int, ...],
        num_res_blocks: int,
        resolution: int,
        attn_resolutions: list[int] | tuple[int, ...],
    ):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks: list[nn.Module] = []
        # initial convultion
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        blocks.extend([
            ResBlock(block_in_ch, block_in_ch),
            AttnBlock(block_in_ch),
            ResBlock(block_in_ch, block_in_ch),
            normalize(block_in_ch),
            nn.Conv2d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1),
        ])
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)

        return x


class Codebook(nn.Module):
    def __init__(self, codebook_size: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, emb_dim)

    def lookup(self, indices: Tensor) -> Tensor:
        batch = indices.shape[0]
        x = self.embedding(indices.reshape(-1))
        return x.view(batch, 16, 16, 256).permute(0, 3, 1, 2).contiguous()


class Generator(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.ch_mult = ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        self.out_channels = 3
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)

        blocks: list[nn.Module] = [
            nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1),
            ResBlock(block_in_ch, block_in_ch),
            AttnBlock(block_in_ch),
            ResBlock(block_in_ch, block_in_ch),
        ]

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.extend([
            normalize(block_in_ch),
            nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1),
        ])

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, "The input feature should be 4D tensor."
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dim_mlp: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=0.0)
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, tgt: Tensor, query_pos: Tensor) -> Tensor:
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos
        tgt = tgt + self.self_attn(q, k, value=tgt2)[0]
        tgt2 = self.norm2(tgt)
        return tgt + self.linear2(F.gelu(self.linear1(tgt2)))


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)

        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


class CodeFormer(nn.Module):
    """Inference-only CodeFormer for aligned 512x512 RGB faces."""

    _CONNECT_LIST = ("32", "64", "128", "256")
    _FUSE_ENCODER_SIZES: ClassVar[dict[int, str]] = {5: "256", 8: "128", 11: "64", 14: "32"}
    _FUSE_GENERATOR_SIZES: ClassVar[dict[int, str]] = {9: "32", 12: "64", 15: "128", 18: "256"}

    def __init__(self):
        super().__init__()
        ch_mult = (1, 2, 2, 4, 4, 8)
        self.encoder = Encoder(3, 64, 256, ch_mult, 2, 512, (16,))
        self.quantize = Codebook(1024, 256)
        self.generator = Generator(64, 256, ch_mult, 2, 512, (16,))

        self.position_emb = nn.Parameter(torch.zeros(256, 512))
        self.feat_emb = nn.Linear(256, 512)
        self.ft_layers = nn.Sequential(*[TransformerSALayer(512, nhead=8, dim_mlp=1024) for _ in range(9)])
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1024, bias=False))

        channels = {"32": 256, "64": 256, "128": 128, "256": 128}
        self.fuse_convs_dict = nn.ModuleDict({size: Fuse_sft_block(channels[size], channels[size]) for size in self._CONNECT_LIST})

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Enhance aligned faces.

        Args:
            x: RGB float tensor shaped [B, 3, 512, 512] in approximately [-1, 1].
            w: Fidelity weight tensor shaped [1]. The intended range is [0, 1].

        Returns:
            RGB float tensor shaped [B, 3, 512, 512] in approximately [-1, 1].
        """
        enc_feat_dict: dict[str, Tensor] = {}
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in self._FUSE_ENCODER_SIZES:
                enc_feat_dict[self._FUSE_ENCODER_SIZES[i]] = x
        lq_feat = x

        pos_emb = self.position_emb.unsqueeze(1).expand(-1, x.shape[0], -1)
        query_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        logits = self.idx_pred_layer(query_emb).permute(1, 0, 2)
        top_idx = logits.argmax(dim=2)
        x = self.quantize.lookup(top_idx)
        x = adaptive_instance_normalization(x, lq_feat)

        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if i in self._FUSE_GENERATOR_SIZES:
                f_size = self._FUSE_GENERATOR_SIZES[i]
                x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size], x, w)
        return x.reshape(x.shape[0], 3, 512, 512)
