"""用于训练损失的可微 FACS Action Unit 模型。

OpenFace 原生 AU 分析器依赖对齐、HOG 与独立的分类/回归预测器，并不是适合
端到端反传的 PyTorch 网络。这里采用 OpenGraphAU / ME-GraphAU 的 MEFARG
Stage-2 作为可微代理，并显式保留 OpenFace AU 子集映射。

网络结构基于 OpenGraphAU / ME-GraphAU；预训练权重来自 mexca 发布的
``mefarg-open-graph-au-resnet50-stage-2``。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from torchvision.models import resnet50

from .. import ImageInputRange

MEFARG_REPOSITORY_ID = "mexca/mefarg-open-graph-au-resnet50-stage-2"
MEFARG_REVISION = "9b25f1a31b79954b415eb7b1dedcb2338c36149f"
MEFARG_WEIGHT_FILENAME = "pytorch_model.bin"

# OpenGraphAU 27 个全局 AU 的固定输出顺序。
MAIN_AU_CODES: tuple[int, ...] = (
    1,
    2,
    4,
    5,
    6,
    7,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    32,
    38,
    39,
)

# 后 14 项是 7 个 AU 的左右侧预测。
SUB_AU_CODES: tuple[str, ...] = (
    "L1",
    "R1",
    "L2",
    "R2",
    "L4",
    "R4",
    "L6",
    "R6",
    "L10",
    "R10",
    "L12",
    "R12",
    "L14",
    "R14",
)

AU_CODES: tuple[int | str, ...] = MAIN_AU_CODES + SUB_AU_CODES

# OpenFace 的 presence/intensity 是两套独立预测器，官方输出集合并不完全相同。
OPENFACE_PRESENCE_AU_CODES: tuple[int, ...] = (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45)
OPENFACE_INTENSITY_AU_CODES: tuple[int, ...] = (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45)

# MEFARG 与两套 OpenFace 集合的公共部分相同：AU28 与 AU45 都无法由本模型提供。
OPENFACE_SUPPORTED_AU_CODES: tuple[int, ...] = tuple(code for code in OPENFACE_INTENSITY_AU_CODES if code in MAIN_AU_CODES)
OPENFACE_UNSUPPORTED_PRESENCE_AU_CODES: tuple[int, ...] = tuple(code for code in OPENFACE_PRESENCE_AU_CODES if code not in MAIN_AU_CODES)
OPENFACE_UNSUPPORTED_INTENSITY_AU_CODES: tuple[int, ...] = tuple(code for code in OPENFACE_INTENSITY_AU_CODES if code not in MAIN_AU_CODES)
_OPENFACE_SUPPORTED_INDICES: tuple[int, ...] = tuple(MAIN_AU_CODES.index(code) for code in OPENFACE_SUPPORTED_AU_CODES)


class _LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)

        self.fc.weight.data.normal_(0.0, math.sqrt(2.0 / out_features))
        self.bn.weight.data.fill_(1.0)
        self.bn.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x


class _AUPredictor(nn.Module):
    def __init__(self, in_features: int, n_main_nodes: int = 27, n_sub_nodes: int = 14) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_main_nodes = n_main_nodes
        self.n_sub_nodes = n_sub_nodes

        self.main_sc = nn.Parameter(torch.zeros(n_main_nodes, in_features))
        self.sub_sc = nn.Parameter(torch.zeros(n_sub_nodes, in_features))
        self.relu = nn.ReLU()
        self.sub_list = (0, 1, 2, 4, 7, 8, 11)

        nn.init.xavier_uniform_(self.main_sc)
        nn.init.xavier_uniform_(self.sub_sc)

    def forward(self, x: Tensor) -> Tensor:
        batch, nodes, channels = x.shape

        main_sc = F.normalize(self.relu(self.main_sc), p=2, dim=-1)
        main_cl = F.normalize(x, p=2, dim=-1)
        main_cl = (main_cl * main_sc.view(1, nodes, channels)).sum(dim=-1)

        sub_cl: list[Tensor] = []
        for i, main_index in enumerate(self.sub_list):
            main_au = F.normalize(x[:, main_index], p=2, dim=-1)
            sc_l = F.normalize(self.relu(self.sub_sc[2 * i]), p=2, dim=-1)
            sc_r = F.normalize(self.relu(self.sub_sc[2 * i + 1]), p=2, dim=-1)
            sub_sc = torch.stack((sc_l, sc_r), dim=0)
            cl = (main_au.unsqueeze(0) * sub_sc.view(2, 1, channels)).sum(dim=-1)
            sub_cl.extend((cl[0].unsqueeze(1), cl[1].unsqueeze(1)))

        return torch.cat((main_cl, torch.cat(sub_cl, dim=-1)), dim=-1)


class _CrossAttention(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        hidden = in_features // 2
        self.linear_q = nn.Linear(in_features, hidden)
        self.linear_k = nn.Linear(in_features, hidden)
        self.linear_v = nn.Linear(in_features, in_features)
        self.scale = hidden**-0.5
        self.attention = nn.Softmax(dim=-1)

        self.linear_q.weight.data.normal_(0.0, math.sqrt(2.0 / hidden))
        self.linear_k.weight.data.normal_(0.0, math.sqrt(2.0 / hidden))
        self.linear_v.weight.data.normal_(0.0, math.sqrt(2.0 / in_features))

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        return self.attention(torch.matmul(query, key.transpose(-2, -1)) * self.scale) @ value


class _GraphEdgeModel(nn.Module):
    def __init__(self, in_features: int, n_nodes: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_nodes = n_nodes
        self.fam = _CrossAttention(in_features)
        self.arm = _CrossAttention(in_features)
        self.edge_proj = nn.Linear(in_features, in_features)
        self.bn = nn.BatchNorm2d(n_nodes * n_nodes)

        self.edge_proj.weight.data.normal_(0.0, math.sqrt(2.0 / in_features))
        self.bn.weight.data.fill_(1.0)
        self.bn.bias.data.zero_()

    def forward(self, node_feature: Tensor, global_feature: Tensor) -> Tensor:
        batch, nodes, spatial, channels = node_feature.shape
        global_feature = global_feature.repeat(1, nodes, 1).view(batch, nodes, spatial, channels)

        feature = self.fam(node_feature, global_feature)
        feature_end = feature.repeat(1, 1, nodes, 1).view(batch, -1, spatial, channels)
        feature_start = feature.repeat(1, nodes, 1, 1).view(batch, -1, spatial, channels)
        return self.bn(self.edge_proj(self.arm(feature_start, feature_end)))


class _GatedGNNLayer(nn.Module):
    def __init__(self, in_features: int, n_nodes: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_nodes = n_nodes

        self.linear_u = nn.Linear(in_features, in_features, bias=False)
        self.linear_v = nn.Linear(in_features, in_features, bias=False)
        self.linear_a = nn.Linear(in_features, in_features, bias=False)
        self.linear_b = nn.Linear(in_features, in_features, bias=False)
        self.linear_e = nn.Linear(in_features, in_features, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.bnv = nn.BatchNorm1d(n_nodes)
        self.bne = nn.BatchNorm1d(n_nodes * n_nodes)
        self.act = nn.ReLU()

        scale = math.sqrt(2.0 / in_features)
        for layer in (self.linear_u, self.linear_v, self.linear_a, self.linear_b, self.linear_e):
            layer.weight.data.normal_(0.0, scale)
        self.bnv.weight.data.fill_(1.0)
        self.bnv.bias.data.zero_()
        self.bne.weight.data.fill_(1.0)
        self.bne.bias.data.zero_()

    def forward(self, x: Tensor, edge: Tensor, start: Tensor, end: Tensor) -> tuple[Tensor, Tensor]:
        residual = x
        v_ix = self.linear_a(x)
        v_jx = self.linear_b(x)
        projected_edge = self.linear_e(edge)

        edge = edge + self.act(
            self.bne(
                torch.einsum("ev,bvc->bec", end, v_ix)
                + torch.einsum("ev,bvc->bec", start, v_jx)
                + projected_edge
            )
        )

        gates = self.sigmoid(edge)
        batch, _, channels = gates.shape
        gates = self.softmax(gates.view(batch, self.n_nodes, self.n_nodes, channels)).view(batch, -1, channels)

        u_jx = torch.einsum("ev,bvc->bec", start, self.linear_v(x))
        x = self.linear_u(x) + torch.einsum("ve,bec->bvc", end.t(), gates * u_jx) / self.n_nodes
        return residual + self.act(self.bnv(x)), edge


class _GatedGNN(nn.Module):
    def __init__(self, in_features: int, n_nodes: int, n_layers: int = 2) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_nodes = n_nodes

        start = torch.eye(n_nodes).repeat(n_nodes, 1)
        end = torch.eye(n_nodes).repeat_interleave(n_nodes, dim=0)
        self.register_buffer("start", start, persistent=False)
        self.register_buffer("end", end, persistent=False)
        self.graph_layers = nn.ModuleList(_GatedGNNLayer(in_features, n_nodes) for _ in range(n_layers))

    def forward(self, x: Tensor, edge: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.graph_layers:
            x, edge = layer(x, edge, self.start, self.end)
        return x, edge


class _MEFL(_AUPredictor):
    def __init__(self, in_features: int, n_main_nodes: int = 27, n_sub_nodes: int = 14) -> None:
        super().__init__(in_features, n_main_nodes, n_sub_nodes)
        self.main_node_linear_layers = nn.ModuleList(_LinearBlock(in_features, in_features) for _ in range(n_main_nodes))
        self.edge_extractor = _GraphEdgeModel(in_features, n_main_nodes)
        self.gnn = _GatedGNN(in_features, n_main_nodes, n_layers=2)

    def forward(self, x: Tensor) -> Tensor:
        node_features = torch.cat([layer(x).unsqueeze(1) for layer in self.main_node_linear_layers], dim=1)
        node_vectors = node_features.mean(dim=-2)
        edge_features = self.edge_extractor(node_features, x).mean(dim=-2)
        node_vectors, _ = self.gnn(node_vectors, edge_features)
        return super().forward(node_vectors)


class OpenFaceAU(nn.Module):
    """冻结的、可对输入反传的 FACS AU 特征提取器。

    ``forward`` 返回 OpenGraphAU 的全部 41 个 occurrence activation score；顺序由
    :data:`AU_CODES` 定义。score 本身就是模型训练使用的连续 [0, 1] 激活量，不再额外
    施加 sigmoid。

    ``forward_openface`` 返回与 OpenFace AU 集合相交的 16 个全局 AU，顺序由
    :data:`OPENFACE_SUPPORTED_AU_CODES` 定义。OpenGraphAU 不预测 AU28 与 AU45，
    因此这两个 AU 不能从该模型中伪造出来。
    """

    INPUT_SIZE = 224
    NUM_MAIN_AUS = 27
    NUM_SUB_AUS = 14
    NUM_AUS = NUM_MAIN_AUS + NUM_SUB_AUS

    def __init__(
        self,
        input_range: ImageInputRange = ImageInputRange.MINUS_ONE_TO_ONE,
        *,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(input_range, ImageInputRange):
            raise TypeError(f"input_range 必须为 ImageInputRange，实际为 {type(input_range).__name__}")

        self.input_range = input_range
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.n_in_channels = 2048
        self.n_out_channels = self.n_in_channels // 4
        self.linear_global = _LinearBlock(self.n_in_channels, self.n_out_channels)
        self.head = _MEFL(self.n_out_channels, self.NUM_MAIN_AUS, self.NUM_SUB_AUS)

        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("openface_indices", torch.tensor(_OPENFACE_SUPPORTED_INDICES, dtype=torch.long), persistent=False)

        if pretrained:
            checkpoint_path = hf_hub_download(repo_id=MEFARG_REPOSITORY_ID, filename=MEFARG_WEIGHT_FILENAME, revision=MEFARG_REVISION)
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(state_dict, dict):
                raise TypeError(f"MEFARG checkpoint 必须为 state_dict，实际为 {type(state_dict).__name__}")
            self.load_state_dict(state_dict, strict=True)

        # 冻结教师参数，但 forward 绝不能使用 no_grad：loss 需要梯度回到生成图像。
        self.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> OpenFaceAU:
        """冻结教师始终保持 eval，避免外层 loss.train() 改写 BatchNorm/Dropout 状态。"""
        super().train(False)
        return self

    def _prepare_input(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(f"images 必须为 NCHW，实际 shape={tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"images 必须为 RGB 三通道，实际 C={images.shape[1]}")

        x = images.float()
        match self.input_range:
            case ImageInputRange.ZERO_TO_255:
                x = x.div(255.0)
            case ImageInputRange.ZERO_TO_ONE:
                pass
            case ImageInputRange.MINUS_ONE_TO_ONE:
                x = x.add(1.0).mul(0.5)

        if x.shape[-2:] == (256, 256):
            offset = (256 - self.INPUT_SIZE) // 2
            x = x[:, :, offset : offset + self.INPUT_SIZE, offset : offset + self.INPUT_SIZE]
        elif x.shape[-2:] != (self.INPUT_SIZE, self.INPUT_SIZE):
            x = F.interpolate(x, size=(self.INPUT_SIZE, self.INPUT_SIZE), mode="bilinear", align_corners=False, antialias=True)
        return (x - self.get_buffer("image_mean")) / self.get_buffer("image_std")

    def _forward_features(self, x: Tensor) -> Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        batch, channels, _, _ = x.shape
        return x.view(batch, channels, -1).permute(0, 2, 1)

    def forward(self, images: Tensor) -> Tensor:
        """返回全部 41 个 FACS AU occurrence activation score。"""
        x = self._prepare_input(images)
        x = self.linear_global(self._forward_features(x))
        return self.head(x)

    def forward_openface(self, images: Tensor) -> Tensor:
        """返回模型实际支持的 16 个 OpenFace AU score。"""
        return self.forward(images).index_select(1, self.get_buffer("openface_indices"))
