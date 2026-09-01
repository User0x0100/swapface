import copy
import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn.functional as NF
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from torch import Tensor, optim
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from losses import (
    DiscriminatorAdversarialLoss,
    GeneratorAdversarialLoss,
    IdentityLoss,
    WeightedFeatureMatchingLoss,
    make_l1_loss,
    r1_reg_loss,
)
from misc.face_alignment import center_crop_and_resize
from misc.models.id_encoder import IDEncoder, IDEncoderProvider
from models.discriminator import Discriminator
from models.networks import Generator

from .dataloader import DATALOADER_RESERVED_KEYS, DEFAULT_DATALOADER_CONFIG, ImageDecoderBackend, ImageSource, create_dataloader_pipeline

EPS = 1e-8

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("high")

torch.manual_seed(42)


def print_mapping(title: str, mapping: Mapping[Any, Any], indent: int = 0) -> None:
    print(f"{' ' * indent}{title}:")
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            print_mapping(str(key), value, indent + 2)
        else:
            print(f"{' ' * (indent + 2)}{key!s:25}: {value}")


class Trainer:
    def __init__(
        self,
        src: Sequence[ImageSource],
        dst: Sequence[ImageSource],
        batch_size: int = 16,
        lr: float = 1e-4,
        lr_scheduler_t_max: int = 0,
        r1_reg_step: int = 16,
        r1_gamma: float = 10.0,
        bf16: bool = True,
        device: str = "cuda",
        compile_module: bool = True,
        ckpt: str | None = None,
        log_path: str = "train_log/exper_0",
        log_interval: int = 10,
        sample_save_every: int = 1000,
        weight_save_every: int = 10000,
        # 模型与数据管线配置。dataloader_cfg 覆盖 DEFAULT_DATALOADER_CONFIG，
        # batch_size/device_id/img_resolution/src/dst 由 Trainer 管理，禁止在其中重复指定。
        net_g_cfg: dict[str, Any] | None = None,
        net_d_cfg: dict[str, Any] | None = None,
        dataloader_cfg: dict[str, Any] | None = None,
        # 身份损失
        id_loss_weight: float = 10.0,
        # 重建损失
        enable_rec_loss: bool = True,
        rec_loss_weight: float = 10.0,
        # 判别器中间特征的弱特征匹配
        enable_wfm_loss: bool = False,
        wfm_loss_weight: dict[int, float] | None = None,
    ):
        if dataloader_cfg is None:
            dataloader_cfg = dict(DEFAULT_DATALOADER_CONFIG)
        else:
            reserved_keys = DATALOADER_RESERVED_KEYS.intersection(dataloader_cfg)
            if reserved_keys:
                names = ", ".join(sorted(reserved_keys))
                raise ValueError(f"dataloader_cfg 不能覆盖保留字段：{names}")
            dataloader_cfg = DEFAULT_DATALOADER_CONFIG | dataloader_cfg

        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正数，实际为 {batch_size}")
        if lr <= 0.0:
            raise ValueError(f"lr 必须为正数，实际为 {lr}")
        if lr_scheduler_t_max < 0:
            raise ValueError(f"lr_scheduler_t_max 不能为负数，实际为 {lr_scheduler_t_max}")
        if r1_reg_step <= 0:
            raise ValueError(f"r1_reg_step 必须为正数，实际为 {r1_reg_step}")
        if r1_gamma < 0.0:
            raise ValueError(f"r1_gamma 不能为负数，实际为 {r1_gamma}")
        if log_interval <= 0 or sample_save_every <= 0 or weight_save_every <= 0:
            raise ValueError("log_interval、sample_save_every 和 weight_save_every 必须为正数")
        if enable_wfm_loss and not wfm_loss_weight:
            raise ValueError("enable_wfm_loss=True 时 wfm_loss_weight 不能为空")
        if wfm_loss_weight is None:
            wfm_loss_weight = {}

        args = locals().copy()
        for k in ["src", "dst", "self"]:
            args.pop(k)

        print_mapping("训练信息", args)

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Trainer 仅支持 NVIDIA CUDA 设备")

        self.batch_size = batch_size
        self.r1_reg_step = r1_reg_step
        self.r1_gamma = r1_gamma
        self.enable_rec_loss = enable_rec_loss
        self.enable_wfm_loss = enable_wfm_loss

        self.bf16 = bool(bf16 and torch.cuda.is_bf16_supported())
        if bf16 and not self.bf16:
            print("警告：当前显卡不支持 BF16")

        self.sample_save_every = sample_save_every
        self.weight_save_every = weight_save_every
        self.log_interval = log_interval
        self.use_cosine_lr = lr_scheduler_t_max > 0

        # ========================= 初始化模型 =========================

        if ckpt is not None:
            if not Path(ckpt).exists():
                raise FileNotFoundError(f"找不到检查点文件：{ckpt}")

            print(f"正在加载检查点：{ckpt}")
            ckpt: dict[str, Any] = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=False)

            self.iter = ckpt["iter"]

            print(f"检查点信息：\n  {'迭代次数':25}: {self.iter}")
            print_mapping("net_g", ckpt["net_g"]["network_cfg"])
            print_mapping("net_d", ckpt["net_d"]["network_cfg"])

            self.img_resolution = ckpt["net_g"]["network_cfg"]["img_resolution"]

            net_g = Generator(**ckpt["net_g"]["network_cfg"])
            net_d = Discriminator(**ckpt["net_d"]["network_cfg"])
            net_g.load_state_dict(ckpt["net_g"]["state_dict"])
            net_d.load_state_dict(ckpt["net_d"]["state_dict"])

        else:
            if net_g_cfg is None or net_d_cfg is None:
                raise ValueError("net_g_cfg 和 net_d_cfg 在未提供 ckpt 时不能为空")
            self.iter, self.img_resolution = 0, net_g_cfg["img_resolution"]
            net_g = Generator(**net_g_cfg)
            net_d = Discriminator(**net_d_cfg)

        d_resolution = int(net_d.network_cfg["img_resolution"])
        if d_resolution != self.img_resolution:
            raise ValueError(f"生成器与判别器分辨率不一致：{self.img_resolution} != {d_resolution}")

        group_size = min(int(net_d.network_cfg["group_size"]), self.batch_size)
        if self.batch_size % group_size != 0:
            raise ValueError(f"batch_size={self.batch_size} 必须能被判别器 minibatch group_size={group_size} 整除")

        self.net_g = net_g.to(self.device).train()
        self.net_d = net_d.to(self.device).train()

        self.net_g_ema = copy.deepcopy(self.net_g)
        self.net_g_ema.eval().requires_grad_(False)

        # ========================= 优化器 =========================
        self.optim_g = optim.Adam(self.net_g.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)
        self.optim_d = optim.Adam(self.net_d.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)

        if self.use_cosine_lr:
            self.lr_scheduler_g = CosineAnnealingLR(self.optim_g, T_max=lr_scheduler_t_max, eta_min=lr * 0.1)
            self.lr_scheduler_d = CosineAnnealingLR(self.optim_d, T_max=lr_scheduler_t_max, eta_min=lr * 0.1)

        # ========================= 损失 =========================

        self.d_loss = DiscriminatorAdversarialLoss(weight=1.0, reduction="mean").to(self.device)
        self.gan_loss = GeneratorAdversarialLoss(weight=1.0, reduction="mean").to(self.device)

        self.generator_id_encoder = IDEncoder(IDEncoderProvider.BLENDFACE).to(self.device).eval().requires_grad_(False)
        self.id_loss = IdentityLoss(weight=id_loss_weight, provider=IDEncoderProvider.MS1MV3_ARCFACE_R50_FP16).to(self.device)

        if self.enable_rec_loss:
            self.rec_loss = make_l1_loss(weight=rec_loss_weight, reduction="mean")

        if self.enable_wfm_loss:
            self.wfm_loss = WeightedFeatureMatchingLoss(layer_weights=wfm_loss_weight, criterion="l1").to(self.device)

        # ========================= 日志 =========================
        base_log_path = Path(log_path)
        self.ckpt_dir = base_log_path / "ckpt"
        self.sample_dir = base_log_path / "sample"
        self.tensorboard_dir = base_log_path / "tensorboard"

        for path in (self.ckpt_dir, self.sample_dir, self.tensorboard_dir):
            path.mkdir(exist_ok=True, parents=True)

        self.log_writer = SummaryWriter(self.tensorboard_dir)

        # ========================= 数据采样 =========================

        device_id = self.device.index if self.device.index is not None else torch.cuda.current_device()
        pipe = create_dataloader_pipeline(
            batch_size=self.batch_size,
            device_id=device_id,
            img_resolution=self.img_resolution,
            src=src,
            dst=dst,
            **dataloader_cfg,
        )

        self.sample_output_map = ["src", "dst", "theta_restore"]
        self.dataset = DALIGenericIterator(pipelines=pipe, output_map=self.sample_output_map, auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)

        # ========================= 编译模型 =========================
        if compile_module:
            self.train_g = torch.compile(self.net_g, fullgraph=True, dynamic=False, options={"max_autotune": True, "epilogue_fusion": True})
            self.train_d = torch.compile(self.net_d, fullgraph=True, dynamic=False, options={"max_autotune": True, "epilogue_fusion": True})
        else:
            self.train_g = self.net_g
            self.train_d = self.net_d

    @torch.no_grad()
    def log(self, key: str, value: Tensor) -> None:
        if self.iter % self.log_interval == 0:
            self.log_writer.add_scalar(f"Loss/{key}", value.detach().mean().item(), self.iter)

    @torch.no_grad()
    def fetch_sample(self) -> tuple[Tensor, Tensor, Tensor]:
        data: dict[str, Tensor] = self.dataset.next()[0]
        src, dst, theta_restore = (data[k] for k in self.sample_output_map)
        return src, dst, theta_restore

    @torch.no_grad()
    def update_ema(self, decay: float = 0.999) -> None:

        decay = min(decay, 1 - 1 / (self.iter + 1))
        alpha = 1.0 - decay

        for p_ema, p_train in zip(self.net_g_ema.parameters(), self.net_g.parameters()):
            p_ema.lerp_(p_train, alpha)

    @torch.no_grad()
    def save_ckpt(self) -> None:

        net_g = {
            "network_cfg": self.net_g_ema.network_cfg,
            "state_dict": self.net_g_ema.state_dict(),
        }

        net_d = {
            "network_cfg": self.net_d.network_cfg,
            "state_dict": self.net_d.state_dict(),
        }

        state_dict = {
            "iter": self.iter,
            "net_g": net_g,
            "net_d": net_d,
        }
        try:
            ckpt_file = self.ckpt_dir / f"{self.iter}.pth"
            torch.save(state_dict, ckpt_file)
        except (OSError, RuntimeError) as e:
            print(f"保存检查点失败：{e}")

    def loss_grad_map(self, loss: Tensor, x: Tensor) -> Tensor:
        (grad,) = torch.autograd.grad(outputs=loss.sum(), inputs=x, retain_graph=False, create_graph=False)

        h = grad.detach().float().abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]

        h = torch.log1p(h)
        h = h / h.amax(dim=(2, 3), keepdim=True).clamp_min(EPS)

        h = h.mul(2.0).sub(1.0)  # [0,1] -> [-1,1]
        h = h.expand(-1, 3, -1, -1).contiguous()

        return h

    def train(self) -> None:

        net_g, net_d = self.train_g, self.train_d
        sample_src, sample_dst, sample_theta_restore = self.fetch_sample()

        for iteration in tqdm(itertools.count(start=self.iter), initial=self.iter, mininterval=1.0, bar_format="{n_fmt:7} | 速度 {rate_fmt:3} | 训练时间 {elapsed}"):
            self.iter = iteration
            src, dst, theta_restore = self.fetch_sample()

            torch.compiler.cudagraph_mark_step_begin()

            # ========================= 生成器前向 =========================
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
                with torch.no_grad():
                    source_faces = center_crop_and_resize(src)
                    generator_identity_embeddings = self.generator_id_encoder(source_faces)
                    source_identity_embeddings = self.id_loss.extract_identity_embeddings(source_faces)
                fake: Tensor = net_g(dst, generator_identity_embeddings)

            # ========================= 训练判别器 =========================
            self.net_d.requires_grad_(True)
            self.optim_d.zero_grad(set_to_none=True)
            is_r1_reg_step = self.iter % self.r1_reg_step == 0

            if is_r1_reg_step:
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                    fake_img = fake.detach().float()
                    real_img = dst.detach().float().requires_grad_(True)

                    # R1：使用原始判别器并以 FP32 计算
                    fake_score = self.net_d(fake_img)
                    real_score = self.net_d(real_img)

                    d_loss = self.d_loss(fake_score, real_score)
                    self.log("d_loss", d_loss)

                    r1_loss_raw = r1_reg_loss(real_score, real_img, gamma=self.r1_gamma)
                    self.log("r1_loss_raw", r1_loss_raw)
                    r1_loss = r1_loss_raw * self.r1_reg_step
                    self.log("r1_loss", r1_loss)
                    d_loss = d_loss + r1_loss
            else:
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
                    fake_score = net_d(fake.detach())
                    real_score = net_d(dst.detach())

                    d_loss = self.d_loss(fake_score, real_score)
                    self.log("d_loss", d_loss)

            d_loss.backward()
            self.optim_d.step()

            if self.use_cosine_lr:
                self.lr_scheduler_d.step()

            # ========================= 训练生成器 =========================
            self.net_d.requires_grad_(False)
            self.optim_g.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
                # gan_loss
                if self.enable_wfm_loss:
                    fake_score, fake_feats = net_d(fake, True)
                else:
                    fake_score = net_d(fake)

                gan_loss = self.gan_loss(fake_score)
                self.log("gan_loss", gan_loss)
                g_loss = gan_loss

                # wfm_loss
                if self.enable_wfm_loss:
                    with torch.no_grad():
                        _, real_feats = net_d(dst, True)
                    wfm_loss = self.wfm_loss(fake_feats, real_feats)
                    self.log("wfm_loss", wfm_loss)
                    g_loss = g_loss + wfm_loss

                # id_loss
                grid = NF.affine_grid(theta_restore, size=list(fake.shape), align_corners=False)
                fake_restored = NF.grid_sample(fake, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
                generated_identity_embeddings = self.id_loss.extract_identity_embeddings(center_crop_and_resize(fake_restored))
                id_loss = self.id_loss(generated_identity_embeddings, source_identity_embeddings)
                self.log("id_loss", id_loss)
                g_loss = g_loss + id_loss

                # rec_loss
                if self.enable_rec_loss:
                    rec_loss = self.rec_loss(fake, dst)
                    self.log("rec_loss", rec_loss)
                    g_loss = g_loss + rec_loss

            g_loss.backward()
            self.optim_g.step()

            if self.use_cosine_lr:
                self.lr_scheduler_g.step()

            self.update_ema()

            if self.iter % self.weight_save_every == 0:
                self.save_ckpt()

            if self.iter % self.sample_save_every == 0:
                with torch.no_grad():
                    half = self.batch_size // 2
                    src_vis = torch.cat((sample_src[:half], src[: self.batch_size - half]), dim=0)
                    dst_vis = torch.cat((sample_dst[:half], dst[: self.batch_size - half]), dim=0)

                    theta_restore_vis = torch.cat((sample_theta_restore[:half], theta_restore[: self.batch_size - half]), dim=0)
                    grid_vis = NF.affine_grid(theta_restore_vis, size=list(fake.shape), align_corners=False)
                    dst_restored_vis = NF.grid_sample(dst_vis, grid_vis, mode="bilinear", padding_mode="reflection", align_corners=False)

                    source_faces_vis = center_crop_and_resize(src_vis)
                    generator_identity_embeddings_vis = self.generator_id_encoder(source_faces_vis)
                    source_identity_embeddings_vis = self.id_loss.extract_identity_embeddings(source_faces_vis)
                    fake_vis: Tensor = self.net_g_ema(dst_vis, generator_identity_embeddings_vis)

                    grid = [src_vis, dst_vis, fake_vis, dst_restored_vis]

                    # ========================= GAN 损失梯度图 =========================
                    fake_for_gan_grad = fake_vis.detach().requires_grad_(True)
                    with torch.enable_grad():
                        fake_score_vis = net_d(fake_for_gan_grad)
                        gan_loss_vis = self.gan_loss(fake_score_vis)
                        gan_grad_map = self.loss_grad_map(gan_loss_vis, fake_for_gan_grad)
                    grid.append(gan_grad_map)

                    # ========================= 身份损失梯度图 =========================
                    with torch.enable_grad():
                        fake_for_id_grad = fake_vis.detach().requires_grad_(True)
                        grid_id_vis = NF.affine_grid(theta_restore_vis, size=list(fake_for_id_grad.shape), align_corners=False)
                        fake_for_id_grad_restored = NF.grid_sample(fake_for_id_grad, grid_id_vis, mode="bilinear", padding_mode="reflection", align_corners=False)

                        generated_identity_embeddings_vis = self.id_loss.extract_identity_embeddings(center_crop_and_resize(fake_for_id_grad_restored))
                        id_loss_vis = self.id_loss(generated_identity_embeddings_vis, source_identity_embeddings_vis.detach())
                        id_grad_map = self.loss_grad_map(id_loss_vis, fake_for_id_grad)

                    grid.append(id_grad_map)

                    grid = torch.cat(grid, dim=0)
                    grid.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
                    grid = make_grid(grid, nrow=self.batch_size)[[2, 1, 0], :, :]  # RGB → BGR
                    grid = grid.permute(1, 2, 0)  # CHW → HWC
                    grid_cpu = grid.to(device="cpu", dtype=torch.uint8).numpy()
                cv2.imwrite(self.sample_dir / f"{self.iter}.png", grid_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3])


if __name__ == "__main__":
    src = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
        # ("/opt/share/deepfake/dataset_1/vggface2_hq512/align_result", 0.0),
    ]

    dst = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/vggface2_hq512/align_result", 0.0),
        ("/opt/share/deepfake/dataset_1/RealOcc/image/realign_arcface_dst", 1.0),
        # ("/opt/share/deepfake/dataset_1/oneman/1_align_results/", 0.0),
    ]

    def_config: dict[str, Any] = {
        "src": src,
        "dst": dst,
        "ckpt": "train_log/512-MS1MV3_ARCFACE_R50_FP16/ckpt/328696.pth",
        "log_path": "train_log/512-MS1MV3_ARCFACE_R50_FP16",
        "batch_size": 16,
        "dataloader_cfg": {
            "num_threads": 16,
            "prefetch_queue_depth": 4,
            "py_num_workers": 8,
            "py_start_method": "spawn",
            "reader_prefetch_queue_depth": 2,
            "decoder_backend": ImageDecoderBackend.MIXED,
            "decoder_hw_load": 0.75,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "flip_prob": 0.5,
            "rotation_range": (-10.0, 10.0),
            "scale_factor_range": (1.0 / 1.3, 1.25),
            "tx_range": (-0.15, 0.15),
            "ty_range": (-0.15, 0.15),
        },
        "net_g_cfg": {
            "img_resolution": 512,
            "img_channels": 3,
            "num_depth": 5,
            "num_latent": 6,
            "base_ch": 16,
            "max_ch": 2048,
            "id_dim": 512,
            "skip": True,
        },
        "net_d_cfg": {
            "img_resolution": 512,
            "img_channels": 3,
            "base_ch": 64,
            "max_ch": 512,
            "group_size": 4,
        },
    }

    trainer = Trainer(**def_config)

    try:
        trainer.train()
    finally:
        trainer.save_ckpt()
