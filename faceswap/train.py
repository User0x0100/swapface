import argparse
import copy
import inspect
import itertools
import tomllib
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
    VGGPerceptualLoss,
    WeightedFeatureMatchingLoss,
    make_l1_loss,
    r1_reg_loss,
)
from misc.face_alignment import ffhq_to_arcface_112, make_ffhq_to_arcface_112_grid, transform_sampling_grid
from misc.models.id_encoder import IDEncoder, IDEncoderProvider
from models.discriminator import Discriminator
from models.discriminator.upfirdn2d import initialize_upfirdn2d
from models.networks import Generator

from .contracts import CHECKPOINT_VERSION
from .dataloader import DATALOADER_RESERVED_KEYS, DEFAULT_DATALOADER_CONFIG, ImageDecoderBackend, ImageSource, create_dataloader_pipeline
from .experiment import RunLock, RunPaths, append_resume_event, config_sha256, create_run, load_resolved_config, read_json, resolve_resume_target, update_metadata, write_latest

EPS = 1e-8
DEFAULT_GENERATOR_ID_ENCODER_PROVIDER = IDEncoderProvider.BLENDFACE
DEFAULT_IDENTITY_LOSS_PROVIDER = IDEncoderProvider.MS1MV3_ARCFACE_R50_FP16
DEFAULT_VGG_PERCEPTUAL_LOSS_WEIGHT: dict[str, float] = {
    "conv1_2": 2.5,
    "conv2_2": 2.5,
    "conv3_3": 2.5,
    "conv4_3": 2.5,
}
DEFAULT_WFM_LOSS_WEIGHT: dict[int, float] = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "experiments" / "train.toml"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "experiments" / "runs"

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
        log_interval: int = 10,
        sample_save_every: int = 1000,
        weight_save_every: int = 10000,
        # 模型与数据管线配置。dataloader_cfg 覆盖 DEFAULT_DATALOADER_CONFIG，
        # batch_size/device_id/img_resolution/src/dst 由 Trainer 管理，禁止在其中重复指定。
        net_g_cfg: dict[str, Any] | None = None,
        net_d_cfg: dict[str, Any] | None = None,
        dataloader_cfg: dict[str, Any] | None = None,
        # 身份编码与身份损失
        generator_id_encoder_provider: IDEncoderProvider = DEFAULT_GENERATOR_ID_ENCODER_PROVIDER,
        identity_loss_provider: IDEncoderProvider = DEFAULT_IDENTITY_LOSS_PROVIDER,
        id_loss_weight: float = 10.0,
        # 重建损失
        enable_rec_loss: bool = True,
        rec_loss_weight: float = 10.0,
        # VGG19 感知特征损失
        enable_perceptual_loss: bool = True,
        perceptual_loss_weight: dict[str, float] | None = None,
        # 判别器浅层/中层特征的弱特征匹配
        enable_wfm_loss: bool = True,
        wfm_loss_weight: dict[int, float] | None = None,
        run_dir: str | Path = "train_log/exper_0",
        resume_checkpoint: str | Path | None = None,
        run_id: str | None = None,
        resolved_config_sha256: str | None = None,
        strict_bf16_resume: bool = False,
    ):
        if dataloader_cfg is None:
            dataloader_cfg = dict(DEFAULT_DATALOADER_CONFIG)
        else:
            reserved_keys = DATALOADER_RESERVED_KEYS.intersection(dataloader_cfg)
            if reserved_keys:
                names = ", ".join(sorted(reserved_keys))
                raise ValueError(f"dataloader_cfg 不能覆盖保留字段：{names}")
            dataloader_cfg = DEFAULT_DATALOADER_CONFIG | dataloader_cfg

        if not isinstance(generator_id_encoder_provider, IDEncoderProvider):
            raise TypeError(f"generator_id_encoder_provider 必须为 IDEncoderProvider，实际为 {type(generator_id_encoder_provider).__name__}")
        if not isinstance(identity_loss_provider, IDEncoderProvider):
            raise TypeError(f"identity_loss_provider 必须为 IDEncoderProvider，实际为 {type(identity_loss_provider).__name__}")
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
        if perceptual_loss_weight is None:
            perceptual_loss_weight = dict(DEFAULT_VGG_PERCEPTUAL_LOSS_WEIGHT)
        if enable_perceptual_loss and not perceptual_loss_weight:
            raise ValueError("enable_perceptual_loss=True 时 perceptual_loss_weight 不能为空")
        if wfm_loss_weight is None:
            wfm_loss_weight = dict(DEFAULT_WFM_LOSS_WEIGHT)
        if enable_wfm_loss and not wfm_loss_weight:
            raise ValueError("enable_wfm_loss=True 时 wfm_loss_weight 不能为空")

        args = locals().copy()
        for k in ["src", "dst", "self"]:
            args.pop(k)

        print_mapping("训练信息", args)

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Trainer 仅支持 NVIDIA CUDA 设备")

        self.batch_size = batch_size
        self.generator_id_encoder_provider = generator_id_encoder_provider
        self.identity_loss_provider = identity_loss_provider
        self.r1_reg_step = r1_reg_step
        self.r1_gamma = r1_gamma
        self.enable_rec_loss = enable_rec_loss
        self.enable_perceptual_loss = enable_perceptual_loss
        self.enable_wfm_loss = enable_wfm_loss

        device_id = self.device.index if self.device.index is not None else torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(device_id)
        bf16_runtime_supported = torch.cuda.is_bf16_supported()
        bf16_compile_supported = compute_capability[0] >= 8
        self.bf16 = bool(bf16 and bf16_runtime_supported and (not compile_module or bf16_compile_supported))
        if bf16 and not self.bf16:
            if compile_module and bf16_runtime_supported and not bf16_compile_supported:
                print(f"警告：当前 GPU SM{compute_capability[0]}{compute_capability[1]} 可运行 BF16，但 torch.compile 不支持该架构的 BF16，训练自动回退 FP32")
            else:
                print("警告：当前显卡不支持 BF16，训练自动回退 FP32")

        self.sample_save_every = sample_save_every
        self.weight_save_every = weight_save_every
        self.log_interval = log_interval
        self.use_cosine_lr = lr_scheduler_t_max > 0
        self.training_config = {
            "batch_size": self.batch_size,
            "bf16": self.bf16,
            "lr": lr,
            "r1_reg_step": self.r1_reg_step,
            "r1_gamma": self.r1_gamma,
            "id_loss_weight": id_loss_weight,
            "enable_rec_loss": self.enable_rec_loss,
            "rec_loss_weight": rec_loss_weight,
            "enable_perceptual_loss": self.enable_perceptual_loss,
            "perceptual_loss_weight": dict(perceptual_loss_weight),
            "enable_wfm_loss": self.enable_wfm_loss,
            "wfm_loss_weight": dict(wfm_loss_weight),
            "lr_scheduler_t_max": lr_scheduler_t_max,
        }

        # ========================= 初始化模型 =========================

        checkpoint: dict[str, Any] | None = None
        training_state: dict[str, Any] | None = None
        if resume_checkpoint is not None:
            checkpoint_path = Path(resume_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"找不到检查点文件：{resume_checkpoint}")

            print(f"正在加载检查点：{resume_checkpoint}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_version = int(checkpoint["version"])
            if checkpoint_version != CHECKPOINT_VERSION:
                raise ValueError(f"不支持的检查点版本：{checkpoint_version}，当前仅支持 v{CHECKPOINT_VERSION}")

            completed_iter = int(checkpoint["iter"])
            completed_step = int(checkpoint.get("step", completed_iter + 1))
            self.iter = int(checkpoint["next_iter"])

            print(f"检查点信息：\n  {'版本':25}: {checkpoint_version}\n  {'已完成 step':25}: {completed_step}\n  {'恢复内部 iter':25}: {self.iter}")
            print_mapping("net_g", checkpoint["net_g"]["network_cfg"])
            print_mapping("net_d", checkpoint["net_d"]["network_cfg"])

            checkpoint_run = checkpoint.get("run")
            if resolved_config_sha256 is not None or run_id is not None:
                if not isinstance(checkpoint_run, dict):
                    raise ValueError("标准 run 的 checkpoint 缺少 run 元数据，不能安全 resume")
                saved_digest = checkpoint_run.get("config_sha256")
                if resolved_config_sha256 is not None and saved_digest != resolved_config_sha256:
                    raise ValueError(f"检查点与 run 的 resolved config 不一致：{saved_digest} != {resolved_config_sha256}")
                saved_run_id = checkpoint_run.get("id")
                if run_id is not None and saved_run_id != run_id:
                    raise ValueError(f"检查点所属 run 不匹配：{saved_run_id} != {run_id}")

            saved_training_config = checkpoint["training_config"]
            strict_resume_keys = [
                "lr",
                "lr_scheduler_t_max",
                "r1_reg_step",
                "r1_gamma",
                "id_loss_weight",
                "enable_rec_loss",
                "rec_loss_weight",
                "enable_perceptual_loss",
                "perceptual_loss_weight",
                "enable_wfm_loss",
                "wfm_loss_weight",
            ]
            if strict_bf16_resume:
                strict_resume_keys.append("bf16")
            mismatches = {key: (saved_training_config[key], self.training_config[key]) for key in strict_resume_keys if saved_training_config[key] != self.training_config[key]}
            if mismatches:
                raise ValueError(f"检查点训练目标配置与当前配置不一致：{mismatches}")

            identity_encoders = checkpoint["identity_encoders"]
            saved_generator_provider = identity_encoders["generator"]
            saved_loss_provider = identity_encoders["identity_loss"]
            if saved_generator_provider != self.generator_id_encoder_provider.name:
                raise ValueError(f"检查点 Generator 身份编码器不匹配：{saved_generator_provider} != {self.generator_id_encoder_provider.name}")
            if saved_loss_provider != self.identity_loss_provider.name:
                raise ValueError(f"检查点身份损失编码器不匹配：{saved_loss_provider} != {self.identity_loss_provider.name}")

            self.img_resolution = int(checkpoint["net_g"]["network_cfg"]["img_resolution"])
            net_g = Generator(**checkpoint["net_g"]["network_cfg"])
            net_d = Discriminator(**checkpoint["net_d"]["network_cfg"])
            net_d.load_state_dict(checkpoint["net_d"]["state_dict"])

            training_state = checkpoint["training_state"]
            net_g.load_state_dict(training_state["net_g"])
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
        if checkpoint is not None:
            self.net_g_ema.load_state_dict(checkpoint["net_g"]["state_dict"])
        self.net_g_ema.eval().requires_grad_(False)
        self._ema_params = tuple(self.net_g_ema.parameters())
        self._train_g_params = tuple(self.net_g.parameters())

        # ========================= 优化器 =========================
        self.optim_g = optim.Adam(self.net_g.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)
        self.optim_d = optim.Adam(self.net_d.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)

        if self.use_cosine_lr:
            self.lr_scheduler_g = CosineAnnealingLR(self.optim_g, T_max=lr_scheduler_t_max, eta_min=lr * 0.1)
            self.lr_scheduler_d = CosineAnnealingLR(self.optim_d, T_max=lr_scheduler_t_max, eta_min=lr * 0.1)

        if training_state is not None:
            self.optim_g.load_state_dict(training_state["optim_g"])
            self.optim_d.load_state_dict(training_state["optim_d"])
            saved_scheduler_g = training_state["lr_scheduler_g"]
            saved_scheduler_d = training_state["lr_scheduler_d"]
            if self.use_cosine_lr:
                if saved_scheduler_g is None or saved_scheduler_d is None:
                    raise ValueError("检查点未包含余弦学习率调度器状态，但当前配置启用了调度器")
                self.lr_scheduler_g.load_state_dict(saved_scheduler_g)
                self.lr_scheduler_d.load_state_dict(saved_scheduler_d)
            elif saved_scheduler_g is not None or saved_scheduler_d is not None:
                raise ValueError("检查点包含学习率调度器状态，但当前配置未启用调度器")
            print("已恢复 Generator、Discriminator、优化器和学习率调度器训练状态")

        # ========================= 损失 =========================

        self.d_loss = DiscriminatorAdversarialLoss(weight=1.0, reduction="mean").to(self.device)
        self.gan_loss = GeneratorAdversarialLoss(weight=1.0, reduction="mean").to(self.device)

        self.generator_id_encoder = IDEncoder(self.generator_id_encoder_provider).to(self.device).eval().requires_grad_(False)
        self.id_loss = IdentityLoss(weight=id_loss_weight, provider=self.identity_loss_provider).to(self.device)
        # 训练数据默认使用 FFHQ canonical alignment。Generator 身份编码器与 Identity Loss
        # 共用同一套 FFHQ -> ArcFace 112 canonical 映射；sampling grid 仅初始化一次。
        self.identity_encoder_grid = make_ffhq_to_arcface_112_grid(self.img_resolution, self.batch_size, self.device)

        if self.enable_rec_loss:
            self.rec_loss = make_l1_loss(weight=rec_loss_weight, reduction="mean")

        if self.enable_perceptual_loss:
            self.perceptual_loss = VGGPerceptualLoss(layer_weights=perceptual_loss_weight, reduction="mean").to(self.device)

        if self.enable_wfm_loss:
            feature_count = len(self.net_d.down_blocks)
            invalid_layers = sorted(index for index in wfm_loss_weight if index >= feature_count)
            if invalid_layers:
                raise ValueError(f"wfm_loss_weight 层索引超出判别器特征范围 0~{feature_count - 1}：{invalid_layers}")
            self.wfm_loss = WeightedFeatureMatchingLoss(layer_weights=wfm_loss_weight, criterion="l1").to(self.device)
            self.wfm_max_layer = max(wfm_loss_weight)

        # ========================= Run 输出 =========================
        self.run_paths = RunPaths.from_root(run_dir)
        self.run_id = run_id
        self.resolved_config_sha256 = resolved_config_sha256
        self.ckpt_dir = self.run_paths.checkpoints
        self.sample_dir = self.run_paths.samples
        self.tensorboard_dir = self.run_paths.tensorboard

        for path in (self.ckpt_dir, self.sample_dir, self.tensorboard_dir):
            path.mkdir(exist_ok=True, parents=True)

        # ========================= 数据采样 =========================

        pipe = create_dataloader_pipeline(batch_size=self.batch_size, device_id=device_id, img_resolution=self.img_resolution, src=src, dst=dst, **dataloader_cfg)

        self.sample_output_map = ["src", "dst", "theta_restore"]
        self.dataset = DALIGenericIterator(pipelines=pipe, output_map=self.sample_output_map, auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)

        # ========================= 编译模型 =========================
        if compile_module:
            initialize_upfirdn2d()
            self.train_g = torch.compile(self.net_g, fullgraph=True, dynamic=False, options={"max_autotune": True, "epilogue_fusion": True})
            self.train_d = torch.compile(self.net_d, fullgraph=True, dynamic=False, options={"max_autotune": True, "epilogue_fusion": True})
            self.generator_id_encoder_forward = torch.compile(self.generator_id_encoder, fullgraph=True, dynamic=False, options={"max_autotune": True, "epilogue_fusion": True})
            if self.enable_wfm_loss:
                self.train_d_features = torch.compile(self.net_d.get_feats, fullgraph=True, dynamic=False, options={"max_autotune": True, "epilogue_fusion": True})
        else:
            self.train_g = self.net_g
            self.train_d = self.net_d
            self.generator_id_encoder_forward = self.generator_id_encoder
            if self.enable_wfm_loss:
                self.train_d_features = self.net_d.get_feats

        # TensorBoard writer 最后创建，避免初始化模型/数据管线失败时遗留后台资源。
        tensorboard_purge_step = None
        if checkpoint is not None:
            checkpoint_step = int(checkpoint.get("step", int(checkpoint["iter"]) + 1))
            tensorboard_purge_step = checkpoint_step + 1
        self.log_writer = SummaryWriter(self.tensorboard_dir, purge_step=tensorboard_purge_step)
        self._log_buffer: dict[str, Tensor] = {}

    @property
    def completed_step(self) -> int:
        """已完成的优化 step 数；内部 iter 仍保持 0-based 以兼容 checkpoint v2。"""
        return self.iter + 1

    @torch.no_grad()
    def log(self, key: str, value: Tensor) -> None:
        if self.completed_step % self.log_interval == 0:
            self._log_buffer[key] = value.detach().mean()

    @torch.no_grad()
    def flush_logs(self) -> None:
        if not self._log_buffer:
            return
        keys = tuple(self._log_buffer)
        values = torch.stack(tuple(self._log_buffer[key] for key in keys)).float().cpu().tolist()
        self._log_buffer.clear()
        for key, value in zip(keys, values):
            self.log_writer.add_scalar(f"Loss/{key}", value, self.completed_step)

    @torch.no_grad()
    def fetch_sample(self) -> tuple[Tensor, Tensor, Tensor]:
        data: dict[str, Tensor] = self.dataset.next()[0]
        src, dst, theta_restore = (data[k] for k in self.sample_output_map)
        return src, dst, theta_restore

    def prepare_identity_encoder_faces(self, faces: Tensor, theta_restore: Tensor | None = None) -> Tensor:
        """将 FFHQ canonical 训练人脸映射为身份编码器使用的 ArcFace 112 canonical 输入。"""
        if theta_restore is None:
            return ffhq_to_arcface_112(faces, self.identity_encoder_grid)
        grid = transform_sampling_grid(self.identity_encoder_grid, theta_restore)
        return ffhq_to_arcface_112(faces, grid, padding_mode="reflection")

    @torch.no_grad()
    def update_ema(self, decay: float = 0.999) -> None:

        decay = min(decay, 1 - 1 / (self.iter + 1))
        alpha = 1.0 - decay

        torch._foreach_lerp_(self._ema_params, self._train_g_params, alpha)

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
        training_state = {
            "net_g": self.net_g.state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "lr_scheduler_g": self.lr_scheduler_g.state_dict() if self.use_cosine_lr else None,
            "lr_scheduler_d": self.lr_scheduler_d.state_dict() if self.use_cosine_lr else None,
        }
        completed_step = self.completed_step
        state_dict = {
            "version": CHECKPOINT_VERSION,
            "iter": self.iter,
            "next_iter": self.iter + 1,
            "step": completed_step,
            "run": {
                "id": self.run_id,
                "config_sha256": self.resolved_config_sha256,
            },
            "identity_encoders": {
                "generator": self.generator_id_encoder_provider.name,
                "identity_loss": self.identity_loss_provider.name,
            },
            "training_config": self.training_config,
            "net_g": net_g,
            "net_d": net_d,
            "training_state": training_state,
        }

        ckpt_file = self.ckpt_dir / f"step_{completed_step:09d}.pth"
        temp_file = ckpt_file.with_suffix(".pth.tmp")
        try:
            torch.save(state_dict, temp_file)
            temp_file.replace(ckpt_file)
            write_latest(self.run_paths, ckpt_file, completed_step)
        except (OSError, RuntimeError) as e:
            temp_file.unlink(missing_ok=True)
            raise RuntimeError(f"保存检查点失败：{e}") from e

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
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16), torch.no_grad():
                source_identity_faces = self.prepare_identity_encoder_faces(src)
                generator_identity_embeddings = self.generator_id_encoder_forward(source_identity_faces)
                source_identity_embeddings = self.id_loss.extract_identity_embeddings(source_identity_faces)
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
                        real_feats = self.train_d_features(dst, self.wfm_max_layer)
                    wfm_loss = self.wfm_loss(fake_feats, real_feats)
                    self.log("wfm_loss", wfm_loss)
                    g_loss = g_loss + wfm_loss

                # id_loss
                generated_identity_embeddings = self.id_loss.extract_identity_embeddings(self.prepare_identity_encoder_faces(fake, theta_restore))
                id_loss = self.id_loss(generated_identity_embeddings, source_identity_embeddings)
                self.log("id_loss", id_loss)
                g_loss = g_loss + id_loss

                # perceptual_loss
                if self.enable_perceptual_loss:
                    perceptual_loss = self.perceptual_loss(fake, dst)
                    self.log("perceptual_loss", perceptual_loss)
                    g_loss = g_loss + perceptual_loss

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
            self.flush_logs()

            if self.completed_step % self.weight_save_every == 0:
                self.save_ckpt()

            if self.completed_step % self.sample_save_every == 0:
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
                    half = self.batch_size // 2
                    src_vis = torch.cat((sample_src[:half], src[: self.batch_size - half]), dim=0)
                    dst_vis = torch.cat((sample_dst[:half], dst[: self.batch_size - half]), dim=0)

                    theta_restore_vis = torch.cat((sample_theta_restore[:half], theta_restore[: self.batch_size - half]), dim=0)
                    grid_vis = NF.affine_grid(theta_restore_vis, size=list(fake.shape), align_corners=False)
                    dst_restored_vis = NF.grid_sample(dst_vis, grid_vis, mode="bilinear", padding_mode="reflection", align_corners=False)

                    source_identity_faces_vis = self.prepare_identity_encoder_faces(src_vis)
                    generator_identity_embeddings_vis = self.generator_id_encoder_forward(source_identity_faces_vis)
                    source_identity_embeddings_vis = self.id_loss.extract_identity_embeddings(source_identity_faces_vis)
                    fake_vis: Tensor = self.net_g_ema(dst_vis, generator_identity_embeddings_vis)

                    # Identity Loss 编码器真正接收的图像；直接组合 restore + FFHQ->112，
                    # 避免先恢复到全分辨率再二次重采样。仅为 sample grid 显示再放大回训练分辨率。
                    identity_encoder_input_vis = self.prepare_identity_encoder_faces(fake_vis, theta_restore_vis)
                    identity_encoder_input_display_vis = NF.interpolate(identity_encoder_input_vis, size=fake_vis.shape[2:], mode="bilinear", align_corners=False)

                    grid = [src_vis, dst_vis, fake_vis, dst_restored_vis, identity_encoder_input_display_vis]

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
                        generated_identity_embeddings_vis = self.id_loss.extract_identity_embeddings(self.prepare_identity_encoder_faces(fake_for_id_grad, theta_restore_vis))
                        id_loss_vis = self.id_loss(generated_identity_embeddings_vis, source_identity_embeddings_vis.detach())
                        id_grad_map = self.loss_grad_map(id_loss_vis, fake_for_id_grad)

                    grid.append(id_grad_map)

                    grid = torch.cat(grid, dim=0)
                    grid.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
                    grid = make_grid(grid, nrow=self.batch_size)[[2, 1, 0], :, :]  # RGB → BGR
                    grid = grid.permute(1, 2, 0)  # CHW → HWC
                    grid_cpu = grid.to(device="cpu", dtype=torch.uint8).numpy()
                sample_file = self.sample_dir / f"step_{self.completed_step:09d}.png"
                if not cv2.imwrite(sample_file, grid_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise OSError(f"保存训练 sample 失败：{sample_file}")


TRAIN_SECTION_PARAMETERS = (
    "batch_size",
    "lr",
    "lr_scheduler_t_max",
    "r1_reg_step",
    "r1_gamma",
    "bf16",
    "device",
    "compile_module",
    "log_interval",
    "sample_save_every",
    "weight_save_every",
)
DATALOADER_RANGE_KEYS = ("rotation_range", "scale_factor_range", "tx_range", "ty_range")


def _parameter_defaults(callable_obj: Any, values: dict[str, Any], names: Sequence[str], section: str) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError(f"{section} 包含未知字段：{sorted(unknown)}")

    resolved: dict[str, Any] = {}
    for name in names:
        if name in values:
            resolved[name] = values[name]
            continue
        default = parameters[name].default
        if default is inspect.Parameter.empty:
            raise ValueError(f"{section}.{name} 必须显式配置")
        resolved[name] = default
    return resolved


def _resolve_model_config(model_type: type[Any], values: dict[str, Any], section: str) -> dict[str, Any]:
    parameters = inspect.signature(model_type.__init__).parameters
    names = tuple(name for name in parameters if name != "self")
    return _parameter_defaults(model_type.__init__, values, names, section)


def _normalize_image_sources(entries: object, section: str) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"[[{section}]] 数据源不能为空")

    normalized: list[dict[str, Any]] = []
    allowed_fields = {"path", "adjustment"}
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise TypeError(f"[[{section}]] 第 {index} 项必须是表/对象")
        entry = dict(raw_entry)
        unknown = set(entry) - allowed_fields
        if unknown:
            raise ValueError(f"[[{section}]] 第 {index} 项包含未知字段：{sorted(unknown)}；训练数据源仅支持本地 path/adjustment")

        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"[[{section}]] 第 {index} 项 path 必须为非空字符串")
        try:
            adjustment = float(entry.get("adjustment", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[[{section}]] 第 {index} 项 adjustment 必须为数值") from exc
        normalized.append({"path": path, "adjustment": adjustment})
    return normalized


def _load_image_sources(entries: list[dict[str, Any]]) -> list[ImageSource]:
    return [(str(entry["path"]), float(entry["adjustment"])) for entry in entries]


def _normalize_dataloader(values: dict[str, Any]) -> dict[str, Any]:
    unknown = set(values) - set(DEFAULT_DATALOADER_CONFIG)
    if unknown:
        raise ValueError(f"[dataloader] 包含未知字段：{sorted(unknown)}")
    config = dict(DEFAULT_DATALOADER_CONFIG) | values

    try:
        decoder_backend = config["decoder_backend"]
        if not isinstance(decoder_backend, ImageDecoderBackend):
            decoder_backend = ImageDecoderBackend(decoder_backend)
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in ImageDecoderBackend)
        raise ValueError(f"dataloader.decoder_backend={config['decoder_backend']!r} 无效，可选：{supported}") from exc
    config["decoder_backend"] = decoder_backend.value

    for key in DATALOADER_RANGE_KEYS:
        value = config[key]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"dataloader.{key} 必须为包含两个数值的数组")
        config[key] = [float(value[0]), float(value[1])]
    return config


def resolve_train_config(config: dict[str, Any]) -> dict[str, Any]:
    """校验配置并展开所有代码默认值，得到可哈希、可持久化的规范配置。"""
    allowed_sections = {"train", "identity", "loss", "dataloader", "generator", "discriminator", "src", "dst"}
    unknown_sections = set(config) - allowed_sections
    if unknown_sections:
        raise ValueError(f"配置包含未知顶层字段：{sorted(unknown_sections)}")

    for section in ("train", "identity", "loss", "dataloader", "generator", "discriminator"):
        value = config.get(section, {})
        if not isinstance(value, dict):
            raise TypeError(f"[{section}] 必须为表/对象")

    train = _parameter_defaults(Trainer.__init__, dict(config.get("train", {})), TRAIN_SECTION_PARAMETERS, "[train]")

    identity = dict(config.get("identity", {}))
    unknown_identity = set(identity) - {"generator_provider", "loss_provider", "loss_weight"}
    if unknown_identity:
        raise ValueError(f"[identity] 包含未知字段：{sorted(unknown_identity)}")
    generator_provider = str(identity.get("generator_provider", DEFAULT_GENERATOR_ID_ENCODER_PROVIDER.name))
    loss_provider = str(identity.get("loss_provider", DEFAULT_IDENTITY_LOSS_PROVIDER.name))
    try:
        IDEncoderProvider[generator_provider]
        IDEncoderProvider[loss_provider]
    except KeyError as exc:
        supported = ", ".join(provider.name for provider in IDEncoderProvider)
        raise ValueError(f"身份编码器类型无效，可选：{supported}") from exc

    loss = dict(config.get("loss", {}))
    unknown_loss = set(loss) - {"enable_rec_loss", "rec_loss_weight", "vgg", "wfm"}
    if unknown_loss:
        raise ValueError(f"[loss] 包含未知字段：{sorted(unknown_loss)}")

    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    enable_rec_loss = bool(loss.get("enable_rec_loss", trainer_parameters["enable_rec_loss"].default))
    rec_loss_weight = float(loss.get("rec_loss_weight", trainer_parameters["rec_loss_weight"].default))

    vgg = loss.get("vgg", {})
    if not isinstance(vgg, dict):
        raise TypeError("[loss.vgg] 必须为表/对象")
    unknown_vgg = set(vgg) - {"enable", "weights"}
    if unknown_vgg:
        raise ValueError(f"[loss.vgg] 包含未知字段：{sorted(unknown_vgg)}")
    enable_vgg = bool(vgg.get("enable", trainer_parameters["enable_perceptual_loss"].default))
    raw_vgg_weights = vgg.get("weights", DEFAULT_VGG_PERCEPTUAL_LOSS_WEIGHT)
    if not isinstance(raw_vgg_weights, dict) or (enable_vgg and not raw_vgg_weights):
        raise ValueError("[loss.vgg.weights] 必须为非空表/对象")
    vgg_weights = {str(layer): float(weight) for layer, weight in raw_vgg_weights.items()}

    wfm = loss.get("wfm", {})
    if not isinstance(wfm, dict):
        raise TypeError("[loss.wfm] 必须为表/对象")
    unknown_wfm = set(wfm) - {"enable", "weights"}
    if unknown_wfm:
        raise ValueError(f"[loss.wfm] 包含未知字段：{sorted(unknown_wfm)}")
    enable_wfm = bool(wfm.get("enable", trainer_parameters["enable_wfm_loss"].default))
    raw_wfm_weights = wfm.get("weights", DEFAULT_WFM_LOSS_WEIGHT)
    if not isinstance(raw_wfm_weights, dict) or (enable_wfm and not raw_wfm_weights):
        raise ValueError("[loss.wfm.weights] 必须为非空表/对象")
    try:
        wfm_weights = {str(int(index)): float(weight) for index, weight in raw_wfm_weights.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("[loss.wfm.weights] 的键必须是整数层索引，值必须是数值") from exc

    return {
        "train": train,
        "identity": {
            "generator_provider": generator_provider,
            "loss_provider": loss_provider,
            "loss_weight": float(identity.get("loss_weight", trainer_parameters["id_loss_weight"].default)),
        },
        "loss": {
            "enable_rec_loss": enable_rec_loss,
            "rec_loss_weight": rec_loss_weight,
            "vgg": {"enable": enable_vgg, "weights": vgg_weights},
            "wfm": {"enable": enable_wfm, "weights": wfm_weights},
        },
        "dataloader": _normalize_dataloader(dict(config.get("dataloader", {}))),
        "generator": _resolve_model_config(Generator, dict(config.get("generator", {})), "[generator]"),
        "discriminator": _resolve_model_config(Discriminator, dict(config.get("discriminator", {})), "[discriminator]"),
        "src": _normalize_image_sources(config.get("src"), "src"),
        "dst": _normalize_image_sources(config.get("dst"), "dst"),
    }


def _runtime_train_config(resolved: dict[str, Any]) -> dict[str, Any]:
    dataloader = dict(resolved["dataloader"])
    dataloader["decoder_backend"] = ImageDecoderBackend(dataloader["decoder_backend"])
    for key in DATALOADER_RANGE_KEYS:
        dataloader[key] = tuple(float(value) for value in dataloader[key])

    identity = resolved["identity"]
    loss = resolved["loss"]
    return {
        **resolved["train"],
        "src": _load_image_sources(resolved["src"]),
        "dst": _load_image_sources(resolved["dst"]),
        "generator_id_encoder_provider": IDEncoderProvider[identity["generator_provider"]],
        "identity_loss_provider": IDEncoderProvider[identity["loss_provider"]],
        "id_loss_weight": float(identity["loss_weight"]),
        "enable_rec_loss": bool(loss["enable_rec_loss"]),
        "rec_loss_weight": float(loss["rec_loss_weight"]),
        "enable_perceptual_loss": bool(loss["vgg"]["enable"]),
        "perceptual_loss_weight": {str(layer): float(weight) for layer, weight in loss["vgg"]["weights"].items()},
        "enable_wfm_loss": bool(loss["wfm"]["enable"]),
        "wfm_loss_weight": {int(index): float(weight) for index, weight in loss["wfm"]["weights"].items()},
        "dataloader_cfg": dataloader,
        "net_g_cfg": dict(resolved["generator"]),
        "net_d_cfg": dict(resolved["discriminator"]),
    }


def load_train_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取用户 TOML，返回 Trainer 参数与完整 resolved config。"""
    config_path = Path(path)
    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)
    resolved = resolve_train_config(raw_config)
    return _runtime_train_config(resolved), resolved


def _load_run_config(paths: RunPaths) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = load_resolved_config(paths)
    canonical = resolve_train_config(resolved)
    if canonical != resolved:
        raise ValueError(f"run 的 resolved config 不是当前格式的规范表示：{paths.resolved_config}")
    return _runtime_train_config(canonical), canonical


def _framework_metadata(requested_device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": requested_device,
    }
    try:
        device = torch.device(requested_device)
    except (TypeError, RuntimeError):
        return result
    if device.type == "cuda" and torch.cuda.is_available():
        device_id = device.index if device.index is not None else torch.cuda.current_device()
        if 0 <= device_id < torch.cuda.device_count():
            result["cuda_device"] = torch.cuda.get_device_name(device_id)
            result["compute_capability"] = list(torch.cuda.get_device_capability(device_id))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceSwap 训练")
    parser.add_argument("--config", type=Path, default=None, help=f"新 run 的训练 TOML；未指定 --resume 时默认：{DEFAULT_TRAIN_CONFIG_PATH}")
    parser.add_argument("--name", type=str, default=None, help="新 run 的可选短标签；run ID 仍包含唯一时间戳")
    parser.add_argument("--runs-root", type=Path, default=None, help=f"新 run 根目录，默认：{DEFAULT_RUNS_ROOT}")
    parser.add_argument("--resume", type=Path, default=None, help="恢复标准 run 目录，或其中 checkpoints/*.pth；恢复时使用 run 冻结配置")
    parser.add_argument("--strict-bf16", action="store_true", help="仅用于 resume：要求当前实际 BF16/FP32 模式与 checkpoint 一致；默认允许变化")
    args = parser.parse_args()

    if args.resume is not None and args.config is not None:
        parser.error("--resume 与 --config 不能同时使用；resume 必须使用原 run 的冻结配置")
    if args.resume is not None and args.name is not None:
        parser.error("--resume 与 --name 不能同时使用；resume 继续写入原 run")
    if args.resume is not None and args.runs_root is not None:
        parser.error("--resume 与 --runs-root 不能同时使用；resume 继续写入原 run")
    if args.strict_bf16 and args.resume is None:
        parser.error("--strict-bf16 仅用于 --resume；fresh training 无需设置 resume 兼容策略")

    is_resume = args.resume is not None
    if not is_resume:
        config_path = args.config or DEFAULT_TRAIN_CONFIG_PATH
        runs_root = args.runs_root or DEFAULT_RUNS_ROOT
        trainer_config, resolved = load_train_config(config_path)
        run_paths = create_run(runs_root, config_path, resolved, name=args.name, project_root=PROJECT_ROOT)
        resume_checkpoint = None
    else:
        run_paths, resume_checkpoint = resolve_resume_target(args.resume)
        trainer_config, resolved = _load_run_config(run_paths)

    run_metadata = read_json(run_paths.metadata)
    if not isinstance(run_metadata, dict) or not isinstance(run_metadata.get("run_id"), str):
        raise TypeError(f"无效的 run_id：{run_paths.metadata}")
    run_id = run_metadata["run_id"]
    digest = config_sha256(resolved)
    framework = _framework_metadata(str(resolved["train"]["device"]))

    with RunLock(run_paths):
        if is_resume:
            if resume_checkpoint is None:
                raise AssertionError("resume checkpoint 未解析")
            append_resume_event(run_paths, resume_checkpoint)
            update_metadata(run_paths, framework=framework)
        else:
            update_metadata(run_paths, status="running", framework=framework)

        trainer: Trainer | None = None
        try:
            trainer = Trainer(
                **trainer_config,
                run_dir=run_paths.root,
                resume_checkpoint=resume_checkpoint,
                run_id=run_id,
                resolved_config_sha256=digest,
                strict_bf16_resume=args.strict_bf16,
            )
            print(f"Run 目录：{run_paths.root}")
            trainer.train()
        except KeyboardInterrupt:
            if trainer is not None:
                trainer.save_ckpt()
            update_metadata(run_paths, status="interrupted")
            print("训练已中断；已保存当前 checkpoint，可使用 --resume 继续该 run")
            raise SystemExit(130) from None
        except BaseException as exc:
            update_metadata(run_paths, status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        else:
            update_metadata(run_paths, status="completed")
        finally:
            if trainer is not None:
                trainer.log_writer.close()


if __name__ == "__main__":
    main()
