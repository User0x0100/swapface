import argparse
import copy
import signal
from collections.abc import Iterable, Mapping
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import cv2
import torch
import torch.nn.functional as NF
from torch import Tensor, optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from losses import (
    DiscriminatorAdversarialLoss,
    FACSConsistencyLoss,
    GazeLoss,
    GeneratorAdversarialLoss,
    HRFFAFacialGeometryLoss,
    IdentityLoss,
    VGGPerceptualLoss,
    WeightedFeatureMatchingLoss,
    make_l1_loss,
    r1_reg_loss,
)
from misc.face_alignment import ffhq_to_arcface_112, make_ffhq_to_arcface_112_grid, transform_sampling_grid
from misc.models.id_encoder import IDEncoder, IDEncoderProvider
from models.discriminator import Discriminator
from models.discriminator.upfirdn2d import initialize_upfirdn2d, is_rocm_gfx1100
from models.networks import Generator

from .config import load_train_config, resolve_train_config
from .contracts import CHECKPOINT_VERSION
from .dataloader import ImageDecoderBackend, TrainingDataLoader
from .experiment import (
    RunLock,
    RunPaths,
    checkpoint_step_from_name,
    config_sha256,
    create_run,
    load_metadata,
    load_resolved_config,
    resolve_branch_target,
    resolve_resume_target,
    update_metadata,
    write_latest,
)

EPS = 1e-8
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "experiments" / "train.toml"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "experiments" / "runs"
MAX_AMP_OVERFLOW_RETRIES = 8
TRAINING_SEMANTICS_VERSION = 3


def _configure_training_runtime() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = torch.version.hip is None
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)


def _supports_compiled_bf16(device_id: int) -> bool:
    """ROCm 由 PyTorch 报告 BF16 能力；NVIDIA 继续要求 Ampere(SM80)+。"""
    if torch.version.hip is not None:
        return True
    return torch.cuda.get_device_capability(device_id)[0] >= 8


def _compile_training_callable(fn: Any) -> Any:
    mode = "default" if torch.version.hip is not None else "max-autotune-no-cudagraphs"
    return torch.compile(fn, fullgraph=True, dynamic=False, mode=mode)


def _ensure_finite_gradients(name: str, named_parameters: Iterable[tuple[str, Tensor]]) -> None:
    """在 optimizer.step 前阻止 BF16/FP32 非有限梯度污染参数与优化器状态。"""
    gradients = [(parameter_name, parameter.grad) for parameter_name, parameter in named_parameters if parameter.grad is not None]
    if not gradients:
        return

    # infinity norm 不会像 L2 norm 那样因平方/求和而把有限大值误判为 Inf；
    # foreach 让常规路径按 tensor list 批量归约，最终只同步一次标量结果。
    gradient_norms = torch._foreach_norm([gradient.detach() for _, gradient in gradients], ord=float("inf"))
    if bool(torch.isfinite(torch.stack(gradient_norms)).all().item()):
        return

    for (parameter_name, gradient), gradient_norm in zip(gradients, gradient_norms, strict=True):
        if bool(torch.isfinite(gradient_norm).item()):
            continue
        detached = gradient.detach()
        finite_count = int(torch.isfinite(detached).sum().item())
        total_count = detached.numel()
        raise FloatingPointError(f"{name} gradient 出现 NaN/Inf：parameter={parameter_name}, shape={tuple(detached.shape)}, dtype={detached.dtype}, finite={finite_count}/{total_count}")
    raise AssertionError("检测到非有限梯度，但未找到对应参数")


def _scaled_backward_step(
    loss: Tensor,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    *,
    name: str,
    named_parameters: Iterable[tuple[str, Tensor]],
) -> bool:
    """执行 backward/optimizer step；FP16 overflow 由 GradScaler 恢复，其余精度先验证梯度。"""
    scale_before = scaler.get_scale()
    scaler.scale(loss).backward()

    if not scaler.is_enabled():
        _ensure_finite_gradients(name, named_parameters)
        optimizer.step()
        return True

    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= scale_before


def _ensure_finite_loss(name: str, loss: Tensor) -> None:
    """在进入 backward 前阻止非有限标量损失污染训练状态。"""
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise FloatingPointError(f"{name} 出现 NaN/Inf：{loss.detach().float().cpu().item()}")


def _require_resume_training_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """读取 resume 真正依赖的运行时状态；旧 checkpoint 的额外字段允许保留。"""
    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError("checkpoint.training_config 缺失或无效")

    semantics_version = training_config.get("semantics_version")
    if not isinstance(semantics_version, int) or isinstance(semantics_version, bool):
        raise ValueError("checkpoint.training_config.semantics_version 缺失或无效")
    if semantics_version != TRAINING_SEMANTICS_VERSION:
        raise ValueError(f"训练语义版本不匹配：checkpoint={semantics_version}, current={TRAINING_SEMANTICS_VERSION}")

    precision = training_config.get("precision")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"checkpoint.training_config.precision 无效：{precision!r}")

    return {"semantics_version": semantics_version, "precision": precision}


def _reduce_reconstruction_loss(
    rec_per_sample: Tensor,
    same_mask: Tensor,
    scope: Literal["same", "all"],
) -> Tensor:
    """按配置范围聚合逐样本 reconstruction loss。"""
    if scope == "all":
        return rec_per_sample.mean()
    if scope == "same":
        same_weight = same_mask.to(dtype=rec_per_sample.dtype)
        return (rec_per_sample * same_weight).sum() / same_weight.sum().clamp_min(1.0)
    raise ValueError(f"reconstruction_scope 无效：{scope!r}")


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
        config: Mapping[str, Any],
        *,
        run_dir: str | Path,
        checkpoint_path: str | Path | None = None,
        run_id: str | None = None,
        resolved_config_sha256: str | None = None,
        strict_precision_resume: bool = False,
        checkpoint_mode: Literal["resume", "branch"] = "resume",
        preloaded_checkpoint: dict[str, Any] | None = None,
        reset_discriminator: bool = False,
        start_step: int | None = None,
    ) -> None:
        # config 必须是 resolve_train_config() 产生的 canonical schema。Trainer 不再维护第二套默认值/配置协议。
        train_config = config["train"]
        optimizer_config = config["optimizer"]
        scheduler_config = config["scheduler"]
        identity_config = config["identity"]
        loss_config = config["loss"]
        data_config = config["data"]
        net_g_cfg = dict(config["generator"])
        net_d_cfg = dict(config["discriminator"])
        coarse_resolution = int(net_g_cfg["coarse_resolution"])
        net_d_coarse_cfg = {**net_d_cfg, "img_resolution": coarse_resolution}

        batch_size = int(train_config["batch_size"])
        lr = float(optimizer_config["lr"])
        compile_module = bool(train_config["compile_module"])
        checkpoint_save_every = int(train_config["checkpoint_save_every"])
        self.log_interval = int(train_config["log_interval"])
        self.sample_save_every = int(train_config["sample_save_every"])
        self.checkpoint_save_every = checkpoint_save_every
        self.batch_size = batch_size

        reconstruction_scope = str(loss_config["reconstruction"]["scope"])
        gan_config = loss_config["gan"]
        identity_loss_config = loss_config["identity"]
        l1_config = loss_config["l1"]
        r1_config = loss_config["r1"]
        gaze_config = loss_config["gaze"]
        hrffa_config = loss_config["hrffa"]
        facs_config = loss_config["facs"]
        vgg_config = loss_config["vgg"]
        wfm_config = loss_config["wfm"]

        self.generator_id_encoder_provider = IDEncoderProvider[str(identity_config["provider"])]
        self.identity_loss_provider = IDEncoderProvider[str(identity_loss_config["provider"])]
        self.reconstruction_scope = reconstruction_scope
        self.enable_r1_loss = bool(r1_config["enable"])
        self.r1_reg_step = int(r1_config["interval"])
        self.r1_gamma = float(r1_config["gamma"])
        self.enable_l1_loss = bool(l1_config["enable"])
        self.enable_gaze_loss = bool(gaze_config["enable"])
        self.enable_hrffa_loss = bool(hrffa_config["enable"])
        self.enable_facs_loss = bool(facs_config["enable"])
        self.enable_vgg_loss = bool(vgg_config["enable"])
        self.enable_wfm_loss = bool(wfm_config["enable"])

        dataloader_cfg = {**data_config["loader"], **data_config["augmentation"], **data_config["sampling"]}
        dataloader_cfg["decoder_backend"] = ImageDecoderBackend(dataloader_cfg["decoder_backend"])
        src = [(str(entry["path"]), float(entry["adjustment"])) for entry in data_config["src"]]
        dst = [(str(entry["path"]), float(entry["adjustment"])) for entry in data_config["dst"]]

        print_mapping("训练配置", config)

        self.device = torch.device(str(train_config["device"]))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Trainer 仅支持 PyTorch CUDA/HIP GPU 设备")

        device_id = self.device.index if self.device.index is not None else torch.cuda.current_device()
        requested_precision = str(train_config["precision"])
        if requested_precision == "bf16":
            bf16_runtime_supported = torch.cuda.is_bf16_supported()
            bf16_compile_supported = _supports_compiled_bf16(device_id)
            if bf16_runtime_supported and (not compile_module or bf16_compile_supported):
                self.precision = "bf16"
                self.amp_dtype = torch.bfloat16
            else:
                self.precision = "fp32"
                self.amp_dtype = None
                if compile_module and bf16_runtime_supported and not bf16_compile_supported:
                    major, minor = torch.cuda.get_device_capability(device_id)
                    print(f"警告：当前 GPU SM{major}{minor} 可运行 BF16，但 torch.compile 不支持该架构的 BF16，训练自动回退 FP32")
                else:
                    print("警告：当前显卡不支持 BF16，训练自动回退 FP32")
        elif requested_precision == "fp16":
            self.precision = "fp16"
            self.amp_dtype = torch.float16
        else:
            self.precision = "fp32"
            self.amp_dtype = None
        self.amp_enabled = self.amp_dtype is not None

        scheduler_type = str(scheduler_config["type"])
        self.use_cosine_lr = scheduler_type == "cosine"
        scheduler_t_max = int(scheduler_config["t_max"])
        scheduler_min_lr_ratio = float(scheduler_config["min_lr_ratio"])
        self.training_config = {
            "semantics_version": TRAINING_SEMANTICS_VERSION,
            "precision": self.precision,
        }

        if checkpoint_mode not in ("resume", "branch"):
            raise ValueError(f"checkpoint_mode 无效：{checkpoint_mode!r}")

        self._completed_step = 0
        self._step_in_progress = False
        self._stop_requested = False

        checkpoint: dict[str, Any] | None = preloaded_checkpoint
        training_state: dict[str, Any] | None = None
        saved_precision: str | None = None
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"找不到检查点文件：{checkpoint_path}")

            print(f"正在加载检查点：{checkpoint_path}")
            if checkpoint is None:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_version = checkpoint["version"]
            if checkpoint_version != CHECKPOINT_VERSION:
                raise ValueError(f"不支持的 checkpoint version：{checkpoint_version}，当前仅支持 v{CHECKPOINT_VERSION}")

            checkpoint_step = int(checkpoint["step"])
            filename_step = checkpoint_step_from_name(checkpoint_path.name)
            if filename_step != checkpoint_step:
                raise ValueError(f"checkpoint 文件名 step 与内部状态不一致：filename={filename_step}, checkpoint={checkpoint_step}")

            print(f"检查点信息：\n  {'版本':25}: {checkpoint_version}\n  {'已完成 step':25}: {checkpoint_step}")
            print_mapping("net_g", checkpoint["net_g"]["network_cfg"])
            print_mapping("net_d", checkpoint["net_d"]["network_cfg"])

            saved_net_g_cfg = dict(checkpoint["net_g"]["network_cfg"])
            if saved_net_g_cfg != net_g_cfg:
                raise ValueError("checkpoint Generator 架构与当前配置不一致")

            self.img_resolution = int(net_g_cfg["img_resolution"])
            net_g = Generator(**net_g_cfg)
            net_d = Discriminator(**net_d_cfg)
            net_d_coarse = Discriminator(**net_d_coarse_cfg)

            if checkpoint_mode == "resume":
                self._completed_step = checkpoint_step
                checkpoint_run = checkpoint["run"]
                if checkpoint_run["id"] != run_id or checkpoint_run["config_sha256"] != resolved_config_sha256:
                    raise ValueError("checkpoint 不属于当前 run 或冻结配置已变化")

                saved_training_config = _require_resume_training_config(checkpoint)
                saved_precision = str(saved_training_config["precision"])
                if strict_precision_resume and saved_precision != self.precision:
                    raise ValueError(f"训练精度不一致：checkpoint={saved_precision}，current={self.precision}")

                saved_generator_provider = checkpoint["identity_encoders"]["generator"]
                if saved_generator_provider != self.generator_id_encoder_provider.name:
                    raise ValueError(f"Generator 身份编码器不匹配：{saved_generator_provider} != {self.generator_id_encoder_provider.name}")

                saved_net_d_cfg = dict(checkpoint["net_d"]["network_cfg"])
                if saved_net_d_cfg != net_d_cfg:
                    raise ValueError("checkpoint Discriminator 架构与当前配置不一致")
                saved_net_d_coarse_cfg = dict(checkpoint["net_d_coarse"]["network_cfg"])
                if saved_net_d_coarse_cfg != net_d_coarse_cfg:
                    raise ValueError("checkpoint Coarse Discriminator 架构与当前配置不一致")

                training_state = checkpoint["training_state"]
                net_g.load_state_dict(training_state["net_g"])
                net_d.load_state_dict(checkpoint["net_d"]["state_dict"])
                net_d_coarse.load_state_dict(checkpoint["net_d_coarse"]["state_dict"])
            else:
                # Branch 继承父模型，但 optimizer/scheduler/scaler 重新初始化。
                self._completed_step = checkpoint_step if start_step is None else start_step
                branch_g_state, branch_d_state, branch_d_coarse_state = _branch_model_states(
                    checkpoint, reset_discriminator=reset_discriminator
                )
                net_g.load_state_dict(branch_g_state)
                if branch_d_state is not None:
                    net_d.load_state_dict(branch_d_state)
                if branch_d_coarse_state is not None:
                    net_d_coarse.load_state_dict(branch_d_coarse_state)
        else:
            if checkpoint_mode == "branch":
                raise ValueError("branch 必须提供父 checkpoint")
            self.img_resolution = int(net_g_cfg["img_resolution"])
            net_g = Generator(**net_g_cfg)
            net_d = Discriminator(**net_d_cfg)
            net_d_coarse = Discriminator(**net_d_coarse_cfg)

        d_resolution = int(net_d.network_cfg["img_resolution"])
        if d_resolution != self.img_resolution:
            raise ValueError(f"生成器与判别器分辨率不一致：{self.img_resolution} != {d_resolution}")

        coarse_d_resolution = int(net_d_coarse.network_cfg["img_resolution"])
        if coarse_d_resolution != coarse_resolution:
            raise ValueError(f"Coarse 与 Coarse Discriminator 分辨率不一致：{coarse_resolution} != {coarse_d_resolution}")

        group_size = min(int(net_d.network_cfg["group_size"]), self.batch_size)
        if self.batch_size % group_size != 0:
            raise ValueError(f"batch_size={self.batch_size} 必须能被判别器 minibatch group_size={group_size} 整除")

        self.net_g = net_g.to(self.device).train()
        self.net_d = net_d.to(self.device).train()
        self.net_d_coarse = net_d_coarse.to(self.device).train()

        self.net_g_ema = copy.deepcopy(self.net_g)
        if checkpoint is not None and checkpoint_mode == "resume":
            self.net_g_ema.load_state_dict(checkpoint["net_g"]["state_dict"])
        self.net_g_ema.eval().requires_grad_(False)
        self._ema_params = tuple(self.net_g_ema.parameters())
        self._train_g_params = tuple(self.net_g.parameters())

        # ========================= 优化器 / 调度器 =========================
        self.optim_g = optim.Adam(self.net_g.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)
        self.optim_d = optim.Adam(
            chain(self.net_d.parameters(), self.net_d_coarse.parameters()), lr=lr, betas=(0.0, 0.99), fused=True
        )
        self._d_named_parameters = tuple(
            [(f"final.{name}", parameter) for name, parameter in self.net_d.named_parameters()]
            + [(f"coarse.{name}", parameter) for name, parameter in self.net_d_coarse.named_parameters()]
        )
        scaler_enabled = self.precision == "fp16"
        self.scaler_g = GradScaler("cuda", enabled=scaler_enabled)
        self.scaler_d = GradScaler("cuda", enabled=scaler_enabled)

        if training_state is not None:
            self.optim_g.load_state_dict(training_state["optim_g"])
            self.optim_d.load_state_dict(training_state["optim_d"])

        if training_state is not None and saved_precision == self.precision == "fp16":
            scaler_g_state = training_state.get("scaler_g")
            scaler_d_state = training_state.get("scaler_d")
            if scaler_g_state is not None:
                self.scaler_g.load_state_dict(scaler_g_state)
            if scaler_d_state is not None:
                self.scaler_d.load_state_dict(scaler_d_state)

        if self.use_cosine_lr:
            self.lr_scheduler_g = CosineAnnealingLR(self.optim_g, T_max=scheduler_t_max, eta_min=lr * scheduler_min_lr_ratio)
            self.lr_scheduler_d = CosineAnnealingLR(self.optim_d, T_max=scheduler_t_max, eta_min=lr * scheduler_min_lr_ratio)
            if training_state is not None:
                self.lr_scheduler_g.load_state_dict(training_state["lr_scheduler_g"])
                self.lr_scheduler_d.load_state_dict(training_state["lr_scheduler_d"])

        # ========================= 损失 =========================
        self.d_loss = DiscriminatorAdversarialLoss(weight=1.0, reduction="mean").to(self.device)
        self.gan_loss = GeneratorAdversarialLoss(weight=float(gan_config["weight"]), reduction="mean").to(self.device)

        self.generator_id_encoder = IDEncoder(self.generator_id_encoder_provider).to(self.device).eval().requires_grad_(False)
        self.id_loss = IdentityLoss(weight=float(identity_loss_config["weight"]), provider=self.identity_loss_provider).to(self.device)
        self.coarse_resolution = int(self.net_g.network_cfg["coarse_resolution"])
        self.identity_encoder_grid = make_ffhq_to_arcface_112_grid(self.img_resolution, self.batch_size, self.device)
        self.coarse_identity_encoder_grid = make_ffhq_to_arcface_112_grid(self.coarse_resolution, self.batch_size, self.device)

        if self.enable_l1_loss:
            self.l1_loss = make_l1_loss(weight=float(l1_config["weight"]), reduction="none")

        if self.enable_gaze_loss:
            self.gaze_loss = GazeLoss(
                weight=float(gaze_config["weight"]),
                distribution_weight=float(gaze_config["distribution_weight"]),
                confidence_weighted=bool(gaze_config["confidence_weighted"]),
            ).to(self.device)

        if self.enable_hrffa_loss:
            self.hrffa_loss = HRFFAFacialGeometryLoss(
                pose_weight=float(hrffa_config["pose_weight"]),
                eye_weight=float(hrffa_config["eye_weight"]),
                mouth_weight=float(hrffa_config["mouth_weight"]),
                contour_weight=float(hrffa_config["contour_weight"]),
                contour_shape_weight=float(hrffa_config["contour_shape_weight"]),
                occluded_geometry_weight=float(hrffa_config["occluded_geometry_weight"]),
            ).to(self.device)

        if self.enable_facs_loss:
            self.facs_loss = FACSConsistencyLoss(
                weight=float(facs_config["weight"]),
                brow_weight=float(facs_config["brow_weight"]),
                eye_weight=float(facs_config["eye_weight"]),
                nose_weight=float(facs_config["nose_weight"]),
                mouth_weight=float(facs_config["mouth_weight"]),
                lower_face_weight=float(facs_config["lower_face_weight"]),
                asymmetry_weight=float(facs_config["asymmetry_weight"]),
            ).to(self.device)

        if self.enable_vgg_loss:
            self.vgg_loss = VGGPerceptualLoss(layer_weights=vgg_config["weights"], reduction="none").to(self.device)

        if self.enable_wfm_loss:
            wfm_weights = {int(index): float(weight) for index, weight in wfm_config["weights"].items()}
            feature_count = len(self.net_d.down_blocks)
            invalid_layers = sorted(index for index in wfm_weights if index >= feature_count)
            if invalid_layers:
                raise ValueError(f"loss.wfm.weights 层索引超出判别器特征范围 0~{feature_count - 1}：{invalid_layers}")
            self.wfm_loss = WeightedFeatureMatchingLoss(layer_weights=wfm_weights, criterion="l1").to(self.device)
            self.wfm_max_layer = max(wfm_weights)

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
        self.dataset = TrainingDataLoader(
            batch_size=self.batch_size,
            device=self.device,
            img_resolution=self.img_resolution,
            src=src,
            dst=dst,
            **dataloader_cfg,
        )

        # ========================= 编译模型 =========================
        # Coarse/HQ 分开前向：HQ 只读取 detached Coarse，避免 final losses 反向改写 Coarse。
        if compile_module:
            initialize_upfirdn2d()
            self.train_coarse = _compile_training_callable(self.net_g.coarse)
            self.train_hq = _compile_training_callable(self.net_g.hq)
            self.train_d = _compile_training_callable(self.net_d)
            self.train_d_coarse = _compile_training_callable(self.net_d_coarse)
            self.generator_id_encoder_forward = _compile_training_callable(self.generator_id_encoder)
            self.identity_embeddings_forward = _compile_training_callable(self.id_loss.extract_identity_embeddings)
            if self.enable_gaze_loss:
                self.gaze_loss_forward = _compile_training_callable(self.gaze_loss)
            if self.enable_hrffa_loss:
                self.hrffa_loss.hrffa.network = _compile_training_callable(self.hrffa_loss.hrffa.network)
            if self.enable_facs_loss:
                if is_rocm_gfx1100(device_id):
                    print("警告：gfx1100 上 FACS/OpenGraphAU 保持 eager，避免 compiled 混合精度数值错误")
                else:
                    self.facs_loss.au_model = _compile_training_callable(self.facs_loss.au_model)
            if self.enable_vgg_loss:
                self.vgg_loss_forward = _compile_training_callable(self.vgg_loss)
            if self.enable_wfm_loss:
                self.train_d_features = _compile_training_callable(self.net_d.get_feats)
        else:
            self.train_coarse = self.net_g.coarse
            self.train_hq = self.net_g.hq
            self.train_d = self.net_d
            self.train_d_coarse = self.net_d_coarse
            self.generator_id_encoder_forward = self.generator_id_encoder
            self.identity_embeddings_forward = self.id_loss.extract_identity_embeddings
            if self.enable_gaze_loss:
                self.gaze_loss_forward = self.gaze_loss
            if self.enable_vgg_loss:
                self.vgg_loss_forward = self.vgg_loss
            if self.enable_wfm_loss:
                self.train_d_features = self.net_d.get_feats

        tensorboard_purge_step = None
        if checkpoint is not None and checkpoint_mode == "resume":
            tensorboard_purge_step = self.completed_step + 1
        self.log_writer = SummaryWriter(self.tensorboard_dir, purge_step=tensorboard_purge_step)
        self._log_buffer: dict[str, Tensor] = {}

    @property
    def completed_step(self) -> int:
        """已经完整完成 D/G/scheduler/EMA 更新的 step 数。"""
        return self._completed_step

    @property
    def can_save_checkpoint(self) -> bool:
        """当前内存状态是否位于可安全持久化的完整 step 边界。"""
        return not self._step_in_progress and self.completed_step > 0

    def request_stop(self) -> None:
        """请求在当前完整 step 结束后停止训练。"""
        self._stop_requested = True

    @torch.no_grad()
    def log(self, key: str, value: Tensor, *, force: bool = False) -> None:
        current_step = self.completed_step + 1
        if force or current_step % self.log_interval == 0:
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
    def fetch_sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.dataset.next()

    def prepare_identity_encoder_faces(self, faces: Tensor, theta_restore: Tensor | None = None) -> Tensor:
        """将 full/coarse FFHQ aligned 人脸映射为身份编码器使用的 ArcFace 112 输入。"""
        spatial = tuple(faces.shape[-2:])
        if spatial == (self.img_resolution, self.img_resolution):
            grid = self.identity_encoder_grid
        elif spatial == (self.coarse_resolution, self.coarse_resolution):
            grid = self.coarse_identity_encoder_grid
        else:
            raise ValueError(f"身份编码器输入分辨率无效：{spatial}")
        if theta_restore is not None:
            grid = transform_sampling_grid(grid, theta_restore)
            return ffhq_to_arcface_112(faces, grid, padding_mode="reflection")
        return ffhq_to_arcface_112(faces, grid)

    @torch.no_grad()
    def update_ema(self, decay: float = 0.999) -> None:

        decay = min(decay, 1 - 1 / (self.completed_step + 1))
        alpha = 1.0 - decay

        torch._foreach_lerp_(self._ema_params, self._train_g_params, alpha)

    @torch.no_grad()
    def save_ckpt(self) -> None:
        if not self.can_save_checkpoint:
            raise RuntimeError("当前训练状态不在完整 step 边界，拒绝保存 partial checkpoint")

        net_g = {
            "network_cfg": self.net_g_ema.network_cfg,
            "state_dict": self.net_g_ema.state_dict(),
        }
        net_d = {
            "network_cfg": self.net_d.network_cfg,
            "state_dict": self.net_d.state_dict(),
        }
        net_d_coarse = {
            "network_cfg": self.net_d_coarse.network_cfg,
            "state_dict": self.net_d_coarse.state_dict(),
        }
        training_state = {
            "net_g": self.net_g.state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "scaler_g": self.scaler_g.state_dict() if self.precision == "fp16" else None,
            "scaler_d": self.scaler_d.state_dict() if self.precision == "fp16" else None,
            "lr_scheduler_g": self.lr_scheduler_g.state_dict() if self.use_cosine_lr else None,
            "lr_scheduler_d": self.lr_scheduler_d.state_dict() if self.use_cosine_lr else None,
        }
        completed_step = self.completed_step
        state_dict = {
            "version": CHECKPOINT_VERSION,
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
            "net_d_coarse": net_d_coarse,
            "training_state": training_state,
        }

        ckpt_file = self.ckpt_dir / f"step_{completed_step:09d}.pth"
        temp_file = ckpt_file.with_suffix(".pth.tmp")
        try:
            torch.save(state_dict, temp_file)
            temp_file.replace(ckpt_file)
            write_latest(self.run_paths, ckpt_file)
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

    def _save_sample(self, sample_batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], current_batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> None:
        sample_src, sample_dst, sample_dst_canonical, sample_theta_restore, _sample_same_mask = sample_batch
        src, dst, dst_canonical, theta_restore, _same_mask = current_batch
        with torch.no_grad(), autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
            half = self.batch_size // 2
            src_vis = torch.cat((sample_src[:half], src[: self.batch_size - half]), dim=0)
            dst_vis = torch.cat((sample_dst[:half], dst[: self.batch_size - half]), dim=0)

            dst_canonical_vis = torch.cat((sample_dst_canonical[:half], dst_canonical[: self.batch_size - half]), dim=0)
            theta_restore_vis = torch.cat((sample_theta_restore[:half], theta_restore[: self.batch_size - half]), dim=0)

            source_identity_faces_vis = self.prepare_identity_encoder_faces(src_vis)
            generator_identity_embeddings_vis = self.generator_id_encoder_forward(source_identity_faces_vis)
            source_identity_embeddings_vis = self.identity_embeddings_forward(source_identity_faces_vis)
            fake_vis, coarse_vis = self.net_g_ema(dst_vis, generator_identity_embeddings_vis, return_coarse=True)
            coarse_display_vis = NF.interpolate(coarse_vis, size=fake_vis.shape[2:], mode="bilinear", align_corners=False)

            # Identity Loss 编码器真正接收的图像；直接组合 restore + FFHQ->112，
            # 避免先恢复到全分辨率再二次重采样。仅为 sample grid 显示再放大回训练分辨率。
            identity_encoder_input_vis = self.prepare_identity_encoder_faces(fake_vis, theta_restore_vis)
            identity_encoder_input_display_vis = NF.interpolate(identity_encoder_input_vis, size=fake_vis.shape[2:], mode="bilinear", align_corners=False)

            grid = [src_vis, dst_vis, coarse_display_vis, fake_vis, dst_canonical_vis, identity_encoder_input_display_vis]

            # ========================= GAN 损失梯度图 =========================
            fake_for_gan_grad = fake_vis.detach().requires_grad_(True)
            with torch.enable_grad():
                fake_score_vis = self.train_d(fake_for_gan_grad)
                gan_loss_vis = self.gan_loss(fake_score_vis)
                gan_grad_map = self.loss_grad_map(gan_loss_vis, fake_for_gan_grad)
            grid.append(gan_grad_map)

            # ========================= 身份损失梯度图 =========================
            with torch.enable_grad():
                fake_for_id_grad = fake_vis.detach().requires_grad_(True)
                generated_identity_embeddings_vis = self.identity_embeddings_forward(self.prepare_identity_encoder_faces(fake_for_id_grad, theta_restore_vis))
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

    def train(self) -> None:

        net_coarse, net_hq = self.train_coarse, self.train_hq
        net_d, net_d_coarse = self.train_d, self.train_d_coarse
        sample_batch = self.fetch_sample()
        d_overflow_streak = 0

        with tqdm(total=None, initial=self.completed_step, mininterval=1.0, bar_format="{n_fmt:7} | 速度 {rate_fmt:3} | 训练时间 {elapsed}") as progress:
            while True:
                if self._stop_requested:
                    raise KeyboardInterrupt

                src, dst, dst_canonical, theta_restore, same_mask = self.fetch_sample()
                self._step_in_progress = True

                # ========================= 生成器前向 =========================
                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                    with torch.no_grad():
                        source_identity_faces = self.prepare_identity_encoder_faces(src)
                        generator_identity_embeddings = self.generator_id_encoder_forward(source_identity_faces)
                        source_identity_embeddings = self.identity_embeddings_forward(source_identity_faces)
                    coarse = net_coarse(dst, generator_identity_embeddings)
                    fake = net_hq(dst, coarse.detach())

                # ========================= 训练判别器 =========================
                self.net_d.requires_grad_(True)
                self.net_d_coarse.requires_grad_(True)
                self.optim_d.zero_grad(set_to_none=True)
                is_r1_reg_step = self.enable_r1_loss and self.completed_step % self.r1_reg_step == 0

                if is_r1_reg_step:
                    with autocast(device_type="cuda", enabled=False):
                        fake_img = fake.detach().float()
                        real_img = dst.detach().float().requires_grad_(True)
                        fake_coarse_img = coarse.detach().float()
                        real_coarse_img = NF.interpolate(
                            dst.detach().float(), size=(self.coarse_resolution, self.coarse_resolution), mode="bilinear", align_corners=False
                        ).requires_grad_(True)

                        # Final/Coarse R1 都使用原始判别器并以 FP32 计算。
                        fake_score = self.net_d(fake_img)
                        real_score = self.net_d(real_img)
                        fake_coarse_score = self.net_d_coarse(fake_coarse_img)
                        real_coarse_score = self.net_d_coarse(real_coarse_img)

                        final_d_loss = self.d_loss(fake_score, real_score)
                        coarse_d_loss = self.d_loss(fake_coarse_score, real_coarse_score)
                        self.log("d_loss", final_d_loss)
                        self.log("coarse_d_loss", coarse_d_loss)
                        d_loss = final_d_loss + coarse_d_loss

                        r1_loss_raw = r1_reg_loss(real_score, real_img, gamma=self.r1_gamma)
                        coarse_r1_loss_raw = r1_reg_loss(real_coarse_score, real_coarse_img, gamma=self.r1_gamma)
                        self.log("r1_loss_raw", r1_loss_raw, force=True)
                        self.log("coarse_r1_loss_raw", coarse_r1_loss_raw, force=True)
                        r1_loss = r1_loss_raw * self.r1_reg_step
                        coarse_r1_loss = coarse_r1_loss_raw * self.r1_reg_step
                        self.log("r1_loss", r1_loss, force=True)
                        self.log("coarse_r1_loss", coarse_r1_loss, force=True)
                        d_loss = d_loss + r1_loss + coarse_r1_loss
                else:
                    with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                        fake_score = net_d(fake.detach())
                        real_score = net_d(dst.detach())
                        real_coarse = NF.interpolate(
                            dst.detach(), size=(self.coarse_resolution, self.coarse_resolution), mode="bilinear", align_corners=False
                        )
                        fake_coarse_score = net_d_coarse(coarse.detach())
                        real_coarse_score = net_d_coarse(real_coarse)

                        final_d_loss = self.d_loss(fake_score, real_score)
                        coarse_d_loss = self.d_loss(fake_coarse_score, real_coarse_score)
                        self.log("d_loss", final_d_loss)
                        self.log("coarse_d_loss", coarse_d_loss)
                        d_loss = final_d_loss + coarse_d_loss

                _ensure_finite_loss("d_loss", d_loss)
                d_updated = _scaled_backward_step(
                    d_loss,
                    self.optim_d,
                    self.scaler_d,
                    name="Discriminator",
                    named_parameters=self._d_named_parameters,
                )
                if not d_updated:
                    d_overflow_streak += 1
                    self.optim_d.zero_grad(set_to_none=True)
                    self._log_buffer.clear()
                    self._step_in_progress = False
                    if d_overflow_streak >= MAX_AMP_OVERFLOW_RETRIES:
                        raise FloatingPointError(f"Discriminator 连续 {MAX_AMP_OVERFLOW_RETRIES} 次 FP16 gradient overflow，停止训练")
                    continue
                d_overflow_streak = 0

                if self.use_cosine_lr:
                    self.lr_scheduler_d.step()

                # ========================= 训练生成器 =========================
                self.net_d.requires_grad_(False)
                self.net_d_coarse.requires_grad_(False)
                g_overflow_retries = 0

                while True:
                    self.optim_g.zero_grad(set_to_none=True)
                    if g_overflow_retries > 0:
                        # D 已经成功更新，G overflow 时只重算 G，避免重复执行 D/R1。
                        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                            coarse = net_coarse(dst, generator_identity_embeddings)
                            fake = net_hq(dst, coarse.detach())

                    with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                        # gan_loss
                        if self.enable_wfm_loss:
                            fake_score, fake_feats = net_d(fake, True)
                        else:
                            fake_score = net_d(fake)

                        gan_loss = self.gan_loss(fake_score)
                        coarse_gan_loss = self.gan_loss(net_d_coarse(coarse))
                        self.log("gan_loss", gan_loss)
                        self.log("coarse_gan_loss", coarse_gan_loss)
                        g_loss = gan_loss + coarse_gan_loss

                        # wfm_loss
                        if self.enable_wfm_loss:
                            with torch.no_grad():
                                real_feats = self.train_d_features(dst, self.wfm_max_layer)
                            wfm_loss = self.wfm_loss(fake_feats, real_feats)
                            self.log("wfm_loss", wfm_loss)
                            g_loss = g_loss + wfm_loss

                        # id_loss
                        generated_identity_embeddings = self.identity_embeddings_forward(self.prepare_identity_encoder_faces(fake, theta_restore))
                        id_loss = self.id_loss(generated_identity_embeddings, source_identity_embeddings)
                        self.log("id_loss", id_loss)
                        g_loss = g_loss + id_loss

                        # Coarse 负责身份迁移本身。Final/HQ losses 通过 coarse.detach() 与 Coarse 参数隔离。
                        coarse_identity_embeddings = self.identity_embeddings_forward(self.prepare_identity_encoder_faces(coarse, theta_restore))
                        coarse_id_loss = self.id_loss(coarse_identity_embeddings, source_identity_embeddings)
                        self.log("coarse_id_loss", coarse_id_loss)
                        g_loss = g_loss + coarse_id_loss

                        # Gaze / HRFFA / FACS：Final 和 Coarse 都保持 target 的 canonical 属性。
                        # Coarse 先上采样到训练分辨率，再与 Final 共用同一个 restore grid。
                        if self.enable_gaze_loss or self.enable_hrffa_loss or self.enable_facs_loss:
                            with autocast(device_type="cuda", enabled=False):
                                restore_grid = NF.affine_grid(theta_restore.float(), size=list(fake.shape), align_corners=False)
                                fake_restored = NF.grid_sample(fake.float(), restore_grid, mode="bilinear", padding_mode="reflection", align_corners=False)
                                coarse_full = NF.interpolate(coarse.float(), size=fake.shape[-2:], mode="bilinear", align_corners=False)
                                coarse_restored = NF.grid_sample(coarse_full, restore_grid, mode="bilinear", padding_mode="reflection", align_corners=False)

                        # gaze_loss：Final/Coarse 都保持 canonical dst 的视线方向。
                        if self.enable_gaze_loss:
                            gaze_loss = self.gaze_loss_forward(fake_restored, dst_canonical)
                            coarse_gaze_loss = self.gaze_loss_forward(coarse_restored, dst_canonical)
                            self.log("gaze_loss", gaze_loss)
                            self.log("coarse_gaze_loss", coarse_gaze_loss)
                            g_loss = g_loss + gaze_loss + coarse_gaze_loss

                        # HRFFA：姿态、眼睑、嘴部开合和 target 外轮廓。
                        # compile_module 仅编译 HRFFA 神经网络主体；FP32 几何求解保持 eager。
                        if self.enable_hrffa_loss:
                            hrffa_components = self.hrffa_loss.forward_components(fake_restored, dst_canonical)
                            coarse_hrffa_components = self.hrffa_loss.forward_components(coarse_restored, dst_canonical)
                            for name, component in hrffa_components.items():
                                self.log(f"hrffa_{name}_loss", component)
                            for name, component in coarse_hrffa_components.items():
                                self.log(f"coarse_hrffa_{name}_loss", component)
                            g_loss = g_loss + torch.stack(tuple(hrffa_components.values())).sum()
                            g_loss = g_loss + torch.stack(tuple(coarse_hrffa_components.values())).sum()

                        # FACS：Final/Coarse 都保持 canonical dst 的连续 Action Unit 与左右非对称表情。
                        if self.enable_facs_loss:
                            facs_components = self.facs_loss.forward_components(fake_restored, dst_canonical)
                            coarse_facs_components = self.facs_loss.forward_components(coarse_restored, dst_canonical)
                            for name, component in facs_components.items():
                                self.log(f"facs_{name}_loss", component)
                            for name, component in coarse_facs_components.items():
                                self.log(f"coarse_facs_{name}_loss", component)
                            g_loss = g_loss + torch.stack(tuple(facs_components.values())).sum()
                            g_loss = g_loss + torch.stack(tuple(coarse_facs_components.values())).sum()

                        # VGG/L1 reconstruction 共用 loss.reconstruction.scope。
                        if self.enable_vgg_loss:
                            vgg_per_sample = self.vgg_loss_forward(fake, dst)
                            vgg_loss = _reduce_reconstruction_loss(vgg_per_sample, same_mask, self.reconstruction_scope)
                            self.log("vgg_loss", vgg_loss)
                            g_loss = g_loss + vgg_loss

                        if self.enable_l1_loss:
                            l1_per_sample = self.l1_loss(fake, dst).flatten(1).mean(dim=1)
                            l1_loss = _reduce_reconstruction_loss(l1_per_sample, same_mask, self.reconstruction_scope)
                            self.log("l1_loss", l1_loss)
                            g_loss = g_loss + l1_loss

                    _ensure_finite_loss("g_loss", g_loss)
                    if _scaled_backward_step(
                        g_loss,
                        self.optim_g,
                        self.scaler_g,
                        name="Generator",
                        named_parameters=self.net_g.named_parameters(),
                    ):
                        break

                    g_overflow_retries += 1
                    self.optim_g.zero_grad(set_to_none=True)
                    if g_overflow_retries >= MAX_AMP_OVERFLOW_RETRIES:
                        raise FloatingPointError(f"Generator 连续 {MAX_AMP_OVERFLOW_RETRIES} 次 FP16 gradient overflow，停止训练")

                if self.use_cosine_lr:
                    self.lr_scheduler_g.step()

                self.update_ema()
                self._completed_step += 1
                self._step_in_progress = False
                self.flush_logs()
                progress.update(1)

                if self._stop_requested:
                    raise KeyboardInterrupt

                if self.completed_step % self.checkpoint_save_every == 0:
                    self.save_ckpt()

                if self.completed_step % self.sample_save_every == 0:
                    self._save_sample(sample_batch, (src, dst, dst_canonical, theta_restore, same_mask))


def _load_run_config(paths: RunPaths) -> dict[str, Any]:
    resolved = load_resolved_config(paths)
    canonical = resolve_train_config(resolved)
    if canonical != resolved:
        raise ValueError(f"run 的 resolved config 不是当前格式的规范表示：{paths.resolved_config}")
    return resolved


def _assert_branch_generator_compatible(parent: dict[str, Any], branch: dict[str, Any]) -> None:
    if parent["generator"] != branch["generator"]:
        raise ValueError("branch 不能修改 Generator 架构")



def _branch_model_states(
    checkpoint: Mapping[str, Any], *, reset_discriminator: bool
) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor] | None, Mapping[str, Tensor] | None]:
    """Branch 继承训练态 G；两个 D 要么同时继承，要么同时重建。"""
    g_state = checkpoint["training_state"]["net_g"]
    if reset_discriminator:
        return g_state, None, None
    return g_state, checkpoint["net_d"]["state_dict"], checkpoint["net_d_coarse"]["state_dict"]

def _load_branch_checkpoint(
    checkpoint_path: Path,
    resolved: dict[str, Any],
    *,
    reset_discriminator: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_version = checkpoint["version"]
    if checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError(f"不支持的 checkpoint version：{checkpoint_version}，当前仅支持 v{CHECKPOINT_VERSION}")

    checkpoint_step = int(checkpoint["step"])
    filename_step = checkpoint_step_from_name(checkpoint_path.name)
    if filename_step != checkpoint_step:
        raise ValueError(f"checkpoint 文件名 step 与内部状态不一致：filename={filename_step}, checkpoint={checkpoint_step}")

    _assert_branch_generator_compatible(
        {"generator": dict(checkpoint["net_g"]["network_cfg"])},
        resolved,
    )
    # 当前训练 checkpoint 必须同时包含 Final/Coarse 两个判别器。
    checkpoint["net_d_coarse"]
    if not reset_discriminator:
        if dict(checkpoint["net_d"]["network_cfg"]) != resolved["discriminator"]:
            raise ValueError("branch 默认恢复 Discriminator，因此架构必须一致；如需新建请使用 --reset-discriminator")
        expected_coarse_d_cfg = {**resolved["discriminator"], "img_resolution": int(resolved["generator"]["coarse_resolution"])}
        if dict(checkpoint["net_d_coarse"]["network_cfg"]) != expected_coarse_d_cfg:
            raise ValueError("branch 恢复 Coarse Discriminator 时要求架构一致；如需新建请使用 --reset-discriminator")
    _branch_model_states(checkpoint, reset_discriminator=reset_discriminator)
    checkpoint_run = checkpoint["run"]
    parent = {
        "run_id": checkpoint_run["id"],
        "checkpoint": checkpoint_path.name,
        "step": checkpoint_step,
        "config_sha256": checkpoint_run["config_sha256"],
        "discriminator": "reset" if reset_discriminator else "inherit",
    }
    return checkpoint, parent


def main() -> None:
    _configure_training_runtime()
    parser = argparse.ArgumentParser(description="SwapFace 训练")
    parser.add_argument("--config", type=Path, default=None, help=f"fresh/branch 的训练 TOML；fresh 默认：{DEFAULT_TRAIN_CONFIG_PATH}")
    parser.add_argument("--name", type=str, default=None, help="新 run 的可选短标签；run ID 仍包含唯一时间戳")
    parser.add_argument("--runs-root", type=Path, default=None, help=f"新 run 根目录，默认：{DEFAULT_RUNS_ROOT}")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--resume", type=Path, default=None, help="严格恢复原 run，只允许 latest checkpoint，并使用原 run 冻结配置")
    source_group.add_argument("--branch-from", type=Path, default=None, help="从 checkpoint 创建新 run；继承训练态 Generator，默认也继承 Discriminator 权重")
    parser.add_argument("--reset-discriminator", action="store_true", help="仅用于 branch：不恢复父 Discriminator，按新配置重新初始化")
    parser.add_argument("--step", type=int, default=None, help="仅用于 branch：新 run 的起始 step；默认继承父 checkpoint step")
    parser.add_argument("--strict-precision", action="store_true", help="仅用于 resume：要求当前实际训练精度与 checkpoint 一致；默认允许变化")
    args = parser.parse_args()

    is_resume = args.resume is not None
    is_branch = args.branch_from is not None

    if is_resume and args.config is not None:
        parser.error("--resume 与 --config 不能同时使用；resume 必须使用原 run 的冻结配置")
    if is_resume and args.name is not None:
        parser.error("--resume 与 --name 不能同时使用；resume 继续写入原 run")
    if is_resume and args.runs_root is not None:
        parser.error("--resume 与 --runs-root 不能同时使用；resume 继续写入原 run")
    if is_branch and args.config is None:
        parser.error("--branch-from 必须显式配合 --config，branch 会创建使用该配置的新 run")
    if args.reset_discriminator and not is_branch:
        parser.error("--reset-discriminator 仅用于 --branch-from")
    if args.step is not None and not is_branch:
        parser.error("--step 仅用于 --branch-from")
    if args.step is not None and args.step < 0:
        parser.error("--step 必须 >= 0")
    if args.strict_precision and not is_resume:
        parser.error("--strict-precision 仅用于 --resume")

    checkpoint_mode: Literal["resume", "branch"] = "resume"
    preloaded_checkpoint: dict[str, Any] | None = None
    start_step: int | None = None

    if is_resume:
        assert args.resume is not None
        run_paths, checkpoint_path = resolve_resume_target(args.resume)
        resolved = _load_run_config(run_paths)
    elif is_branch:
        assert args.branch_from is not None and args.config is not None
        checkpoint_path = resolve_branch_target(args.branch_from)
        resolved = load_train_config(args.config)

        preloaded_checkpoint, parent = _load_branch_checkpoint(
            checkpoint_path,
            resolved,
            reset_discriminator=args.reset_discriminator,
        )
        start_step = int(preloaded_checkpoint["step"]) if args.step is None else args.step

        run_paths = create_run(args.runs_root or DEFAULT_RUNS_ROOT, args.config, resolved, name=args.name, parent=parent)
        checkpoint_mode = "branch"
    else:
        config_path = args.config or DEFAULT_TRAIN_CONFIG_PATH
        resolved = load_train_config(config_path)
        run_paths = create_run(args.runs_root or DEFAULT_RUNS_ROOT, config_path, resolved, name=args.name)
        checkpoint_path = None

    run_metadata = load_metadata(run_paths)
    run_id = run_metadata["run_id"]
    digest = config_sha256(resolved)

    with RunLock(run_paths):
        if is_resume:
            # latest 只能在线性历史上继续；拿到锁后重新解析，避免锁前竞态。
            _, checkpoint_path = resolve_resume_target(run_paths.root)

        update_metadata(run_paths, status="running", error=None)

        trainer: Trainer | None = None
        try:
            trainer = Trainer(
                resolved,
                run_dir=run_paths.root,
                checkpoint_path=checkpoint_path,
                run_id=run_id,
                resolved_config_sha256=digest,
                strict_precision_resume=args.strict_precision,
                checkpoint_mode=checkpoint_mode,
                preloaded_checkpoint=preloaded_checkpoint,
                reset_discriminator=args.reset_discriminator,
                start_step=start_step,
            )
            preloaded_checkpoint = None
            if is_branch:
                print(f"Branch 来源：{checkpoint_path}")
            print(f"Run 目录：{run_paths.root}")

            previous_sigint_handler = signal.getsignal(signal.SIGINT)
            sigint_requested = False

            def _handle_sigint(_signum: int, _frame: Any) -> None:
                nonlocal sigint_requested
                if sigint_requested:
                    signal.signal(signal.SIGINT, previous_sigint_handler)
                    raise KeyboardInterrupt
                sigint_requested = True
                trainer.request_stop()
                print("收到 Ctrl+C；将在当前完整 step 结束后保存 checkpoint 并退出。再次 Ctrl+C 可立即中断。")

            signal.signal(signal.SIGINT, _handle_sigint)
            try:
                trainer.train()
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)
        except KeyboardInterrupt:
            if trainer is not None and trainer.can_save_checkpoint:
                trainer.save_ckpt()
                message = f"训练已中断；已保存 completed step {trainer.completed_step} checkpoint"
            else:
                message = "训练已中断；当前状态不在完整 step 边界，未保存 partial checkpoint，latest 保持不变"
            update_metadata(run_paths, status="interrupted")
            print(message)
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
