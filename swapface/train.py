import argparse
import copy
import signal
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
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
    WeakFeatureMatchingLoss,
    make_blurred_l1_loss,
    make_l1_loss,
    r1_reg_loss,
)
from misc.face_alignment import ffhq_to_arcface_112, make_ffhq_to_arcface_112_grid, transform_sampling_grid
from misc.models.id_encoder import IDEncoder, IDEncoderProvider
from models.discriminator import DiscriminatorType, build_discriminator
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
MAX_AMP_OVERFLOW_RETRIES = 16
TRAINING_SEMANTICS_VERSION = 21
SAMPLE_LABEL_WIDTH = 300
SAMPLE_GRID_PADDING = 2


def _sample_row_bgr(images: Tensor, label: str, nrow: int) -> np.ndarray:
    """把一个 sample row 转为 BGR，并在最左侧添加含义标签。"""
    images = images.detach().float().add(1.0).mul(127.5).clamp(0.0, 255.0)
    grid = make_grid(images, nrow=nrow, padding=SAMPLE_GRID_PADDING)[[2, 1, 0], :, :]
    image = grid.permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
    label_panel = np.full((image.shape[0], SAMPLE_LABEL_WIDTH, 3), 20, dtype=np.uint8)
    text_y = max(28, image.shape[0] // 2)
    cv2.putText(label_panel, label, (14, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate((label_panel, image), axis=1)

COARSE_GENERATOR_CONFIG_KEYS = (
    "img_channels",
    "id_dim",
    "coarse_resolution",
    "coarse_latent_resolution",
    "coarse_num_latent",
    "coarse_base_ch",
    "coarse_max_ch",
)


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


def _require_training_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """严格校验当前 checkpoint 的训练运行时协议。"""
    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, dict):
        raise TypeError("checkpoint.training_config 必须为 dict")

    expected_keys = {"semantics_version", "precision", "stage"}
    actual_keys = set(training_config)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        raise ValueError(f"checkpoint.training_config 不是当前协议：missing={missing}, unknown={unknown}")

    semantics_version = training_config["semantics_version"]
    if not isinstance(semantics_version, int) or isinstance(semantics_version, bool):
        raise TypeError("checkpoint.training_config.semantics_version 必须为 int")
    if semantics_version != TRAINING_SEMANTICS_VERSION:
        raise ValueError(f"训练语义版本不匹配：checkpoint={semantics_version}, current={TRAINING_SEMANTICS_VERSION}")

    precision = training_config["precision"]
    if not isinstance(precision, str):
        raise TypeError("checkpoint.training_config.precision 必须为 str")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"checkpoint.training_config.precision 无效：{precision!r}")

    stage = training_config["stage"]
    if not isinstance(stage, str) or stage not in {"joint", "coarse", "hq"}:
        raise ValueError(f"checkpoint.training_config.stage 无效：{stage!r}")

    return training_config


def _reduce_reconstruction_loss(rec_per_sample: Tensor, same_mask: Tensor, scope: Literal["same", "all"]) -> Tensor:
    """按配置范围聚合逐样本 reconstruction loss。"""
    if scope == "all":
        return rec_per_sample.mean()
    if scope == "same":
        same_weight = same_mask.to(dtype=rec_per_sample.dtype)
        return (rec_per_sample * same_weight).sum() / same_weight.sum().clamp_min(1.0)
    raise ValueError(f"reconstruction_scope 无效：{scope!r}")


def _canonical_checkpoint_discriminator_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """把新增 type 字段前保存的当前 FIR D checkpoint 映射到新判别器配置协议。"""
    result = dict(config)
    result.setdefault("type", DiscriminatorType.FIR_MINIBATCH_STD.name)
    return result


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
        reset_hq_discriminator: bool = False,
        reset_coarse_discriminator: bool = False,
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
        net_d_cfg = dict(config["discriminator"]["hq"])
        net_d_coarse_cfg = dict(config["discriminator"]["coarse"])
        coarse_resolution = int(net_g_cfg["coarse_resolution"])

        generator_optimizer_config = optimizer_config["generator"]
        hq_d_optimizer_config = optimizer_config["hq_discriminator"]
        coarse_d_optimizer_config = optimizer_config["coarse_discriminator"]
        generator_scheduler_config = scheduler_config["generator"]
        hq_d_scheduler_config = scheduler_config["hq_discriminator"]
        coarse_d_scheduler_config = scheduler_config["coarse_discriminator"]
        generator_lr = float(generator_optimizer_config["lr"])
        hq_d_lr = float(hq_d_optimizer_config["lr"])
        coarse_d_lr = float(coarse_d_optimizer_config["lr"])
        compile_module = bool(train_config["compile_module"])
        self.train_stage = str(train_config["stage"])
        self.coarse_stage_active = self.train_stage in {"joint", "coarse"}
        self.hq_stage_active = self.train_stage in {"joint", "hq"}
        self.batch_size = int(train_config["batch_size"])
        self.log_interval = int(train_config["log_interval"])
        self.sample_save_every = int(train_config["sample_save_every"])
        self.checkpoint_save_every = int(train_config["checkpoint_save_every"])

        coarse_loss_config = loss_config["coarse"]
        hq_loss_config = loss_config["hq"]
        coarse_gan_config = coarse_loss_config["gan"]
        hq_gan_config = hq_loss_config["gan"]
        coarse_identity_config = coarse_loss_config["identity"]
        hq_identity_config = hq_loss_config["identity"]
        coarse_r1_config = coarse_loss_config["r1"]
        hq_r1_config = hq_loss_config["r1"]
        coarse_wfm_config = coarse_loss_config["wfm"]
        hq_wfm_config = hq_loss_config["wfm"]
        coarse_gaze_config = coarse_loss_config["gaze"]
        hq_gaze_config = hq_loss_config["gaze"]
        coarse_hrffa_config = coarse_loss_config["hrffa"]
        hq_hrffa_config = hq_loss_config["hrffa"]
        coarse_facs_config = coarse_loss_config["facs"]
        hq_facs_config = hq_loss_config["facs"]
        coarse_reconstruction_scope = str(coarse_loss_config["reconstruction"]["scope"])
        hq_reconstruction_scope = str(hq_loss_config["reconstruction"]["scope"])
        coarse_l1_config = coarse_loss_config["l1"]
        hq_l1_config = hq_loss_config["l1"]
        coarse_vgg_config = coarse_loss_config["vgg"]
        hq_vgg_config = hq_loss_config["vgg"]

        self.generator_id_encoder_provider = IDEncoderProvider[str(identity_config["provider"])]
        self.coarse_identity_loss_provider = IDEncoderProvider[str(coarse_identity_config["provider"])]
        self.hq_identity_loss_provider = IDEncoderProvider[str(hq_identity_config["provider"])]
        self.reuse_generator_identity_for_coarse_source = self.coarse_stage_active and self.generator_id_encoder_provider is self.coarse_identity_loss_provider
        self.reuse_generator_identity_for_hq_source = self.hq_stage_active and self.generator_id_encoder_provider is self.hq_identity_loss_provider
        self.coarse_reconstruction_scope = coarse_reconstruction_scope
        self.hq_reconstruction_scope = hq_reconstruction_scope
        self.enable_coarse_l1_loss = self.coarse_stage_active and bool(coarse_l1_config["enable"])
        self.enable_hq_l1_loss = self.hq_stage_active and bool(hq_l1_config["enable"])
        self.enable_coarse_r1_loss = self.coarse_stage_active and bool(coarse_r1_config["enable"])
        self.coarse_r1_reg_step = int(coarse_r1_config["interval"])
        self.coarse_r1_gamma = float(coarse_r1_config["gamma"])
        self.enable_hq_r1_loss = self.hq_stage_active and bool(hq_r1_config["enable"])
        self.hq_r1_reg_step = int(hq_r1_config["interval"])
        self.hq_r1_gamma = float(hq_r1_config["gamma"])
        self.enable_coarse_wfm_loss = self.coarse_stage_active and bool(coarse_wfm_config["enable"])
        self.enable_hq_wfm_loss = self.hq_stage_active and bool(hq_wfm_config["enable"])
        self.enable_coarse_gaze_loss = self.coarse_stage_active and bool(coarse_gaze_config["enable"])
        self.enable_hq_gaze_loss = self.hq_stage_active and bool(hq_gaze_config["enable"])
        self.enable_coarse_hrffa_loss = self.coarse_stage_active and bool(coarse_hrffa_config["enable"])
        self.enable_hq_hrffa_loss = self.hq_stage_active and bool(hq_hrffa_config["enable"])
        self.enable_coarse_facs_loss = self.coarse_stage_active and bool(coarse_facs_config["enable"])
        self.enable_hq_facs_loss = self.hq_stage_active and bool(hq_facs_config["enable"])
        self.enable_coarse_vgg_loss = self.coarse_stage_active and bool(coarse_vgg_config["enable"])
        self.enable_hq_vgg_loss = self.hq_stage_active and bool(hq_vgg_config["enable"])

        dataloader_cfg = {**data_config["loader"], **data_config["augmentation"], **data_config["sampling"]}
        dataloader_cfg["decoder_backend"] = ImageDecoderBackend(dataloader_cfg["decoder_backend"])
        src = [(str(entry["path"]), float(entry["adjustment"]), str(entry.get("alignment", "ffhq"))) for entry in data_config["src"]]
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

        self.use_cosine_lr_g = str(generator_scheduler_config["type"]) == "cosine"
        self.use_cosine_lr_d_hq = self.hq_stage_active and str(hq_d_scheduler_config["type"]) == "cosine"
        self.use_cosine_lr_d_coarse = self.coarse_stage_active and str(coarse_d_scheduler_config["type"]) == "cosine"
        self.training_config = {
            "semantics_version": TRAINING_SEMANTICS_VERSION,
            "precision": self.precision,
            "stage": self.train_stage,
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
            branch_mode: Literal["inherit", "hq_rebuild"] = "inherit"
            if checkpoint_mode == "resume":
                if saved_net_g_cfg != net_g_cfg:
                    raise ValueError("checkpoint Generator 架构与当前配置不一致")
            else:
                branch_mode = _branch_generator_mode(checkpoint, config)

            self.img_resolution = int(net_g_cfg["img_resolution"])
            net_g = Generator(**net_g_cfg)
            net_d = build_discriminator(net_d_cfg)
            net_d_coarse = build_discriminator(net_d_coarse_cfg)

            if checkpoint_mode == "resume":
                self._completed_step = checkpoint_step
                checkpoint_run = checkpoint["run"]
                if checkpoint_run["id"] != run_id or checkpoint_run["config_sha256"] != resolved_config_sha256:
                    raise ValueError("checkpoint 不属于当前 run 或冻结配置已变化")

                saved_training_config = _require_training_config(checkpoint)
                saved_precision = str(saved_training_config["precision"])
                saved_stage = str(saved_training_config["stage"])
                if saved_stage != self.train_stage:
                    raise ValueError(f"训练 stage 不一致：checkpoint={saved_stage}, config={self.train_stage}")
                if strict_precision_resume and saved_precision != self.precision:
                    raise ValueError(f"训练精度不一致：checkpoint={saved_precision}，current={self.precision}")

                saved_identity_encoders = checkpoint["identity_encoders"]
                if saved_identity_encoders["generator"] != self.generator_id_encoder_provider.name:
                    raise ValueError(f"Generator 身份编码器不匹配：{saved_identity_encoders['generator']} != {self.generator_id_encoder_provider.name}")
                if saved_identity_encoders["coarse_identity_loss"] != self.coarse_identity_loss_provider.name:
                    raise ValueError("checkpoint Coarse Identity Loss teacher 与当前配置不一致")
                if saved_identity_encoders["hq_identity_loss"] != self.hq_identity_loss_provider.name:
                    raise ValueError("checkpoint HQ Identity Loss teacher 与当前配置不一致")

                saved_net_d_cfg = _canonical_checkpoint_discriminator_config(checkpoint["net_d"]["network_cfg"])
                if saved_net_d_cfg != net_d_cfg:
                    raise ValueError("checkpoint Discriminator 架构与当前配置不一致")
                saved_net_d_coarse_cfg = _canonical_checkpoint_discriminator_config(checkpoint["net_d_coarse"]["network_cfg"])
                if saved_net_d_coarse_cfg != net_d_coarse_cfg:
                    raise ValueError("checkpoint Coarse Discriminator 架构与当前配置不一致")

                training_state = checkpoint["training_state"]
                net_g.load_state_dict(training_state["net_g"])
                net_d.load_state_dict(checkpoint["net_d"]["state_dict"])
                net_d_coarse.load_state_dict(checkpoint["net_d_coarse"]["state_dict"])
            else:
                # Branch 不恢复 optimizer/scheduler/scaler。HQ rebuild 只继承训练态 Coarse，HQ/HQ D 从零初始化。
                hq_rebuild = branch_mode == "hq_rebuild"
                if hq_rebuild:
                    if start_step not in (None, 0):
                        raise ValueError("HQ rebuild 必须从 step 0 开始")
                    self._completed_step = 0
                else:
                    self._completed_step = checkpoint_step if start_step is None else start_step

                branch_g_state, branch_d_state, branch_d_coarse_state = _branch_model_states(
                    checkpoint,
                    reset_hq_discriminator=reset_hq_discriminator,
                    reset_coarse_discriminator=reset_coarse_discriminator,
                    hq_rebuild=hq_rebuild,
                )
                if hq_rebuild:
                    net_g.coarse.load_state_dict(_submodule_state_dict(branch_g_state, "coarse"), strict=True)
                else:
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
            net_d = build_discriminator(net_d_cfg)
            net_d_coarse = build_discriminator(net_d_coarse_cfg)

        d_resolution = int(net_d.network_cfg["img_resolution"])
        if d_resolution != self.img_resolution:
            raise ValueError(f"生成器与 HQ Discriminator 分辨率不一致：{self.img_resolution} != {d_resolution}")

        coarse_d_resolution = int(net_d_coarse.network_cfg["img_resolution"])
        if coarse_d_resolution != coarse_resolution:
            raise ValueError(f"Coarse 与 Coarse Discriminator 分辨率不一致：{coarse_resolution} != {coarse_d_resolution}")

        generator_channels = int(net_g.network_cfg["img_channels"])
        hq_d_channels = int(net_d.network_cfg["img_channels"])
        if hq_d_channels != generator_channels:
            raise ValueError(f"Generator 与 HQ Discriminator 通道数不一致：{generator_channels} != {hq_d_channels}")

        coarse_d_channels = int(net_d_coarse.network_cfg["img_channels"])
        if coarse_d_channels != generator_channels:
            raise ValueError(f"Generator 与 Coarse Discriminator 通道数不一致：{generator_channels} != {coarse_d_channels}")

        self.net_g = net_g.to(self.device).train()
        self.net_g.coarse.requires_grad_(self.coarse_stage_active)
        self.net_g.hq.requires_grad_(self.hq_stage_active)
        if not self.coarse_stage_active:
            self.net_g.coarse.eval()
        if not self.hq_stage_active:
            self.net_g.hq.eval()

        self.net_d = net_d.train(self.hq_stage_active).requires_grad_(self.hq_stage_active)
        self.net_d_coarse = net_d_coarse.train(self.coarse_stage_active).requires_grad_(self.coarse_stage_active)
        if self.hq_stage_active:
            self.net_d.to(self.device)
        if self.coarse_stage_active:
            self.net_d_coarse.to(self.device)

        self.net_g_ema = copy.deepcopy(self.net_g)
        if checkpoint is not None and checkpoint_mode == "resume":
            self.net_g_ema.load_state_dict(checkpoint["net_g"]["state_dict"])
        self.net_g_ema.eval().requires_grad_(False)
        if self.train_stage == "joint":
            self._ema_params = tuple(self.net_g_ema.parameters())
            self._train_g_params = tuple(self.net_g.parameters())
        elif self.coarse_stage_active:
            self._ema_params = tuple(self.net_g_ema.coarse.parameters())
            self._train_g_params = tuple(self.net_g.coarse.parameters())
        else:
            self._ema_params = tuple(self.net_g_ema.hq.parameters())
            self._train_g_params = tuple(self.net_g.hq.parameters())

        # ========================= 优化器 / 调度器 =========================
        if self.train_stage == "joint":
            self._g_named_parameters = tuple(self.net_g.named_parameters())
        elif self.coarse_stage_active:
            self._g_named_parameters = tuple((f"coarse.{name}", parameter) for name, parameter in self.net_g.coarse.named_parameters())
        else:
            self._g_named_parameters = tuple((f"hq.{name}", parameter) for name, parameter in self.net_g.hq.named_parameters())
        self.optim_g = optim.Adam((parameter for _, parameter in self._g_named_parameters), lr=generator_lr, betas=(0.0, 0.99), fused=True)
        self.optim_d_hq = optim.Adam(self.net_d.parameters(), lr=hq_d_lr, betas=(0.0, 0.99), fused=True) if self.hq_stage_active else None
        self.optim_d_coarse = optim.Adam(self.net_d_coarse.parameters(), lr=coarse_d_lr, betas=(0.0, 0.99), fused=True) if self.coarse_stage_active else None
        self._hq_d_named_parameters = tuple((f"hq.{name}", parameter) for name, parameter in self.net_d.named_parameters()) if self.hq_stage_active else ()
        self._coarse_d_named_parameters = tuple((f"coarse.{name}", parameter) for name, parameter in self.net_d_coarse.named_parameters()) if self.coarse_stage_active else ()
        scaler_enabled = self.precision == "fp16"
        self.scaler_g = GradScaler("cuda", enabled=scaler_enabled)
        self.scaler_d_hq = GradScaler("cuda", enabled=scaler_enabled) if self.hq_stage_active else None
        self.scaler_d_coarse = GradScaler("cuda", enabled=scaler_enabled) if self.coarse_stage_active else None

        if training_state is not None:
            self.optim_g.load_state_dict(training_state["optim_g"])
            if self.hq_stage_active:
                assert self.optim_d_hq is not None
                self.optim_d_hq.load_state_dict(training_state["optim_d_hq"])
            if self.coarse_stage_active:
                assert self.optim_d_coarse is not None
                self.optim_d_coarse.load_state_dict(training_state["optim_d_coarse"])

        if training_state is not None and saved_precision == self.precision == "fp16":
            scaler_g_state = training_state.get("scaler_g")
            if scaler_g_state is not None:
                self.scaler_g.load_state_dict(scaler_g_state)
            if self.hq_stage_active:
                assert self.scaler_d_hq is not None
                scaler_d_hq_state = training_state.get("scaler_d_hq")
                if scaler_d_hq_state is not None:
                    self.scaler_d_hq.load_state_dict(scaler_d_hq_state)
            if self.coarse_stage_active:
                assert self.scaler_d_coarse is not None
                scaler_d_coarse_state = training_state.get("scaler_d_coarse")
                if scaler_d_coarse_state is not None:
                    self.scaler_d_coarse.load_state_dict(scaler_d_coarse_state)

        if self.use_cosine_lr_g:
            self.lr_scheduler_g = CosineAnnealingLR(
                self.optim_g,
                T_max=int(generator_scheduler_config["t_max"]),
                eta_min=generator_lr * float(generator_scheduler_config["min_lr_ratio"]),
            )
            if training_state is not None:
                self.lr_scheduler_g.load_state_dict(training_state["lr_scheduler_g"])
        if self.use_cosine_lr_d_hq:
            assert self.optim_d_hq is not None
            self.lr_scheduler_d_hq = CosineAnnealingLR(
                self.optim_d_hq,
                T_max=int(hq_d_scheduler_config["t_max"]),
                eta_min=hq_d_lr * float(hq_d_scheduler_config["min_lr_ratio"]),
            )
            if training_state is not None:
                self.lr_scheduler_d_hq.load_state_dict(training_state["lr_scheduler_d_hq"])
        if self.use_cosine_lr_d_coarse:
            assert self.optim_d_coarse is not None
            self.lr_scheduler_d_coarse = CosineAnnealingLR(
                self.optim_d_coarse,
                T_max=int(coarse_d_scheduler_config["t_max"]),
                eta_min=coarse_d_lr * float(coarse_d_scheduler_config["min_lr_ratio"]),
            )
            if training_state is not None:
                self.lr_scheduler_d_coarse.load_state_dict(training_state["lr_scheduler_d_coarse"])

        # ========================= 损失 =========================
        self.d_loss = DiscriminatorAdversarialLoss(weight=1.0, reduction="mean").to(self.device)
        if self.coarse_stage_active:
            self.coarse_gan_loss = GeneratorAdversarialLoss(weight=float(coarse_gan_config["weight"]), reduction="mean").to(self.device)
        if self.hq_stage_active:
            self.hq_gan_loss = GeneratorAdversarialLoss(weight=float(hq_gan_config["weight"]), reduction="mean").to(self.device)
        if self.enable_coarse_wfm_loss:
            self.coarse_wfm_loss = WeakFeatureMatchingLoss(coarse_wfm_config["weights"]).to(self.device)
        if self.enable_hq_wfm_loss:
            self.hq_wfm_loss = WeakFeatureMatchingLoss(hq_wfm_config["weights"]).to(self.device)

        self.generator_id_encoder = IDEncoder(self.generator_id_encoder_provider).to(self.device).eval().requires_grad_(False)
        if self.hq_stage_active:
            self.hq_id_loss = IdentityLoss(weight=float(hq_identity_config["weight"]), provider=self.hq_identity_loss_provider).to(self.device)
        if self.coarse_stage_active:
            self.coarse_id_loss = IdentityLoss(weight=float(coarse_identity_config["weight"]), provider=self.coarse_identity_loss_provider).to(self.device)
            if self.hq_stage_active and self.coarse_identity_loss_provider is self.hq_identity_loss_provider:
                self.coarse_id_loss.id_encoder = self.hq_id_loss.id_encoder
        self.coarse_resolution = int(self.net_g.network_cfg["coarse_resolution"])
        self.identity_encoder_grid = make_ffhq_to_arcface_112_grid(self.img_resolution, self.batch_size, self.device)
        self.coarse_identity_encoder_grid = make_ffhq_to_arcface_112_grid(self.coarse_resolution, self.batch_size, self.device)

        if self.enable_coarse_l1_loss:
            if bool(coarse_l1_config["gaussian_blur"]):
                self.coarse_l1_loss = make_blurred_l1_loss(weight=float(coarse_l1_config["weight"]), resolution=self.coarse_resolution, reduction="none")
            else:
                self.coarse_l1_loss = make_l1_loss(weight=float(coarse_l1_config["weight"]), reduction="none")
        if self.enable_hq_l1_loss:
            if bool(hq_l1_config["gaussian_blur"]):
                self.hq_l1_loss = make_blurred_l1_loss(weight=float(hq_l1_config["weight"]), resolution=self.img_resolution, reduction="none")
            else:
                self.hq_l1_loss = make_l1_loss(weight=float(hq_l1_config["weight"]), reduction="none")

        if self.enable_hq_gaze_loss:
            self.hq_gaze_loss = GazeLoss(
                weight=float(hq_gaze_config["weight"]),
                distribution_weight=float(hq_gaze_config["distribution_weight"]),
                confidence_weighted=bool(hq_gaze_config["confidence_weighted"]),
            )
        if self.enable_coarse_gaze_loss:
            self.coarse_gaze_loss = GazeLoss(
                weight=float(coarse_gaze_config["weight"]),
                distribution_weight=float(coarse_gaze_config["distribution_weight"]),
                confidence_weighted=bool(coarse_gaze_config["confidence_weighted"]),
            )
            if self.enable_hq_gaze_loss:
                self.coarse_gaze_loss.gaze_model = self.hq_gaze_loss.gaze_model
        if self.enable_hq_gaze_loss:
            self.hq_gaze_loss.to(self.device)
        if self.enable_coarse_gaze_loss:
            self.coarse_gaze_loss.to(self.device)

        if self.enable_hq_hrffa_loss:
            self.hq_hrffa_loss = HRFFAFacialGeometryLoss(
                pose_weight=float(hq_hrffa_config["pose_weight"]),
                eye_weight=float(hq_hrffa_config["eye_weight"]),
                mouth_weight=float(hq_hrffa_config["mouth_weight"]),
                contour_weight=float(hq_hrffa_config["contour_weight"]),
                contour_shape_weight=float(hq_hrffa_config["contour_shape_weight"]),
                occluded_geometry_weight=float(hq_hrffa_config["occluded_geometry_weight"]),
            )
        if self.enable_coarse_hrffa_loss:
            self.coarse_hrffa_loss = HRFFAFacialGeometryLoss(
                pose_weight=float(coarse_hrffa_config["pose_weight"]),
                eye_weight=float(coarse_hrffa_config["eye_weight"]),
                mouth_weight=float(coarse_hrffa_config["mouth_weight"]),
                contour_weight=float(coarse_hrffa_config["contour_weight"]),
                contour_shape_weight=float(coarse_hrffa_config["contour_shape_weight"]),
                occluded_geometry_weight=float(coarse_hrffa_config["occluded_geometry_weight"]),
            )
            if self.enable_hq_hrffa_loss:
                self.coarse_hrffa_loss.hrffa = self.hq_hrffa_loss.hrffa
        if self.enable_hq_hrffa_loss:
            self.hq_hrffa_loss.to(self.device)
        if self.enable_coarse_hrffa_loss:
            self.coarse_hrffa_loss.to(self.device)

        if self.enable_hq_facs_loss:
            self.hq_facs_loss = FACSConsistencyLoss(
                weight=float(hq_facs_config["weight"]),
                brow_weight=float(hq_facs_config["brow_weight"]),
                eye_weight=float(hq_facs_config["eye_weight"]),
                nose_weight=float(hq_facs_config["nose_weight"]),
                mouth_weight=float(hq_facs_config["mouth_weight"]),
                lower_face_weight=float(hq_facs_config["lower_face_weight"]),
                asymmetry_weight=float(hq_facs_config["asymmetry_weight"]),
            )
        if self.enable_coarse_facs_loss:
            self.coarse_facs_loss = FACSConsistencyLoss(
                weight=float(coarse_facs_config["weight"]),
                brow_weight=float(coarse_facs_config["brow_weight"]),
                eye_weight=float(coarse_facs_config["eye_weight"]),
                nose_weight=float(coarse_facs_config["nose_weight"]),
                mouth_weight=float(coarse_facs_config["mouth_weight"]),
                lower_face_weight=float(coarse_facs_config["lower_face_weight"]),
                asymmetry_weight=float(coarse_facs_config["asymmetry_weight"]),
            )
            if self.enable_hq_facs_loss:
                self.coarse_facs_loss.au_model = self.hq_facs_loss.au_model
        if self.enable_hq_facs_loss:
            self.hq_facs_loss.to(self.device)
        if self.enable_coarse_facs_loss:
            self.coarse_facs_loss.to(self.device)

        if self.enable_hq_vgg_loss:
            self.hq_vgg_loss = VGGPerceptualLoss(layer_weights=hq_vgg_config["weights"], reduction="none").to(self.device)
        if self.enable_coarse_vgg_loss:
            if self.enable_hq_vgg_loss and coarse_vgg_config["weights"] == hq_vgg_config["weights"]:
                self.coarse_vgg_loss = self.hq_vgg_loss
            else:
                self.coarse_vgg_loss = VGGPerceptualLoss(layer_weights=coarse_vgg_config["weights"], reduction="none").to(self.device)

        # ========================= Run 输出 =========================
        self.run_paths = RunPaths.from_root(run_dir)
        self.run_id = run_id
        self.resolved_config_sha256 = resolved_config_sha256
        for path in (self.run_paths.checkpoints, self.run_paths.samples, self.run_paths.tensorboard):
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
        # Coarse 始终需要前向；HQ/D/loss teacher 仅为 active stage 构建训练热路径。
        if compile_module:
            initialize_upfirdn2d()
            self.train_coarse = _compile_training_callable(self.net_g.coarse)
            if self.hq_stage_active:
                self.train_hq = _compile_training_callable(self.net_g.hq)
                self.train_d = _compile_training_callable(self.net_d)
            if self.coarse_stage_active:
                self.train_d_coarse = _compile_training_callable(self.net_d_coarse)
            self.generator_id_encoder_forward = _compile_training_callable(self.generator_id_encoder)
            if self.hq_stage_active:
                self.hq_identity_embeddings_forward = _compile_training_callable(self.hq_id_loss.extract_identity_embeddings)
            if self.coarse_stage_active:
                if self.hq_stage_active and self.coarse_identity_loss_provider is self.hq_identity_loss_provider:
                    self.coarse_identity_embeddings_forward = self.hq_identity_embeddings_forward
                else:
                    self.coarse_identity_embeddings_forward = _compile_training_callable(self.coarse_id_loss.extract_identity_embeddings)
            if self.enable_hq_gaze_loss:
                self.hq_gaze_loss_forward = _compile_training_callable(self.hq_gaze_loss)
            if self.enable_coarse_gaze_loss:
                self.coarse_gaze_loss_forward = _compile_training_callable(self.coarse_gaze_loss)
            if self.enable_hq_hrffa_loss or self.enable_coarse_hrffa_loss:
                hrffa_loss = self.hq_hrffa_loss if self.enable_hq_hrffa_loss else self.coarse_hrffa_loss
                hrffa_loss.hrffa.network = _compile_training_callable(hrffa_loss.hrffa.network)
            if self.enable_hq_facs_loss or self.enable_coarse_facs_loss:
                if is_rocm_gfx1100(device_id):
                    print("警告：gfx1100 上 FACS/OpenGraphAU 保持 eager，避免 compiled 混合精度数值错误")
                else:
                    facs_loss = self.hq_facs_loss if self.enable_hq_facs_loss else self.coarse_facs_loss
                    compiled_au_model = _compile_training_callable(facs_loss.au_model)
                    if self.enable_hq_facs_loss:
                        self.hq_facs_loss.au_model = compiled_au_model
                    if self.enable_coarse_facs_loss:
                        self.coarse_facs_loss.au_model = compiled_au_model
            if self.enable_hq_vgg_loss:
                self.hq_vgg_loss_forward = _compile_training_callable(self.hq_vgg_loss)
            if self.enable_coarse_vgg_loss:
                if self.enable_hq_vgg_loss and self.coarse_vgg_loss is self.hq_vgg_loss:
                    self.coarse_vgg_loss_forward = self.hq_vgg_loss_forward
                else:
                    self.coarse_vgg_loss_forward = _compile_training_callable(self.coarse_vgg_loss)
        else:
            self.train_coarse = self.net_g.coarse
            if self.hq_stage_active:
                self.train_hq = self.net_g.hq
                self.train_d = self.net_d
                self.hq_identity_embeddings_forward = self.hq_id_loss.extract_identity_embeddings
            if self.coarse_stage_active:
                self.train_d_coarse = self.net_d_coarse
                if self.hq_stage_active and self.coarse_identity_loss_provider is self.hq_identity_loss_provider:
                    self.coarse_identity_embeddings_forward = self.hq_identity_embeddings_forward
                else:
                    self.coarse_identity_embeddings_forward = self.coarse_id_loss.extract_identity_embeddings
            self.generator_id_encoder_forward = self.generator_id_encoder
            if self.enable_hq_gaze_loss:
                self.hq_gaze_loss_forward = self.hq_gaze_loss
            if self.enable_coarse_gaze_loss:
                self.coarse_gaze_loss_forward = self.coarse_gaze_loss
            if self.enable_hq_vgg_loss:
                self.hq_vgg_loss_forward = self.hq_vgg_loss
            if self.enable_coarse_vgg_loss:
                self.coarse_vgg_loss_forward = self.coarse_vgg_loss

        tensorboard_purge_step = None
        if checkpoint is not None and checkpoint_mode == "resume":
            tensorboard_purge_step = self.completed_step + 1
        self.log_writer = SummaryWriter(self.run_paths.tensorboard, purge_step=tensorboard_purge_step)
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

    def _discriminator_stage_loss(
        self,
        stage: Literal["hq", "coarse"],
        fake: Tensor,
        real: Tensor,
        net: torch.nn.Module,
        train_net: Any,
        *,
        r1_enabled: bool,
        r1_interval: int,
        r1_gamma: float,
    ) -> Tensor:
        """计算单个全局判别器阶段的 adversarial + lazy R1。"""
        r1_step = r1_enabled and self.completed_step % r1_interval == 0
        if r1_step:
            # R1 涉及输入梯度的二阶反传，固定走未编译的 FP32 D，避免 AMP/compile
            # 改变梯度惩罚的数值路径。fake/real 分开前向也保持 MinibatchStd 统计隔离。
            with autocast(device_type="cuda", enabled=False):
                fake_img = fake.detach().float()
                real_img = real.detach().float().requires_grad_(True)
                fake_score = net(fake_img)
                real_score = net(real_img)
                adversarial_loss = self.d_loss(fake_score, real_score)
                r1_loss_raw = r1_reg_loss(real_score, real_img, gamma=r1_gamma)
                r1_loss = r1_loss_raw * r1_interval
            self.log(f"{stage}_r1_loss_raw", r1_loss_raw, force=True)
            self.log(f"{stage}_r1_loss", r1_loss, force=True)
            total = adversarial_loss + r1_loss
        else:
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                scores = train_net(
                    torch.cat((fake.detach(), real.detach()), dim=0),
                    True,
                )
                fake_score, real_score = scores.chunk(2, dim=0)
                adversarial_loss = self.d_loss(fake_score, real_score)
            total = adversarial_loss

        self.log(f"{stage}_d_loss", adversarial_loss)
        return total

    def _update_discriminator_stage(
        self,
        stage: Literal["hq", "coarse"],
        fake: Tensor,
        real: Tensor,
        net: torch.nn.Module,
        train_net: Any,
        optimizer: optim.Optimizer,
        scaler: GradScaler,
        named_parameters: Iterable[tuple[str, Tensor]],
        *,
        r1_enabled: bool,
        r1_interval: int,
        r1_gamma: float,
    ) -> None:
        """独立完成一个判别器 stage 的 backward/step；FP16 overflow 只重试当前 stage。"""
        overflow_retries = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            loss = self._discriminator_stage_loss(
                stage,
                fake,
                real,
                net,
                train_net,
                r1_enabled=r1_enabled,
                r1_interval=r1_interval,
                r1_gamma=r1_gamma,
            )
            _ensure_finite_loss(f"{stage}_d_loss", loss)
            if _scaled_backward_step(
                loss,
                optimizer,
                scaler,
                name=f"{stage} Discriminator",
                named_parameters=named_parameters,
            ):
                return

            overflow_retries += 1
            optimizer.zero_grad(set_to_none=True)
            if overflow_retries >= MAX_AMP_OVERFLOW_RETRIES:
                raise FloatingPointError(f"{stage} Discriminator 连续 {MAX_AMP_OVERFLOW_RETRIES} 次 FP16 gradient overflow，停止训练")

    def _coarse_generator_stage_loss(
        self,
        coarse: Tensor,
        coarse_resize_in: Tensor,
        dst_canonical: Tensor,
        theta_restore: Tensor,
        same_mask: Tensor,
        source_identity_embeddings: Tensor,
        restore_grid: Tensor | None,
    ) -> Tensor:
        """计算 Coarse stage 的全部 Generator 训练目标。"""
        if self.enable_coarse_wfm_loss:
            coarse_score, coarse_fake_feats = self.train_d_coarse(coarse, False, True)
        else:
            coarse_score = self.train_d_coarse(coarse)
        coarse_gan_loss = self.coarse_gan_loss(coarse_score)
        self.log("coarse_gan_loss", coarse_gan_loss)
        total = coarse_gan_loss

        if self.enable_coarse_wfm_loss:
            with torch.no_grad():
                _, coarse_real_feats = self.train_d_coarse(coarse_resize_in, False, True)
            coarse_wfm_loss = self.coarse_wfm_loss(coarse_fake_feats, coarse_real_feats)
            self.log("coarse_wfm_loss", coarse_wfm_loss)
            total = total + coarse_wfm_loss

        coarse_identity_embeddings = self.coarse_identity_embeddings_forward(self.prepare_identity_encoder_faces(coarse, theta_restore))
        coarse_id_loss = self.coarse_id_loss(coarse_identity_embeddings, source_identity_embeddings)
        self.log("coarse_id_loss", coarse_id_loss)
        total = total + coarse_id_loss

        geometry_enabled = self.enable_coarse_gaze_loss or self.enable_coarse_hrffa_loss or self.enable_coarse_facs_loss
        if geometry_enabled:
            if restore_grid is None:
                raise RuntimeError("Coarse geometry loss 需要 restore_grid")
            with autocast(device_type="cuda", enabled=False):
                coarse_full = NF.interpolate(coarse.float(), size=restore_grid.shape[1:3], mode="bilinear", align_corners=False)
                coarse_restored = NF.grid_sample(coarse_full, restore_grid, mode="bilinear", padding_mode="reflection", align_corners=False)

            if self.enable_coarse_gaze_loss:
                coarse_gaze_loss = self.coarse_gaze_loss_forward(coarse_restored, dst_canonical)
                self.log("coarse_gaze_loss", coarse_gaze_loss)
                total = total + coarse_gaze_loss

            if self.enable_coarse_hrffa_loss:
                coarse_hrffa_components = self.coarse_hrffa_loss.forward_components(coarse_restored, dst_canonical)
                for name, component in coarse_hrffa_components.items():
                    self.log(f"coarse_hrffa_{name}_loss", component)
                total = total + torch.stack(tuple(coarse_hrffa_components.values())).sum()

            if self.enable_coarse_facs_loss:
                coarse_facs_components = self.coarse_facs_loss.forward_components(coarse_restored, dst_canonical)
                for name, component in coarse_facs_components.items():
                    self.log(f"coarse_facs_{name}_loss", component)
                total = total + torch.stack(tuple(coarse_facs_components.values())).sum()

        if self.enable_coarse_vgg_loss:
            coarse_vgg_per_sample = self.coarse_vgg_loss_forward(coarse, coarse_resize_in)
            coarse_vgg_loss = _reduce_reconstruction_loss(coarse_vgg_per_sample, same_mask, self.coarse_reconstruction_scope)
            self.log("coarse_vgg_loss", coarse_vgg_loss)
            total = total + coarse_vgg_loss

        if self.enable_coarse_l1_loss:
            coarse_l1_per_sample = self.coarse_l1_loss(coarse, coarse_resize_in).flatten(1).mean(dim=1)
            coarse_l1_loss = _reduce_reconstruction_loss(coarse_l1_per_sample, same_mask, self.coarse_reconstruction_scope)
            self.log("coarse_l1_loss", coarse_l1_loss)
            total = total + coarse_l1_loss

        return total

    def _hq_generator_stage_loss(
        self,
        fake: Tensor,
        dst: Tensor,
        dst_canonical: Tensor,
        theta_restore: Tensor,
        same_mask: Tensor,
        source_identity_embeddings: Tensor,
        restore_grid: Tensor | None,
    ) -> Tensor:
        """计算 HQ stage 的全部 Generator 训练目标。"""
        if self.enable_hq_wfm_loss:
            hq_score, hq_fake_feats = self.train_d(fake, False, True)
        else:
            hq_score = self.train_d(fake)
        hq_gan_loss = self.hq_gan_loss(hq_score)
        self.log("hq_gan_loss", hq_gan_loss)
        total = hq_gan_loss

        if self.enable_hq_wfm_loss:
            with torch.no_grad():
                _, hq_real_feats = self.train_d(dst, False, True)
            hq_wfm_loss = self.hq_wfm_loss(hq_fake_feats, hq_real_feats)
            self.log("hq_wfm_loss", hq_wfm_loss)
            total = total + hq_wfm_loss

        hq_identity_embeddings = self.hq_identity_embeddings_forward(self.prepare_identity_encoder_faces(fake, theta_restore))
        hq_id_loss = self.hq_id_loss(hq_identity_embeddings, source_identity_embeddings)
        self.log("hq_id_loss", hq_id_loss)
        total = total + hq_id_loss

        geometry_enabled = self.enable_hq_gaze_loss or self.enable_hq_hrffa_loss or self.enable_hq_facs_loss
        if geometry_enabled:
            if restore_grid is None:
                raise RuntimeError("HQ geometry loss 需要 restore_grid")
            with autocast(device_type="cuda", enabled=False):
                fake_restored = NF.grid_sample(fake.float(), restore_grid, mode="bilinear", padding_mode="reflection", align_corners=False)

            if self.enable_hq_gaze_loss:
                hq_gaze_loss = self.hq_gaze_loss_forward(fake_restored, dst_canonical)
                self.log("hq_gaze_loss", hq_gaze_loss)
                total = total + hq_gaze_loss

            if self.enable_hq_hrffa_loss:
                hq_hrffa_components = self.hq_hrffa_loss.forward_components(fake_restored, dst_canonical)
                for name, component in hq_hrffa_components.items():
                    self.log(f"hq_hrffa_{name}_loss", component)
                total = total + torch.stack(tuple(hq_hrffa_components.values())).sum()

            if self.enable_hq_facs_loss:
                hq_facs_components = self.hq_facs_loss.forward_components(fake_restored, dst_canonical)
                for name, component in hq_facs_components.items():
                    self.log(f"hq_facs_{name}_loss", component)
                total = total + torch.stack(tuple(hq_facs_components.values())).sum()

        if self.enable_hq_vgg_loss:
            hq_vgg_per_sample = self.hq_vgg_loss_forward(fake, dst)
            hq_vgg_loss = _reduce_reconstruction_loss(hq_vgg_per_sample, same_mask, self.hq_reconstruction_scope)
            self.log("hq_vgg_loss", hq_vgg_loss)
            total = total + hq_vgg_loss

        if self.enable_hq_l1_loss:
            hq_l1_per_sample = self.hq_l1_loss(fake, dst).flatten(1).mean(dim=1)
            hq_l1_loss = _reduce_reconstruction_loss(hq_l1_per_sample, same_mask, self.hq_reconstruction_scope)
            self.log("hq_l1_loss", hq_l1_loss)
            total = total + hq_l1_loss

        return total

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
            "optim_d_hq": self.optim_d_hq.state_dict() if self.optim_d_hq is not None else None,
            "optim_d_coarse": self.optim_d_coarse.state_dict() if self.optim_d_coarse is not None else None,
            "scaler_g": self.scaler_g.state_dict() if self.precision == "fp16" else None,
            "scaler_d_hq": self.scaler_d_hq.state_dict() if self.precision == "fp16" and self.scaler_d_hq is not None else None,
            "scaler_d_coarse": self.scaler_d_coarse.state_dict() if self.precision == "fp16" and self.scaler_d_coarse is not None else None,
            "lr_scheduler_g": self.lr_scheduler_g.state_dict() if self.use_cosine_lr_g else None,
            "lr_scheduler_d_hq": self.lr_scheduler_d_hq.state_dict() if self.use_cosine_lr_d_hq else None,
            "lr_scheduler_d_coarse": self.lr_scheduler_d_coarse.state_dict() if self.use_cosine_lr_d_coarse else None,
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
                "coarse_identity_loss": self.coarse_identity_loss_provider.name,
                "hq_identity_loss": self.hq_identity_loss_provider.name,
            },
            "training_config": self.training_config,
            "net_g": net_g,
            "net_d": net_d,
            "net_d_coarse": net_d_coarse,
            "training_state": training_state,
        }

        ckpt_file = self.run_paths.checkpoints / f"step_{completed_step:09d}.pth"
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

    def _sample_stage_grad_maps(
        self,
        stage: Literal["coarse", "hq"],
        generated: Tensor,
        reconstruction_target: Tensor,
        dst_canonical: Tensor,
        theta_restore: Tensor,
        same_mask: Tensor,
        source_identity_embeddings: Tensor,
        restore_grid: Tensor | None,
        display_size: tuple[int, int],
    ) -> list[tuple[str, Tensor]]:
        """仅为当前启用的 Generator loss 生成旧版风格梯度图。R1 不参与。"""
        if stage == "coarse":
            prefix = "COARSE"
            discriminator = self.net_d_coarse
            gan_loss_fn = self.coarse_gan_loss
            identity_embeddings_forward = self.coarse_identity_embeddings_forward
            identity_loss_fn = self.coarse_id_loss
            wfm_enabled = self.enable_coarse_wfm_loss
            wfm_loss_fn = self.coarse_wfm_loss if wfm_enabled else None
            gaze_enabled = self.enable_coarse_gaze_loss
            gaze_loss_fn = self.coarse_gaze_loss_forward if gaze_enabled else None
            hrffa_enabled = self.enable_coarse_hrffa_loss
            hrffa_loss_fn = self.coarse_hrffa_loss if hrffa_enabled else None
            facs_enabled = self.enable_coarse_facs_loss
            facs_loss_fn = self.coarse_facs_loss if facs_enabled else None
            vgg_enabled = self.enable_coarse_vgg_loss
            vgg_loss_fn = self.coarse_vgg_loss_forward if vgg_enabled else None
            l1_enabled = self.enable_coarse_l1_loss
            l1_loss_fn = self.coarse_l1_loss if l1_enabled else None
            reconstruction_scope = self.coarse_reconstruction_scope
        else:
            prefix = "HQ"
            discriminator = self.net_d
            gan_loss_fn = self.hq_gan_loss
            identity_embeddings_forward = self.hq_identity_embeddings_forward
            identity_loss_fn = self.hq_id_loss
            wfm_enabled = self.enable_hq_wfm_loss
            wfm_loss_fn = self.hq_wfm_loss if wfm_enabled else None
            gaze_enabled = self.enable_hq_gaze_loss
            gaze_loss_fn = self.hq_gaze_loss_forward if gaze_enabled else None
            hrffa_enabled = self.enable_hq_hrffa_loss
            hrffa_loss_fn = self.hq_hrffa_loss if hrffa_enabled else None
            facs_enabled = self.enable_hq_facs_loss
            facs_loss_fn = self.hq_facs_loss if facs_enabled else None
            vgg_enabled = self.enable_hq_vgg_loss
            vgg_loss_fn = self.hq_vgg_loss_forward if vgg_enabled else None
            l1_enabled = self.enable_hq_l1_loss
            l1_loss_fn = self.hq_l1_loss if l1_enabled else None
            reconstruction_scope = self.hq_reconstruction_scope

        rows: list[tuple[str, Tensor]] = []

        def add_row(label: str, grad_map: Tensor) -> None:
            if tuple(grad_map.shape[-2:]) != display_size:
                grad_map = NF.interpolate(grad_map, size=display_size, mode="bilinear", align_corners=False)
            rows.append((label, grad_map))

        discriminator_training = discriminator.training
        discriminator.eval()
        try:
            with torch.enable_grad():
                x = generated.detach().requires_grad_(True)
                score = discriminator(x)
                add_row(f"{prefix} GAN GRAD", self.loss_grad_map(gan_loss_fn(score), x))

            if wfm_enabled:
                assert wfm_loss_fn is not None
                with torch.enable_grad():
                    x = generated.detach().requires_grad_(True)
                    _, fake_features = discriminator(x, False, True)
                    with torch.no_grad():
                        _, real_features = discriminator(reconstruction_target, False, True)
                    add_row(f"{prefix} WFM GRAD", self.loss_grad_map(wfm_loss_fn(fake_features, real_features), x))
        finally:
            discriminator.train(discriminator_training)

        with torch.enable_grad():
            x = generated.detach().requires_grad_(True)
            generated_identity_embeddings = identity_embeddings_forward(self.prepare_identity_encoder_faces(x, theta_restore))
            add_row(f"{prefix} ID GRAD", self.loss_grad_map(identity_loss_fn(generated_identity_embeddings, source_identity_embeddings.detach()), x))

        geometry_enabled = gaze_enabled or hrffa_enabled or facs_enabled
        if geometry_enabled and restore_grid is None:
            raise RuntimeError(f"{stage} sample geometry gradient 需要 restore_grid")

        def restored_input(x: Tensor) -> Tensor:
            assert restore_grid is not None
            with autocast(device_type="cuda", enabled=False):
                if stage == "coarse":
                    x = NF.interpolate(x.float(), size=restore_grid.shape[1:3], mode="bilinear", align_corners=False)
                return NF.grid_sample(x.float(), restore_grid, mode="bilinear", padding_mode="reflection", align_corners=False)

        if gaze_enabled:
            assert gaze_loss_fn is not None
            with torch.enable_grad():
                x = generated.detach().requires_grad_(True)
                add_row(f"{prefix} GAZE GRAD", self.loss_grad_map(gaze_loss_fn(restored_input(x), dst_canonical), x))

        if hrffa_enabled:
            assert hrffa_loss_fn is not None
            with torch.enable_grad():
                x = generated.detach().requires_grad_(True)
                components = hrffa_loss_fn.forward_components(restored_input(x), dst_canonical)
                loss = torch.stack(tuple(components.values())).sum()
                add_row(f"{prefix} HRFFA GRAD", self.loss_grad_map(loss, x))

        if facs_enabled:
            assert facs_loss_fn is not None
            with torch.enable_grad():
                x = generated.detach().requires_grad_(True)
                components = facs_loss_fn.forward_components(restored_input(x), dst_canonical)
                loss = torch.stack(tuple(components.values())).sum()
                add_row(f"{prefix} FACS GRAD", self.loss_grad_map(loss, x))

        if vgg_enabled:
            assert vgg_loss_fn is not None
            with torch.enable_grad():
                x = generated.detach().requires_grad_(True)
                per_sample = vgg_loss_fn(x, reconstruction_target)
                loss = _reduce_reconstruction_loss(per_sample, same_mask, reconstruction_scope)
                add_row(f"{prefix} VGG GRAD", self.loss_grad_map(loss, x))

        if l1_enabled:
            assert l1_loss_fn is not None
            with torch.enable_grad():
                x = generated.detach().requires_grad_(True)
                per_sample = l1_loss_fn(x, reconstruction_target).flatten(1).mean(dim=1)
                loss = _reduce_reconstruction_loss(per_sample, same_mask, reconstruction_scope)
                add_row(f"{prefix} L1 GRAD", self.loss_grad_map(loss, x))

        return rows

    def _save_sample(
        self,
        sample_reference: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        current_batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> None:
        sample_src, sample_dst, sample_dst_canonical, sample_theta_restore, sample_same_mask, sample_src_identity_faces = (
            tensor.to(self.device) for tensor in sample_reference
        )
        src, dst, dst_canonical, theta_restore, same_mask, source_identity_faces = current_batch
        with torch.no_grad(), autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
            half = sample_src.shape[0]
            dynamic_count = self.batch_size - half
            src_vis = torch.cat((sample_src, src[:dynamic_count]), dim=0)
            dst_vis = torch.cat((sample_dst, dst[:dynamic_count]), dim=0)
            dst_canonical_vis = torch.cat((sample_dst_canonical, dst_canonical[:dynamic_count]), dim=0)
            theta_restore_vis = torch.cat((sample_theta_restore, theta_restore[:dynamic_count]), dim=0)
            same_mask_vis = torch.cat((sample_same_mask, same_mask[:dynamic_count]), dim=0)
            source_identity_faces_vis = torch.cat((sample_src_identity_faces, source_identity_faces[:dynamic_count]), dim=0)

            generator_identity_embeddings_vis = self.generator_id_encoder_forward(source_identity_faces_vis)
            if self.hq_stage_active:
                if self.reuse_generator_identity_for_hq_source:
                    hq_source_identity_embeddings_vis = generator_identity_embeddings_vis
                else:
                    hq_source_identity_embeddings_vis = self.hq_identity_embeddings_forward(source_identity_faces_vis)
            if self.coarse_stage_active:
                if self.reuse_generator_identity_for_coarse_source:
                    coarse_source_identity_embeddings_vis = generator_identity_embeddings_vis
                elif self.hq_stage_active and self.coarse_identity_loss_provider is self.hq_identity_loss_provider:
                    coarse_source_identity_embeddings_vis = hq_source_identity_embeddings_vis
                else:
                    coarse_source_identity_embeddings_vis = self.coarse_identity_embeddings_forward(source_identity_faces_vis)

            fake_vis, coarse_vis = self.net_g_ema(dst_vis, generator_identity_embeddings_vis, return_coarse=True)
            coarse_resize_in_vis = NF.interpolate(dst_vis, size=self.coarse_resolution, mode="bilinear", align_corners=False)
            coarse_display_vis = NF.interpolate(coarse_vis, size=fake_vis.shape[2:], mode="bilinear", align_corners=False)
            identity_encoder_input_vis = self.prepare_identity_encoder_faces(fake_vis, theta_restore_vis)
            identity_encoder_input_display_vis = NF.interpolate(identity_encoder_input_vis, size=fake_vis.shape[2:], mode="bilinear", align_corners=False)

            restore_grid_vis = None
            if (
                self.enable_hq_gaze_loss
                or self.enable_hq_hrffa_loss
                or self.enable_hq_facs_loss
                or self.enable_coarse_gaze_loss
                or self.enable_coarse_hrffa_loss
                or self.enable_coarse_facs_loss
            ):
                with autocast(device_type="cuda", enabled=False):
                    restore_grid_vis = NF.affine_grid(theta_restore_vis.float(), size=list(dst_vis.shape), align_corners=False)

            rows: list[tuple[str, Tensor]] = [
                ("SOURCE / SRC", src_vis),
                ("TARGET / DST", dst_vis),
                ("COARSE OUTPUT", coarse_display_vis),
                ("FINAL OUTPUT", fake_vis),
                ("DST CANONICAL", dst_canonical_vis),
                ("FINAL ID INPUT", identity_encoder_input_display_vis),
            ]

            if self.coarse_stage_active:
                rows.extend(
                    self._sample_stage_grad_maps(
                        "coarse",
                        coarse_vis,
                        coarse_resize_in_vis,
                        dst_canonical_vis,
                        theta_restore_vis,
                        same_mask_vis,
                        coarse_source_identity_embeddings_vis,
                        restore_grid_vis,
                        tuple(fake_vis.shape[-2:]),
                    )
                )

            if self.hq_stage_active:
                rows.extend(
                    self._sample_stage_grad_maps(
                        "hq",
                        fake_vis,
                        dst_vis,
                        dst_canonical_vis,
                        theta_restore_vis,
                        same_mask_vis,
                        hq_source_identity_embeddings_vis,
                        restore_grid_vis,
                        tuple(fake_vis.shape[-2:]),
                    )
                )

            rendered_rows = [_sample_row_bgr(images, label, self.batch_size) for label, images in rows]
            grid_cpu = np.concatenate(rendered_rows, axis=0)

        sample_file = self.run_paths.samples / f"step_{self.completed_step:09d}.png"
        if not cv2.imwrite(sample_file, grid_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"保存训练 sample 失败：{sample_file}")

    def train(self) -> None:

        net_coarse = self.train_coarse
        sample_src, sample_dst, sample_dst_canonical, sample_theta_restore, sample_same_mask, sample_src_identity_faces = self.dataset.next()
        half = self.batch_size // 2
        sample_reference = tuple(
            tensor[:half].detach().cpu()
            for tensor in (sample_src, sample_dst, sample_dst_canonical, sample_theta_restore, sample_same_mask, sample_src_identity_faces)
        )
        with tqdm(total=None, initial=self.completed_step, mininterval=1.0, bar_format="{n_fmt:7} | 速度 {rate_fmt:3} | 训练时间 {elapsed}") as progress:
            while True:
                if self._stop_requested:
                    raise KeyboardInterrupt

                src, dst, dst_canonical, theta_restore, same_mask, source_identity_faces = self.dataset.next()
                self._step_in_progress = True

                # ========================= 生成器前向 =========================
                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                    with torch.no_grad():
                        generator_identity_embeddings = self.generator_id_encoder_forward(source_identity_faces)

                    if self.coarse_stage_active:
                        coarse, coarse_resize_in = net_coarse(dst, generator_identity_embeddings, return_resize_in=True)
                    else:
                        with torch.no_grad():
                            coarse = net_coarse(dst, generator_identity_embeddings)

                    if self.hq_stage_active:
                        fake = self.train_hq(dst, coarse.detach())

                # ========================= 训练判别器 =========================
                self.net_d.requires_grad_(self.hq_stage_active)
                self.net_d_coarse.requires_grad_(self.coarse_stage_active)

                if self.hq_stage_active:
                    assert self.optim_d_hq is not None and self.scaler_d_hq is not None
                    self._update_discriminator_stage(
                        "hq",
                        fake,
                        dst,
                        self.net_d,
                        self.train_d,
                        self.optim_d_hq,
                        self.scaler_d_hq,
                        self._hq_d_named_parameters,
                        r1_enabled=self.enable_hq_r1_loss,
                        r1_interval=self.hq_r1_reg_step,
                        r1_gamma=self.hq_r1_gamma,
                    )
                    if self.use_cosine_lr_d_hq:
                        self.lr_scheduler_d_hq.step()

                if self.coarse_stage_active:
                    assert self.optim_d_coarse is not None and self.scaler_d_coarse is not None
                    self._update_discriminator_stage(
                        "coarse",
                        coarse,
                        coarse_resize_in,
                        self.net_d_coarse,
                        self.train_d_coarse,
                        self.optim_d_coarse,
                        self.scaler_d_coarse,
                        self._coarse_d_named_parameters,
                        r1_enabled=self.enable_coarse_r1_loss,
                        r1_interval=self.coarse_r1_reg_step,
                        r1_gamma=self.coarse_r1_gamma,
                    )
                    if self.use_cosine_lr_d_coarse:
                        self.lr_scheduler_d_coarse.step()

                # source/reference 与 generated/fake identity teacher 必须使用同一 AMP 精度，
                # 避免余弦损失两端落在不同数值空间；reference 无需梯度，但仍跟随训练 autocast。
                # 只为 active stage 计算，并跨 G overflow retry 复用。
                with torch.no_grad(), autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                    if self.hq_stage_active:
                        if self.reuse_generator_identity_for_hq_source:
                            hq_source_identity_embeddings = generator_identity_embeddings
                        else:
                            hq_source_identity_embeddings = self.hq_identity_embeddings_forward(source_identity_faces)
                    if self.coarse_stage_active:
                        if self.reuse_generator_identity_for_coarse_source:
                            coarse_source_identity_embeddings = generator_identity_embeddings
                        elif self.hq_stage_active and self.coarse_identity_loss_provider is self.hq_identity_loss_provider:
                            coarse_source_identity_embeddings = hq_source_identity_embeddings
                        else:
                            coarse_source_identity_embeddings = self.coarse_identity_embeddings_forward(source_identity_faces)

                # ========================= 训练生成器 =========================
                self.net_d.requires_grad_(False)
                self.net_d_coarse.requires_grad_(False)
                hq_d_training = self.net_d.training
                coarse_d_training = self.net_d_coarse.training
                self.net_d.eval()
                self.net_d_coarse.eval()
                g_overflow_retries = 0

                while True:
                    self.optim_g.zero_grad(set_to_none=True)
                    if g_overflow_retries > 0:
                        # D 已经成功更新，G overflow 时只重算 active Generator stage。
                        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                            if self.coarse_stage_active:
                                coarse, coarse_resize_in = net_coarse(dst, generator_identity_embeddings, return_resize_in=True)
                            if self.hq_stage_active:
                                fake = self.train_hq(dst, coarse.detach())

                    with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                        restore_grid = None
                        if self.enable_hq_gaze_loss or self.enable_hq_hrffa_loss or self.enable_hq_facs_loss or self.enable_coarse_gaze_loss or self.enable_coarse_hrffa_loss or self.enable_coarse_facs_loss:
                            with autocast(device_type="cuda", enabled=False):
                                restore_grid = NF.affine_grid(theta_restore.float(), size=list(dst.shape), align_corners=False)

                        if self.coarse_stage_active:
                            coarse_g_loss = self._coarse_generator_stage_loss(
                                coarse,
                                coarse_resize_in,
                                dst_canonical,
                                theta_restore,
                                same_mask,
                                coarse_source_identity_embeddings,
                                restore_grid,
                            )
                        if self.hq_stage_active:
                            hq_g_loss = self._hq_generator_stage_loss(
                                fake,
                                dst,
                                dst_canonical,
                                theta_restore,
                                same_mask,
                                hq_source_identity_embeddings,
                                restore_grid,
                            )

                        if self.train_stage == "joint":
                            g_loss = coarse_g_loss + hq_g_loss
                        elif self.hq_stage_active:
                            g_loss = hq_g_loss
                        else:
                            g_loss = coarse_g_loss

                    _ensure_finite_loss("g_loss", g_loss)
                    if _scaled_backward_step(
                        g_loss,
                        self.optim_g,
                        self.scaler_g,
                        name="Generator",
                        named_parameters=self._g_named_parameters,
                    ):
                        break

                    g_overflow_retries += 1
                    self.optim_g.zero_grad(set_to_none=True)
                    if g_overflow_retries >= MAX_AMP_OVERFLOW_RETRIES:
                        raise FloatingPointError(f"Generator 连续 {MAX_AMP_OVERFLOW_RETRIES} 次 FP16 gradient overflow，停止训练")

                self.net_d.train(hq_d_training)
                self.net_d_coarse.train(coarse_d_training)

                if self.use_cosine_lr_g:
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
                    self._save_sample(sample_reference, (src, dst, dst_canonical, theta_restore, same_mask, source_identity_faces))


def _load_run_config(paths: RunPaths) -> dict[str, Any]:
    resolved = load_resolved_config(paths)
    canonical = resolve_train_config(resolved)
    if canonical != resolved:
        raise ValueError(f"run 的 resolved config 不是当前格式的规范表示：{paths.resolved_config}")
    return resolved


def _coarse_generator_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """提取决定 Coarse 参数结构与 identity 输入空间的 Generator 配置。"""
    return {key: config[key] for key in COARSE_GENERATOR_CONFIG_KEYS}


def _branch_generator_mode(checkpoint: Mapping[str, Any], resolved: Mapping[str, Any]) -> Literal["inherit", "hq_rebuild"]:
    """决定 branch 是完整继承 Generator，还是仅继承 Coarse 并重建 HQ。"""
    parent_generator = dict(checkpoint["net_g"]["network_cfg"])
    current_generator = dict(resolved["generator"])
    hq_only = resolved["train"]["stage"] == "hq"

    if hq_only:
        identity_encoders = checkpoint.get("identity_encoders")
        if not isinstance(identity_encoders, Mapping):
            raise TypeError("HQ-only branch 要求父 checkpoint 记录 identity_encoders")
        parent_generator_provider = identity_encoders.get("generator")
        current_generator_provider = resolved["identity"]["provider"]
        if parent_generator_provider != current_generator_provider:
            raise ValueError(f"HQ-only branch 要求 Generator identity provider 一致：checkpoint={parent_generator_provider!r}, config={current_generator_provider!r}")

    if parent_generator == current_generator:
        return "inherit"
    if not hq_only:
        raise ValueError("branch 修改 Generator 架构仅允许 train.stage='hq' 的 HQ rebuild")
    if _coarse_generator_config(parent_generator) != _coarse_generator_config(current_generator):
        raise ValueError("HQ rebuild 要求 Coarse 架构保持一致")
    return "hq_rebuild"


def _resolve_branch_start_step(branch_mode: Literal["inherit", "hq_rebuild"], checkpoint_step: int, requested_step: int | None) -> int:
    """普通 branch 可继承/重设 step；HQ rebuild 始终从 0 开始。"""
    if branch_mode == "hq_rebuild":
        if requested_step not in (None, 0):
            raise ValueError("HQ rebuild 必须从 --step 0 开始；省略 --step 即自动从 0 开始")
        return 0
    return checkpoint_step if requested_step is None else requested_step


def _print_hq_rebuild_summary(checkpoint: Mapping[str, Any], resolved: Mapping[str, Any], start_step: int) -> None:
    """打印 HQ rebuild 的关键继承/重建语义和实际 HQ 配置变化。"""
    parent_generator = checkpoint["net_g"]["network_cfg"]
    current_generator = resolved["generator"]
    changed_hq_config = [(key, parent_generator.get(key), value) for key, value in current_generator.items() if key not in COARSE_GENERATOR_CONFIG_KEYS and parent_generator.get(key) != value]

    print("HQ rebuild:")
    print("  HQ 配置变化:")
    for key, old_value, new_value in changed_hq_config:
        print(f"    {key:26}: {old_value} -> {new_value}")
    print("  Coarse                  : inherit + frozen")
    print("  HQ                      : reset")
    print("  HQ Discriminator        : reset")
    print(f"  Generator identity      : {resolved['identity']['provider']}")
    print(f"  起始 step                : {start_step}")


def _submodule_state_dict(state_dict: Mapping[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """从完整 Generator state_dict 中严格提取一个子模块的 state_dict。"""
    full_prefix = f"{prefix}."
    result = {key.removeprefix(full_prefix): value for key, value in state_dict.items() if key.startswith(full_prefix)}
    if not result:
        raise ValueError(f"Generator checkpoint 缺少 {prefix} state_dict")
    return result


def _branch_model_states(
    checkpoint: Mapping[str, Any],
    *,
    reset_hq_discriminator: bool,
    reset_coarse_discriminator: bool,
    hq_rebuild: bool = False,
) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor] | None, Mapping[str, Tensor] | None]:
    """Branch 继承训练态 G；HQ rebuild 时 HQ D 必定重建。"""
    g_state = checkpoint["training_state"]["net_g"]
    hq_d_state = None if (reset_hq_discriminator or hq_rebuild) else checkpoint["net_d"]["state_dict"]
    coarse_d_state = None if reset_coarse_discriminator else checkpoint["net_d_coarse"]["state_dict"]
    return g_state, hq_d_state, coarse_d_state


def _load_branch_checkpoint(
    checkpoint_path: Path,
    resolved: dict[str, Any],
    *,
    reset_hq_discriminator: bool,
    reset_coarse_discriminator: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_version = checkpoint["version"]
    if checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError(f"不支持的 checkpoint version：{checkpoint_version}，当前仅支持 v{CHECKPOINT_VERSION}")

    checkpoint_step = int(checkpoint["step"])
    filename_step = checkpoint_step_from_name(checkpoint_path.name)
    if filename_step != checkpoint_step:
        raise ValueError(f"checkpoint 文件名 step 与内部状态不一致：filename={filename_step}, checkpoint={checkpoint_step}")

    # Branch 创建新的训练 run，只要求 checkpoint 的模型/状态结构可被当前配置安全载入。
    # training_config/semantics_version 只用于严格 resume，不限制 branch。
    branch_mode = _branch_generator_mode(checkpoint, resolved)
    hq_rebuild = branch_mode == "hq_rebuild"
    effective_reset_hq_discriminator = reset_hq_discriminator or hq_rebuild

    if not effective_reset_hq_discriminator and _canonical_checkpoint_discriminator_config(checkpoint["net_d"]["network_cfg"]) != resolved["discriminator"]["hq"]:
        raise ValueError("branch 继承 HQ Discriminator 时要求架构一致；如需新建请使用 --reset-hq-discriminator")

    if not reset_coarse_discriminator and _canonical_checkpoint_discriminator_config(checkpoint["net_d_coarse"]["network_cfg"]) != resolved["discriminator"]["coarse"]:
        raise ValueError("branch 继承 Coarse Discriminator 时要求架构一致；如需新建请使用 --reset-coarse-discriminator")

    _branch_model_states(
        checkpoint,
        reset_hq_discriminator=reset_hq_discriminator,
        reset_coarse_discriminator=reset_coarse_discriminator,
        hq_rebuild=hq_rebuild,
    )
    checkpoint_run = checkpoint["run"]
    parent = {
        "run_id": checkpoint_run["id"],
        "checkpoint": checkpoint_path.name,
        "step": checkpoint_step,
        "config_sha256": checkpoint_run["config_sha256"],
        "branch_mode": branch_mode,
        "generator": {
            "coarse": "inherit",
            "hq": "reset" if hq_rebuild else "inherit",
        },
        "discriminator": {
            "hq": "reset" if effective_reset_hq_discriminator else "inherit",
            "coarse": "reset" if reset_coarse_discriminator else "inherit",
        },
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
    source_group.add_argument("--branch-from", type=Path, default=None, help="从 checkpoint 创建新 run；默认继承训练态 Generator，HQ rebuild 仅继承 Coarse")
    parser.add_argument("--reset-hq-discriminator", action="store_true", help="仅用于 branch：重新初始化 HQ Discriminator；HQ rebuild 会自动执行")
    parser.add_argument("--reset-coarse-discriminator", action="store_true", help="仅用于 branch：重新初始化 Coarse Discriminator")
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
    if args.reset_hq_discriminator and not is_branch:
        parser.error("--reset-hq-discriminator 仅用于 --branch-from")
    if args.reset_coarse_discriminator and not is_branch:
        parser.error("--reset-coarse-discriminator 仅用于 --branch-from")
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
            reset_hq_discriminator=args.reset_hq_discriminator,
            reset_coarse_discriminator=args.reset_coarse_discriminator,
        )
        try:
            start_step = _resolve_branch_start_step(parent["branch_mode"], int(preloaded_checkpoint["step"]), args.step)
        except ValueError as error:
            parser.error(str(error))
        if parent["branch_mode"] == "hq_rebuild":
            _print_hq_rebuild_summary(preloaded_checkpoint, resolved, start_step)

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
                reset_hq_discriminator=args.reset_hq_discriminator,
                reset_coarse_discriminator=args.reset_coarse_discriminator,
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
