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
    GazeLoss,
    GeneratorAdversarialLoss,
    IdentityLoss,
    SMIRKMotionLoss,
    VGGPerceptualLoss,
    WeakFeatureMatchingLoss,
    make_blurred_l1_loss,
    make_l1_loss,
    r1_reg_loss,
)
from misc.face_alignment import arcface_to_identity_112, ffhq_to_arcface_112, make_arcface128_to_arcface112_grid, make_ffhq_to_arcface_112_grid, transform_sampling_grid
from misc.models.face_mask import FaceMasker
from misc.models.id_encoder import IDEncoder, IDEncoderProvider
from models.discriminator import UNetDiscriminatorSN, build_discriminator
from models.discriminator.upfirdn2d import initialize_upfirdn2d
from models.networks import Generator

from .config import load_train_config, resolve_train_config
from .contracts import CHECKPOINT_VERSION
from .dataloader import ImageDecoderBackend, TrainingDataLoader
from .dataloader_common import is_identity_geometry
from .experiment import (
    RunLock,
    RunPaths,
    config_sha256,
    create_run,
    load_metadata,
    load_resolved_config,
    resolve_branch_target,
    resolve_resume_target,
    update_metadata,
    write_latest,
)
from .occlusion import OcclusionAugmentor, gate_identity_gradient

EPS = 1e-8
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "experiments" / "train.toml"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "experiments" / "runs"
MAX_AMP_OVERFLOW_RETRIES = 16
# ArcFace 输入由两级缩放/仿射改为原图直接抗混叠采样，不能静默 resume 旧预处理状态。
TRAINING_SEMANTICS_VERSION = 25
SAMPLE_LABEL_WIDTH = 300
SAMPLE_GRID_PADDING = 2
# 默认保留原有逐损失亮度归一化；改为 "shared" 可直观比较同 stage 各梯度图的绝对强度。
SAMPLE_GRADIENT_HEATMAP_SCALE: Literal["independent", "shared"] = "independent"
# 合成梯度低于各项梯度范数总和的 20% 时，EFF 投影比例容易被抵消放大，显示 N/A。
SAMPLE_MIN_RESULTANT_RATIO_FOR_EFFECT = 0.2


def _sample_row_bgr(images: Tensor, label: str, nrow: int) -> np.ndarray:
    """把一个 sample row 转为 BGR，并在最左侧添加含义标签。"""
    images = images.detach().float().add(1.0).mul(127.5).clamp(0.0, 255.0)
    grid = make_grid(images, nrow=nrow, padding=SAMPLE_GRID_PADDING)[[2, 1, 0], :, :]
    image = grid.permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
    label_panel = np.full((image.shape[0], SAMPLE_LABEL_WIDTH, 3), 20, dtype=np.uint8)
    label_lines = label.splitlines()
    line_height = 26
    first_baseline = (image.shape[0] - (len(label_lines) - 1) * line_height) // 2 + 6
    for index, line in enumerate(label_lines):
        cv2.putText(label_panel, line, (14, first_baseline + index * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate((label_panel, image), axis=1)


def _format_sample_gradient_rows(
    rows: list[tuple[str, Tensor]],
    gradients: list[Tensor],
    *,
    heatmap_scale: Literal["independent", "shared"] = "independent",
) -> list[tuple[str, Tensor]]:
    """输出已加权 loss 的梯度强度、合成方向投影与 stage 抵消率。

    指标在 Generator 输出图像空间计算，而非模型参数空间。热力图默认逐损失归一化；
    shared 模式在当前 stage 的所有损失之间使用同一份逐样本亮度标尺。
    """
    if len(rows) != len(gradients):
        raise ValueError("gradient row/count mismatch")
    if not rows:
        return []
    if heatmap_scale not in ("independent", "shared"):
        raise ValueError(f"unsupported gradient heatmap scale: {heatmap_scale}")
    if any(grad.shape != gradients[0].shape for grad in gradients):
        raise ValueError("gradient shapes mismatch")

    # 保留梯度直到所有启用的 loss 计算完成；不需要再次运行任何损失网络。
    grads = [grad.detach().float() for grad in gradients]
    norms = torch.stack([torch.linalg.vector_norm(grad) for grad in grads])
    total_grad = grads[0].clone()
    for grad in grads[1:]:
        total_grad.add_(grad)
    total_norm = torch.linalg.vector_norm(total_grad)
    sum_norm = norms.sum()
    values = torch.cat((norms, total_norm[None], sum_norm[None])).cpu().tolist()
    if not all(np.isfinite(value) and value >= 0 for value in values):
        raise FloatingPointError("sample loss gradient norms contain invalid values")

    strength_denominator = sum_norm.clamp_min(torch.finfo(torch.float32).tiny)
    strengths = norms / strength_denominator * 100.0
    rms = norms / (grads[0].numel() ** 0.5)
    # DIR 仅在总梯度近乎为 0 时无定义；EFF 则在高抵消率下提前隐藏，
    # 避免方向仍可辨认但投影占比被很小的总梯度放大为极端百分比。
    direction_defined = values[-1] > 0.0 and values[-2] > 1e-4 * values[-1]
    effect_defined = direction_defined and values[-2] > SAMPLE_MIN_RESULTANT_RATIO_FOR_EFFECT * values[-1]
    if direction_defined:
        dots = torch.stack([torch.sum(grad * total_grad) for grad in grads])
        alignments = dots / (norms * total_norm).clamp_min(torch.finfo(torch.float32).tiny)
        effects = dots / total_norm.square() if effect_defined else torch.zeros_like(norms)
    else:
        alignments = torch.zeros_like(norms)
        effects = torch.zeros_like(norms)

    metrics = torch.stack((rms, strengths, alignments, effects), dim=1).cpu().tolist()
    cancel_pct = min(100.0, max(0.0, (1.0 - values[-2] / values[-1]) * 100.0)) if values[-1] > 0 else None
    if not all(all(np.isfinite(v) for v in metric) for metric in metrics):
        raise FloatingPointError("sample loss gradient metrics contain invalid values")

    # 归一化只影响显示亮度，不能参与梯度强度或方向的统计。
    maps = [torch.log1p(heatmap.detach().float()) for _, heatmap in rows]
    if heatmap_scale == "shared":
        shared_max = maps[0].amax(dim=(2, 3), keepdim=True)
        for heatmap in maps[1:]:
            shared_max = torch.maximum(shared_max, heatmap.amax(dim=(2, 3), keepdim=True))
    result: list[tuple[str, Tensor]] = []
    for index, ((label, _), heatmap, (grad_rms, strength, alignment, effect)) in enumerate(zip(rows, maps, metrics, strict=True)):
        if heatmap_scale == "independent":
            max_value = heatmap.amax(dim=(2, 3), keepdim=True)
        else:
            max_value = shared_max
        display_map = heatmap / max_value.clamp_min(EPS)
        display_map = display_map.mul(2.0).sub(1.0).expand(-1, 3, -1, -1).contiguous()

        direction_text = f"{alignment:+.2f}" if direction_defined and values[index] > 0 else "N/A"
        effective_text = f"{effect * 100.0:+.1f}%" if effect_defined else "N/A"
        label_lines = [label]
        if index == 0:
            label_lines.append(f"STAGE CANCEL {cancel_pct:.1f}%" if cancel_pct is not None else "STAGE CANCEL N/A")
        label_lines.extend((
            f"RMS {grad_rms:.2e}  STR {strength:.1f}%",
            f"DIR {direction_text}  EFF {effective_text}",
        ))
        result.append(("\n".join(label_lines), display_map))
    return result


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


def _compile_training_callable(fn: Any, mode: str) -> Any:
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

    expected_keys = {"semantics_version", "precision"}
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

    return training_config


def _reduce_reconstruction_loss(rec_per_sample: Tensor, same_mask: Tensor, scope: Literal["same", "all"]) -> Tensor:
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


def _validate_checkpoint_identity_providers(
    checkpoint: Mapping[str, Any],
    *,
    generator_provider: str,
    identity_loss_provider: str,
    checkpoint_mode: Literal["resume", "branch"],
) -> None:
    """Only an actual Resume needs to preserve the checkpoint's identity providers."""
    if checkpoint_mode != "resume":
        return
    providers = checkpoint["identity_encoders"]
    if providers["generator"] != generator_provider:
        raise ValueError("checkpoint generator identity provider mismatch")
    if providers["identity_loss"] != identity_loss_provider:
        raise ValueError("checkpoint identity loss provider mismatch")


def _branch_generator_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Tensor]:
    """Prefer training weights; also accept exported/EMA-only or raw state dicts."""
    training = checkpoint.get("training_state")
    if isinstance(training, Mapping) and isinstance(training.get("net_g"), Mapping):
        return training["net_g"]
    generator = checkpoint.get("net_g")
    if isinstance(generator, Mapping) and isinstance(generator.get("state_dict"), Mapping):
        return generator["state_dict"]
    if checkpoint and all(isinstance(key, str) and isinstance(value, Tensor) for key, value in checkpoint.items()):
        return checkpoint
    raise ValueError("Branch checkpoint does not contain loadable Generator weights")


def _branch_discriminator_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Tensor] | None:
    discriminator = checkpoint.get("net_d")
    if isinstance(discriminator, Mapping) and isinstance(discriminator.get("state_dict"), Mapping):
        return discriminator["state_dict"]
    return None


def _state_dict_shapes_match(model: torch.nn.Module, weights: Mapping[str, Tensor]) -> bool:
    """Check strict-load keys/shapes before copying any checkpoint tensors."""
    expected = model.state_dict()
    return expected.keys() == weights.keys() and all(
        isinstance(weights[key], Tensor) and expected[key].shape == weights[key].shape
        for key in expected
    )


def _branch_checkpoint_step(checkpoint: Mapping[str, Any]) -> int:
    step = checkpoint.get("step", 0)
    return step if type(step) is int and step >= 0 else 0


class Trainer:
    """Single-generator, single-discriminator training loop for both architectures."""

    def __init__(
        self, config: Mapping[str, Any], *, run_dir: str | Path,
        checkpoint_path: str | Path | None = None, run_id: str | None = None,
        resolved_config_sha256: str | None = None, strict_precision_resume: bool = False,
        checkpoint_mode: Literal["resume", "branch"] = "resume",
        preloaded_checkpoint: dict[str, Any] | None = None,
        reset_discriminator: bool = False, start_step: int | None = None,
    ) -> None:
        train_cfg = config["train"]
        generator_cfg = dict(config["generator"])
        discriminator_cfg = dict(config["discriminator"])
        loss_cfg = config["loss"]
        data_cfg = config["data"]
        self.img_resolution = generator_cfg["img_resolution"]
        self.face_alignment = data_cfg["alignment"]
        self.batch_size = train_cfg["batch_size"]
        self.log_interval = train_cfg["log_interval"]
        self.sample_save_every = train_cfg["sample_save_every"]
        self.checkpoint_save_every = train_cfg["checkpoint_save_every"]
        self._completed_step = 0
        self._step_in_progress = False
        self._stop_requested = False
        self.device = torch.device(train_cfg["device"])
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Trainer requires a CUDA/HIP GPU")
        print_mapping("训练配置", config)
        compile_module = train_cfg["compile_module"]
        compile_mode = train_cfg["compile_mode"]
        req = train_cfg["precision"]
        device_id = self.device.index if self.device.index is not None else torch.cuda.current_device()
        if req == "bf16" and not (torch.cuda.is_bf16_supported() and (not compile_module or _supports_compiled_bf16(device_id))):
            print("警告：当前设备不支持当前编译模式的 BF16，回退 FP32")
            req = "fp32"
        self.precision = req
        self.amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[req]
        self.amp_enabled = self.amp_dtype is not None
        self.training_config = {"semantics_version": TRAINING_SEMANTICS_VERSION, "precision": req}

        if checkpoint_mode not in ("resume", "branch"):
            raise ValueError("checkpoint_mode must be resume or branch")
        checkpoint = preloaded_checkpoint
        state = None
        saved_precision = None
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            if checkpoint is None:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, Mapping):
                raise ValueError("checkpoint must be a state dictionary")
            if checkpoint_mode == "resume":
                if checkpoint["version"] != CHECKPOINT_VERSION:
                    raise ValueError("checkpoint version mismatch")
                if checkpoint.get("face_alignment") != self.face_alignment:
                    raise ValueError("checkpoint alignment mismatch")
                if _canonical_generator_config(checkpoint["net_g"]["network_cfg"]) != _canonical_generator_config(generator_cfg):
                    raise ValueError("checkpoint Generator architecture mismatch")
                prior = _require_training_config(checkpoint)
                saved_precision = prior["precision"]
                if strict_precision_resume and saved_precision != self.precision:
                    raise ValueError("checkpoint precision mismatch")
                _validate_checkpoint_identity_providers(
                    checkpoint,
                    generator_provider=config["identity"]["provider"],
                    identity_loss_provider=loss_cfg["identity"]["provider"],
                    checkpoint_mode=checkpoint_mode,
                )
                if checkpoint["run"] != {"id": run_id, "config_sha256": resolved_config_sha256}:
                    raise ValueError("checkpoint run does not match frozen config")
                if checkpoint["net_d"]["network_cfg"] != discriminator_cfg:
                    raise ValueError("checkpoint discriminator mismatch")
                self._completed_step = checkpoint["step"]
                state = checkpoint["training_state"]
            else:
                self._completed_step = _branch_checkpoint_step(checkpoint) if start_step is None else start_step

        self.net_g = Generator(**generator_cfg).to(self.device).train()
        self.net_d = build_discriminator(discriminator_cfg).to(self.device).train()
        self.branch_discriminator_state: Literal["inherit", "reset"] | None = None
        if checkpoint is not None:
            if state is not None:
                self.net_g.load_state_dict(state["net_g"], strict=True)
                self.net_d.load_state_dict(checkpoint["net_d"]["state_dict"], strict=True)
            else:
                try:
                    self.net_g.load_state_dict(_branch_generator_state(checkpoint), strict=True)
                except RuntimeError as error:
                    raise ValueError(f"Branch Generator weights cannot be loaded: {error}") from error
                self.branch_discriminator_state = "reset"
                if not reset_discriminator:
                    d_weights = _branch_discriminator_state(checkpoint)
                    if d_weights is None:
                        print("Branch: no Discriminator weights in checkpoint; using a new Discriminator")
                    elif not _state_dict_shapes_match(self.net_d, d_weights):
                        print("Branch: Discriminator keys/shapes differ; using a new Discriminator")
                    else:
                        try:
                            self.net_d.load_state_dict(d_weights, strict=True)
                        except RuntimeError as error:
                            # load_state_dict can partially mutate a module before raising.
                            self.net_d = build_discriminator(discriminator_cfg).to(self.device).train()
                            print(f"Branch: Discriminator weights cannot be loaded; reinitializing: {str(error).splitlines()[0]}")
                        else:
                            self.branch_discriminator_state = "inherit"
        self.net_g_ema = copy.deepcopy(self.net_g).eval().requires_grad_(False)
        if state is not None:
            self.net_g_ema.load_state_dict(checkpoint["net_g"]["state_dict"], strict=True)
        self._train_g_params = tuple(self.net_g.parameters())
        self._ema_params = tuple(self.net_g_ema.parameters())
        self._g_named_parameters = tuple(self.net_g.named_parameters())
        self._d_named_parameters = tuple(self.net_d.named_parameters())
        opt = config["optimizer"]
        schedule = config["scheduler"]
        self.optim_g = optim.Adam(self._train_g_params, lr=opt["generator"]["lr"], betas=(0.0, .99), fused=True)
        self.optim_d = optim.Adam(self.net_d.parameters(), lr=opt["discriminator"]["lr"], betas=(0.0, .99), fused=True)
        self.scaler_g = GradScaler("cuda", enabled=self.precision == "fp16")
        self.scaler_d = GradScaler("cuda", enabled=self.precision == "fp16")
        if state is not None:
            self.optim_g.load_state_dict(state["optim_g"])
            self.optim_d.load_state_dict(state["optim_d"])
            if saved_precision == self.precision == "fp16":
                if state["scaler_g"] is not None: self.scaler_g.load_state_dict(state["scaler_g"])
                if state["scaler_d"] is not None: self.scaler_d.load_state_dict(state["scaler_d"])
        self.use_cosine_lr_g = schedule["generator"]["type"] == "cosine"
        self.use_cosine_lr_d = schedule["discriminator"]["type"] == "cosine"
        if self.use_cosine_lr_g:
            self.lr_scheduler_g = CosineAnnealingLR(self.optim_g, T_max=schedule["generator"]["t_max"], eta_min=opt["generator"]["lr"] * schedule["generator"]["min_lr_ratio"])
            if state is not None: self.lr_scheduler_g.load_state_dict(state["lr_scheduler_g"])
        if self.use_cosine_lr_d:
            self.lr_scheduler_d = CosineAnnealingLR(self.optim_d, T_max=schedule["discriminator"]["t_max"], eta_min=opt["discriminator"]["lr"] * schedule["discriminator"]["min_lr_ratio"])
            if state is not None: self.lr_scheduler_d.load_state_dict(state["lr_scheduler_d"])

        self.generator_id_encoder_provider = IDEncoderProvider[config["identity"]["provider"]]
        self.identity_loss_provider = IDEncoderProvider[loss_cfg["identity"]["provider"]]
        self.reuse_generator_identity_for_source = self.identity_loss_provider is self.generator_id_encoder_provider
        self.reconstruction_scope = loss_cfg["reconstruction"]["scope"]
        self.mask_enabled = generator_cfg["type"] == "stylegan2" and generator_cfg["mask_enabled"]
        self.mask_weight = loss_cfg.get("mask", {}).get("weight", 0.0)
        self.enable_mask_loss = self.mask_enabled and self.mask_weight > 0
        self.enable_l1_loss = loss_cfg["l1"]["enable"]
        self.enable_vgg_loss = loss_cfg["vgg"]["enable"]
        self.enable_wfm_loss = loss_cfg["wfm"]["enable"]
        self.enable_gaze_loss = loss_cfg["gaze"]["enable"]
        self.enable_smirk_loss = loss_cfg["smirk"]["enable"]
        self.enable_r1_loss = loss_cfg["r1"]["enable"]
        self.r1_reg_step = loss_cfg["r1"]["interval"]
        self.r1_gamma = loss_cfg["r1"]["gamma"]
        self.excluded_parsing_labels = tuple(loss_cfg["identity"]["exclude_parsing_labels"])
        self.d_loss = DiscriminatorAdversarialLoss(weight=1, reduction="mean").to(self.device)
        self.gan_loss = GeneratorAdversarialLoss(weight=loss_cfg["gan"]["weight"], reduction="mean").to(self.device)
        if self.enable_wfm_loss: self.wfm_loss = WeakFeatureMatchingLoss(loss_cfg["wfm"]["weights"]).to(self.device)
        self.generator_id_encoder = IDEncoder(self.generator_id_encoder_provider).to(self.device).eval().requires_grad_(False)
        self.id_loss = IdentityLoss(weight=loss_cfg["identity"]["weight"], provider=self.identity_loss_provider,
                                    min_similarity=loss_cfg["identity"].get("min_similarity"),
                                    transition_width=loss_cfg["identity"].get("transition_width", .03)).to(self.device)
        self.identity_encoder_grid = make_ffhq_to_arcface_112_grid(self.img_resolution, self.batch_size, self.device)
        if self.face_alignment == "arcface":
            self.arcface_identity_grid = make_arcface128_to_arcface112_grid(self.img_resolution, self.batch_size, self.device)
        if self.enable_l1_loss:
            l1 = loss_cfg["l1"]
            self.l1_loss = (make_blurred_l1_loss(weight=l1["weight"], resolution=self.img_resolution, reduction="none", blur_divisor=l1["gaussian_blur"])
                            if l1["gaussian_blur"] else make_l1_loss(weight=l1["weight"], reduction="none"))
        if self.enable_gaze_loss:
            gaze = loss_cfg["gaze"]
            self.gaze_loss = GazeLoss(weight=gaze["weight"], distribution_weight=gaze["distribution_weight"], confidence_weighted=gaze["confidence_weighted"]).to(self.device)
        if self.enable_smirk_loss:
            smirk = loss_cfg["smirk"]
            self.smirk_loss = SMIRKMotionLoss(pose_weight=smirk["pose_weight"], expression_weight=smirk["expression_weight"],
                                              jaw_weight=smirk["jaw_weight"], eyelid_weight=smirk["eyelid_weight"]).to(self.device)
        if self.enable_vgg_loss:
            self.vgg_loss = VGGPerceptualLoss(layer_weights=loss_cfg["vgg"]["weights"], reduction="none").to(self.device)
        self.face_parser: FaceMasker | None = None
        if self.excluded_parsing_labels or self.enable_mask_loss:
            self.face_parser = FaceMasker(mode="parsing").to(self.device).eval().requires_grad_(False)
        self.parsing_exclusion_indices = torch.tensor(self.excluded_parsing_labels, device=self.device, dtype=torch.long) if self.excluded_parsing_labels else None

        augmentation = data_cfg["augmentation"]
        self.geometry_augmentation_enabled = not is_identity_geometry(tuple(augmentation["rotation_range"]), tuple(augmentation["scale_factor_range"]), tuple(augmentation["tx_range"]), tuple(augmentation["ty_range"]))
        self.dataset = TrainingDataLoader(batch_size=self.batch_size, device=self.device, img_resolution=self.img_resolution,
                                          src=[(e["path"], e["adjustment"], e["alignment"]) for e in data_cfg["src"]],
                                          dst=[(e["path"], e["adjustment"]) for e in data_cfg["dst"]],
                                          **(data_cfg["loader"] | augmentation | data_cfg["sampling"] | {"alignment": self.face_alignment, "decoder_backend": ImageDecoderBackend(data_cfg["loader"]["decoder_backend"])}))
        self.occlusion = None
        occlusion = data_cfg.get("occlusion", {})
        if occlusion.get("prob", 0) > 0:
            self.occlusion = OcclusionAugmentor(probability=occlusion["prob"], assets_dir=occlusion["assets_dir"], device=self.device,
                                                size_range_pct=occlusion["size_range_pct"], center_x_range_pct=occlusion["center_x_range_pct"],
                                                center_y_range_pct=occlusion["center_y_range_pct"], rotation_range_deg=occlusion["rotation_range_deg"])
        if compile_module:
            initialize_upfirdn2d()
            self.train_g = _compile_training_callable(self.net_g.core, compile_mode)
            self.train_d = _compile_training_callable(self.net_d, compile_mode)
            self.generator_id_encoder_forward = _compile_training_callable(self.generator_id_encoder, compile_mode)
            self.identity_embeddings_forward = _compile_training_callable(self.id_loss.extract_identity_embeddings, compile_mode)
            if self.enable_gaze_loss: self.gaze_loss_forward = _compile_training_callable(self.gaze_loss, compile_mode)
            if self.enable_vgg_loss: self.vgg_loss_forward = _compile_training_callable(self.vgg_loss, compile_mode)
        else:
            self.train_g = self.net_g.core
            self.train_d = self.net_d
            self.generator_id_encoder_forward = self.generator_id_encoder
            self.identity_embeddings_forward = self.id_loss.extract_identity_embeddings
            if self.enable_gaze_loss: self.gaze_loss_forward = self.gaze_loss
            if self.enable_vgg_loss: self.vgg_loss_forward = self.vgg_loss

        self.run_paths = RunPaths.from_root(run_dir)
        self.run_id = run_id
        self.resolved_config_sha256 = resolved_config_sha256
        for path in (self.run_paths.checkpoints, self.run_paths.samples, self.run_paths.tensorboard):
            path.mkdir(parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(self.run_paths.tensorboard, purge_step=self.completed_step + 1 if state is not None else None)
        self._log_buffer = {}

    @property
    def completed_step(self) -> int:
        return self._completed_step

    @property
    def can_save_checkpoint(self) -> bool:
        return not self._step_in_progress and self.completed_step > 0

    def request_stop(self) -> None:
        self._stop_requested = True

    @torch.no_grad()
    def log(self, key: str, value: Tensor, *, force: bool = False) -> None:
        if force or (self.completed_step + 1) % self.log_interval == 0:
            self._log_buffer[key] = value.detach().mean()

    @torch.no_grad()
    def flush_logs(self) -> None:
        if not self._log_buffer: return
        keys = tuple(self._log_buffer)
        values = torch.stack(tuple(self._log_buffer[key] for key in keys)).float().cpu().tolist()
        self._log_buffer.clear()
        for key, val in zip(keys, values, strict=True):
            self.log_writer.add_scalar(f"Loss/{key}", val, self.completed_step)

    @torch.no_grad()
    def _log_identity_threshold_metrics(self, criterion: IdentityLoss, generated: Tensor, reference: Tensor) -> None:
        if criterion.min_similarity is None or (self.completed_step + 1) % self.log_interval: return
        similarity = NF.cosine_similarity(generated.float(), reference.float(), dim=1)
        excess = (criterion.min_similarity - similarity).clamp_min(0)
        self.log("id_similarity_mean", similarity.mean())
        self.log("id_satisfied_fraction", (excess == 0).float().mean())
        self.log("id_gradient_scale_mean", (excess / criterion.transition_width).clamp(max=1).mean())

    def prepare_identity_encoder_faces(self, faces: Tensor, theta_restore: Tensor | None = None) -> Tensor:
        if tuple(faces.shape[-2:]) != (self.img_resolution, self.img_resolution):
            raise ValueError("identity face resolution mismatch")
        if self.face_alignment == "arcface":
            return arcface_to_identity_112(faces, self.arcface_identity_grid, theta_restore)
        grid = self.identity_encoder_grid
        if theta_restore is not None:
            return ffhq_to_arcface_112(faces, transform_sampling_grid(grid, theta_restore), padding_mode="reflection")
        return ffhq_to_arcface_112(faces, grid)

    def _discriminator_loss(self, fake: Tensor, real: Tensor) -> Tensor:
        r1_step = self.enable_r1_loss and not isinstance(self.net_d, UNetDiscriminatorSN) and self.completed_step % self.r1_reg_step == 0
        if r1_step:
            with autocast(device_type="cuda", enabled=False):
                fake_score = self.net_d(fake.detach().float())
                real_img = real.detach().float().requires_grad_(True)
                real_score = self.net_d(real_img)
                adv = self.d_loss(fake_score, real_score)
                r1_raw = r1_reg_loss(real_score, real_img, gamma=self.r1_gamma)
                r1 = r1_raw * self.r1_reg_step
            self.log("r1_loss_raw", r1_raw, force=True)
            self.log("r1_loss", r1, force=True)
            total = adv + r1
        else:
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                scores = self.train_d(torch.cat((fake.detach(), real.detach()), dim=0), True)
                fake_score, real_score = scores.chunk(2, dim=0)
                adv = self.d_loss(fake_score, real_score)
            total = adv
        self.log("d_loss", adv)
        return total

    def _update_discriminator(self, fake: Tensor, real: Tensor) -> None:
        for retry in range(MAX_AMP_OVERFLOW_RETRIES):
            self.optim_d.zero_grad(set_to_none=True)
            loss = self._discriminator_loss(fake, real)
            _ensure_finite_loss("d_loss", loss)
            if _scaled_backward_step(loss, self.optim_d, self.scaler_d, name="Discriminator", named_parameters=self._d_named_parameters):
                return
        raise FloatingPointError("Discriminator FP16 overflow retries exhausted")

    def _restore_for_geometry_loss(self, fake: Tensor, canonical_size: tuple[int, int], restore_grid: Tensor | None) -> Tensor:
        with autocast(device_type="cuda", enabled=False):
            result = fake.float()
            if tuple(result.shape[-2:]) != canonical_size:
                result = NF.interpolate(result, size=canonical_size, mode="bilinear", align_corners=False)
            if self.geometry_augmentation_enabled:
                if restore_grid is None: raise RuntimeError("missing geometry restoration grid")
                result = NF.grid_sample(result, restore_grid, mode="bilinear", padding_mode="reflection", align_corners=False)
            return result

    def _generator_loss(self, fake: Tensor, target: Tensor, canonical: Tensor, theta_restore: Tensor,
                        same_mask: Tensor, source_identity: Tensor, restore_grid: Tensor | None,
                        id_gate: Tensor | None, smirk_reference: dict[str, Tensor] | None,
                        smirk_sample_weights: Tensor | None,
                        mask_logits: Tensor | None = None, mask_teacher: Tensor | None = None) -> Tensor:
        if self.enable_wfm_loss:
            score, fake_features = self.train_d(fake, False, True)
        else:
            score = self.train_d(fake)
        total = self.gan_loss(score)
        self.log("gan_loss", total)
        if self.enable_mask_loss:
            if mask_logits is None or mask_teacher is None: raise RuntimeError("missing StyleGAN2 mask supervision")
            teacher = NF.interpolate(mask_teacher, size=mask_logits.shape[-2:], mode="area")
            with autocast(device_type="cuda", enabled=False):
                mask_loss = NF.binary_cross_entropy_with_logits(mask_logits.float(), teacher.float()) * self.mask_weight
            self.log("mask_loss", mask_loss)
            total = total + mask_loss
        if self.enable_wfm_loss:
            with torch.no_grad(): _, real_features = self.train_d(target, False, True)
            wfm = self.wfm_loss(fake_features, real_features)
            self.log("wfm_loss", wfm)
            total = total + wfm
        identity_output = self.identity_embeddings_forward(self.prepare_identity_encoder_faces(gate_identity_gradient(fake, id_gate), theta_restore if self.geometry_augmentation_enabled else None))
        id_loss = self.id_loss(identity_output, source_identity)
        self.log("id_loss", id_loss)
        self._log_identity_threshold_metrics(self.id_loss, identity_output, source_identity)
        total = total + id_loss
        if self.enable_gaze_loss or self.enable_smirk_loss:
            restored = self._restore_for_geometry_loss(fake, tuple(canonical.shape[-2:]), restore_grid)
            if self.enable_gaze_loss:
                gaze = self.gaze_loss_forward(restored, canonical)
                self.log("gaze_loss", gaze)
                total = total + gaze
            if self.enable_smirk_loss:
                smirk = self.smirk_loss.forward_components(restored, canonical, reference_features=smirk_reference, sample_weights=smirk_sample_weights)
                for name, value in smirk.items(): self.log(f"smirk_{name}_loss", value)
                if smirk: total = total + torch.stack(tuple(smirk.values())).sum()
        if self.enable_vgg_loss:
            vgg_per_sample = self.vgg_loss_forward(fake, target)
            vgg = _reduce_reconstruction_loss(vgg_per_sample, same_mask, self.reconstruction_scope)
            self.log("vgg_loss", vgg)
            total = total + vgg
        if self.enable_l1_loss:
            l1_per_sample = self.l1_loss(fake, target).flatten(1).mean(dim=1)
            l1 = _reduce_reconstruction_loss(l1_per_sample, same_mask, self.reconstruction_scope)
            self.log("l1_loss", l1)
            total = total + l1
        return total
    @torch.no_grad()
    def update_ema(self, decay: float = .999) -> None:
        decay = min(decay, 1 - 1 / (self.completed_step + 1))
        torch._foreach_lerp_(self._ema_params, self._train_g_params, 1 - decay)

    @torch.no_grad()
    def save_ckpt(self) -> None:
        if not self.can_save_checkpoint:
            raise RuntimeError("cannot save partial training step")
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "step": self.completed_step,
            "face_alignment": self.face_alignment,
            "run": {"id": self.run_id, "config_sha256": self.resolved_config_sha256},
            "identity_encoders": {"generator": self.generator_id_encoder_provider.name, "identity_loss": self.identity_loss_provider.name},
            "training_config": self.training_config,
            "net_g": {"network_cfg": self.net_g_ema.network_cfg, "state_dict": self.net_g_ema.state_dict()},
            "net_d": {"network_cfg": self.net_d.network_cfg, "state_dict": self.net_d.state_dict()},
            "training_state": {
                "net_g": self.net_g.state_dict(), "optim_g": self.optim_g.state_dict(), "optim_d": self.optim_d.state_dict(),
                "scaler_g": self.scaler_g.state_dict() if self.precision == "fp16" else None,
                "scaler_d": self.scaler_d.state_dict() if self.precision == "fp16" else None,
                "lr_scheduler_g": self.lr_scheduler_g.state_dict() if self.use_cosine_lr_g else None,
                "lr_scheduler_d": self.lr_scheduler_d.state_dict() if self.use_cosine_lr_d else None,
            },
        }
        target = self.run_paths.checkpoints / f"step_{self.completed_step:09d}.pth"
        temporary = target.with_suffix(".pth.tmp")
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(target)
            write_latest(self.run_paths, target)
        except (OSError, RuntimeError):
            temporary.unlink(missing_ok=True)
            raise

    def loss_grad_map(self, loss: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        (grad,) = torch.autograd.grad(loss.sum(), x)
        grad = grad.detach().float()
        return grad.abs().mean(dim=1, keepdim=True), grad

    def _sample_grad_maps(self, generated: Tensor, target: Tensor, canonical: Tensor, theta_restore: Tensor | None,
                          same_mask: Tensor, source_identity: Tensor, restore_grid: Tensor | None,
                          gate: Tensor | None, smirk_reference: dict[str, Tensor] | None,
                          weights: Tensor | None) -> list[tuple[str, Tensor]]:
        rows: list[tuple[str, Tensor]] = []
        grads: list[Tensor] = []
        def add(label: str, loss: Tensor, x: Tensor) -> None:
            heat, grad = self.loss_grad_map(loss, x)
            rows.append((label, heat))
            grads.append(grad)
        was_training = self.net_d.training
        self.net_d.eval()
        try:
            with torch.enable_grad():
                x = generated.detach().requires_grad_(True)
                add("GAN GRAD", self.gan_loss(self.net_d(x)), x)
            if self.enable_wfm_loss:
                with torch.enable_grad():
                    x = generated.detach().requires_grad_(True)
                    _, features = self.net_d(x, False, True)
                    with torch.no_grad(): _, teacher_features = self.net_d(target, False, True)
                    add("WFM GRAD", self.wfm_loss(features, teacher_features), x)
        finally:
            self.net_d.train(was_training)
        with torch.enable_grad():
            x = generated.detach().requires_grad_(True)
            face = self.prepare_identity_encoder_faces(gate_identity_gradient(x, gate), theta_restore)
            embeddings = self.identity_embeddings_forward(face)
            add("ID GRAD", self.id_loss(embeddings, source_identity.detach()), x)
        def restored(x: Tensor) -> Tensor:
            return self._restore_for_geometry_loss(x, tuple(canonical.shape[-2:]), restore_grid)
        if self.enable_gaze_loss:
            with torch.enable_grad():
                x = generated.detach().requires_grad_(True)
                add("GAZE GRAD", self.gaze_loss_forward(restored(x), canonical), x)
        if self.enable_smirk_loss:
            with torch.enable_grad():
                x = generated.detach().requires_grad_(True)
                components = self.smirk_loss.forward_components(restored(x), canonical, reference_features=smirk_reference, sample_weights=weights)
                if components: add("SMIRK GRAD", torch.stack(tuple(components.values())).sum(), x)
        if self.enable_vgg_loss:
            with torch.enable_grad():
                x = generated.detach().requires_grad_(True)
                value = _reduce_reconstruction_loss(self.vgg_loss_forward(x, target), same_mask, self.reconstruction_scope)
                add("VGG GRAD", value, x)
        if self.enable_l1_loss:
            with torch.enable_grad():
                x = generated.detach().requires_grad_(True)
                value = _reduce_reconstruction_loss(self.l1_loss(x, target).flatten(1).mean(dim=1), same_mask, self.reconstruction_scope)
                add("L1 GRAD", value, x)
        return _format_sample_gradient_rows(rows, grads, heatmap_scale=SAMPLE_GRADIENT_HEATMAP_SCALE)

    def _next_training_batch(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        src, dst, canonical, theta_restore, same_mask, identity_faces = self.dataset.next()
        labels = None
        if self.face_parser is not None:
            with torch.no_grad(), autocast(device_type="cuda", enabled=False):
                labels = self.face_parser.predict_labels(dst.float())
        gate = None
        if self.occlusion is not None:
            dst, gate = self.occlusion.apply(dst)
        weights = gate.detach().float().flatten(1).mean(dim=1).clamp(0, 1) if gate is not None else None
        if self.parsing_exclusion_indices is not None:
            assert labels is not None
            keep = (~torch.isin(labels, self.parsing_exclusion_indices)).unsqueeze(1).to(dtype=dst.dtype)
            gate = keep if gate is None else gate * keep
        mask_teacher = None
        if self.enable_mask_loss:
            assert labels is not None and self.face_parser is not None
            indices = self.face_parser.get_buffer("face_part_indices")
            face_mask = torch.isin(labels, indices).unsqueeze(1).float()
            mask_teacher = face_mask if gate is None else face_mask * gate.float()
        return src, dst, canonical, theta_restore, same_mask, identity_faces, gate, weights, mask_teacher

    @torch.no_grad()
    def _save_sample(self, sample_reference: tuple[Tensor, ...], current_batch: tuple[Tensor, ...],
                     sample_gate: Tensor | None = None, current_gate: Tensor | None = None,
                     sample_weights: Tensor | None = None, current_weights: Tensor | None = None) -> None:
        previous = [t.to(self.device) for t in sample_reference]
        half = previous[0].shape[0]
        active = [torch.cat((saved, live[:self.batch_size-half]), dim=0) for saved, live in zip(previous, current_batch, strict=True)]
        src, dst, canonical, theta, same_mask, identity_faces = active
        gate = torch.cat((sample_gate.to(self.device), current_gate[:self.batch_size-half]), dim=0) if sample_gate is not None and current_gate is not None else None
        weights = torch.cat((sample_weights.to(self.device), current_weights[:self.batch_size-half]), dim=0) if sample_weights is not None and current_weights is not None else None
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
            id_vectors = self.generator_id_encoder_forward(identity_faces)
            source_embeddings = id_vectors if self.reuse_generator_identity_for_source else self.identity_embeddings_forward(identity_faces)
            if self.mask_enabled:
                generated, mask = self.net_g_ema(dst, id_vectors, return_mask=True)
            else:
                generated = self.net_g_ema(dst, id_vectors)
                mask = None
            id_input = self.prepare_identity_encoder_faces(generated, theta if self.geometry_augmentation_enabled else None)
            rows = [("SOURCE / SRC", src), ("TARGET / DST", dst), ("OUTPUT", generated), ("DST CANONICAL", canonical),
                    ("ID INPUT", NF.interpolate(id_input, size=generated.shape[-2:], mode="bilinear", align_corners=False))]
            if mask is not None:
                rows.append(("MASK", mask.mul(2).sub(1).expand(-1, 3, -1, -1)))
            grid = None
            if self.geometry_augmentation_enabled and (self.enable_gaze_loss or self.enable_smirk_loss):
                with autocast(device_type="cuda", enabled=False):
                    grid = NF.affine_grid(theta.float(), list(dst.shape), align_corners=False)
            reference = self.smirk_loss.encode_reference(canonical) if self.enable_smirk_loss else None
            rows.extend(self._sample_grad_maps(generated, dst, canonical,
                                               theta if self.geometry_augmentation_enabled else None,
                                               same_mask, source_embeddings, grid, gate, reference, weights))
            image = np.concatenate([_sample_row_bgr(value, label, self.batch_size) for label, value in rows], axis=0)
        sample_file = self.run_paths.samples / f"step_{self.completed_step:09d}.png"
        if not cv2.imwrite(str(sample_file), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"cannot save sample {sample_file}")

    def train(self) -> None:
        sample = self._next_training_batch()
        half = self.batch_size // 2
        sample_ref = tuple(x[:half].detach().cpu() for x in sample[:6])
        sample_gate = sample[6][:half].detach().cpu() if sample[6] is not None else None
        sample_weights = sample[7][:half].detach().cpu() if sample[7] is not None else None
        with tqdm(total=None, initial=self.completed_step, mininterval=1, bar_format="{n_fmt:7} | 速度 {rate_fmt:3} | 训练时间 {elapsed}") as progress:
            while True:
                if self._stop_requested: raise KeyboardInterrupt
                src, dst, canonical, theta, same_mask, id_faces, gate, weights, mask_teacher = self._next_training_batch()
                self._step_in_progress = True
                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                    with torch.no_grad(): id_vectors = self.generator_id_encoder_forward(id_faces)
                    if self.mask_enabled:
                        generated, target, mask_logits = self.train_g(dst, id_vectors, return_resize_in=True, return_mask_logits=True)
                    else:
                        generated, target = self.train_g(dst, id_vectors, return_resize_in=True)
                        mask_logits = None
                self.net_d.requires_grad_(True)
                self._update_discriminator(generated, target)
                if self.use_cosine_lr_d: self.lr_scheduler_d.step()
                with torch.no_grad(), autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                    source_identity = id_vectors if self.reuse_generator_identity_for_source else self.identity_embeddings_forward(id_faces)
                smirk_reference = self.smirk_loss.encode_reference(canonical) if self.enable_smirk_loss else None
                self.net_d.requires_grad_(False)
                was_training = self.net_d.training
                self.net_d.eval()
                for retry in range(MAX_AMP_OVERFLOW_RETRIES):
                    self.optim_g.zero_grad(set_to_none=True)
                    if retry:
                        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                            if self.mask_enabled:
                                generated, target, mask_logits = self.train_g(dst, id_vectors, return_resize_in=True, return_mask_logits=True)
                            else:
                                generated, target = self.train_g(dst, id_vectors, return_resize_in=True)
                    with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                        restore_grid = None
                        if self.geometry_augmentation_enabled and (self.enable_gaze_loss or self.enable_smirk_loss):
                            with autocast(device_type="cuda", enabled=False):
                                restore_grid = NF.affine_grid(theta.float(), list(dst.shape), align_corners=False)
                        g_loss = self._generator_loss(generated, target, canonical, theta, same_mask, source_identity,
                                                      restore_grid, gate, smirk_reference, weights, mask_logits, mask_teacher)
                    _ensure_finite_loss("g_loss", g_loss)
                    if _scaled_backward_step(g_loss, self.optim_g, self.scaler_g, name="Generator", named_parameters=self._g_named_parameters):
                        break
                else:
                    raise FloatingPointError("Generator FP16 overflow retries exhausted")
                self.net_d.train(was_training)
                if self.use_cosine_lr_g: self.lr_scheduler_g.step()
                self.update_ema()
                self._completed_step += 1
                self._step_in_progress = False
                self.flush_logs()
                progress.update(1)
                if self._stop_requested: raise KeyboardInterrupt
                if self.completed_step % self.checkpoint_save_every == 0: self.save_ckpt()
                if self.completed_step % self.sample_save_every == 0:
                    self._save_sample(sample_ref, (src, dst, canonical, theta, same_mask, id_faces), sample_gate, gate, sample_weights, weights)

def _load_run_config(paths: RunPaths) -> dict[str, Any]:
    resolved = load_resolved_config(paths)
    if resolve_train_config(resolved) != resolved:
        raise ValueError("run resolved config is not canonical")
    return resolved


def _canonical_generator_config(config: Mapping[str, Any]) -> dict[str, Any]:
    from .config import _normalize_generator
    if "type" not in config:
        raise ValueError("Generator checkpoint missing type")
    return _normalize_generator({"generator": dict(config)})


def _load_branch_checkpoint(checkpoint_path: Path, *, reset_discriminator: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Branch uses actual state-dict loading rather than checkpoint metadata equality."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Branch checkpoint must be a state dictionary")
    _branch_generator_state(checkpoint)
    run = checkpoint.get("run")
    run_info = run if isinstance(run, Mapping) else {}
    parent = {
        "run_id": run_info.get("id"),
        "checkpoint": checkpoint_path.name,
        "step": _branch_checkpoint_step(checkpoint),
        "config_sha256": run_info.get("config_sha256"),
        "branch_mode": "inherit", "generator": "inherit",
        "discriminator": "reset" if reset_discriminator else "auto",
    }
    return checkpoint, parent


def main() -> None:
    _configure_training_runtime()
    parser = argparse.ArgumentParser(description="SwapFace 单阶段生成器训练")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--resume", type=Path, default=None)
    sources.add_argument("--branch-from", type=Path, default=None)
    parser.add_argument("--reset-discriminator", action="store_true")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--strict-precision", action="store_true")
    args = parser.parse_args()
    is_resume, is_branch = args.resume is not None, args.branch_from is not None
    if is_resume and (args.config is not None or args.name is not None or args.runs_root is not None):
        parser.error("--resume uses the original frozen run configuration")
    if is_branch and args.config is None:
        parser.error("--branch-from requires --config")
    if args.reset_discriminator and not is_branch:
        parser.error("--reset-discriminator requires --branch-from")
    if args.step is not None and (not is_branch or args.step < 0):
        parser.error("--step requires --branch-from and nonnegative value")
    if args.strict_precision and not is_resume:
        parser.error("--strict-precision only applies to resume")
    checkpoint = None
    preloaded = None
    checkpoint_mode: Literal["resume", "branch"] = "resume"
    if is_resume:
        paths, checkpoint = resolve_resume_target(args.resume)
        resolved = _load_run_config(paths)
    elif is_branch:
        checkpoint = resolve_branch_target(args.branch_from)
        resolved = load_train_config(args.config)
        preloaded, parent = _load_branch_checkpoint(checkpoint, reset_discriminator=args.reset_discriminator)
        paths = create_run(args.runs_root or DEFAULT_RUNS_ROOT, args.config, resolved, name=args.name, parent=parent)
        checkpoint_mode = "branch"
    else:
        source_path = args.config or DEFAULT_TRAIN_CONFIG_PATH
        resolved = load_train_config(source_path)
        paths = create_run(args.runs_root or DEFAULT_RUNS_ROOT, source_path, resolved, name=args.name)
    metadata = load_metadata(paths)
    with RunLock(paths):
        if is_resume:
            _, checkpoint = resolve_resume_target(paths.root)
        update_metadata(paths, status="running", error=None)
        trainer = None
        try:
            trainer = Trainer(resolved, run_dir=paths.root, checkpoint_path=checkpoint,
                              run_id=metadata["run_id"], resolved_config_sha256=config_sha256(resolved),
                              strict_precision_resume=args.strict_precision, checkpoint_mode=checkpoint_mode,
                              preloaded_checkpoint=preloaded, reset_discriminator=args.reset_discriminator, start_step=args.step)
            if is_branch:
                parent = dict(metadata["parent"])
                parent["discriminator"] = trainer.branch_discriminator_state
                update_metadata(paths, parent=parent)
                print(f"Branch Discriminator: {trainer.branch_discriminator_state}")
            previous_handler = signal.getsignal(signal.SIGINT)
            already_interrupted = False
            def on_signal(_signum: int, _frame: Any) -> None:
                nonlocal already_interrupted
                if already_interrupted:
                    signal.signal(signal.SIGINT, previous_handler)
                    raise KeyboardInterrupt
                already_interrupted = True
                trainer.request_stop()
                print("收到 Ctrl+C；将在完整 step 结束后保存 checkpoint")
            signal.signal(signal.SIGINT, on_signal)
            try:
                trainer.train()
            finally:
                signal.signal(signal.SIGINT, previous_handler)
        except KeyboardInterrupt:
            if trainer is not None and trainer.can_save_checkpoint:
                trainer.save_ckpt()
                print(f"Saved interrupted checkpoint: {trainer.completed_step}")
            update_metadata(paths, status="interrupted")
            raise SystemExit(130) from None
        except BaseException as exc:
            update_metadata(paths, status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        else:
            update_metadata(paths, status="completed")
        finally:
            if trainer is not None:
                trainer.log_writer.close()


if __name__ == "__main__":
    main()
