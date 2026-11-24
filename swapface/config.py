"""训练实验配置协议：默认值、严格校验与 canonical config 生成。"""

import math
import tomllib
from pathlib import Path
from typing import Any

from losses.functional import _gaussian_blur_parameters
from losses.identity import MIN_IDENTITY_TRANSITION_WIDTH
from misc.models.face_mask import FACE_PARSING_NUM_CLASSES
from misc.models.id_encoder import IDEncoderProvider
from models.discriminator import INSTANCE_NORM_RESIDUAL_MIN_RESOLUTION, DiscriminatorType

from .dataloader_common import ImageDecoderBackend

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "batch_size": 16,
    "precision": "bf16",
    "device": "cuda",
    "compile_module": True,
    "compile_mode": "max-autotune-no-cudagraphs",
    "log_interval": 10,
    "sample_save_every": 1000,
    "checkpoint_save_every": 10000,
}
DEFAULT_OPTIMIZER_STAGE_CONFIG: dict[str, Any] = {
    "lr": 1e-4,
}
DEFAULT_OPTIMIZER_CONFIG: dict[str, Any] = {
    "generator": dict(DEFAULT_OPTIMIZER_STAGE_CONFIG),
    "discriminator": dict(DEFAULT_OPTIMIZER_STAGE_CONFIG),
}
DEFAULT_SCHEDULER_STAGE_CONFIG: dict[str, Any] = {
    "type": "none",
    "t_max": 20000,
    "min_lr_ratio": 0.1,
}
DEFAULT_SCHEDULER_CONFIG: dict[str, Any] = {
    "generator": dict(DEFAULT_SCHEDULER_STAGE_CONFIG),
    "discriminator": dict(DEFAULT_SCHEDULER_STAGE_CONFIG),
}
DEFAULT_GENERATOR_CONFIG: dict[str, Any] = {
    "type": "inswapper", "img_resolution": 512, "id_dim": 512,
}
IN_SWAPPER_CONFIG_DEFAULTS = {
    "latent_resolution": 32, "num_latent": 6, "base_ch": 64, "max_ch": 256,
    "skip_mode": "cat", "skip_layers": [],
}
STYLEGAN2_CONFIG_DEFAULTS = {
    "w_dim": 512, "mapping_layers": 8, "start_resolution": 4, "channels": [128, 512],
    "target_inject_mode": "concat", "mask_enabled": False, "mask_noise_blank_level": 9,
    "mask_guidance_epsilon": 1e-4, "mask_guidance_interpolation": "bilinear",
}
DEFAULT_DISCRIMINATOR_CONFIG: dict[str, Any] = {
    "type": DiscriminatorType.FIR_MINIBATCH_STD.name,
    "img_resolution": 512, "base_ch": 64,
    "max_ch": 512, "minibatch_std_group_size": 4,
}
DEFAULT_GENERATOR_ID_ENCODER_PROVIDER = IDEncoderProvider.BLENDFACE
DEFAULT_IDENTITY_LOSS_PROVIDER = IDEncoderProvider.MS1MV3_ARCFACE_R50_FP16
DEFAULT_IDENTITY_CONFIG: dict[str, Any] = {
    "provider": DEFAULT_GENERATOR_ID_ENCODER_PROVIDER.name,
}
DEFAULT_VGG_PERCEPTUAL_LOSS_WEIGHT: dict[str, float] = {
    "conv1_2": 2.5,
    "conv2_2": 2.5,
    "conv3_3": 2.5,
    "conv4_3": 2.5,
}
DEFAULT_GAN_LOSS_CONFIG = {"weight": 1.0}
DEFAULT_IDENTITY_LOSS_CONFIG = {"provider": DEFAULT_IDENTITY_LOSS_PROVIDER.name, "weight": 10.0, "exclude_parsing_labels": []}
DEFAULT_R1_LOSS_CONFIG = {"enable": True, "interval": 16, "gamma": 10.0}
DEFAULT_WFM_LOSS_CONFIG = {"enable": False, "weights": {"0": 0.5, "1": 1.0, "2": 1.0, "3": 1.0}}
DEFAULT_GAZE_LOSS_CONFIG = {"enable": False, "weight": 1.0, "distribution_weight": 0.1, "confidence_weighted": True}
DEFAULT_SMIRK_LOSS_CONFIG = {
    "enable": False,
    "pose_weight": 1.0,
    "expression_weight": 1.0,
    "jaw_weight": 1.0,
    "eyelid_weight": 1.0,
}
DEFAULT_LOSS_CONFIG: dict[str, Any] = {
    "gan": dict(DEFAULT_GAN_LOSS_CONFIG),
    "identity": dict(DEFAULT_IDENTITY_LOSS_CONFIG),
    "r1": dict(DEFAULT_R1_LOSS_CONFIG),
    "wfm": dict(DEFAULT_WFM_LOSS_CONFIG),
    "mask": {"weight": 1.0},
    "gaze": dict(DEFAULT_GAZE_LOSS_CONFIG),
    "smirk": dict(DEFAULT_SMIRK_LOSS_CONFIG),
    "reconstruction": {"scope": "same"},
    "l1": {"enable": True, "weight": 10.0, "gaussian_blur": 0},
    "vgg": {"enable": True, "weights": dict(DEFAULT_VGG_PERCEPTUAL_LOSS_WEIGHT)},
}

DEFAULT_DATA_CONFIG: dict[str, Any] = {
    "loader": {
        "num_threads": 16,
        "prefetch_queue_depth": 4,
        "py_num_workers": 8,
        "py_start_method": "spawn",
        "reader_prefetch_queue_depth": 2,
        "decoder_backend": ImageDecoderBackend.MIXED.value,
        "decoder_hw_load": 0.75,
    },
    "augmentation": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "flip_prob": 0.5,
        "rotation_range": (-10.0, 10.0),
        "scale_factor_range": (1.0 / 1.3, 1.25),
        "tx_range": (-0.15, 0.15),
        "ty_range": (-0.15, 0.15),
    },
    "sampling": {
        "same_prob": 0.2,
    },
}
# 遮挡尺寸和位置均按训练图像边长的百分比配置，角度独立使用度数。
DEFAULT_OCCLUSION_CONFIG: dict[str, Any] = {
    "prob": 0.0,
    "assets_dir": "",
    "size_range_pct": [40.0, 90.0],
    "center_x_range_pct": [-15.0, 15.0],
    "center_y_range_pct": [-15.0, 15.0],
    "rotation_range_deg": [-40.0, 40.0],
}
DATALOADER_RANGE_KEYS = ("rotation_range", "scale_factor_range", "tx_range", "ty_range")


def _with_defaults(values: dict[str, Any], defaults: dict[str, Any], section: str) -> dict[str, Any]:
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"{section} 包含未知字段：{sorted(unknown)}")
    return defaults | values


def _table(parent: dict[str, Any], key: str, section: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{section} 必须为表/对象")
    return dict(value)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} 必须为 bool，实际为 {type(value).__name__}")
    return value


def _int(value: object, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} 必须为 int，实际为 {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，实际为 {value}")
    return value


def _float(value: object, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须为数值，实际为 {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须为有限值，实际为 {result}")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，实际为 {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} 必须 <= {maximum}，实际为 {result}")
    return result


def _range(value: object, name: str, *, positive: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} 必须为包含两个数值的数组")
    low = _float(value[0], f"{name}[0]")
    high = _float(value[1], f"{name}[1]")
    if low > high:
        raise ValueError(f"{name} 必须按从小到大排列，实际为 [{low}, {high}]")
    if positive and low <= 0.0:
        raise ValueError(f"{name} 必须为正数范围，实际为 [{low}, {high}]")
    return [low, high]


def _provider(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} 必须为字符串，实际为 {type(value).__name__}")
    try:
        return IDEncoderProvider[value].name
    except KeyError as exc:
        supported = ", ".join(provider.name for provider in IDEncoderProvider)
        raise ValueError(f"{name}={value!r} 无效，可选：{supported}") from exc


def _normalize_image_sources(entries: object, section: str) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"[[{section}]] 数据源不能为空")

    normalized: list[dict[str, Any]] = []
    allowed_fields = {"path", "adjustment", "alignment", "layout"}
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise TypeError(f"[[{section}]] 第 {index} 项必须是表/对象")
        entry = dict(raw_entry)
        unknown = set(entry) - allowed_fields
        if unknown:
            raise ValueError(f"[[{section}]] 第 {index} 项包含未知字段：{sorted(unknown)}；训练数据源仅支持本地 path/adjustment/alignment/layout")
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"[[{section}]] 第 {index} 项 path 必须为非空字符串")
        alignment = entry.get("alignment", "ffhq")
        if alignment not in ("ffhq", "arcface"):
            raise ValueError(f"{section}[{index}].alignment 必须为 ffhq 或 arcface，实际为 {alignment!r}")
        if section == "data.dst" and alignment != "ffhq":
            raise ValueError("data.dst 仅支持 ffhq 对齐；arcface 仅用于 source identity")
        item = {"path": path, "adjustment": _float(entry.get("adjustment", 0.0), f"{section}[{index}].adjustment")}
        item["alignment"] = alignment
        layout = entry.get("layout", "flat")
        if layout not in ("flat", "identity"):
            raise ValueError(f"{section}[{index}].layout 必须为 flat 或 identity，实际为 {layout!r}")
        # flat 为历史默认值，不写入 canonical config，保持旧实验哈希。
        if layout != "flat":
            item["layout"] = layout
        normalized.append(item)
    return normalized


def _normalize_train(config: dict[str, Any]) -> dict[str, Any]:
    raw = _table(config, "train", "[train]")
    if "precision" not in raw:
        raise ValueError("[train].precision 必须显式配置为 fp32、fp16 或 bf16")
    result = _with_defaults(raw, DEFAULT_TRAIN_CONFIG, "[train]")
    result["batch_size"] = _int(result["batch_size"], "train.batch_size", minimum=1)
    if not isinstance(result["precision"], str):
        raise TypeError(f"train.precision 必须为字符串，实际为 {type(result['precision']).__name__}")
    result["precision"] = result["precision"].lower()
    if result["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"train.precision={result['precision']!r} 无效，可选：fp32、fp16、bf16")
    if not isinstance(result["device"], str) or not result["device"]:
        raise TypeError("train.device 必须为非空字符串")
    result["compile_module"] = _bool(result["compile_module"], "train.compile_module")
    for key in ("log_interval", "sample_save_every", "checkpoint_save_every"):
        result[key] = _int(result[key], f"train.{key}", minimum=1)
    return result


def _normalize_optimizer(config: dict[str, Any]) -> dict[str, Any]:
    optimizer = _table(config, "optimizer", "[optimizer]")
    unknown = set(optimizer) - set(DEFAULT_OPTIMIZER_CONFIG)
    if unknown:
        raise ValueError(f"[optimizer] 包含未知字段：{sorted(unknown)}")

    result: dict[str, Any] = {}
    for stage, defaults in DEFAULT_OPTIMIZER_CONFIG.items():
        stage_config = _with_defaults(
            _table(optimizer, stage, f"[optimizer.{stage}]"),
            defaults,
            f"[optimizer.{stage}]",
        )
        stage_config["lr"] = _float(stage_config["lr"], f"optimizer.{stage}.lr", minimum=1e-30)
        result[stage] = stage_config
    return result


def _normalize_scheduler(config: dict[str, Any]) -> dict[str, Any]:
    scheduler = _table(config, "scheduler", "[scheduler]")
    unknown = set(scheduler) - set(DEFAULT_SCHEDULER_CONFIG)
    if unknown:
        raise ValueError(f"[scheduler] 包含未知字段：{sorted(unknown)}")

    result: dict[str, Any] = {}
    for stage, defaults in DEFAULT_SCHEDULER_CONFIG.items():
        stage_config = _with_defaults(
            _table(scheduler, stage, f"[scheduler.{stage}]"),
            defaults,
            f"[scheduler.{stage}]",
        )
        scheduler_type = stage_config["type"]
        if not isinstance(scheduler_type, str):
            raise TypeError(f"scheduler.{stage}.type 必须为字符串，实际为 {type(scheduler_type).__name__}")
        scheduler_type = scheduler_type.lower()
        if scheduler_type not in {"none", "cosine"}:
            raise ValueError(f"scheduler.{stage}.type={scheduler_type!r} 无效，可选：none、cosine")
        stage_config["type"] = scheduler_type
        stage_config["t_max"] = _int(stage_config["t_max"], f"scheduler.{stage}.t_max", minimum=1)
        stage_config["min_lr_ratio"] = _float(stage_config["min_lr_ratio"], f"scheduler.{stage}.min_lr_ratio", minimum=0.0, maximum=1.0)
        result[stage] = stage_config
    return result


def _normalize_identity(config: dict[str, Any]) -> dict[str, Any]:
    result = _with_defaults(_table(config, "identity", "[identity]"), DEFAULT_IDENTITY_CONFIG, "[identity]")
    result["provider"] = _provider(result["provider"], "identity.provider")
    return result


def _normalize_loss_config(loss: dict[str, Any], *, mask_enabled: bool = False) -> dict[str, Any]:
    defaults = dict(DEFAULT_LOSS_CONFIG)
    if not mask_enabled:
        defaults.pop("mask")
    unknown = set(loss) - set(defaults)
    if unknown:
        raise ValueError(f"[loss] 包含未知字段：{sorted(unknown)}")
    prefix = "loss"

    raw_gan = _table(loss, "gan", f"[{prefix}.gan]")
    gan = _with_defaults({key: value for key, value in raw_gan.items() if key != "spatial_weight"}, defaults["gan"], f"[{prefix}.gan]")
    gan["weight"] = _float(gan["weight"], f"{prefix}.gan.weight", minimum=0.0)
    if "spatial_weight" in raw_gan:
        gan["spatial_weight"] = _float(raw_gan["spatial_weight"], f"{prefix}.gan.spatial_weight", minimum=0.0)

    raw_identity = _table(loss, "identity", f"[{prefix}.identity]")
    identity = _with_defaults(
        {key: value for key, value in raw_identity.items() if key not in {"min_similarity", "transition_width"}},
        defaults["identity"],
        f"[{prefix}.identity]",
    )
    identity["provider"] = _provider(identity["provider"], f"{prefix}.identity.provider")
    identity["weight"] = _float(identity["weight"], f"{prefix}.identity.weight", minimum=0.0)
    if "min_similarity" in raw_identity:
        identity["min_similarity"] = _float(
            raw_identity["min_similarity"], f"{prefix}.identity.min_similarity", minimum=-1.0, maximum=1.0
        )
        identity["transition_width"] = _float(
            raw_identity.get("transition_width", 0.03),
            f"{prefix}.identity.transition_width",
            minimum=MIN_IDENTITY_TRANSITION_WIDTH,
        )
    elif "transition_width" in raw_identity:
        raise ValueError(f"{prefix}.identity.transition_width 要求同时提供 min_similarity")
    labels = identity["exclude_parsing_labels"]
    if not isinstance(labels, list):
        raise TypeError(f"{prefix}.identity.exclude_parsing_labels 必须为整数数组")
    validated: list[int] = []
    for index, label in enumerate(labels):
        label = _int(label, f"{prefix}.identity.exclude_parsing_labels[{index}]", minimum=0)
        if label >= FACE_PARSING_NUM_CLASSES:
            raise ValueError(f"{prefix}.identity.exclude_parsing_labels[{index}] 超出 [0, {FACE_PARSING_NUM_CLASSES - 1}]")
        if label in validated:
            raise ValueError(f"{prefix}.identity.exclude_parsing_labels 不允许重复标签：{label}")
        validated.append(label)
    identity["exclude_parsing_labels"] = sorted(validated)

    r1 = _with_defaults(_table(loss, "r1", f"[{prefix}.r1]"), defaults["r1"], f"[{prefix}.r1]")
    r1["enable"] = _bool(r1["enable"], f"{prefix}.r1.enable")
    r1["interval"] = _int(r1["interval"], f"{prefix}.r1.interval", minimum=1)
    r1["gamma"] = _float(r1["gamma"], f"{prefix}.r1.gamma", minimum=0.0)

    wfm = _with_defaults(_table(loss, "wfm", f"[{prefix}.wfm]"), defaults["wfm"], f"[{prefix}.wfm]")
    wfm["enable"] = _bool(wfm["enable"], f"{prefix}.wfm.enable")
    raw_wfm_weights = wfm["weights"]
    if not isinstance(raw_wfm_weights, dict) or (wfm["enable"] and not raw_wfm_weights):
        raise ValueError(f"[{prefix}.wfm.weights] 必须为非空表/对象")
    normalized_wfm_weights: dict[str, float] = {}
    for raw_index, raw_weight in raw_wfm_weights.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix}.wfm.weights 的层索引必须为非负整数，实际为 {raw_index!r}") from exc
        if index < 0 or str(index) != str(raw_index):
            raise ValueError(f"{prefix}.wfm.weights 的层索引必须为规范非负整数，实际为 {raw_index!r}")
        normalized_wfm_weights[str(index)] = _float(raw_weight, f"{prefix}.wfm.weights.{index}", minimum=0.0)
    wfm["weights"] = normalized_wfm_weights

    gaze = _with_defaults(_table(loss, "gaze", f"[{prefix}.gaze]"), defaults["gaze"], f"[{prefix}.gaze]")
    gaze["enable"] = _bool(gaze["enable"], f"{prefix}.gaze.enable")
    gaze["weight"] = _float(gaze["weight"], f"{prefix}.gaze.weight", minimum=0.0)
    gaze["distribution_weight"] = _float(gaze["distribution_weight"], f"{prefix}.gaze.distribution_weight", minimum=0.0)
    gaze["confidence_weighted"] = _bool(gaze["confidence_weighted"], f"{prefix}.gaze.confidence_weighted")

    smirk = _with_defaults(_table(loss, "smirk", f"[{prefix}.smirk]"), defaults["smirk"], f"[{prefix}.smirk]")
    smirk["enable"] = _bool(smirk["enable"], f"{prefix}.smirk.enable")
    for key in ("pose_weight", "expression_weight", "jaw_weight", "eyelid_weight"):
        smirk[key] = _float(smirk[key], f"{prefix}.smirk.{key}", minimum=0.0)

    result = {"gan": gan, "identity": identity, "r1": r1, "wfm": wfm, "gaze": gaze, "smirk": smirk}
    if mask_enabled:
        mask = _with_defaults(_table(loss, "mask", "[loss.mask]"), defaults["mask"], "[loss.mask]")
        mask["weight"] = _float(mask["weight"], "loss.mask.weight", minimum=0.0)
        result["mask"] = mask

    reconstruction = _with_defaults(_table(loss, "reconstruction", f"[{prefix}.reconstruction]"), defaults["reconstruction"], f"[{prefix}.reconstruction]")
    scope = reconstruction["scope"]
    if not isinstance(scope, str):
        raise TypeError(f"{prefix}.reconstruction.scope 必须为字符串，实际为 {type(scope).__name__}")
    if scope not in {"same", "all"}:
        raise ValueError(f"{prefix}.reconstruction.scope={scope!r} 无效，可选：same、all")

    l1 = _with_defaults(_table(loss, "l1", f"[{prefix}.l1]"), defaults["l1"], f"[{prefix}.l1]")
    l1["enable"] = _bool(l1["enable"], f"{prefix}.l1.enable")
    l1["weight"] = _float(l1["weight"], f"{prefix}.l1.weight", minimum=0.0)
    l1["gaussian_blur"] = _int(l1["gaussian_blur"], f"{prefix}.l1.gaussian_blur", minimum=0)

    vgg = _with_defaults(_table(loss, "vgg", f"[{prefix}.vgg]"), defaults["vgg"], f"[{prefix}.vgg]")
    vgg["enable"] = _bool(vgg["enable"], f"{prefix}.vgg.enable")
    raw_vgg_weights = vgg["weights"]
    if not isinstance(raw_vgg_weights, dict) or (vgg["enable"] and not raw_vgg_weights):
        raise ValueError(f"[{prefix}.vgg.weights] 必须为非空表/对象")
    vgg["weights"] = {str(layer): _float(weight, f"{prefix}.vgg.weights.{layer}", minimum=0.0) for layer, weight in raw_vgg_weights.items()}

    result.update({"reconstruction": reconstruction, "l1": l1, "vgg": vgg})
    return result


def _normalize_loss(config: dict[str, Any], *, mask_enabled: bool) -> dict[str, Any]:
    return _normalize_loss_config(_table(config, "loss", "[loss]"), mask_enabled=mask_enabled)


def _normalize_generator(config: dict[str, Any]) -> dict[str, Any]:
    raw = _table(config, "generator", "[generator]")
    kind = raw.get("type", "inswapper")
    if kind not in ("inswapper", "stylegan2"):
        raise ValueError("generator.type only supports inswapper/stylegan2")
    defaults = DEFAULT_GENERATOR_CONFIG | (IN_SWAPPER_CONFIG_DEFAULTS if kind == "inswapper" else STYLEGAN2_CONFIG_DEFAULTS)
    if kind == "stylegan2":
        defaults = defaults | {"target_inject_layers": []}
    result = _with_defaults(raw, defaults, "[generator]")
    for key in ("img_resolution", "id_dim"):
        result[key] = _int(result[key], f"generator.{key}", minimum=1)
    resolution = result["img_resolution"]
    if resolution < 4 or resolution > 1024 or resolution & (resolution - 1):
        raise ValueError("generator.img_resolution must be a power of 2 in [4,1024]")
    if kind == "inswapper":
        for key in ("latent_resolution", "num_latent", "base_ch", "max_ch"):
            result[key] = _int(result[key], f"generator.{key}", minimum=1)
        if result["base_ch"] < 2 or result["max_ch"] < result["base_ch"]:
            raise ValueError("generator.base_ch/max_ch invalid")
        latent = result["latent_resolution"]
        if latent < 2 or latent >= resolution or resolution % latent or (resolution // latent) & (resolution // latent - 1):
            raise ValueError("generator.latent_resolution must be >= 2 and img_resolution / latent_resolution must be a power of 2 >= 2")
        if result["skip_mode"] not in ("cat", "add"):
            raise ValueError("generator.skip_mode must be cat/add")
        max_layer = (resolution // latent).bit_length() - 2
        layers = result["skip_layers"]
        if not isinstance(layers, (list, tuple)) or any(type(i) is not int or i < 0 or i > max_layer for i in layers) or len(set(layers)) != len(layers):
            raise ValueError(f"generator.skip_layers must be unique integer indices in [0,{max_layer}]")
        result["skip_layers"] = sorted(layers)
    else:
        for key in ("w_dim", "mapping_layers", "start_resolution"):
            result[key] = _int(result[key], f"generator.{key}", minimum=1)
        start = result["start_resolution"]
        if start < 4 or start > resolution or start & (start - 1):
            raise ValueError("generator.start_resolution must be power of 2 <= img_resolution")
        channels = result["channels"]
        if not isinstance(channels, (list, tuple)) or len(channels) != 2 or any(type(ch) is not int or ch <= 0 for ch in channels) or channels[0] > channels[1]:
            raise ValueError("generator.channels must contain [base_ch,max_ch]")
        result["channels"] = list(channels)
        max_layer = resolution.bit_length() - start.bit_length()
        if "target_inject_layers" not in raw:
            result["target_inject_layers"] = list(range(max_layer + 1))
        layers = result["target_inject_layers"]
        if not isinstance(layers, (list, tuple)) or any(type(i) is not int or i < 0 or i > max_layer for i in layers) or len(set(layers)) != len(layers):
            raise ValueError(f"generator.target_inject_layers must contain unique indices in [0,{max_layer}]")
        result["target_inject_layers"] = sorted(layers)
        if result["target_inject_mode"] not in ("concat", "add"):
            raise ValueError("generator.target_inject_mode must be concat/add")
        result["mask_enabled"] = _bool(result["mask_enabled"], "generator.mask_enabled")
        result["mask_noise_blank_level"] = _int(result["mask_noise_blank_level"], "generator.mask_noise_blank_level", minimum=-1)
        result["mask_guidance_epsilon"] = _float(result["mask_guidance_epsilon"], "generator.mask_guidance_epsilon", minimum=1e-8)
        if result["mask_guidance_interpolation"] not in ("bilinear", "nearest"):
            raise ValueError("generator.mask_guidance_interpolation must be bilinear/nearest")
    return result


def _normalize_discriminator_type(value: object, name: str) -> DiscriminatorType:
    if not isinstance(value, str):
        raise TypeError(f"{name} 必须为字符串，实际为 {type(value).__name__}")
    try:
        return DiscriminatorType[value]
    except KeyError:
        try:
            return DiscriminatorType(value.lower())
        except ValueError as exc:
            supported = ", ".join(item.name for item in DiscriminatorType)
            raise ValueError(f"{name}={value!r} 无效，可选：{supported}") from exc


def _normalize_discriminator(config: dict[str, Any]) -> dict[str, Any]:
    raw = _table(config, "discriminator", "[discriminator]")
    discriminator_type = _normalize_discriminator_type(
        raw.get("type", DEFAULT_DISCRIMINATOR_CONFIG["type"]), "discriminator.type",
    )
    common_keys = {"type", "img_resolution", "base_ch"}
    if discriminator_type is DiscriminatorType.UNET:
        allowed = common_keys | {"skip_connection"}
        defaults = {
            "type": discriminator_type.name,
            "img_resolution": DEFAULT_DISCRIMINATOR_CONFIG["img_resolution"],
            "base_ch": 64,
            "skip_connection": True,
        }
    elif discriminator_type is DiscriminatorType.FIR_MINIBATCH_STD:
        allowed = common_keys | {"max_ch", "minibatch_std_group_size"}
        defaults = dict(DEFAULT_DISCRIMINATOR_CONFIG)
    else:
        allowed = common_keys | {"max_ch"}
        defaults = {key: value for key, value in DEFAULT_DISCRIMINATOR_CONFIG.items() if key in allowed}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"[discriminator] {discriminator_type.name} 不支持字段：{sorted(unknown)}")

    result = defaults | raw
    result["type"] = discriminator_type.name
    for key in ("img_resolution", "base_ch"):
        result[key] = _int(result[key], f"discriminator.{key}", minimum=1)
    if discriminator_type is DiscriminatorType.UNET:
        result["skip_connection"] = _bool(result["skip_connection"], "discriminator.skip_connection")
    else:
        result["max_ch"] = _int(result["max_ch"], "discriminator.max_ch", minimum=1)
        if result["base_ch"] > result["max_ch"]:
            raise ValueError("discriminator.base_ch > max_ch")
    resolution = result["img_resolution"]
    if resolution < 16 or resolution & (resolution - 1):
        raise ValueError("discriminator.img_resolution must be a power of 2 >= 16")
    if discriminator_type is DiscriminatorType.FIR_MINIBATCH_STD:
        result["minibatch_std_group_size"] = _int(
            result["minibatch_std_group_size"], "discriminator.minibatch_std_group_size", minimum=1,
        )
    return result


def _normalize_data(config: dict[str, Any]) -> dict[str, Any]:
    data = _table(config, "data", "[data]")
    allowed = {"alignment", "loader", "augmentation", "sampling", "occlusion", "src", "dst"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"[data] 包含未知字段：{sorted(unknown)}")

    loader = _with_defaults(_table(data, "loader", "[data.loader]"), DEFAULT_DATA_CONFIG["loader"], "[data.loader]")
    for key in ("num_threads", "prefetch_queue_depth", "reader_prefetch_queue_depth"):
        loader[key] = _int(loader[key], f"data.loader.{key}", minimum=1)
    loader["py_num_workers"] = _int(loader["py_num_workers"], "data.loader.py_num_workers", minimum=0)
    if not isinstance(loader["py_start_method"], str) or loader["py_start_method"] not in {"spawn", "fork", "forkserver"}:
        raise ValueError(f"data.loader.py_start_method={loader['py_start_method']!r} 无效，可选：spawn、fork、forkserver")
    try:
        loader["decoder_backend"] = ImageDecoderBackend(loader["decoder_backend"]).value
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in ImageDecoderBackend)
        raise ValueError(f"data.loader.decoder_backend={loader['decoder_backend']!r} 无效，可选：{supported}") from exc
    loader["decoder_hw_load"] = _float(loader["decoder_hw_load"], "data.loader.decoder_hw_load", minimum=0.0, maximum=1.0)

    augmentation = _with_defaults(_table(data, "augmentation", "[data.augmentation]"), DEFAULT_DATA_CONFIG["augmentation"], "[data.augmentation]")
    for key in ("brightness", "contrast", "saturation"):
        augmentation[key] = _float(augmentation[key], f"data.augmentation.{key}", minimum=0.0)
    augmentation["flip_prob"] = _float(augmentation["flip_prob"], "data.augmentation.flip_prob", minimum=0.0, maximum=1.0)
    for key in DATALOADER_RANGE_KEYS:
        augmentation[key] = _range(augmentation[key], f"data.augmentation.{key}", positive=key == "scale_factor_range")

    src = _normalize_image_sources(data.get("src"), "data.src")
    dst = _normalize_image_sources(data.get("dst"), "data.dst")
    has_identity_dst = any(entry.get("layout") == "identity" for entry in dst)
    sampling_defaults = DEFAULT_DATA_CONFIG["sampling"]
    if has_identity_dst:
        sampling_defaults = sampling_defaults | {"cross_view_prob": 0.75}
    sampling = _with_defaults(_table(data, "sampling", "[data.sampling]"), sampling_defaults, "[data.sampling]")
    if has_identity_dst:
        sampling["cross_view_prob"] = _float(sampling["cross_view_prob"], "data.sampling.cross_view_prob", minimum=0.0, maximum=1.0)
    sampling["same_prob"] = _float(sampling["same_prob"], "data.sampling.same_prob", minimum=0.0, maximum=1.0)

    occlusion = None
    if "occlusion" in data:
        occlusion = _with_defaults(_table(data, "occlusion", "[data.occlusion]"), DEFAULT_OCCLUSION_CONFIG, "[data.occlusion]")
        occlusion["prob"] = _float(occlusion["prob"], "data.occlusion.prob", minimum=0.0, maximum=1.0)
        if not isinstance(occlusion["assets_dir"], str):
            raise TypeError("data.occlusion.assets_dir 必须是字符串")
        for field in ("size_range_pct", "center_x_range_pct", "center_y_range_pct", "rotation_range_deg"):
            occlusion[field] = _range(occlusion[field], f"data.occlusion.{field}", positive=field == "size_range_pct")

    alignment = data.get("alignment", "ffhq")
    if alignment not in ("ffhq", "arcface"):
        raise ValueError(f"data.alignment 必须为 ffhq 或 arcface，实际为 {alignment!r}")
    result = {
        "loader": loader,
        "augmentation": augmentation,
        "sampling": sampling,
        "src": src,
        "dst": dst,
    }
    result["alignment"] = alignment
    if occlusion is not None:
        result["occlusion"] = occlusion
    return result


def resolve_train_config(config: dict[str, Any]) -> dict[str, Any]:
    """严格校验配置并展开默认值，得到唯一 canonical config。"""
    allowed_sections = {"train", "optimizer", "scheduler", "identity", "loss", "data", "generator", "discriminator"}
    unknown_sections = set(config) - allowed_sections
    if unknown_sections:
        raise ValueError(f"配置包含未知顶层字段：{sorted(unknown_sections)}")

    train = _normalize_train(config)
    discriminator = _normalize_discriminator(config)
    generator = _normalize_generator(config)
    loss = _normalize_loss(config, mask_enabled=generator["type"] == "stylegan2" and generator["mask_enabled"])
    divisor = loss["l1"]["gaussian_blur"]
    if divisor:
        _gaussian_blur_parameters(generator["img_resolution"], divisor)
    if discriminator["img_resolution"] != generator["img_resolution"]:
        raise ValueError("discriminator.img_resolution must match generator.img_resolution")
    discriminator_type = DiscriminatorType[discriminator["type"]]
    if discriminator_type is DiscriminatorType.UNET:
        loss["r1"]["enable"] = False
    if "spatial_weight" in loss["gan"]:
        raise ValueError("loss.gan.spatial_weight is no longer supported")
    if discriminator_type is DiscriminatorType.FIR_MINIBATCH_STD:
        group_size = discriminator["minibatch_std_group_size"]
        if train["batch_size"] % group_size:
            raise ValueError("train.batch_size must be divisible by discriminator.minibatch_std_group_size")
        feature_count = int(math.log2(generator["img_resolution"])) - 2
    elif discriminator_type is DiscriminatorType.UNET:
        feature_count = 4
    else:
        feature_count = int(math.log2(generator["img_resolution"] // INSTANCE_NORM_RESIDUAL_MIN_RESOLUTION))
    if loss["wfm"]["enable"] and max(map(int, loss["wfm"]["weights"]), default=-1) >= feature_count:
        raise ValueError("loss.wfm.weights exceeds discriminator feature count")
    return {
        "train": train,
        "optimizer": _normalize_optimizer(config),
        "scheduler": _normalize_scheduler(config),
        "identity": _normalize_identity(config),
        "loss": loss,
        "data": _normalize_data(config),
        "generator": generator,
        "discriminator": discriminator,
    }


def load_train_config(path: str | Path) -> dict[str, Any]:
    """读取用户 TOML 并返回唯一 canonical config。"""
    config_path = Path(path)
    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)
    return resolve_train_config(raw_config)
