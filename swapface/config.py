"""训练实验配置协议：默认值、严格校验与 canonical config 生成。"""

import math
import tomllib
from pathlib import Path
from typing import Any

from misc.models.id_encoder import IDEncoderProvider

from .dataloader_common import ImageDecoderBackend

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "batch_size": 16,
    "stage": "joint",
    "precision": "bf16",
    "device": "cuda",
    "compile_module": True,
    "log_interval": 10,
    "sample_save_every": 1000,
    "checkpoint_save_every": 10000,
}
DEFAULT_OPTIMIZER_STAGE_CONFIG: dict[str, Any] = {
    "lr": 1e-4,
}
DEFAULT_OPTIMIZER_CONFIG: dict[str, Any] = {
    "generator": dict(DEFAULT_OPTIMIZER_STAGE_CONFIG),
    "hq_discriminator": dict(DEFAULT_OPTIMIZER_STAGE_CONFIG),
    "coarse_discriminator": dict(DEFAULT_OPTIMIZER_STAGE_CONFIG),
}
DEFAULT_SCHEDULER_STAGE_CONFIG: dict[str, Any] = {
    "type": "none",
    "t_max": 20000,
    "min_lr_ratio": 0.1,
}
DEFAULT_SCHEDULER_CONFIG: dict[str, Any] = {
    "generator": dict(DEFAULT_SCHEDULER_STAGE_CONFIG),
    "hq_discriminator": dict(DEFAULT_SCHEDULER_STAGE_CONFIG),
    "coarse_discriminator": dict(DEFAULT_SCHEDULER_STAGE_CONFIG),
}
DEFAULT_GENERATOR_CONFIG: dict[str, Any] = {
    "img_resolution": 512,
    "img_channels": 3,
    "id_dim": 512,
    "coarse_resolution": 128,
    "coarse_latent_resolution": 32,
    "coarse_num_latent": 8,
    "coarse_base_ch": 64,
    "coarse_max_ch": 512,
    "hq_bottleneck_resolution": 16,
    "hq_base_ch": 8,
    "hq_max_ch": 128,
}

DEFAULT_DISCRIMINATOR_CONFIG: dict[str, Any] = {
    "hq": {
        "img_resolution": 512,
        "img_channels": 3,
        "base_ch": 64,
        "max_ch": 512,
    },
    "coarse": {
        "img_resolution": 128,
        "img_channels": 3,
        "base_ch": 64,
        "max_ch": 512,
    },
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
DEFAULT_WFM_LOSS_WEIGHT: dict[int, float] = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}
DEFAULT_GAN_LOSS_CONFIG = {"weight": 1.0}
DEFAULT_IDENTITY_LOSS_CONFIG = {"provider": DEFAULT_IDENTITY_LOSS_PROVIDER.name, "weight": 10.0}
DEFAULT_GAZE_LOSS_CONFIG = {"enable": False, "weight": 1.0, "distribution_weight": 0.1, "confidence_weighted": True}
DEFAULT_HRFFA_LOSS_CONFIG = {
    "enable": False,
    "pose_weight": 1.0,
    "eye_weight": 1.0,
    "mouth_weight": 1.0,
    "contour_weight": 1.0,
    "contour_shape_weight": 0.5,
    "occluded_geometry_weight": 0.25,
}
DEFAULT_FACS_LOSS_CONFIG = {
    "enable": False,
    "weight": 1.0,
    "brow_weight": 1.0,
    "eye_weight": 1.0,
    "nose_weight": 1.0,
    "mouth_weight": 1.0,
    "lower_face_weight": 1.0,
    "asymmetry_weight": 1.0,
}
DEFAULT_COARSE_LOSS_CONFIG: dict[str, Any] = {
    "gan": dict(DEFAULT_GAN_LOSS_CONFIG),
    "identity": dict(DEFAULT_IDENTITY_LOSS_CONFIG),
    "gaze": dict(DEFAULT_GAZE_LOSS_CONFIG),
    "hrffa": dict(DEFAULT_HRFFA_LOSS_CONFIG),
    "facs": dict(DEFAULT_FACS_LOSS_CONFIG),
    "reconstruction": {"scope": "same"},
    "l1": {"enable": True, "weight": 10.0},
    "vgg": {"enable": True, "weights": dict(DEFAULT_VGG_PERCEPTUAL_LOSS_WEIGHT)},
}
DEFAULT_HQ_LOSS_CONFIG: dict[str, Any] = {
    "gan": dict(DEFAULT_GAN_LOSS_CONFIG),
    "identity": dict(DEFAULT_IDENTITY_LOSS_CONFIG),
    "gaze": dict(DEFAULT_GAZE_LOSS_CONFIG),
    "hrffa": dict(DEFAULT_HRFFA_LOSS_CONFIG),
    "facs": dict(DEFAULT_FACS_LOSS_CONFIG),
    "reconstruction": {"scope": "same"},
    "l1": {"enable": True, "weight": 10.0},
    "vgg": {"enable": True, "weights": dict(DEFAULT_VGG_PERCEPTUAL_LOSS_WEIGHT)},
    "wfm": {"enable": True, "weights": dict(DEFAULT_WFM_LOSS_WEIGHT)},
}
DEFAULT_LOSS_CONFIG: dict[str, Any] = {
    "coarse": DEFAULT_COARSE_LOSS_CONFIG,
    "hq": DEFAULT_HQ_LOSS_CONFIG,
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
        normalized.append({"path": path, "adjustment": _float(entry.get("adjustment", 0.0), f"{section}[{index}].adjustment")})
    return normalized


def _normalize_train(config: dict[str, Any]) -> dict[str, Any]:
    raw = _table(config, "train", "[train]")
    if "precision" not in raw:
        raise ValueError("[train].precision 必须显式配置为 fp32、fp16 或 bf16")
    result = _with_defaults(raw, DEFAULT_TRAIN_CONFIG, "[train]")
    result["batch_size"] = _int(result["batch_size"], "train.batch_size", minimum=1)
    if not isinstance(result["stage"], str):
        raise TypeError(f"train.stage 必须为字符串，实际为 {type(result['stage']).__name__}")
    result["stage"] = result["stage"].lower()
    if result["stage"] not in {"joint", "coarse", "hq"}:
        raise ValueError(f"train.stage={result['stage']!r} 无效，可选：joint、coarse、hq")
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


def _normalize_stage_loss(loss: dict[str, Any], stage: str, *, hq: bool) -> dict[str, Any]:
    defaults = DEFAULT_HQ_LOSS_CONFIG if hq else DEFAULT_COARSE_LOSS_CONFIG
    unknown = set(loss) - set(defaults)
    if unknown:
        raise ValueError(f"[loss.{stage}] 包含未知字段：{sorted(unknown)}")
    prefix = f"loss.{stage}"

    gan = _with_defaults(_table(loss, "gan", f"[{prefix}.gan]"), defaults["gan"], f"[{prefix}.gan]")
    gan["weight"] = _float(gan["weight"], f"{prefix}.gan.weight", minimum=0.0)

    identity = _with_defaults(_table(loss, "identity", f"[{prefix}.identity]"), defaults["identity"], f"[{prefix}.identity]")
    identity["provider"] = _provider(identity["provider"], f"{prefix}.identity.provider")
    identity["weight"] = _float(identity["weight"], f"{prefix}.identity.weight", minimum=0.0)

    gaze = _with_defaults(_table(loss, "gaze", f"[{prefix}.gaze]"), defaults["gaze"], f"[{prefix}.gaze]")
    gaze["enable"] = _bool(gaze["enable"], f"{prefix}.gaze.enable")
    gaze["weight"] = _float(gaze["weight"], f"{prefix}.gaze.weight", minimum=0.0)
    gaze["distribution_weight"] = _float(gaze["distribution_weight"], f"{prefix}.gaze.distribution_weight", minimum=0.0)
    gaze["confidence_weighted"] = _bool(gaze["confidence_weighted"], f"{prefix}.gaze.confidence_weighted")

    hrffa = _with_defaults(_table(loss, "hrffa", f"[{prefix}.hrffa]"), defaults["hrffa"], f"[{prefix}.hrffa]")
    hrffa["enable"] = _bool(hrffa["enable"], f"{prefix}.hrffa.enable")
    for key in ("pose_weight", "eye_weight", "mouth_weight", "contour_weight", "contour_shape_weight"):
        hrffa[key] = _float(hrffa[key], f"{prefix}.hrffa.{key}", minimum=0.0)
    hrffa["occluded_geometry_weight"] = _float(hrffa["occluded_geometry_weight"], f"{prefix}.hrffa.occluded_geometry_weight", minimum=0.0, maximum=1.0)

    facs = _with_defaults(_table(loss, "facs", f"[{prefix}.facs]"), defaults["facs"], f"[{prefix}.facs]")
    facs["enable"] = _bool(facs["enable"], f"{prefix}.facs.enable")
    for key in ("weight", "brow_weight", "eye_weight", "nose_weight", "mouth_weight", "lower_face_weight", "asymmetry_weight"):
        facs[key] = _float(facs[key], f"{prefix}.facs.{key}", minimum=0.0)

    result = {"gan": gan, "identity": identity, "gaze": gaze, "hrffa": hrffa, "facs": facs}

    reconstruction = _with_defaults(_table(loss, "reconstruction", f"[{prefix}.reconstruction]"), defaults["reconstruction"], f"[{prefix}.reconstruction]")
    scope = reconstruction["scope"]
    if not isinstance(scope, str):
        raise TypeError(f"{prefix}.reconstruction.scope 必须为字符串，实际为 {type(scope).__name__}")
    if scope not in {"same", "all"}:
        raise ValueError(f"{prefix}.reconstruction.scope={scope!r} 无效，可选：same、all")

    l1 = _with_defaults(_table(loss, "l1", f"[{prefix}.l1]"), defaults["l1"], f"[{prefix}.l1]")
    l1["enable"] = _bool(l1["enable"], f"{prefix}.l1.enable")
    l1["weight"] = _float(l1["weight"], f"{prefix}.l1.weight", minimum=0.0)

    vgg = _with_defaults(_table(loss, "vgg", f"[{prefix}.vgg]"), defaults["vgg"], f"[{prefix}.vgg]")
    vgg["enable"] = _bool(vgg["enable"], f"{prefix}.vgg.enable")
    raw_vgg_weights = vgg["weights"]
    if not isinstance(raw_vgg_weights, dict) or (vgg["enable"] and not raw_vgg_weights):
        raise ValueError(f"[{prefix}.vgg.weights] 必须为非空表/对象")
    vgg["weights"] = {str(layer): _float(weight, f"{prefix}.vgg.weights.{layer}", minimum=0.0) for layer, weight in raw_vgg_weights.items()}

    result.update({"reconstruction": reconstruction, "l1": l1, "vgg": vgg})
    if not hq:
        return result

    wfm = _with_defaults(_table(loss, "wfm", f"[{prefix}.wfm]"), defaults["wfm"], f"[{prefix}.wfm]")
    wfm["enable"] = _bool(wfm["enable"], f"{prefix}.wfm.enable")
    raw_wfm_weights = wfm["weights"]
    if not isinstance(raw_wfm_weights, dict) or (wfm["enable"] and not raw_wfm_weights):
        raise ValueError(f"[{prefix}.wfm.weights] 必须为非空表/对象")
    normalized_wfm_weights: dict[str, float] = {}
    for index, weight in raw_wfm_weights.items():
        try:
            layer_index = int(index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[{prefix}.wfm.weights] 的键必须是非负整数层索引") from exc
        if layer_index < 0 or str(layer_index) != str(index):
            raise ValueError(f"{prefix}.wfm.weights 层索引无效：{index!r}")
        normalized_wfm_weights[str(layer_index)] = _float(weight, f"{prefix}.wfm.weights.{layer_index}", minimum=0.0)
    wfm["weights"] = normalized_wfm_weights

    result["wfm"] = wfm
    return result


def _normalize_loss(config: dict[str, Any]) -> dict[str, Any]:
    loss = _table(config, "loss", "[loss]")
    unknown = set(loss) - set(DEFAULT_LOSS_CONFIG)
    if unknown:
        raise ValueError(f"[loss] 包含未知字段：{sorted(unknown)}")
    return {
        "coarse": _normalize_stage_loss(_table(loss, "coarse", "[loss.coarse]"), "coarse", hq=False),
        "hq": _normalize_stage_loss(_table(loss, "hq", "[loss.hq]"), "hq", hq=True),
    }


def _normalize_generator(config: dict[str, Any]) -> dict[str, Any]:
    result = _with_defaults(_table(config, "generator", "[generator]"), DEFAULT_GENERATOR_CONFIG, "[generator]")

    integer_keys = (
        "img_resolution",
        "img_channels",
        "id_dim",
        "coarse_resolution",
        "coarse_latent_resolution",
        "coarse_num_latent",
        "coarse_base_ch",
        "coarse_max_ch",
        "hq_bottleneck_resolution",
        "hq_base_ch",
        "hq_max_ch",
    )
    for key in integer_keys:
        result[key] = _int(result[key], f"generator.{key}", minimum=1)

    if result["coarse_resolution"] > result["img_resolution"]:
        raise ValueError("generator.coarse_resolution 不能大于 generator.img_resolution")
    if result["coarse_max_ch"] < result["coarse_base_ch"]:
        raise ValueError("generator.coarse_max_ch 必须 >= generator.coarse_base_ch")
    if result["hq_max_ch"] < result["hq_base_ch"]:
        raise ValueError("generator.hq_max_ch 必须 >= generator.hq_base_ch")

    for resolution_key, latent_key in (
        ("coarse_resolution", "coarse_latent_resolution"),
        ("img_resolution", "hq_bottleneck_resolution"),
    ):
        resolution = result[resolution_key]
        latent = result[latent_key]
        if resolution < latent or resolution % latent != 0:
            raise ValueError(f"generator.{resolution_key} 必须是 generator.{latent_key} 的整数倍")
        ratio = resolution // latent
        if ratio & (ratio - 1):
            raise ValueError(f"generator.{resolution_key}/generator.{latent_key} 必须为 2 的整数次幂")

    # HQRefiner 至少需要一级下采样；其 bottleneck 不能等于最终输出分辨率。
    if result["hq_bottleneck_resolution"] == result["img_resolution"]:
        raise ValueError("generator.hq_bottleneck_resolution 必须小于 generator.img_resolution")

    return result


def _normalize_discriminator(config: dict[str, Any]) -> dict[str, Any]:
    discriminator = _table(config, "discriminator", "[discriminator]")
    unknown = set(discriminator) - set(DEFAULT_DISCRIMINATOR_CONFIG)
    if unknown:
        raise ValueError(f"[discriminator] 包含未知字段：{sorted(unknown)}")

    result: dict[str, Any] = {}
    for stage, defaults in DEFAULT_DISCRIMINATOR_CONFIG.items():
        stage_config = _with_defaults(
            _table(discriminator, stage, f"[discriminator.{stage}]"),
            defaults,
            f"[discriminator.{stage}]",
        )
        for key in ("img_resolution", "img_channels", "base_ch", "max_ch"):
            stage_config[key] = _int(stage_config[key], f"discriminator.{stage}.{key}", minimum=1)
        if stage_config["max_ch"] < stage_config["base_ch"]:
            raise ValueError(f"discriminator.{stage}.max_ch 必须 >= discriminator.{stage}.base_ch")
        if stage_config["img_resolution"] % 8 != 0:
            raise ValueError(f"discriminator.{stage}.img_resolution 必须能被 8 整除")
        result[stage] = stage_config
    return result


def _normalize_data(config: dict[str, Any]) -> dict[str, Any]:
    data = _table(config, "data", "[data]")
    allowed = {"loader", "augmentation", "sampling", "src", "dst"}
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

    sampling = _with_defaults(_table(data, "sampling", "[data.sampling]"), DEFAULT_DATA_CONFIG["sampling"], "[data.sampling]")
    sampling["same_prob"] = _float(sampling["same_prob"], "data.sampling.same_prob", minimum=0.0, maximum=1.0)

    return {
        "loader": loader,
        "augmentation": augmentation,
        "sampling": sampling,
        "src": _normalize_image_sources(data.get("src"), "data.src"),
        "dst": _normalize_image_sources(data.get("dst"), "data.dst"),
    }


def resolve_train_config(config: dict[str, Any]) -> dict[str, Any]:
    """严格校验配置并展开默认值，得到唯一 canonical config。"""
    allowed_sections = {"train", "optimizer", "scheduler", "identity", "loss", "data", "generator", "discriminator"}
    unknown_sections = set(config) - allowed_sections
    if unknown_sections:
        raise ValueError(f"配置包含未知顶层字段：{sorted(unknown_sections)}")

    return {
        "train": _normalize_train(config),
        "optimizer": _normalize_optimizer(config),
        "scheduler": _normalize_scheduler(config),
        "identity": _normalize_identity(config),
        "loss": _normalize_loss(config),
        "data": _normalize_data(config),
        "generator": _normalize_generator(config),
        "discriminator": _normalize_discriminator(config),
    }


def load_train_config(path: str | Path) -> dict[str, Any]:
    """读取用户 TOML 并返回唯一 canonical config。"""
    config_path = Path(path)
    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)
    return resolve_train_config(raw_config)
