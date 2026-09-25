"""训练实验配置协议：默认值、严格校验与 canonical config 生成。"""

import math
import tomllib
from pathlib import Path
from typing import Any

from misc.models.id_encoder import IDEncoderProvider

from .dataloader_common import ImageDecoderBackend

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "batch_size": 16,
    "precision": "bf16",
    "device": "cuda",
    "compile_module": True,
    "log_interval": 10,
    "sample_save_every": 1000,
    "checkpoint_save_every": 10000,
}
DEFAULT_OPTIMIZER_CONFIG: dict[str, Any] = {
    "lr": 1e-4,
}
DEFAULT_SCHEDULER_CONFIG: dict[str, Any] = {
    "type": "none",
    "t_max": 20000,
    "min_lr_ratio": 0.1,
}
DEFAULT_GENERATOR_CONFIG: dict[str, Any] = {
    "img_resolution": 256,
    "img_channels": 3,
    "num_depth": 3,
    "num_latent": 6,
    "base_ch": 256,
    "max_ch": 1024,
    "id_dim": 512,
    "aad_skip_layers": (),
}
DEFAULT_DISCRIMINATOR_CONFIG: dict[str, Any] = {
    "img_resolution": 256,
    "img_channels": 3,
    "base_ch": 64,
    "max_ch": 512,
    "group_size": 4,
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
DEFAULT_LOSS_CONFIG: dict[str, Any] = {
    "reconstruction": {"scope": "same"},
    "gan": {"weight": 1.0},
    "identity": {"provider": DEFAULT_IDENTITY_LOSS_PROVIDER.name, "weight": 10.0},
    "l1": {"enable": True, "weight": 10.0},
    "r1": {"enable": True, "interval": 16, "gamma": 10.0},
    "gaze": {"enable": False, "weight": 1.0, "distribution_weight": 0.1, "confidence_weighted": True},
    "hrffa": {
        "enable": False,
        "pose_weight": 1.0,
        "eye_weight": 1.0,
        "mouth_weight": 1.0,
        "contour_weight": 1.0,
        "contour_shape_weight": 0.5,
        "occluded_geometry_weight": 0.25,
    },
    "facs": {
        "enable": False,
        "weight": 1.0,
        "brow_weight": 1.0,
        "eye_weight": 1.0,
        "nose_weight": 1.0,
        "mouth_weight": 1.0,
        "lower_face_weight": 1.0,
        "asymmetry_weight": 1.0,
    },
    "vgg": {"enable": True, "weights": DEFAULT_VGG_PERCEPTUAL_LOSS_WEIGHT},
    "wfm": {"enable": True, "weights": DEFAULT_WFM_LOSS_WEIGHT},
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
    result = _with_defaults(_table(config, "optimizer", "[optimizer]"), DEFAULT_OPTIMIZER_CONFIG, "[optimizer]")
    result["lr"] = _float(result["lr"], "optimizer.lr", minimum=1e-30)
    return result


def _normalize_scheduler(config: dict[str, Any]) -> dict[str, Any]:
    result = _with_defaults(_table(config, "scheduler", "[scheduler]"), DEFAULT_SCHEDULER_CONFIG, "[scheduler]")
    scheduler_type = result["type"]
    if not isinstance(scheduler_type, str):
        raise TypeError(f"scheduler.type 必须为字符串，实际为 {type(scheduler_type).__name__}")
    scheduler_type = scheduler_type.lower()
    if scheduler_type not in {"none", "cosine"}:
        raise ValueError(f"scheduler.type={scheduler_type!r} 无效，可选：none、cosine")
    result["type"] = scheduler_type
    result["t_max"] = _int(result["t_max"], "scheduler.t_max", minimum=1)
    result["min_lr_ratio"] = _float(result["min_lr_ratio"], "scheduler.min_lr_ratio", minimum=0.0, maximum=1.0)
    return result


def _normalize_identity(config: dict[str, Any]) -> dict[str, Any]:
    result = _with_defaults(_table(config, "identity", "[identity]"), DEFAULT_IDENTITY_CONFIG, "[identity]")
    result["provider"] = _provider(result["provider"], "identity.provider")
    return result


def _normalize_loss(config: dict[str, Any]) -> dict[str, Any]:
    loss = _table(config, "loss", "[loss]")
    unknown = set(loss) - set(DEFAULT_LOSS_CONFIG)
    if unknown:
        raise ValueError(f"[loss] 包含未知字段：{sorted(unknown)}")

    reconstruction = _with_defaults(_table(loss, "reconstruction", "[loss.reconstruction]"), DEFAULT_LOSS_CONFIG["reconstruction"], "[loss.reconstruction]")
    scope = reconstruction["scope"]
    if not isinstance(scope, str):
        raise TypeError(f"loss.reconstruction.scope 必须为字符串，实际为 {type(scope).__name__}")
    if scope not in {"same", "all"}:
        raise ValueError(f"loss.reconstruction.scope={scope!r} 无效，可选：same、all")

    gan = _with_defaults(_table(loss, "gan", "[loss.gan]"), DEFAULT_LOSS_CONFIG["gan"], "[loss.gan]")
    gan["weight"] = _float(gan["weight"], "loss.gan.weight", minimum=0.0)

    identity = _with_defaults(_table(loss, "identity", "[loss.identity]"), DEFAULT_LOSS_CONFIG["identity"], "[loss.identity]")
    identity["provider"] = _provider(identity["provider"], "loss.identity.provider")
    identity["weight"] = _float(identity["weight"], "loss.identity.weight", minimum=0.0)

    l1 = _with_defaults(_table(loss, "l1", "[loss.l1]"), DEFAULT_LOSS_CONFIG["l1"], "[loss.l1]")
    l1["enable"] = _bool(l1["enable"], "loss.l1.enable")
    l1["weight"] = _float(l1["weight"], "loss.l1.weight", minimum=0.0)

    r1 = _with_defaults(_table(loss, "r1", "[loss.r1]"), DEFAULT_LOSS_CONFIG["r1"], "[loss.r1]")
    r1["enable"] = _bool(r1["enable"], "loss.r1.enable")
    r1["interval"] = _int(r1["interval"], "loss.r1.interval", minimum=1)
    r1["gamma"] = _float(r1["gamma"], "loss.r1.gamma", minimum=0.0)

    gaze = _with_defaults(_table(loss, "gaze", "[loss.gaze]"), DEFAULT_LOSS_CONFIG["gaze"], "[loss.gaze]")
    gaze["enable"] = _bool(gaze["enable"], "loss.gaze.enable")
    gaze["weight"] = _float(gaze["weight"], "loss.gaze.weight", minimum=0.0)
    gaze["distribution_weight"] = _float(gaze["distribution_weight"], "loss.gaze.distribution_weight", minimum=0.0)
    gaze["confidence_weighted"] = _bool(gaze["confidence_weighted"], "loss.gaze.confidence_weighted")

    hrffa = _with_defaults(_table(loss, "hrffa", "[loss.hrffa]"), DEFAULT_LOSS_CONFIG["hrffa"], "[loss.hrffa]")
    hrffa["enable"] = _bool(hrffa["enable"], "loss.hrffa.enable")
    for key in ("pose_weight", "eye_weight", "mouth_weight", "contour_weight", "contour_shape_weight"):
        hrffa[key] = _float(hrffa[key], f"loss.hrffa.{key}", minimum=0.0)
    hrffa["occluded_geometry_weight"] = _float(hrffa["occluded_geometry_weight"], "loss.hrffa.occluded_geometry_weight", minimum=0.0, maximum=1.0)

    facs = _with_defaults(_table(loss, "facs", "[loss.facs]"), DEFAULT_LOSS_CONFIG["facs"], "[loss.facs]")
    facs["enable"] = _bool(facs["enable"], "loss.facs.enable")
    for key in ("weight", "brow_weight", "eye_weight", "nose_weight", "mouth_weight", "lower_face_weight", "asymmetry_weight"):
        facs[key] = _float(facs[key], f"loss.facs.{key}", minimum=0.0)

    vgg = _with_defaults(_table(loss, "vgg", "[loss.vgg]"), DEFAULT_LOSS_CONFIG["vgg"], "[loss.vgg]")
    vgg["enable"] = _bool(vgg["enable"], "loss.vgg.enable")
    raw_vgg_weights = vgg["weights"]
    if not isinstance(raw_vgg_weights, dict) or (vgg["enable"] and not raw_vgg_weights):
        raise ValueError("[loss.vgg.weights] 必须为非空表/对象")
    vgg["weights"] = {str(layer): _float(weight, f"loss.vgg.weights.{layer}", minimum=0.0) for layer, weight in raw_vgg_weights.items()}

    wfm = _with_defaults(_table(loss, "wfm", "[loss.wfm]"), DEFAULT_LOSS_CONFIG["wfm"], "[loss.wfm]")
    wfm["enable"] = _bool(wfm["enable"], "loss.wfm.enable")
    raw_wfm_weights = wfm["weights"]
    if not isinstance(raw_wfm_weights, dict) or (wfm["enable"] and not raw_wfm_weights):
        raise ValueError("[loss.wfm.weights] 必须为非空表/对象")
    normalized_wfm_weights: dict[str, float] = {}
    for index, weight in raw_wfm_weights.items():
        try:
            layer_index = int(index)
        except (TypeError, ValueError) as exc:
            raise ValueError("[loss.wfm.weights] 的键必须是非负整数层索引") from exc
        if layer_index < 0 or str(layer_index) != str(index):
            raise ValueError(f"loss.wfm.weights 层索引无效：{index!r}")
        normalized_wfm_weights[str(layer_index)] = _float(weight, f"loss.wfm.weights.{layer_index}", minimum=0.0)
    wfm["weights"] = normalized_wfm_weights

    return {
        "reconstruction": reconstruction,
        "gan": gan,
        "identity": identity,
        "l1": l1,
        "r1": r1,
        "gaze": gaze,
        "hrffa": hrffa,
        "facs": facs,
        "vgg": vgg,
        "wfm": wfm,
    }


def _normalize_generator(config: dict[str, Any]) -> dict[str, Any]:
    result = _with_defaults(_table(config, "generator", "[generator]"), DEFAULT_GENERATOR_CONFIG, "[generator]")
    for key in ("img_resolution", "img_channels", "num_depth", "num_latent", "base_ch", "max_ch", "id_dim"):
        result[key] = _int(result[key], f"generator.{key}")

    skip_layers = result["aad_skip_layers"]
    if not isinstance(skip_layers, (list, tuple)):
        raise TypeError(f"generator.aad_skip_layers 必须为整数数组，实际为 {type(skip_layers).__name__}")
    normalized_skip_layers: list[int] = []
    for index, layer in enumerate(skip_layers):
        normalized_skip_layers.append(_int(layer, f"generator.aad_skip_layers[{index}]"))
    result["aad_skip_layers"] = normalized_skip_layers
    return result


def _normalize_discriminator(config: dict[str, Any]) -> dict[str, Any]:
    result = _with_defaults(_table(config, "discriminator", "[discriminator]"), DEFAULT_DISCRIMINATOR_CONFIG, "[discriminator]")
    for key in ("img_resolution", "img_channels", "base_ch", "max_ch"):
        result[key] = _int(result[key], f"discriminator.{key}")
    result["group_size"] = _int(result["group_size"], "discriminator.group_size", minimum=1)
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
