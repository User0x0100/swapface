"""训练实验配置协议：显式默认值、TOML 解析与 runtime 参数转换。"""

import math
import tomllib
from pathlib import Path
from typing import Any

from misc.models.id_encoder import IDEncoderProvider

from .dataloader_common import DEFAULT_DATALOADER_CONFIG, ImageDecoderBackend, ImageSource

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "batch_size": 16,
    "lr": 1e-4,
    "lr_scheduler_t_max": 0,
    "r1_reg_step": 16,
    "r1_gamma": 10.0,
    "precision": "bf16",
    "device": "cuda",
    "compile_module": True,
    "log_interval": 10,
    "sample_save_every": 1000,
    "weight_save_every": 10000,
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
DEFAULT_VGG_PERCEPTUAL_LOSS_WEIGHT: dict[str, float] = {
    "conv1_2": 2.5,
    "conv2_2": 2.5,
    "conv3_3": 2.5,
    "conv4_3": 2.5,
}
DEFAULT_WFM_LOSS_WEIGHT: dict[int, float] = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}
DEFAULT_IDENTITY_CONFIG: dict[str, Any] = {
    "generator_provider": DEFAULT_GENERATOR_ID_ENCODER_PROVIDER.name,
    "loss_provider": DEFAULT_IDENTITY_LOSS_PROVIDER.name,
    "loss_weight": 10.0,
}
DEFAULT_LOSS_CONFIG: dict[str, Any] = {
    "enable_rec_loss": True,
    "rec_loss_weight": 10.0,
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
DATALOADER_RANGE_KEYS = ("rotation_range", "scale_factor_range", "tx_range", "ty_range")


def _with_defaults(values: dict[str, Any], defaults: dict[str, Any], section: str) -> dict[str, Any]:
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"{section} 包含未知字段：{sorted(unknown)}")
    return defaults | values


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

    same_prob = float(config["same_prob"])
    if not math.isfinite(same_prob) or not 0.0 <= same_prob <= 1.0:
        raise ValueError(f"dataloader.same_prob 必须位于 [0, 1]，实际为 {same_prob}")
    config["same_prob"] = same_prob
    return config


def resolve_train_config(config: dict[str, Any]) -> dict[str, Any]:
    """校验配置并展开显式协议默认值，得到可哈希、可持久化的规范配置。"""
    allowed_sections = {"train", "identity", "loss", "dataloader", "generator", "discriminator", "src", "dst"}
    unknown_sections = set(config) - allowed_sections
    if unknown_sections:
        raise ValueError(f"配置包含未知顶层字段：{sorted(unknown_sections)}")

    for section in ("train", "identity", "loss", "dataloader", "generator", "discriminator"):
        value = config.get(section, {})
        if not isinstance(value, dict):
            raise TypeError(f"[{section}] 必须为表/对象")

    raw_train = dict(config.get("train", {}))
    if "precision" not in raw_train:
        raise ValueError("[train].precision 必须显式配置为 fp32、fp16 或 bf16")
    train = _with_defaults(raw_train, DEFAULT_TRAIN_CONFIG, "[train]")
    precision = train["precision"]
    if not isinstance(precision, str):
        raise TypeError(f"train.precision 必须为字符串，实际为 {type(precision).__name__}")
    precision = precision.lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"train.precision={precision!r} 无效，可选：fp32、fp16、bf16")
    train["precision"] = precision

    identity = dict(config.get("identity", {}))
    unknown_identity = set(identity) - set(DEFAULT_IDENTITY_CONFIG)
    if unknown_identity:
        raise ValueError(f"[identity] 包含未知字段：{sorted(unknown_identity)}")
    generator_provider = str(identity.get("generator_provider", DEFAULT_IDENTITY_CONFIG["generator_provider"]))
    loss_provider = str(identity.get("loss_provider", DEFAULT_IDENTITY_CONFIG["loss_provider"]))
    try:
        IDEncoderProvider[generator_provider]
        IDEncoderProvider[loss_provider]
    except KeyError as exc:
        supported = ", ".join(provider.name for provider in IDEncoderProvider)
        raise ValueError(f"身份编码器类型无效，可选：{supported}") from exc

    loss = dict(config.get("loss", {}))
    unknown_loss = set(loss) - set(DEFAULT_LOSS_CONFIG)
    if unknown_loss:
        raise ValueError(f"[loss] 包含未知字段：{sorted(unknown_loss)}")

    enable_rec_loss = bool(loss.get("enable_rec_loss", DEFAULT_LOSS_CONFIG["enable_rec_loss"]))
    rec_loss_weight = float(loss.get("rec_loss_weight", DEFAULT_LOSS_CONFIG["rec_loss_weight"]))

    gaze = loss.get("gaze", {})
    if not isinstance(gaze, dict):
        raise TypeError("[loss.gaze] 必须为表/对象")
    unknown_gaze = set(gaze) - set(DEFAULT_LOSS_CONFIG["gaze"])
    if unknown_gaze:
        raise ValueError(f"[loss.gaze] 包含未知字段：{sorted(unknown_gaze)}")
    gaze_config = {
        "enable": bool(gaze.get("enable", DEFAULT_LOSS_CONFIG["gaze"]["enable"])),
        "weight": float(gaze.get("weight", DEFAULT_LOSS_CONFIG["gaze"]["weight"])),
        "distribution_weight": float(gaze.get("distribution_weight", DEFAULT_LOSS_CONFIG["gaze"]["distribution_weight"])),
        "confidence_weighted": bool(gaze.get("confidence_weighted", DEFAULT_LOSS_CONFIG["gaze"]["confidence_weighted"])),
    }

    hrffa = loss.get("hrffa", {})
    if not isinstance(hrffa, dict):
        raise TypeError("[loss.hrffa] 必须为表/对象")
    unknown_hrffa = set(hrffa) - set(DEFAULT_LOSS_CONFIG["hrffa"])
    if unknown_hrffa:
        raise ValueError(f"[loss.hrffa] 包含未知字段：{sorted(unknown_hrffa)}")
    hrffa_config = {key: bool(hrffa.get(key, default)) if key == "enable" else float(hrffa.get(key, default)) for key, default in DEFAULT_LOSS_CONFIG["hrffa"].items()}
    hrffa_numeric = {key: value for key, value in hrffa_config.items() if key != "enable"}
    non_finite_hrffa = {key: value for key, value in hrffa_numeric.items() if not math.isfinite(value)}
    if non_finite_hrffa:
        raise ValueError(f"[loss.hrffa] 数值必须为有限值：{non_finite_hrffa}")
    negative_hrffa = {key: value for key, value in hrffa_numeric.items() if value < 0.0}
    if negative_hrffa:
        raise ValueError(f"[loss.hrffa] 权重必须非负：{negative_hrffa}")
    if hrffa_config["occluded_geometry_weight"] > 1.0:
        raise ValueError(f"loss.hrffa.occluded_geometry_weight 必须位于 [0, 1]，实际为 {hrffa_config['occluded_geometry_weight']}")

    facs = loss.get("facs", {})
    if not isinstance(facs, dict):
        raise TypeError("[loss.facs] 必须为表/对象")
    unknown_facs = set(facs) - set(DEFAULT_LOSS_CONFIG["facs"])
    if unknown_facs:
        raise ValueError(f"[loss.facs] 包含未知字段：{sorted(unknown_facs)}")
    facs_config = {key: bool(facs.get(key, default)) if key == "enable" else float(facs.get(key, default)) for key, default in DEFAULT_LOSS_CONFIG["facs"].items()}
    invalid_facs = {key: value for key, value in facs_config.items() if key != "enable" and (not math.isfinite(value) or value < 0.0)}
    if invalid_facs:
        raise ValueError(f"[loss.facs] 权重必须为有限非负数：{invalid_facs}")

    vgg = loss.get("vgg", {})
    if not isinstance(vgg, dict):
        raise TypeError("[loss.vgg] 必须为表/对象")
    unknown_vgg = set(vgg) - set(DEFAULT_LOSS_CONFIG["vgg"])
    if unknown_vgg:
        raise ValueError(f"[loss.vgg] 包含未知字段：{sorted(unknown_vgg)}")
    enable_vgg = bool(vgg.get("enable", DEFAULT_LOSS_CONFIG["vgg"]["enable"]))
    raw_vgg_weights = vgg.get("weights", DEFAULT_LOSS_CONFIG["vgg"]["weights"])
    if not isinstance(raw_vgg_weights, dict) or (enable_vgg and not raw_vgg_weights):
        raise ValueError("[loss.vgg.weights] 必须为非空表/对象")
    vgg_weights = {str(layer): float(weight) for layer, weight in raw_vgg_weights.items()}

    wfm = loss.get("wfm", {})
    if not isinstance(wfm, dict):
        raise TypeError("[loss.wfm] 必须为表/对象")
    unknown_wfm = set(wfm) - set(DEFAULT_LOSS_CONFIG["wfm"])
    if unknown_wfm:
        raise ValueError(f"[loss.wfm] 包含未知字段：{sorted(unknown_wfm)}")
    enable_wfm = bool(wfm.get("enable", DEFAULT_LOSS_CONFIG["wfm"]["enable"]))
    raw_wfm_weights = wfm.get("weights", DEFAULT_LOSS_CONFIG["wfm"]["weights"])
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
            "loss_weight": float(identity.get("loss_weight", DEFAULT_IDENTITY_CONFIG["loss_weight"])),
        },
        "loss": {
            "enable_rec_loss": enable_rec_loss,
            "rec_loss_weight": rec_loss_weight,
            "gaze": gaze_config,
            "hrffa": hrffa_config,
            "facs": facs_config,
            "vgg": {"enable": enable_vgg, "weights": vgg_weights},
            "wfm": {"enable": enable_wfm, "weights": wfm_weights},
        },
        "dataloader": _normalize_dataloader(dict(config.get("dataloader", {}))),
        "generator": _with_defaults(dict(config.get("generator", {})), DEFAULT_GENERATOR_CONFIG, "[generator]"),
        "discriminator": _with_defaults(dict(config.get("discriminator", {})), DEFAULT_DISCRIMINATOR_CONFIG, "[discriminator]"),
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
        "enable_gaze_loss": bool(loss["gaze"]["enable"]),
        "gaze_loss_weight": float(loss["gaze"]["weight"]),
        "gaze_distribution_weight": float(loss["gaze"]["distribution_weight"]),
        "gaze_confidence_weighted": bool(loss["gaze"]["confidence_weighted"]),
        "enable_hrffa_loss": bool(loss["hrffa"]["enable"]),
        "hrffa_pose_weight": float(loss["hrffa"]["pose_weight"]),
        "hrffa_eye_weight": float(loss["hrffa"]["eye_weight"]),
        "hrffa_mouth_weight": float(loss["hrffa"]["mouth_weight"]),
        "hrffa_contour_weight": float(loss["hrffa"]["contour_weight"]),
        "hrffa_contour_shape_weight": float(loss["hrffa"]["contour_shape_weight"]),
        "hrffa_occluded_geometry_weight": float(loss["hrffa"]["occluded_geometry_weight"]),
        "enable_facs_loss": bool(loss["facs"]["enable"]),
        "facs_loss_weight": float(loss["facs"]["weight"]),
        "facs_brow_weight": float(loss["facs"]["brow_weight"]),
        "facs_eye_weight": float(loss["facs"]["eye_weight"]),
        "facs_nose_weight": float(loss["facs"]["nose_weight"]),
        "facs_mouth_weight": float(loss["facs"]["mouth_weight"]),
        "facs_lower_face_weight": float(loss["facs"]["lower_face_weight"]),
        "facs_asymmetry_weight": float(loss["facs"]["asymmetry_weight"]),
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
