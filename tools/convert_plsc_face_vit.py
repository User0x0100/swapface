from __future__ import annotations

import argparse
import hashlib
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from misc.models.id_encoder.vit import plsc_face_vit_b

PLSC_FACEVIT_B_SOURCE_SHA256 = "2bb7c1a8f1aaf060eb23361aa5d8f16010dec45adc870d017070a2965f71d77b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_plsc_state_dict(path: Path) -> dict[str, np.ndarray]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != PLSC_FACEVIT_B_SOURCE_SHA256:
        raise ValueError(f"Refusing to unpickle an unverified checkpoint: SHA256 {actual_sha256} != official {PLSC_FACEVIT_B_SOURCE_SHA256}")

    with path.open("rb") as f:
        state_dict = pickle.load(f)
    if not isinstance(state_dict, dict):
        raise TypeError(f"PLSC checkpoint must contain a dict, got {type(state_dict).__name__}")
    return state_dict


def convert_plsc_face_vit_b(source: Path) -> OrderedDict[str, Tensor]:
    model = plsc_face_vit_b()
    target_state = model.state_dict()
    linear_weight_keys = {f"{name}.weight" if name else "weight" for name, module in model.named_modules() if isinstance(module, nn.Linear)}

    converted: OrderedDict[str, Tensor] = OrderedDict()
    for source_key, value in _load_plsc_state_dict(source).items():
        if source_key == "StructuredToParameterName@@":
            continue
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{source_key}: expected numpy.ndarray, got {type(value).__name__}")

        target_key = source_key.replace("._mean", ".running_mean").replace("._variance", ".running_var")
        if target_key not in target_state:
            raise KeyError(f"Unexpected PLSC parameter: {source_key} -> {target_key}")

        tensor = torch.from_numpy(value)
        if target_key in linear_weight_keys:
            tensor = tensor.transpose(0, 1).contiguous()

        expected = target_state[target_key]
        if tensor.shape != expected.shape:
            raise ValueError(f"{source_key} -> {target_key}: converted shape {tuple(tensor.shape)} != expected {tuple(expected.shape)}")
        converted[target_key] = tensor.to(dtype=expected.dtype)

    for key, value in target_state.items():
        if key.endswith(".num_batches_tracked"):
            converted[key] = value.clone()

    missing = sorted(set(target_state) - set(converted))
    if missing:
        raise KeyError(f"Missing converted parameters: {missing}")

    model.load_state_dict(converted, strict=True)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the official PLSC WebFace42M FaceViT-B checkpoint to PyTorch.")
    parser.add_argument("source", type=Path, help="Official PLSC .pdparams checkpoint")
    parser.add_argument("output", type=Path, help="Destination PyTorch .pth state_dict")
    args = parser.parse_args()

    converted = convert_plsc_face_vit_b(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    print(f"saved {len(converted)} tensors to {args.output}")
    print(f"output sha256: {_sha256(args.output)}")


if __name__ == "__main__":
    main()
