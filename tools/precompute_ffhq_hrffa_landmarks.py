"""从本地 FFHQ WebDataset tar shards 预计算 HRFFA 关键点。

默认只处理少量样本并生成可视化预览，避免误触发 70k 全量任务。只有显式传入
``--all`` 才会处理选中 shard 的全部图像。

示例:
    uv run python tools/precompute_ffhq_hrffa_landmarks.py --input-dir /path/to/ffhq-wds
    uv run python tools/precompute_ffhq_hrffa_landmarks.py --input-dir /path/to/ffhq-wds --limit 8 --preview-count 8
    uv run python tools/precompute_ffhq_hrffa_landmarks.py --input-dir /path/to/ffhq-wds --all --preview-count 0
"""

import argparse
import json
import math
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.io import decode_image
from torchvision.io.image import ImageReadMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from misc.models import ImageInputRange
from misc.models.hrffa import HRFFALandmarkModel, HRFFAModelVariant, HRFFAScheme, HRFFAVisibility

DEFAULT_INPUT_DIR = PROJECT_ROOT / "datasets" / "ffhq-1024-wds"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "ffhq_hrffa_landmarks"
DEFAULT_SAMPLE_LIMIT = 8

VISIBILITY_NAMES = {
    HRFFAVisibility.OUTSIDE_IMAGE.value: "outside",
    HRFFAVisibility.OCCLUDED.value: "occluded",
    HRFFAVisibility.VISIBLE.value: "visible",
}

# OpenCV BGR colors.
VISIBILITY_COLORS = {
    HRFFAVisibility.OUTSIDE_IMAGE.value: (0, 0, 255),
    HRFFAVisibility.OCCLUDED.value: (0, 165, 255),
    HRFFAVisibility.VISIBLE.value: (0, 220, 0),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从本地 FFHQ WebDataset tar shards 预计算 HRFFA 关键点")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help=f"本地 tar shard 目录，默认 {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"输出目录，默认 {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--scheme", choices=[scheme.value for scheme in HRFFAScheme], default=HRFFAScheme.IBUG68.value, help="HRFFA 关键点拓扑")
    parser.add_argument("--variant", choices=[variant.value for variant in HRFFAModelVariant], default=HRFFAModelVariant.VITT_256.value, help="HRFFA 模型版本，默认 vitt-256")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None, help="可选的 vitl-320 clean_v3 checkpoint；未指定时从上游 release 缓存")
    parser.add_argument("--device", default="cuda", help="HRFFA 推理设备，默认 cuda")
    parser.add_argument("--batch-size", type=int, default=4, help="HRFFA 推理 batch size，默认 4")
    parser.add_argument("--preview-count", type=int, default=DEFAULT_SAMPLE_LIMIT, help="保存前 N 张关键点预览图，0 表示关闭")
    parser.add_argument("--start-shard-index", type=int, default=0, help="从排序后的第几个 shard 开始")
    parser.add_argument("--num-shards", type=int, default=None, help="最多处理多少个 shard；默认不额外限制")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名 landmark 输出")

    run_scope = parser.add_mutually_exclusive_group()
    run_scope.add_argument("--limit", type=int, default=DEFAULT_SAMPLE_LIMIT, help=f"全局最多处理 N 张；默认 {DEFAULT_SAMPLE_LIMIT}")
    run_scope.add_argument("--all", dest="process_all", action="store_true", help="显式处理选中 shard 的全部图像")

    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size 必须 > 0")
    if args.preview_count < 0:
        parser.error("--preview-count 不能为负数")
    if args.start_shard_index < 0:
        parser.error("--start-shard-index 不能为负数")
    if args.num_shards is not None and args.num_shards <= 0:
        parser.error("--num-shards 必须 > 0")
    if not args.process_all and args.limit <= 0:
        parser.error("--limit 必须 > 0；如需全量处理请显式使用 --all")
    return args


def _discover_shards(input_dir: Path, start: int, count: int | None) -> list[Path]:
    root = input_dir.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"input-dir 不是目录: {root}")
    shards = sorted(root.glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"本地目录中没有 tar shard: {root}")
    if start >= len(shards):
        raise ValueError(f"start_shard_index={start} 超出 shard 数量 {len(shards)}")
    return shards[start:] if count is None else shards[start : start + count]


def _decode_webp(data: bytes) -> Tensor:
    # torchvision decode_image 接受 1-D uint8 tensor。bytearray 提供可写 buffer，避免
    # torch.frombuffer 对只读 bytes 发出警告。
    encoded = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return decode_image(encoded, mode=ImageReadMode.RGB)


@torch.inference_mode()
def _infer_batch(
    model: HRFFALandmarkModel,
    device: torch.device,
    keys: list[str],
    images: list[Tensor],
) -> list[tuple[str, Tensor, np.ndarray, np.ndarray]]:
    if not images:
        return []

    batch = torch.stack(images).to(device=device, non_blocking=True)
    predicted, visibility = model(batch, return_visibility=True)
    predicted_np = predicted.cpu().numpy()
    visibility_np = visibility.cpu().numpy()
    return list(zip(keys, images, predicted_np, visibility_np, strict=True))


def _annotate_landmarks(image: Tensor, key: str, landmarks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    canvas = image.permute(1, 2, 0).contiguous().numpy()[..., ::-1].copy()  # RGB -> BGR
    height, width = canvas.shape[:2]

    for index, ((x, y), state) in enumerate(zip(landmarks, visibility, strict=True)):
        px, py = round(float(x)), round(float(y))
        if not (0 <= px < width and 0 <= py < height):
            continue
        color = VISIBILITY_COLORS.get(int(state), (255, 255, 255))
        cv2.circle(canvas, (px, py), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, str(index), (px + 5, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (width - 1, 50), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, key, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    legend_x = max(12, width - 360)
    for offset, state in enumerate((HRFFAVisibility.VISIBLE, HRFFAVisibility.OCCLUDED, HRFFAVisibility.OUTSIDE_IMAGE)):
        x = legend_x + offset * 118
        color = VISIBILITY_COLORS[state.value]
        cv2.circle(canvas, (x, 25), 5, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, VISIBILITY_NAMES[state.value], (x + 9, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return canvas


def _save_contact_sheet(previews: list[tuple[str, np.ndarray]], output_path: Path, tile_size: int = 512, columns: int = 3) -> None:
    if not previews:
        return

    columns = min(columns, len(previews))
    rows = math.ceil(len(previews) / columns)
    sheet = np.zeros((rows * tile_size, columns * tile_size, 3), dtype=np.uint8)

    for index, (_key, image) in enumerate(previews):
        resized = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, columns)
        y0, x0 = row * tile_size, col * tile_size
        sheet[y0 : y0 + tile_size, x0 : x0 + tile_size] = resized

    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _save_shard_landmarks(
    output_dir: Path,
    shard_name: str,
    keys: list[str],
    landmarks: list[np.ndarray],
    visibility: list[np.ndarray],
    image_sizes: list[tuple[int, int]],
    *,
    source_dir: Path,
    scheme: HRFFAScheme,
    variant: HRFFAModelVariant,
    complete: bool,
    overwrite: bool,
) -> Path:
    stem = Path(shard_name).stem
    suffix = "" if complete else f".part-{len(keys):06d}"
    output_path = output_dir / f"{stem}{suffix}.npz"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"landmark 输出已存在: {output_path}；如需覆盖请传 --overwrite")

    np.savez_compressed(
        output_path,
        keys=np.asarray(keys, dtype=str),
        landmarks=np.stack(landmarks).astype(np.float32, copy=False),
        visibility=np.stack(visibility).astype(np.uint8, copy=False),
        image_sizes=np.asarray(image_sizes, dtype=np.int32),
        source_dir=np.asarray(str(source_dir)),
        shard=np.asarray(shard_name),
        scheme=np.asarray(scheme.value),
        variant=np.asarray(variant.value),
        complete=np.asarray(complete),
        visibility_names=np.asarray([VISIBILITY_NAMES[i] for i in range(3)]),
    )
    return output_path


def main() -> None:
    args = _parse_args()
    scheme = HRFFAScheme(args.scheme)
    variant = HRFFAModelVariant(args.variant)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但 torch.cuda.is_available() 为 False")

    output_dir = args.output_dir.resolve()
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_count > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    input_dir = args.input_dir.resolve(strict=True)
    shards = _discover_shards(input_dir, args.start_shard_index, args.num_shards)
    global_limit = None if args.process_all else args.limit

    print(f"input={input_dir}")
    print(f"scheme={scheme.value} variant={variant.value} device={device} batch_size={args.batch_size}")
    print(f"selected_shards={len(shards)} limit={'all' if global_limit is None else global_limit}")
    print(f"output={output_dir}")

    model = HRFFALandmarkModel(scheme=scheme, input_range=ImageInputRange.ZERO_TO_255, variant=variant, teacher_checkpoint=args.teacher_checkpoint).to(device).eval()

    total_processed = 0
    written_files: list[str] = []
    previews: list[tuple[str, np.ndarray]] = []

    for shard_path in shards:
        if global_limit is not None and total_processed >= global_limit:
            break

        shard_name = Path(shard_path).name
        shard_keys: list[str] = []
        shard_landmarks: list[np.ndarray] = []
        shard_visibility: list[np.ndarray] = []
        shard_sizes: list[tuple[int, int]] = []
        batch_keys: list[str] = []
        batch_images: list[Tensor] = []
        shard_complete = True

        print(f"processing {shard_name} ...")
        with tarfile.open(shard_path, mode="r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(".webp"):
                    continue
                if global_limit is not None and total_processed + len(batch_images) >= global_limit:
                    shard_complete = False
                    break

                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"无法读取 tar member: {shard_name}:{member.name}")
                image = _decode_webp(extracted.read())
                if image.ndim != 3 or image.shape[0] != 3 or image.shape[1] != image.shape[2]:
                    raise ValueError(f"HRFFA 需要正方形 RGB 图像，{shard_name}:{member.name} shape={tuple(image.shape)}")

                batch_keys.append(Path(member.name).stem)
                batch_images.append(image)
                if len(batch_images) >= args.batch_size:
                    results = _infer_batch(model, device, batch_keys, batch_images)
                    for key, batch_image, points, states in results:
                        height, width = batch_image.shape[-2:]
                        shard_keys.append(key)
                        shard_landmarks.append(points)
                        shard_visibility.append(states)
                        shard_sizes.append((height, width))
                        if len(previews) < args.preview_count:
                            annotated = _annotate_landmarks(batch_image, key, points, states)
                            preview_path = preview_dir / f"{key}.jpg"
                            cv2.imwrite(str(preview_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 96])
                            previews.append((key, annotated))
                    total_processed += len(batch_images)
                    batch_keys.clear()
                    batch_images.clear()

            results = _infer_batch(model, device, batch_keys, batch_images)
            for key, batch_image, points, states in results:
                height, width = batch_image.shape[-2:]
                shard_keys.append(key)
                shard_landmarks.append(points)
                shard_visibility.append(states)
                shard_sizes.append((height, width))
                if len(previews) < args.preview_count:
                    annotated = _annotate_landmarks(batch_image, key, points, states)
                    preview_path = preview_dir / f"{key}.jpg"
                    cv2.imwrite(str(preview_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 96])
                    previews.append((key, annotated))
            total_processed += len(batch_images)
            batch_keys.clear()
            batch_images.clear()

        if not shard_keys:
            continue

        output_path = _save_shard_landmarks(
            output_dir,
            shard_name,
            shard_keys,
            shard_landmarks,
            shard_visibility,
            shard_sizes,
            source_dir=input_dir,
            scheme=scheme,
            variant=variant,
            complete=shard_complete,
            overwrite=args.overwrite,
        )
        written_files.append(str(output_path))
        print(f"saved {len(shard_keys)} samples -> {output_path.name}")

    if previews:
        _save_contact_sheet(previews, preview_dir / "contact_sheet.jpg")

    manifest = {
        "input_dir": str(input_dir),
        "scheme": scheme.value,
        "variant": variant.value,
        "device": str(device),
        "processed": total_processed,
        "selected_shards": [Path(path).name for path in shards],
        "landmark_files": written_files,
        "preview_count": len(previews),
        "visibility": VISIBILITY_NAMES,
    }
    manifest_path = output_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"processed={total_processed}")
    print(f"manifest={manifest_path}")
    if previews:
        print(f"preview_dir={preview_dir}")
        print(f"contact_sheet={preview_dir / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
