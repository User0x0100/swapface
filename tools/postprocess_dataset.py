#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MANIFEST_NAME = ".postprocess.json"
VERSION = 1
FAIRFACE_VERSION = 2
FAIRFACE_SIZE = 112
FAIRFACE_RACE_NAMES = {
    0: "East Asian",
    1: "Indian",
    2: "Black",
    3: "White",
    4: "Middle Eastern",
    5: "Latino_Hispanic",
    6: "Southeast Asian",
}
FAIRFACE_RACE_GROUPS = {
    0: "asian",
    1: "indian",
    2: "black",
    3: "white",
    4: "middle_eastern",
    5: "latino",
    6: "asian",
}
FAIRFACE_GROUPS = tuple(sorted(set(FAIRFACE_RACE_GROUPS.values())))
FAIRFACE_SHARD_PATTERN = re.compile(r"^(?P<split>.+?)-(?P<shard>\d+)-of-(?P<count>\d+)-")

_WORKER_ARCHIVE: str | None = None
_WORKER_ZIP: zipfile.ZipFile | None = None


@dataclass(frozen=True, slots=True)
class ImageItem:
    archive: Path
    member: str
    output_name: str


@dataclass(frozen=True, slots=True)
class ArchiveIdentity:
    path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class FairFaceItem:
    key: str
    group: str
    image_bytes: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess project datasets. ZIP mode produces a flat RGB PNG dataset; FairFace mode produces race-grouped ArcFace-112 source faces."
    )
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("zip", "fairface"), default="zip")
    parser.add_argument("--size", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 8),
    )
    parser.add_argument("--batch-size", type=int, default=128, help="FairFace RetinaFace inference batch size.")
    parser.add_argument("--device", default="cuda", help="FairFace alignment device. Default: cuda.")
    parser.add_argument("--confidence", type=float, default=0.9, help="FairFace RetinaFace confidence threshold.")
    parser.add_argument("--mobilenet", action="store_true", help="Use MobileNet RetinaFace instead of the default ResNet-50.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.size is None:
        args.size = FAIRFACE_SIZE if args.format == "fairface" else 256
    if args.size <= 0:
        raise ValueError("--size must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be in (0, 1]")
    if args.format == "fairface" and args.size != FAIRFACE_SIZE:
        raise ValueError(f"FairFace output is fixed to ArcFace {FAIRFACE_SIZE}x{FAIRFACE_SIZE}; omit --size or set --size {FAIRFACE_SIZE}")
    for archive in args.archives:
        if not archive.is_file():
            raise FileNotFoundError(archive)
        if args.format == "fairface" and archive.suffix.lower() != ".parquet":
            raise ValueError(f"FairFace input must be a Parquet file: {archive}")


def archive_identity(path: Path) -> ArchiveIdentity:
    resolved = path.resolve()
    stat = resolved.stat()
    return ArchiveIdentity(
        path=str(resolved),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def scan_archives(archives: list[Path]) -> tuple[list[ImageItem], list[ArchiveIdentity]]:
    items: list[ImageItem] = []
    identities: list[ArchiveIdentity] = []
    output_names: dict[str, tuple[Path, str]] = {}

    for archive in archives:
        identities.append(archive_identity(archive))

        with zipfile.ZipFile(archive) as source:
            member_names: set[str] = set()
            for info in source.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in IMAGE_EXTS:
                    continue

                if info.filename in member_names:
                    raise RuntimeError(f"duplicate ZIP member: {archive}::{info.filename}")
                member_names.add(info.filename)

                output_name = Path(info.filename).stem + ".png"
                key = output_name.casefold()
                if key in output_names:
                    previous_archive, previous_member = output_names[key]
                    raise RuntimeError(f"flat-name collision: {output_name}: {previous_archive}::{previous_member} vs {archive}::{info.filename}")
                output_names[key] = (archive, info.filename)
                items.append(
                    ImageItem(
                        archive=archive,
                        member=info.filename,
                        output_name=output_name,
                    )
                )

    return items, identities


def manifest_value(
    size: int,
    archives: list[ArchiveIdentity],
) -> dict[str, object]:
    return {
        "version": VERSION,
        "size": size,
        "archives": [asdict(archive) for archive in archives],
    }


def fairface_manifest_value(args: argparse.Namespace, archives: list[ArchiveIdentity]) -> dict[str, object]:
    return {
        "version": FAIRFACE_VERSION,
        "format": "fairface-arcface112",
        "size": FAIRFACE_SIZE,
        "detector": "mobilenet0.25" if args.mobilenet else "resnet50",
        "confidence": args.confidence,
        "race_names": {str(key): value for key, value in FAIRFACE_RACE_NAMES.items()},
        "race_groups": {str(key): value for key, value in FAIRFACE_RACE_GROUPS.items()},
        "archives": [asdict(archive) for archive in archives],
    }


def write_json_atomic(path: Path, value: object) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def output_lock(output: Path):
    resolved = output.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lock_path = resolved.parent / f".{resolved.name}.postprocess.lock"

    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"output directory is already being processed: {resolved}") from error
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def prepare_output(
    output: Path,
    size: int,
    archives: list[ArchiveIdentity],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST_NAME
    expected = manifest_value(size, archives)

    if manifest_path.exists():
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid manifest: {manifest_path}") from error

        if actual != expected:
            raise RuntimeError(f"output directory belongs to a different postprocess run: {manifest_path}")

        for temporary in output.glob("*.part"):
            temporary.unlink()
        return

    for temporary in output.glob(f"{MANIFEST_NAME}.*.tmp"):
        temporary.unlink()

    if any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty but has no {MANIFEST_NAME}: {output}")

    write_json_atomic(manifest_path, expected)


def prepare_fairface_output(args: argparse.Namespace, archives: list[ArchiveIdentity]) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST_NAME
    expected = fairface_manifest_value(args, archives)

    if manifest_path.exists():
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid manifest: {manifest_path}") from error
        if actual != expected:
            raise RuntimeError(f"output directory belongs to a different postprocess run: {manifest_path}")
        for temporary in output.rglob("*.part"):
            temporary.unlink()
    else:
        for temporary in output.glob(f"{MANIFEST_NAME}.*.tmp"):
            temporary.unlink()
        if any(output.iterdir()):
            raise RuntimeError(f"output directory is not empty but has no {MANIFEST_NAME}: {output}")
        write_json_atomic(manifest_path, expected)

    for group in FAIRFACE_GROUPS:
        (output / group).mkdir(exist_ok=True)


def worker_zip(archive: str) -> zipfile.ZipFile:
    global _WORKER_ARCHIVE, _WORKER_ZIP
    if _WORKER_ARCHIVE != archive or _WORKER_ZIP is None:
        if _WORKER_ZIP is not None:
            _WORKER_ZIP.close()
        _WORKER_ZIP = zipfile.ZipFile(archive)
        _WORKER_ARCHIVE = archive
    return _WORKER_ZIP


def process_one(task: tuple[str, str, str, int]) -> str:
    archive, member, destination, size = task
    target = Path(destination)
    fd, temporary = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".part",
        dir=target.parent,
    )

    try:
        with worker_zip(archive).open(member) as source, Image.open(source) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened)
            if image.mode != "RGB":
                image = image.convert("RGB")
            if image.width != image.height:
                raise RuntimeError(f"expected square image, got {image.width}x{image.height}")
            if image.size != (size, size):
                image = image.resize(
                    (size, size),
                    Image.Resampling.LANCZOS,
                    reducing_gap=3.0,
                )

            with os.fdopen(fd, "wb") as output:
                image.save(output, format="PNG", compress_level=1)

        os.replace(temporary, target)
        return target.name
    except BaseException as error:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError(f"failed to process {archive}::{member} -> {target}") from error


def run(
    args: argparse.Namespace,
    items: list[ImageItem],
    archives: list[ArchiveIdentity],
) -> None:
    with output_lock(args.output):
        prepare_output(args.output, args.size, archives)

        tasks = []
        existing = 0
        for item in items:
            destination = args.output / item.output_name
            if destination.exists():
                if not destination.is_file():
                    raise RuntimeError(f"existing output is not a regular file: {destination}")
                existing += 1
                continue

            tasks.append((
                str(item.archive),
                item.member,
                str(destination),
                args.size,
            ))

        processed = 0
        with Pool(processes=args.workers) as pool:
            for _ in pool.imap_unordered(process_one, tasks, chunksize=1):
                processed += 1
                if processed % 500 == 0 or processed == len(tasks):
                    print(
                        f"processed={processed}/{len(tasks)} existing={existing}",
                        flush=True,
                    )

        print(f"COMPLETE output={args.output} images={len(items)} existing={existing} processed={processed}")


def fairface_item_key(parquet: Path, row_index: int) -> str:
    match = FAIRFACE_SHARD_PATTERN.match(parquet.name)
    if match is None:
        prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", parquet.stem)
    else:
        prefix = f"{match.group('split')}_{int(match.group('shard')):02d}"
    return f"{prefix}_{row_index:06d}"


def fairface_group(race: int) -> str:
    try:
        return FAIRFACE_RACE_GROUPS[int(race)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"unsupported FairFace race label: {race!r}") from error


def fairface_image_bytes(value: object) -> bytes:
    if isinstance(value, dict):
        data = value.get("bytes")
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
        path = value.get("path")
        if isinstance(path, str) and Path(path).is_file():
            return Path(path).read_bytes()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise ValueError(f"unsupported FairFace image value: {type(value).__name__}")


def iter_fairface_batches(parquets: list[Path], batch_size: int):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("FairFace processing requires pyarrow; run uv sync for the project environment") from error

    for parquet in parquets:
        source = pq.ParquetFile(parquet)
        row_index = 0
        for batch in source.iter_batches(batch_size=batch_size, columns=("image", "race")):
            images = batch.column(0).to_pylist()
            races = batch.column(1).to_pylist()
            items = []
            for image_value, race in zip(images, races, strict=True):
                items.append(
                    FairFaceItem(
                        key=fairface_item_key(parquet, row_index),
                        group=fairface_group(race),
                        image_bytes=fairface_image_bytes(image_value),
                    )
                )
                row_index += 1
            yield items


def decode_fairface_image(item: FairFaceItem) -> np.ndarray:
    with Image.open(io.BytesIO(item.image_bytes)) as opened:
        opened.load()
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if image.width != image.height:
            raise RuntimeError(f"FairFace image {item.key} is not square: {image.width}x{image.height}")
        return np.asarray(image, dtype=np.uint8).copy()


def save_fairface_image(task: tuple[Path, np.ndarray]) -> None:
    destination, array = task
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".part",
        dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "wb") as output:
            Image.fromarray(array, mode="RGB").save(output, format="PNG", compress_level=1)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_failed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.split("\t", 1)[0] for line in path.read_text(encoding="utf-8").splitlines() if line}


def run_fairface(args: argparse.Namespace) -> None:
    import torch

    import torch.nn.functional as F

    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from misc.face_alignment import make_alignment_grid_theta
    from misc.models.id_encoder import get_alignment_template
    from misc.models.retinaface import RetinaFace

    archives = [archive_identity(path) for path in args.archives]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA/ROCm device requested but torch.cuda.is_available() is false: {args.device}")

    with output_lock(args.output):
        prepare_fairface_output(args, archives)
        failed_path = args.output / ".failed.txt"
        failed_keys = load_failed_keys(failed_path)

        detector = RetinaFace(use_mobilenet=args.mobilenet).to(device=device).eval()
        template = torch.as_tensor(get_alignment_template(FAIRFACE_SIZE), device=device, dtype=torch.float32)

        existing = 0
        processed = 0
        failed = len(failed_keys)
        seen = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor, failed_path.open("a", encoding="utf-8") as failure_log:
            for items in iter_fairface_batches(args.archives, args.batch_size):
                seen += len(items)
                pending: list[FairFaceItem] = []
                for item in items:
                    destination = args.output / item.group / f"{item.key}.png"
                    if destination.exists():
                        if not destination.is_file():
                            raise RuntimeError(f"existing output is not a regular file: {destination}")
                        existing += 1
                    elif item.key not in failed_keys:
                        pending.append(item)

                if not pending:
                    continue

                arrays = list(executor.map(decode_fairface_image, pending))
                shapes = {array.shape for array in arrays}
                if len(shapes) != 1:
                    raise RuntimeError(f"FairFace batch contains inconsistent image shapes: {sorted(shapes)}")

                batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32, non_blocking=True)
                with torch.inference_mode():
                    detections, lengths = detector.detect(
                        batch,
                        confidence_threshold=args.confidence,
                        iou_threshold=0.2,
                    )

                    selected_indices: list[int] = []
                    selected_landmarks = []
                    offset = 0
                    for index, count in enumerate(lengths):
                        per_image = detections[offset : offset + count]
                        offset += count
                        if count == 0:
                            key = pending[index].key
                            failed_keys.add(key)
                            failure_log.write(f"{key}\tno_face\n")
                            failed += 1
                            continue
                        best = per_image[per_image[:, 0].argmax()]
                        selected_indices.append(index)
                        selected_landmarks.append(best[-10:].reshape(5, 2))

                    if selected_indices:
                        landmarks = torch.stack(selected_landmarks)
                        selected = batch[selected_indices]
                        theta = make_alignment_grid_theta(landmarks, template, selected.shape[-2:], FAIRFACE_SIZE)
                        grid = F.affine_grid(
                            theta,
                            (selected.shape[0], selected.shape[1], FAIRFACE_SIZE, FAIRFACE_SIZE),
                            align_corners=False,
                        )
                        aligned = F.grid_sample(selected, grid, mode="bilinear", align_corners=False)
                        aligned = aligned.clamp_(0.0, 255.0).round_().to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                    else:
                        aligned = np.empty((0, FAIRFACE_SIZE, FAIRFACE_SIZE, 3), dtype=np.uint8)

                save_tasks = []
                for output_image, index in zip(aligned, selected_indices, strict=True):
                    item = pending[index]
                    save_tasks.append((args.output / item.group / f"{item.key}.png", output_image))
                list(executor.map(save_fairface_image, save_tasks))
                processed += len(save_tasks)
                failure_log.flush()

                if seen % 2048 < len(items) or processed + existing + failed == seen:
                    print(
                        f"seen={seen} processed={processed} existing={existing} failed={failed}",
                        flush=True,
                    )

        print(
            f"COMPLETE output={args.output} seen={seen} processed={processed} existing={existing} failed={failed} groups={','.join(FAIRFACE_GROUPS)}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.format == "fairface":
        run_fairface(args)
        return

    items, archives = scan_archives(args.archives)
    if not items:
        raise RuntimeError("no images found")

    print(f"found_images={len(items)}", flush=True)
    run(args, items, archives)


if __name__ == "__main__":
    main()
