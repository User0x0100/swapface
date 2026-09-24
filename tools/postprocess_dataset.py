#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MANIFEST_NAME = ".postprocess.json"
VERSION = 1

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert images in ZIP archives to a flat RGB PNG dataset.")
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 8),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.size <= 0:
        raise ValueError("--size must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    for archive in args.archives:
        if not archive.is_file():
            raise FileNotFoundError(archive)


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


def main() -> None:
    args = parse_args()
    validate_args(args)

    items, archives = scan_archives(args.archives)
    if not items:
        raise RuntimeError("no images found")

    print(f"found_images={len(items)}", flush=True)
    run(args, items, archives)


if __name__ == "__main__":
    main()
