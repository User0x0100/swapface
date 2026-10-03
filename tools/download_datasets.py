#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

HF_ENDPOINTS = {
    "direct": "https://huggingface.co",
    "mirror": "https://hf-mirror.com",
}

DEFAULT_WORKERS = 24
DEFAULT_CHUNK_SIZE_MIB = 128
DEFAULT_CONNECT_TIMEOUT = 15
DEFAULT_MAX_TIME = 300
DEFAULT_SPEED_LIMIT = 1024
DEFAULT_SPEED_TIME = 30


@dataclass(frozen=True, slots=True)
class Artifact:
    name: str
    repo_id: str
    revision: str
    filename: str
    output: Path
    size: int
    sha256: str


DATASETS: dict[str, tuple[Artifact, ...]] = {
    "lpff": (
        Artifact(
            name="LPFF",
            repo_id="onethousand/LPFF",
            revision="f598e6ea5c871f9b557d11b07d8d4248f127ea08",
            filename="LPFF-dataset/crop/stylegan.zip",
            output=Path("lpff/stylegan.zip"),
            size=20_634_073_128,
            sha256="f6481682f744b2b493e8c091d84998e8462b01c2370810ae81da46b6955f8b4a",
        ),
    ),
    "ffhq": (
        Artifact(
            name="FFHQ-1",
            repo_id="Iceclear/FFHQ-HQ1024",
            revision="93e4ac2b99e06b492efcee5d2182bdca6b6f4671",
            filename="FFHQ-1024-1.zip",
            output=Path("ffhq/FFHQ-1024-1.zip"),
            size=46_018_727_889,
            sha256="ef8567e3fce296f40cfddcc6abe130939782b2bfa81c8ed7936e996be8273c52",
        ),
        Artifact(
            name="FFHQ-2",
            repo_id="Iceclear/FFHQ-HQ1024",
            revision="93e4ac2b99e06b492efcee5d2182bdca6b6f4671",
            filename="FFHQ-1024-2.zip",
            output=Path("ffhq/FFHQ-1024-2.zip"),
            size=49_709_577_843,
            sha256="1009dd36ce01a15ed2724233c777f643af9f0f63f884424b4ccee74c58a12de2",
        ),
    ),
    "fairface": (
        Artifact(
            name="FairFace-0.25-0",
            repo_id="HuggingFaceM4/FairFace",
            revision="54d573cdb8b5af490ba8da9da2799628f6e5c496",
            filename="0.25/train-00000-of-00002-d405faba4f4b9b85.parquet",
            output=Path("fairface/part-00000.parquet"),
            size=250_030_031,
            sha256="acebfc4a735050d0ee618b3c690ad91e1b11bcc25151f398f06990bc15e99d06",
        ),
        Artifact(
            name="FairFace-0.25-1",
            repo_id="HuggingFaceM4/FairFace",
            revision="54d573cdb8b5af490ba8da9da2799628f6e5c496",
            filename="0.25/train-00001-of-00002-dd3cb68164727418.parquet",
            output=Path("fairface/part-00001.parquet"),
            size=250_217_804,
            sha256="e8ec1fb27f0745c166b218708bf4923811cd41d85cb998f09f8f624b0eb40326",
        ),
        Artifact(
            name="FairFace-0.25-2",
            repo_id="HuggingFaceM4/FairFace",
            revision="54d573cdb8b5af490ba8da9da2799628f6e5c496",
            filename="0.25/validation-00000-of-00001-951dbd63c8724ee1.parquet",
            output=Path("fairface/part-00002.parquet"),
            size=63_189_799,
            sha256="d0cfac2888356b3d3fa3d837bc30961d8c860934e4c5df862e8cb87812bfa137",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download project datasets with resumable parallel HTTP ranges.")
    parser.add_argument("dataset", nargs="?", choices=(*DATASETS, "all"), default="all")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/downloads"),
        help="Output root. Default: datasets/downloads.",
    )
    parser.add_argument(
        "--source",
        choices=tuple(HF_ENDPOINTS),
        help="Hugging Face endpoint. Omit to use HF_ENDPOINT, otherwise hf-mirror.com.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel HTTP range workers per file. Default: {DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--chunk-size-mib",
        type=int,
        default=DEFAULT_CHUNK_SIZE_MIB,
        help=f"Persistent range chunk size in MiB. Default: {DEFAULT_CHUNK_SIZE_MIB}.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Retries per range after curl errors; 0 retries indefinitely. Default: 0.",
    )
    parser.add_argument("--force", action="store_true", help="Discard an existing destination/partial state and redownload.")
    parser.add_argument("--list", action="store_true", help="Print selected artifacts without downloading.")
    return parser.parse_args()


def selected_artifacts(dataset: str) -> list[Artifact]:
    if dataset == "all":
        return [artifact for artifacts in DATASETS.values() for artifact in artifacts]
    return list(DATASETS[dataset])


def selected_endpoint(source: str | None) -> str:
    if source is not None:
        return HF_ENDPOINTS[source]
    return os.environ.get("HF_ENDPOINT", HF_ENDPOINTS["mirror"]).rstrip("/")


def artifact_url(artifact: Artifact, endpoint: str) -> str:
    filename = quote(artifact.filename, safe="/")
    return f"{endpoint.rstrip('/')}/datasets/{artifact.repo_id}/resolve/{artifact.revision}/{filename}"


@contextmanager
def download_lock(destination: Path):
    resolved = destination.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lock_path = resolved.parent / f".{resolved.name}.download.lock"

    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"artifact is already being downloaded: {resolved}") from error
        try:
            yield lock_file.fileno()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _chunk_ranges(size: int, chunk_size: int) -> list[tuple[int, int, int]]:
    chunks: list[tuple[int, int, int]] = []
    start = 0
    index = 0
    while start < size:
        end = min(size - 1, start + chunk_size - 1)
        chunks.append((index, start, end))
        index += 1
        start = end + 1
    return chunks


def _write_manifest(
    part_dir: Path,
    *,
    url: str,
    size: int,
    sha256: str,
    chunk_size: int,
    force: bool,
) -> None:
    manifest_path = part_dir / "manifest.json"
    expected = {"url": url, "size": size, "sha256": sha256, "chunk_size": chunk_size}

    if force and part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current != expected:
            raise RuntimeError(
                f"partial download settings changed for {part_dir}; use the original source/chunk size or pass --force"
            )
        return

    if any(part_dir.glob("*.part")):
        raise RuntimeError(f"partial directory has no manifest: {part_dir}; pass --force to restart it")

    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)


def _parse_final_headers(path: Path) -> tuple[int | None, dict[str, str]]:
    status: int | None = None
    headers: dict[str, str] = {}
    for raw_line in path.read_text(encoding="iso-8859-1", errors="replace").splitlines():
        line = raw_line.strip("\r")
        if line.startswith("HTTP/"):
            parts = line.split(None, 2)
            status = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None
            headers = {}
        elif ":" in line and status is not None:
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
    return status, headers


def _verified_path(part_path: Path) -> Path:
    return part_path.with_name(part_path.name + ".verified")


def _write_verified_size(part_path: Path, size: int) -> None:
    with part_path.open("rb") as part_file:
        os.fsync(part_file.fileno())

    verified_path = _verified_path(part_path)
    temporary = verified_path.with_suffix(verified_path.suffix + ".tmp")
    temporary.write_text(f"{size}\n", encoding="ascii")
    os.replace(temporary, verified_path)


def _normalize_part(part_path: Path, expected: int) -> int:
    verified_path = _verified_path(part_path)
    if not part_path.exists():
        verified_path.unlink(missing_ok=True)
        return 0

    actual = part_path.stat().st_size
    try:
        verified = int(verified_path.read_text(encoding="ascii").strip())
    except (FileNotFoundError, ValueError):
        verified = 0

    if verified < 0 or verified > expected or verified > actual:
        verified = 0
        verified_path.unlink(missing_ok=True)

    if actual != verified:
        _truncate(part_path, verified)
    if verified == 0:
        verified_path.unlink(missing_ok=True)
    return verified


def _curl_range(
    url: str,
    start: int,
    end: int,
    part_path: Path,
    stop_event: threading.Event,
    active_processes: set[subprocess.Popen[bytes]],
    active_lock: threading.Lock,
    lock_fd: int,
) -> tuple[int, bytes, int | None, str | None]:
    header_path = part_path.with_name(part_path.name + ".headers")
    header_path.unlink(missing_ok=True)
    command = [
        "curl",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--header",
        "Accept-Encoding: identity",
        "--range",
        f"{start}-{end}",
        "--connect-timeout",
        str(DEFAULT_CONNECT_TIMEOUT),
        "--max-time",
        str(DEFAULT_MAX_TIME),
        "--speed-limit",
        str(DEFAULT_SPEED_LIMIT),
        "--speed-time",
        str(DEFAULT_SPEED_TIME),
        "--max-filesize",
        str(end - start + 1),
        "--dump-header",
        str(header_path),
        "--output",
        "-",
        url,
    ]

    try:
        with part_path.open("ab") as output:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.PIPE,
                pass_fds=(lock_fd,),
            )
            with active_lock:
                active_processes.add(process)
            try:
                if stop_event.is_set() and process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                _, stderr = process.communicate()
            finally:
                with active_lock:
                    active_processes.discard(process)

        status, headers = _parse_final_headers(header_path) if header_path.exists() else (None, {})
        return process.returncode, stderr or b"", status, headers.get("content-range")
    finally:
        header_path.unlink(missing_ok=True)


def _truncate(path: Path, size: int) -> None:
    if not path.exists():
        return
    with path.open("r+b") as file:
        file.truncate(size)


def _download_chunk(
    *,
    url: str,
    total_size: int,
    part_path: Path,
    start: int,
    end: int,
    retries: int,
    stop_event: threading.Event,
    active_processes: set[subprocess.Popen[bytes]],
    active_lock: threading.Lock,
    lock_fd: int,
) -> None:
    expected = end - start + 1
    failures = 0

    while not stop_event.is_set():
        have = _normalize_part(part_path, expected)
        if have == expected:
            return

        request_start = start + have
        returncode, stderr, status, content_range = _curl_range(
            url,
            request_start,
            end,
            part_path,
            stop_event,
            active_processes,
            active_lock,
            lock_fd,
        )
        new_size = part_path.stat().st_size if part_path.exists() else 0
        expected_content_range = f"bytes {request_start}-{end}/{total_size}"
        valid_response = status == 206 and content_range == expected_content_range

        if not valid_response or new_size < have or new_size > expected:
            _truncate(part_path, have)
            new_size = have
        elif new_size > have:
            _write_verified_size(part_path, new_size)

        if stop_event.is_set():
            return
        if valid_response and new_size == expected:
            return

        failures += 1
        if retries > 0 and failures >= retries:
            message = stderr.decode(errors="replace").strip()
            if not valid_response:
                message = (
                    f"invalid Range response: status={status}, Content-Range={content_range!r}; {message}"
                ).strip()
            elif returncode == 0:
                message = f"short Range response: {new_size} != {expected}"
            raise RuntimeError(f"range {start}-{end} failed after {failures} attempts: {message}")
        stop_event.wait(min(5.0, 0.5 * failures))


def _terminate_processes(
    active_processes: set[subprocess.Popen[bytes]],
    active_lock: threading.Lock,
) -> None:
    with active_lock:
        processes = tuple(active_processes)
    for process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    time.sleep(0.1)
    for process in processes:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass


def _parts_size(part_dir: Path) -> int:
    return sum(path.stat().st_size for path in part_dir.glob("*.part") if path.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assemble(
    destination: Path,
    part_dir: Path,
    chunks: list[tuple[int, int, int]],
    expected_size: int,
    expected_sha256: str,
) -> None:
    temporary = destination.with_name(destination.name + ".assembling")
    temporary.unlink(missing_ok=True)

    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for index, start, end in chunks:
            part_path = part_dir / f"{index:05d}.part"
            expected = end - start + 1
            actual = part_path.stat().st_size if part_path.exists() else 0
            verified = _normalize_part(part_path, expected)
            if actual != expected or verified != expected:
                raise RuntimeError(
                    f"incomplete or unverified range {index}: size={actual}, verified={verified}, expected={expected}"
                )
            with part_path.open("rb") as source:
                while data := source.read(16 * 1024 * 1024):
                    output.write(data)
                    digest.update(data)
        output.flush()
        os.fsync(output.fileno())

    actual_size = temporary.stat().st_size
    if actual_size != expected_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"assembled size mismatch: {actual_size} != {expected_size}")

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"assembled SHA-256 mismatch: {actual_sha256} != {expected_sha256}; pass --force to redownload"
        )
    os.replace(temporary, destination)


def _download_artifact_locked(
    artifact: Artifact,
    destination: Path,
    endpoint: str,
    *,
    workers: int,
    chunk_size: int,
    retries: int,
    force_download: bool,
    lock_fd: int,
) -> Path:
    part_dir = destination.with_name(destination.name + ".parts")

    if force_download and (destination.exists() or destination.is_symlink()):
        destination.unlink()

    if destination.exists():
        actual_size = destination.stat().st_size
        actual_sha256 = _sha256(destination) if actual_size == artifact.size else ""
        if actual_size == artifact.size and actual_sha256 == artifact.sha256:
            shutil.rmtree(part_dir, ignore_errors=True)
            destination.with_name(destination.name + ".assembling").unlink(missing_ok=True)
            print(f"[{artifact.name}] COMPLETE: {destination} ({actual_size} bytes, already present)")
            return destination
        raise RuntimeError(
            f"existing destination failed integrity check: {destination}; pass --force to replace it"
        )
    if destination.is_symlink():
        raise RuntimeError(f"broken destination symlink: {destination}; pass --force to replace it")

    url = artifact_url(artifact, endpoint)
    _write_manifest(
        part_dir,
        url=url,
        size=artifact.size,
        sha256=artifact.sha256,
        chunk_size=chunk_size,
        force=force_download,
    )
    chunks = _chunk_ranges(artifact.size, chunk_size)
    for index, start, end in chunks:
        _normalize_part(part_dir / f"{index:05d}.part", end - start + 1)

    initial_size = _parts_size(part_dir)
    print(
        f"[{artifact.name}] START: {artifact.size / 1024**3:.2f} GiB, "
        f"workers={workers}, chunk={chunk_size / 1024**2:.0f} MiB, "
        f"resume={initial_size / 1024**3:.2f} GiB"
    )

    started = time.monotonic()
    last_time = started
    last_size = initial_size
    last_print = started
    stop_event = threading.Event()
    active_processes: set[subprocess.Popen[bytes]] = set()
    active_lock = threading.Lock()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    pending: set[concurrent.futures.Future[None]] = set()

    try:
        pending = {
            pool.submit(
                _download_chunk,
                url=url,
                total_size=artifact.size,
                part_path=part_dir / f"{index:05d}.part",
                start=start,
                end=end,
                retries=retries,
                stop_event=stop_event,
                active_processes=active_processes,
                active_lock=active_lock,
                lock_fd=lock_fd,
            )
            for index, start, end in chunks
        }

        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=10.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                future.result()

            now = time.monotonic()
            if now - last_print >= 10.0 or not pending:
                current_size = _parts_size(part_dir)
                recent_speed = (current_size - last_size) / max(now - last_time, 1e-6) / 1024**2
                print(
                    f"[{artifact.name}] {current_size / artifact.size * 100:5.1f}% "
                    f"({current_size / 1024**3:.2f}/{artifact.size / 1024**3:.2f} GiB) "
                    f"{recent_speed:.1f} MiB/s"
                )
                last_size = current_size
                last_time = now
                last_print = now
    except KeyboardInterrupt:
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            stop_event.set()
            _terminate_processes(active_processes, active_lock)
            pool.shutdown(wait=True, cancel_futures=True)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
        raise
    except BaseException:
        stop_event.set()
        _terminate_processes(active_processes, active_lock)
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)

    _assemble(destination, part_dir, chunks, artifact.size, artifact.sha256)
    shutil.rmtree(part_dir)
    elapsed = time.monotonic() - started
    average_speed = (artifact.size - initial_size) / max(elapsed, 1e-6) / 1024**2
    print(f"[{artifact.name}] COMPLETE: {destination} ({artifact.size} bytes, avg {average_speed:.1f} MiB/s)")
    return destination


def download_artifact(
    artifact: Artifact,
    root: Path,
    endpoint: str,
    *,
    workers: int,
    chunk_size: int,
    retries: int,
    force_download: bool,
) -> Path:
    if workers <= 0:
        raise ValueError(f"workers must be > 0, got {workers}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if retries < 0:
        raise ValueError(f"retries must be >= 0, got {retries}")
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required for dataset downloads")

    destination = root / artifact.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with download_lock(destination) as lock_fd:
        return _download_artifact_locked(
            artifact,
            destination,
            endpoint,
            workers=workers,
            chunk_size=chunk_size,
            retries=retries,
            force_download=force_download,
            lock_fd=lock_fd,
        )


def main() -> None:
    args = parse_args()
    artifacts = selected_artifacts(args.dataset)
    endpoint = selected_endpoint(args.source)
    chunk_size = args.chunk_size_mib * 1024 * 1024

    if args.list:
        for artifact in artifacts:
            print(
                f"{artifact.name}\t{artifact.repo_id}\t{artifact.revision}\t"
                f"{artifact.filename}\t{artifact.output}\t{artifact.size}\t{artifact.sha256}"
            )
        return

    try:
        for artifact in artifacts:
            download_artifact(
                artifact,
                args.root,
                endpoint,
                workers=args.workers,
                chunk_size=chunk_size,
                retries=args.retries,
                force_download=args.force,
            )
    except KeyboardInterrupt:
        print("download interrupted; partial ranges were preserved", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
