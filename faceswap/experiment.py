"""训练 run 的目录、配置快照、latest 指针和恢复定位。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

RUN_CONFIG_FILE = "config.toml"
RUN_RESOLVED_CONFIG_FILE = "config.resolved.json"
RUN_METADATA_FILE = "metadata.json"
RUN_LATEST_FILE = "latest.json"
RESOLVED_CONFIG_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    samples: Path
    tensorboard: Path
    config: Path
    resolved_config: Path
    metadata: Path
    latest: Path

    @classmethod
    def from_root(cls, root: str | os.PathLike[str]) -> RunPaths:
        root_path = Path(root).resolve()
        return cls(
            root=root_path,
            checkpoints=root_path / "checkpoints",
            samples=root_path / "samples",
            tensorboard=root_path / "tensorboard",
            config=root_path / RUN_CONFIG_FILE,
            resolved_config=root_path / RUN_RESOLVED_CONFIG_FILE,
            metadata=root_path / RUN_METADATA_FILE,
            latest=root_path / RUN_LATEST_FILE,
        )


class RunLock:
    """避免两个训练进程同时写同一个 run。"""

    def __init__(self, paths: RunPaths):
        self.path = paths.root / ".run.lock"
        self._file: Any | None = None

    def __enter__(self) -> Self:
        lock_file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise RuntimeError(f"run 已被另一个训练进程占用：{self.path.parent}") from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} hostname={socket.gethostname()}\n")
        lock_file.flush()
        self._file = lock_file
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None) -> None:
        if self._file is not None:
            try:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            finally:
                self._file.close()
                self._file = None


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sanitize_name(name: str | None) -> str | None:
    if name is None:
        return None
    value = re.sub(r"[^0-9A-Za-z._-]+", "-", name.strip()).strip("-._")
    if not value:
        raise ValueError("--name 不能为空")
    return value[:80]


def create_run(
    runs_root: str | os.PathLike[str],
    source_config: str | os.PathLike[str],
    resolved_config: dict[str, Any],
    *,
    name: str | None,
    parent: dict[str, Any] | None = None,
) -> RunPaths:
    root = Path(runs_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    label = _sanitize_name(name)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    base_id = f"{timestamp}_{label}" if label else timestamp

    run_id = base_id
    suffix = 2
    while True:
        paths = RunPaths.from_root(root / run_id)
        try:
            paths.root.mkdir()
            break
        except FileExistsError:
            run_id = f"{base_id}-{suffix:02d}"
            suffix += 1

    try:
        for directory in (paths.checkpoints, paths.samples, paths.tensorboard):
            directory.mkdir()
        source_path = Path(source_config).resolve()
        paths.config.write_bytes(source_path.read_bytes())
        atomic_write_json(
            paths.resolved_config,
            {"format_version": RESOLVED_CONFIG_FORMAT_VERSION, "config": resolved_config},
        )
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        atomic_write_json(
            paths.metadata,
            {
                "run_id": run_id,
                "status": "created",
                "created_at": now,
                "updated_at": now,
                "config_sha256": config_sha256(resolved_config),
                "parent": parent,
            },
        )
    except BaseException:
        shutil.rmtree(paths.root, ignore_errors=True)
        raise
    return paths


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_metadata(paths: RunPaths) -> dict[str, Any]:
    metadata = read_json(paths.metadata)
    if not isinstance(metadata, dict):
        raise TypeError(f"无效的 metadata：{paths.metadata}")
    return metadata


def load_resolved_config(paths: RunPaths) -> dict[str, Any]:
    document = read_json(paths.resolved_config)
    if document["format_version"] != RESOLVED_CONFIG_FORMAT_VERSION:
        raise ValueError(f"不支持的 resolved config：{paths.resolved_config}")
    config = document["config"]
    if config_sha256(config) != load_metadata(paths)["config_sha256"]:
        raise ValueError(f"run 的 resolved config 已被修改：{paths.resolved_config}")
    return config


def update_metadata(paths: RunPaths, **changes: Any) -> None:
    metadata = load_metadata(paths)
    metadata.update(changes)
    metadata["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_write_json(paths.metadata, metadata)


def write_latest(paths: RunPaths, checkpoint: Path) -> None:
    atomic_write_json(paths.latest, {"checkpoint": checkpoint.name})


def _validate_run(paths: RunPaths) -> None:
    for path in (paths.config, paths.resolved_config, paths.metadata, paths.checkpoints):
        if not path.exists():
            raise ValueError(f"不是有效的训练 run：缺少 {path}")


def _checkpoint_from_latest(paths: RunPaths) -> Path:
    checkpoint = paths.checkpoints / read_json(paths.latest)["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"latest checkpoint 不存在：{checkpoint}")
    return checkpoint


def checkpoint_step_from_name(name: str) -> int:
    match = re.fullmatch(r"step_(\d+)\.pth", name)
    if match is None:
        raise ValueError(f"无效的 checkpoint 文件名：{name}")
    return int(match.group(1))


def resolve_branch_target(target: str | os.PathLike[str]) -> tuple[RunPaths, Path]:
    """目录使用 latest；显式 checkpoint 允许历史版本。"""
    requested = Path(target).resolve()
    if requested.is_dir():
        paths = RunPaths.from_root(requested)
        _validate_run(paths)
        return paths, _checkpoint_from_latest(paths)

    if not requested.is_file() or requested.parent.name != "checkpoints":
        raise ValueError(f"无效的 branch checkpoint：{requested}")
    paths = RunPaths.from_root(requested.parent.parent)
    _validate_run(paths)
    checkpoint_step_from_name(requested.name)
    return paths, requested


def resolve_resume_target(target: str | os.PathLike[str]) -> tuple[RunPaths, Path]:
    """Resume 只能继续 run 的 latest checkpoint。"""
    requested = Path(target).resolve()
    if requested.is_dir():
        paths = RunPaths.from_root(requested)
        _validate_run(paths)
        return paths, _checkpoint_from_latest(paths)

    if not requested.is_file() or requested.parent.name != "checkpoints":
        raise ValueError(f"无效的 resume checkpoint：{requested}")
    paths = RunPaths.from_root(requested.parent.parent)
    _validate_run(paths)
    latest = _checkpoint_from_latest(paths)
    if requested != latest:
        raise ValueError(f"--resume 只能继续 latest checkpoint：{latest.name}；历史 checkpoint 请使用 --branch-from")
    return paths, requested
