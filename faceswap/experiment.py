"""训练 run 的文件布局、原子元数据写入与恢复定位。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
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
    """对单个 run 持有进程级 advisory lock，进程退出后由内核自动释放。"""

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
        lock_file.write(f"pid={os.getpid()} hostname={socket.gethostname()} acquired_at={datetime.now().astimezone().isoformat(timespec='seconds')}\n")
        lock_file.flush()
        self._file = lock_file
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def _sanitize_name(name: str | None) -> str | None:
    if name is None:
        return None
    value = re.sub(r"[^0-9A-Za-z._-]+", "-", name.strip()).strip("-._")
    if not value:
        raise ValueError("--name 必须至少包含一个字母、数字或可保留的 ._- 字符")
    return value[:80]


def _git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _base_metadata(run_id: str, project_root: Path, source_config: Path, digest: str) -> dict[str, Any]:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "format_version": 1,
        "run_id": run_id,
        "status": "created",
        "created_at": now,
        "updated_at": now,
        "config_sha256": digest,
        "config_source": str(source_config.resolve()),
        "git": _git_metadata(project_root),
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        },
        "resume_history": [],
    }


def create_run(
    runs_root: str | os.PathLike[str],
    source_config: str | os.PathLike[str],
    resolved_config: dict[str, Any],
    *,
    name: str | None,
    project_root: str | os.PathLike[str],
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
            paths.root.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            run_id = f"{base_id}-{suffix:02d}"
            suffix += 1

    try:
        for directory in (paths.checkpoints, paths.samples, paths.tensorboard):
            directory.mkdir()

        source_path = Path(source_config).resolve()
        paths.config.write_bytes(source_path.read_bytes())
        resolved_document = {
            "format_version": RESOLVED_CONFIG_FORMAT_VERSION,
            "config": resolved_config,
        }
        atomic_write_json(paths.resolved_config, resolved_document)
        digest = config_sha256(resolved_config)
        atomic_write_json(paths.metadata, _base_metadata(run_id, Path(project_root).resolve(), source_path, digest))
    except BaseException:
        shutil.rmtree(paths.root, ignore_errors=True)
        raise
    return paths


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_resolved_config(paths: RunPaths) -> dict[str, Any]:
    try:
        document = read_json(paths.resolved_config)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"run 缺少 {RUN_RESOLVED_CONFIG_FILE}：{paths.root}") from exc
    if not isinstance(document, dict) or document.get("format_version") != RESOLVED_CONFIG_FORMAT_VERSION or not isinstance(document.get("config"), dict):
        raise ValueError(f"无效的 {RUN_RESOLVED_CONFIG_FILE}：{paths.resolved_config}")
    config = document["config"]
    metadata = read_json(paths.metadata)
    expected_digest = metadata.get("config_sha256") if isinstance(metadata, dict) else None
    actual_digest = config_sha256(config)
    if expected_digest != actual_digest:
        raise ValueError(f"run 配置摘要不匹配：metadata={expected_digest!r}, actual={actual_digest}")
    return config


def update_metadata(paths: RunPaths, **changes: Any) -> None:
    metadata = read_json(paths.metadata)
    if not isinstance(metadata, dict):
        raise TypeError(f"无效的 run metadata：{paths.metadata}")
    metadata.update(changes)
    metadata["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_write_json(paths.metadata, metadata)


def append_resume_event(paths: RunPaths, checkpoint: Path) -> None:
    metadata = read_json(paths.metadata)
    if not isinstance(metadata, dict):
        raise TypeError(f"无效的 run metadata：{paths.metadata}")
    history = metadata.setdefault("resume_history", [])
    if not isinstance(history, list):
        raise TypeError(f"metadata.resume_history 必须是数组：{paths.metadata}")
    history.append({
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint": checkpoint.name,
    })
    metadata["status"] = "running"
    metadata.pop("error", None)
    metadata["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_write_json(paths.metadata, metadata)


def write_latest(paths: RunPaths, checkpoint: Path, completed_step: int) -> None:
    atomic_write_json(
        paths.latest,
        {
            "step": int(completed_step),
            "checkpoint": checkpoint.name,
        },
    )


def _validate_run(paths: RunPaths) -> None:
    missing = [path.name for path in (paths.config, paths.resolved_config, paths.metadata) if not path.is_file()]
    if missing:
        raise ValueError(f"不是有效的训练 run：{paths.root}，缺少：{', '.join(missing)}")
    if not paths.checkpoints.is_dir():
        raise ValueError(f"run 缺少 checkpoints 目录：{paths.root}")


def _checkpoint_from_latest(paths: RunPaths) -> Path:
    try:
        latest = read_json(paths.latest)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"run 尚无 {RUN_LATEST_FILE}，无法自动恢复：{paths.root}") from exc
    checkpoint_name = latest.get("checkpoint") if isinstance(latest, dict) else None
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError(f"无效的 latest checkpoint 指针：{paths.latest}")
    checkpoint = paths.checkpoints / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"latest checkpoint 不存在：{checkpoint}")
    return checkpoint


def resolve_resume_target(target: str | os.PathLike[str]) -> tuple[RunPaths, Path]:
    requested = Path(target).resolve()
    if requested.is_dir():
        paths = RunPaths.from_root(requested)
        _validate_run(paths)
        return paths, _checkpoint_from_latest(paths)

    if not requested.is_file():
        raise FileNotFoundError(f"恢复目标不存在：{requested}")
    if requested.suffix != ".pth":
        raise ValueError(f"--resume 文件必须是 .pth checkpoint：{requested}")
    if requested.parent.name != "checkpoints":
        raise ValueError("--resume 指定 checkpoint 时，该文件必须位于标准 run/checkpoints/ 目录中")

    paths = RunPaths.from_root(requested.parent.parent)
    _validate_run(paths)
    latest = _checkpoint_from_latest(paths)
    if requested != latest:
        raise ValueError(f"--resume 只能线性继续 latest checkpoint：requested={requested.name}, latest={latest.name}；历史 checkpoint 应作为新实验分支初始化")
    return paths, requested
