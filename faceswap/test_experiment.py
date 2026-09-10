"""训练 run 文件布局的无 GPU 回归检查：.venv/bin/python -m faceswap.test_experiment"""

import json
import tempfile
from pathlib import Path

from faceswap.experiment import RunLock, RunPaths, append_resume_event, config_sha256, create_run, load_resolved_config, resolve_resume_target, update_metadata, write_latest


def main() -> None:
    resolved = {
        "train": {"batch_size": 8},
        "dataloader": {"huggingface_proxy": None},
        "src": [{"backend": "local", "path": "/data/src", "adjustment": 0.0}],
    }

    with tempfile.TemporaryDirectory(prefix="faceswap-run-check-") as temporary:
        root = Path(temporary)
        source_config = root / "train.toml"
        source_config.write_text("[train]\nbatch_size = 8\n", encoding="utf-8")
        runs_root = root / "runs"

        first = create_run(runs_root, source_config, resolved, name="blendface vgg", project_root=root)
        second = create_run(runs_root, source_config, resolved, name="blendface vgg", project_root=root)

        assert first.root != second.root
        assert first.root.name.endswith("_blendface-vgg")
        assert second.root.name.startswith(first.root.name.rsplit("-", 1)[0])
        assert first.config.read_text(encoding="utf-8") == source_config.read_text(encoding="utf-8")
        assert load_resolved_config(first) == resolved
        assert first.checkpoints.is_dir() and first.samples.is_dir() and first.tensorboard.is_dir()

        metadata = json.loads(first.metadata.read_text(encoding="utf-8"))
        assert metadata["config_sha256"] == config_sha256(resolved)
        assert metadata["status"] == "created"

        with RunLock(first):
            try:
                with RunLock(first):
                    raise AssertionError("同一 run 被重复加锁")
            except RuntimeError as error:
                assert "另一个训练进程" in str(error)
        with RunLock(first):
            pass

        checkpoint = first.checkpoints / "step_000010000.pth"
        checkpoint.write_bytes(b"checkpoint")
        write_latest(first, checkpoint, 10_000)

        run_paths, resolved_checkpoint = resolve_resume_target(first.root)
        assert run_paths == RunPaths.from_root(first.root)
        assert resolved_checkpoint == checkpoint

        file_run_paths, file_checkpoint = resolve_resume_target(checkpoint)
        assert file_run_paths == run_paths
        assert file_checkpoint == checkpoint

        update_metadata(first, status="failed", error={"type": "TestError", "message": "test"})
        append_resume_event(first, checkpoint)
        resumed_metadata = json.loads(first.metadata.read_text(encoding="utf-8"))
        assert resumed_metadata["status"] == "running"
        assert "error" not in resumed_metadata
        assert resumed_metadata["resume_history"][-1]["checkpoint"] == checkpoint.name

        historical = first.checkpoints / "step_000005000.pth"
        historical.write_bytes(b"checkpoint")
        try:
            resolve_resume_target(historical)
        except ValueError as error:
            assert "latest checkpoint" in str(error)
        else:
            raise AssertionError("历史 checkpoint 被错误地当作线性 resume 接受")

        document = json.loads(first.resolved_config.read_text(encoding="utf-8"))
        document["config"]["train"]["batch_size"] = 16
        first.resolved_config.write_text(json.dumps(document), encoding="utf-8")
        try:
            load_resolved_config(first)
        except ValueError as error:
            assert "摘要不匹配" in str(error)
        else:
            raise AssertionError("被篡改的 resolved config 未被拒绝")

    print("PASS: unique run IDs, frozen config, digest validation, latest pointer and resume target resolution")


if __name__ == "__main__":
    main()
