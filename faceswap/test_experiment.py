"""训练 run / branch / scheduler 的无 GPU 回归检查。"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from faceswap.experiment import RunLock, RunPaths, config_sha256, create_run, load_resolved_config, resolve_branch_target, resolve_resume_target, write_latest
from faceswap.train import _assert_branch_model_compatible, _load_branch_optimizer_state, _supports_compiled_bf16


def main() -> None:
    # ROCm 不使用 NVIDIA compute capability；CUDA 继续保持 SM80+ 的 compile-BF16 限制。
    with (
        patch("faceswap.train.torch.version.hip", "7.0.0"),
        patch("faceswap.train.torch.cuda.get_device_capability", side_effect=AssertionError("ROCm 不应查询 NVIDIA SM")),
    ):
        assert _supports_compiled_bf16(0)
    with patch("faceswap.train.torch.version.hip", None), patch("faceswap.train.torch.cuda.get_device_capability", return_value=(7, 5)):
        assert not _supports_compiled_bf16(0)
    with patch("faceswap.train.torch.version.hip", None), patch("faceswap.train.torch.cuda.get_device_capability", return_value=(8, 0)):
        assert _supports_compiled_bf16(0)

    resolved = {"train": {"batch_size": 8}, "generator": {"depth": 5}, "discriminator": {"base_ch": 64}, "identity": {"generator_provider": "BLENDFACE"}}

    with tempfile.TemporaryDirectory(prefix="faceswap-run-check-") as temporary:
        root = Path(temporary)
        source_config = root / "train.toml"
        source_config.write_text("[train]\nbatch_size = 8\n", encoding="utf-8")
        runs_root = root / "runs"

        first = create_run(runs_root, source_config, resolved, name="test")
        second = create_run(runs_root, source_config, resolved, name="test")
        assert first.root != second.root
        assert first.config.read_bytes() == source_config.read_bytes()
        assert load_resolved_config(first) == resolved

        metadata = json.loads(first.metadata.read_text(encoding="utf-8"))
        assert metadata["config_sha256"] == config_sha256(resolved)

        with RunLock(first):
            try:
                with RunLock(first):
                    raise AssertionError("同一 run 被重复加锁")
            except RuntimeError:
                pass

        latest = first.checkpoints / "step_000010000.pth"
        historical = first.checkpoints / "step_000005000.pth"
        latest.write_bytes(b"checkpoint")
        historical.write_bytes(b"checkpoint")
        write_latest(first, latest)

        paths, checkpoint = resolve_resume_target(first.root)
        assert paths == RunPaths.from_root(first.root) and checkpoint == latest
        try:
            resolve_resume_target(historical)
        except ValueError:
            pass
        else:
            raise AssertionError("resume 错误接受了历史 checkpoint")
        assert resolve_branch_target(historical)[1] == historical

        parent = {"run_id": metadata["run_id"], "checkpoint": historical.name, "step": 5000, "config_sha256": metadata["config_sha256"]}
        branch = create_run(runs_root, source_config, resolved, name="branch", parent=parent)
        assert json.loads(branch.metadata.read_text(encoding="utf-8"))["parent"] == parent

        _assert_branch_model_compatible(resolved, dict(resolved))
        changed = json.loads(json.dumps(resolved))
        changed["generator"]["depth"] = 6
        try:
            _assert_branch_model_compatible(resolved, changed)
        except ValueError:
            pass
        else:
            raise AssertionError("branch 错误接受了模型架构变更")

        changed_provider = json.loads(json.dumps(resolved))
        changed_provider["identity"]["generator_provider"] = "MS1MV3_ARCFACE_R50_FP16"
        _assert_branch_model_compatible(resolved, changed_provider)

        # scheduler 配置不变时，branch 必须从 checkpoint 的当前 LR 连续运行。
        parent_param = torch.nn.Parameter(torch.tensor(1.0))
        parent_optim = Adam([parent_param], lr=1e-4)
        parent_scheduler = CosineAnnealingLR(parent_optim, T_max=100, eta_min=1e-5)
        for _ in range(20):
            parent_optim.step()
            parent_scheduler.step()

        inherited_param = torch.nn.Parameter(torch.tensor(1.0))
        inherited_optim = Adam([inherited_param], lr=1e-4)
        _load_branch_optimizer_state(inherited_optim, parent_optim.state_dict(), lr=1e-4, reset_lr=False)
        inherited_scheduler = CosineAnnealingLR(inherited_optim, T_max=100, eta_min=1e-5)
        inherited_scheduler.load_state_dict(parent_scheduler.state_dict())
        inherited_optim.step()
        inherited_scheduler.step()

        reference_param = torch.nn.Parameter(torch.tensor(1.0))
        reference_optim = Adam([reference_param], lr=1e-4)
        reference_scheduler = CosineAnnealingLR(reference_optim, T_max=100, eta_min=1e-5)
        for _ in range(21):
            reference_optim.step()
            reference_scheduler.step()
        assert abs(inherited_optim.param_groups[0]["lr"] - reference_optim.param_groups[0]["lr"]) < 1e-15

        changed_optim = Adam([torch.nn.Parameter(torch.tensor(1.0))], lr=5e-5)
        _load_branch_optimizer_state(changed_optim, parent_optim.state_dict(), lr=5e-5, reset_lr=True)
        assert abs(changed_optim.param_groups[0]["lr"] - 5e-5) < 1e-15

        document = json.loads(first.resolved_config.read_text(encoding="utf-8"))
        document["config"]["train"]["batch_size"] = 16
        first.resolved_config.write_text(json.dumps(document), encoding="utf-8")
        try:
            load_resolved_config(first)
        except ValueError:
            pass
        else:
            raise AssertionError("被修改的 resolved config 未被拒绝")

    print("PASS: run layout, latest/resume, branch guard, scheduler continuity and config digest")


if __name__ == "__main__":
    main()
