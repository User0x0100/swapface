"""训练 run / branch / scheduler 的无 GPU 回归检查。"""

import copy
import inspect
import json
import tempfile
import tomllib
from pathlib import Path
from unittest.mock import patch

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from swapface.config import load_train_config, resolve_train_config
from swapface.contracts import CHECKPOINT_VERSION
from swapface.experiment import RunLock, RunPaths, config_sha256, create_run, load_resolved_config, resolve_branch_target, resolve_resume_target, write_latest
from swapface.train import Trainer, _assert_branch_model_compatible, _compile_training_callable, _load_branch_checkpoint, _load_branch_optimizer_state, _load_run_config, _supports_compiled_bf16


def _check_train_config(root: Path) -> None:
    source = root / "canonical.toml"
    source.write_text(
        "[train]\nbatch_size = 8\ncompile_module = false\n"
        "[generator]\naad_skip_layers = []\n"
        '[identity]\ngenerator_provider = "MS1MV3_ARCFACE_R50_FP16"\nloss_provider = "BLENDFACE"\n'
        "[dataloader]\nrotation_range = [-3, 3]\nsame_prob = 0.25\n"
        "[loss.gaze]\nenable = true\nweight = 0.75\ndistribution_weight = 0.2\nconfidence_weighted = false\n"
        "[loss.hrffa]\nenable = true\npose_weight = 0.5\neye_weight = 1.25\nmouth_weight = 1.5\ncontour_weight = 2.0\ncontour_shape_weight = 0.4\noccluded_geometry_weight = 0.1\n"
        "[loss.facs]\nenable = true\nweight = 0.8\nbrow_weight = 1.1\neye_weight = 1.2\nnose_weight = 0.9\nmouth_weight = 1.3\nlower_face_weight = 0.7\nasymmetry_weight = 1.4\n"
        "[loss.wfm.weights]\n2 = 0.25\n"
        '[[src]]\npath = "source"\nadjustment = 1\n'
        '[[dst]]\npath = "target"\nadjustment = -1\n',
        encoding="utf-8",
    )
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    original = copy.deepcopy(raw)
    with patch.object(inspect, "signature", side_effect=AssertionError("配置协议不应依赖 constructor signature")):
        resolved = resolve_train_config(raw)
    assert raw == original
    assert resolve_train_config(resolved) == resolved
    _, loaded = load_train_config(source)
    assert loaded == resolved
    assert resolved["train"]["batch_size"] == 8
    assert resolved["train"]["compile_module"] is False
    assert resolved["identity"]["generator_provider"] == raw["identity"]["generator_provider"]
    assert resolved["identity"]["loss_provider"] == raw["identity"]["loss_provider"]
    assert resolved["dataloader"]["rotation_range"] == [-3.0, 3.0]
    assert abs(resolved["dataloader"]["same_prob"] - 0.25) < 1e-12
    assert resolved["loss"]["gaze"] == {"enable": True, "weight": 0.75, "distribution_weight": 0.2, "confidence_weighted": False}
    assert resolved["loss"]["hrffa"] == {
        "enable": True,
        "pose_weight": 0.5,
        "eye_weight": 1.25,
        "mouth_weight": 1.5,
        "contour_weight": 2.0,
        "contour_shape_weight": 0.4,
        "occluded_geometry_weight": 0.1,
    }
    assert resolved["loss"]["facs"] == {
        "enable": True,
        "weight": 0.8,
        "brow_weight": 1.1,
        "eye_weight": 1.2,
        "nose_weight": 0.9,
        "mouth_weight": 1.3,
        "lower_face_weight": 0.7,
        "asymmetry_weight": 1.4,
    }
    assert resolved["loss"]["wfm"]["weights"] == {"2": 0.25}
    assert resolved["src"][0]["adjustment"] == 1 and resolved["dst"][0]["adjustment"] == -1

    paths = create_run(root / "config-runs", source, resolved, name="canonical")
    assert load_resolved_config(paths) == resolved
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["config_sha256"] == config_sha256(resolved)

    # 旧 schema 不再兼容；缺少当前必需字段的 run 必须拒绝 resume。
    legacy_resolved = copy.deepcopy(resolved)
    legacy_resolved["dataloader"].pop("same_prob")
    legacy_paths = create_run(root / "legacy-runs", source, legacy_resolved, name="legacy")
    try:
        _load_run_config(legacy_paths)
    except ValueError as error:
        assert "不是当前格式" in str(error)
    else:
        raise AssertionError("旧 run schema 被错误接受")

    invalid = copy.deepcopy(raw)
    invalid["train"]["unknown_field"] = True
    try:
        resolve_train_config(invalid)
    except ValueError as error:
        assert "unknown_field" in str(error)
    else:
        raise AssertionError("配置错误接受了未知字段")

    invalid = copy.deepcopy(raw)
    invalid["dataloader"]["same_prob"] = 1.1
    try:
        resolve_train_config(invalid)
    except ValueError as error:
        assert "same_prob" in str(error)
    else:
        raise AssertionError("配置错误接受了 same_prob > 1")

    for key, value in (("pose_weight", float("nan")), ("eye_weight", float("inf")), ("occluded_geometry_weight", float("nan"))):
        invalid = copy.deepcopy(raw)
        invalid["loss"]["hrffa"][key] = value
        try:
            resolve_train_config(invalid)
        except ValueError as error:
            assert "有限" in str(error)
        else:
            raise AssertionError(f"配置错误接受了 HRFFA 非有限数值：{key}={value}")
    print("PASS: TOML fields, canonical config persistence and unknown-field rejection")


def _check_step_boundary() -> None:
    # 只构造进度边界，不创建模型、优化器或 GPU Trainer。
    trainer = Trainer.__new__(Trainer)
    trainer._completed_step = 0
    trainer._step_in_progress = False
    assert not trainer.can_save_checkpoint

    trainer._completed_step = 1
    assert trainer.completed_step == 1 and trainer.can_save_checkpoint

    trainer._step_in_progress = True
    assert trainer.completed_step == 1 and not trainer.can_save_checkpoint
    try:
        trainer.save_ckpt()
    except RuntimeError:
        pass
    else:
        raise AssertionError("未完成的 step 不应允许保存 checkpoint")
    print("PASS: fresh/completed/partial step checkpoint boundary")


def main() -> None:
    _check_step_boundary()
    # ROCm 不使用 NVIDIA compute capability；CUDA 继续保持 SM80+ 的 compile-BF16 限制。
    with (
        patch("swapface.train.torch.version.hip", "7.0.0"),
        patch("swapface.train.torch.cuda.get_device_capability", side_effect=AssertionError("ROCm 不应查询 NVIDIA SM")),
    ):
        assert _supports_compiled_bf16(0)
    with patch("swapface.train.torch.version.hip", None), patch("swapface.train.torch.cuda.get_device_capability", return_value=(7, 5)):
        assert not _supports_compiled_bf16(0)
    with patch("swapface.train.torch.version.hip", None), patch("swapface.train.torch.cuda.get_device_capability", return_value=(8, 0)):
        assert _supports_compiled_bf16(0)

    compile_target = object()
    with patch("swapface.train.torch.version.hip", "7.2.0"), patch("swapface.train.torch.compile", return_value=compile_target) as compile_mock:
        assert _compile_training_callable(object()) is compile_target
        assert compile_mock.call_args.kwargs["mode"] == "default"
    with patch("swapface.train.torch.version.hip", None), patch("swapface.train.torch.compile", return_value=compile_target) as compile_mock:
        assert _compile_training_callable(object()) is compile_target
        assert compile_mock.call_args.kwargs["mode"] == "max-autotune-no-cudagraphs"

    resolved = {"train": {"batch_size": 8}, "generator": {"depth": 5}, "discriminator": {"base_ch": 64}, "identity": {"generator_provider": "BLENDFACE"}}

    with tempfile.TemporaryDirectory(prefix="swapface-run-check-") as temporary:
        root = Path(temporary)
        _check_train_config(root)
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
        assert resolve_resume_target(latest) == (paths, latest)
        try:
            resolve_resume_target(historical)
        except ValueError:
            pass
        else:
            raise AssertionError("resume 错误接受了历史 checkpoint")
        assert resolve_branch_target(first.root) == latest
        assert resolve_branch_target(historical) == historical

        standalone = root / "step_000007500.pth"
        torch.save(
            {
                "version": CHECKPOINT_VERSION,
                "step": 7500,
                "run": {"id": metadata["run_id"], "config_sha256": metadata["config_sha256"]},
                "net_g": {"network_cfg": resolved["generator"]},
                "net_d": {"network_cfg": resolved["discriminator"]},
            },
            standalone,
        )
        assert resolve_branch_target(standalone) == standalone
        loaded, standalone_parent = _load_branch_checkpoint(standalone, resolved)
        assert loaded["step"] == 7500
        assert standalone_parent == {
            "run_id": metadata["run_id"],
            "checkpoint": standalone.name,
            "step": 7500,
            "config_sha256": metadata["config_sha256"],
        }

        parent = {"run_id": metadata["run_id"], "checkpoint": historical.name, "step": 5000, "config_sha256": metadata["config_sha256"]}
        branch = create_run(runs_root, source_config, resolved, name="branch", parent=parent)
        assert json.loads(branch.metadata.read_text(encoding="utf-8"))["parent"] == parent

        _assert_branch_model_compatible(resolved, dict(resolved))
        for section, key, value in (("generator", "depth", 6), ("discriminator", "base_ch", 128)):
            changed = copy.deepcopy(resolved)
            changed[section][key] = value
            try:
                _assert_branch_model_compatible(resolved, changed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"branch 错误接受了 {section} 架构变更")

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
