"""训练 run / branch / scheduler 的无 GPU 回归检查。"""

import copy
import inspect
import json
import tempfile
import tomllib
from pathlib import Path
from unittest.mock import patch

import torch

from swapface.config import load_train_config, resolve_train_config
from swapface.contracts import CHECKPOINT_VERSION
from swapface.experiment import RunLock, RunPaths, config_sha256, create_run, load_resolved_config, resolve_branch_target, resolve_resume_target, write_latest
from swapface.train import (
    TRAINING_SEMANTICS_VERSION,
    Trainer,
    _assert_branch_generator_compatible,
    _compile_training_callable,
    _load_branch_checkpoint,
    _branch_model_states,
    _load_run_config,
    _reduce_reconstruction_loss,
    _require_resume_training_config,
    _scaled_backward_step,
    _supports_compiled_bf16,
)


def _check_train_config(root: Path) -> None:
    source = root / "canonical.toml"
    source.write_text(
        '[train]\nbatch_size = 8\nprecision = "fp16"\ncompile_module = false\n'
        "[optimizer]\nlr = 5e-5\n"
        '[scheduler]\ntype = "cosine"\nt_max = 1234\nmin_lr_ratio = 0.2\n'
        "[generator]\ncoarse_num_latent = 7\n"
        '[identity]\nprovider = "MS1MV3_ARCFACE_R50_FP16"\n'
        '[loss.reconstruction]\nscope = "all"\n'
        "[loss.gan]\nweight = 0.8\n"
        '[loss.identity]\nprovider = "BLENDFACE"\nweight = 9.0\n'
        "[loss.l1]\nenable = true\nweight = 7.0\n"
        "[loss.gaze]\nenable = true\nweight = 0.75\ndistribution_weight = 0.2\nconfidence_weighted = false\n"
        "[loss.hrffa]\nenable = true\npose_weight = 0.5\neye_weight = 1.25\nmouth_weight = 1.5\ncontour_weight = 2.0\ncontour_shape_weight = 0.4\noccluded_geometry_weight = 0.1\n"
        "[loss.facs]\nenable = true\nweight = 0.8\nbrow_weight = 1.1\neye_weight = 1.2\nnose_weight = 0.9\nmouth_weight = 1.3\nlower_face_weight = 0.7\nasymmetry_weight = 1.4\n"
        "[loss.wfm.weights]\n2 = 0.25\n"
        "[data.augmentation]\nrotation_range = [-3, 3]\n"
        "[data.sampling]\nsame_prob = 0.25\n"
        '[[data.src]]\npath = "source"\nadjustment = 1\n'
        '[[data.dst]]\npath = "target"\nadjustment = -1\n',
        encoding="utf-8",
    )
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    original = copy.deepcopy(raw)
    with patch.object(inspect, "signature", side_effect=AssertionError("配置协议不应依赖 constructor signature")):
        resolved = resolve_train_config(raw)
    assert raw == original
    assert resolve_train_config(resolved) == resolved
    assert json.loads(json.dumps(resolved)) == resolved
    loaded = load_train_config(source)
    assert loaded == resolved

    assert resolved["train"]["batch_size"] == 8
    assert resolved["train"]["compile_module"] is False
    assert resolved["train"]["precision"] == "fp16"
    assert resolved["optimizer"] == {"lr": 5e-5}
    assert resolved["scheduler"] == {"type": "cosine", "t_max": 1234, "min_lr_ratio": 0.2}
    assert resolved["identity"]["provider"] == raw["identity"]["provider"]
    assert resolved["loss"]["gan"] == {"weight": 0.8}
    assert resolved["loss"]["identity"] == {"provider": "BLENDFACE", "weight": 9.0}
    assert resolved["loss"]["l1"] == {"enable": True, "weight": 7.0}
    assert resolved["loss"]["reconstruction"] == {"scope": "all"}
    assert resolved["data"]["augmentation"]["rotation_range"] == [-3.0, 3.0]
    assert abs(resolved["data"]["sampling"]["same_prob"] - 0.25) < 1e-12
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
    assert resolved["data"]["src"][0]["adjustment"] == 1 and resolved["data"]["dst"][0]["adjustment"] == -1

    assert resolved["generator"]["coarse_num_latent"] == 7
    default_generator = copy.deepcopy(raw)
    default_generator["generator"].pop("coarse_num_latent")
    default_generator_resolved = resolve_train_config(default_generator)
    assert default_generator_resolved["generator"]["coarse_num_latent"] == 8
    _assert_branch_generator_compatible(json.loads(json.dumps(default_generator_resolved)), default_generator_resolved)

    paths = create_run(root / "config-runs", source, resolved, name="canonical")
    assert load_resolved_config(paths) == resolved
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["config_sha256"] == config_sha256(resolved)

    # frozen run 必须严格等于当前 canonical schema；缺字段/旧字段都不做迁移。
    invalid_frozen_configs = []

    old_precision = copy.deepcopy(resolved)
    old_precision["train"].pop("precision")
    old_precision["train"]["bf16"] = True
    invalid_frozen_configs.append(old_precision)

    missing_scope = copy.deepcopy(resolved)
    missing_scope["loss"]["reconstruction"].pop("scope")
    invalid_frozen_configs.append(missing_scope)

    missing_same_prob = copy.deepcopy(resolved)
    missing_same_prob["data"]["sampling"].pop("same_prob")
    invalid_frozen_configs.append(missing_same_prob)

    for index, frozen in enumerate(invalid_frozen_configs):
        paths = create_run(root / f"invalid-frozen-{index}", source, frozen, name=f"invalid-frozen-{index}")
        try:
            _load_run_config(paths)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("非当前 schema 的 frozen run 被错误接受")

    invalid = copy.deepcopy(raw)
    invalid["train"]["unknown_field"] = True
    try:
        resolve_train_config(invalid)
    except ValueError as error:
        assert "unknown_field" in str(error)
    else:
        raise AssertionError("配置错误接受了未知字段")

    invalid = copy.deepcopy(raw)
    invalid["train"]["precision"] = "fp8"
    try:
        resolve_train_config(invalid)
    except ValueError as error:
        assert "precision" in str(error)
    else:
        raise AssertionError("配置错误接受了未知训练精度")

    invalid = copy.deepcopy(raw)
    invalid["train"].pop("precision")
    try:
        resolve_train_config(invalid)
    except ValueError as error:
        assert "precision" in str(error)
    else:
        raise AssertionError("配置错误接受了未显式声明的训练精度")

    invalid = copy.deepcopy(raw)
    invalid["loss"]["reconstruction"]["scope"] = "cross"
    try:
        resolve_train_config(invalid)
    except ValueError as error:
        assert "loss.reconstruction.scope" in str(error)
    else:
        raise AssertionError("配置错误接受了未知 reconstruction scope")

    obsolete = copy.deepcopy(raw)
    obsolete["loss"]["rec_loss_scope"] = "same"
    try:
        resolve_train_config(obsolete)
    except ValueError as error:
        assert "rec_loss_scope" in str(error)
    else:
        raise AssertionError("配置错误接受了已移除的 rec_loss_scope")

    invalid = copy.deepcopy(raw)
    invalid["data"]["sampling"]["same_prob"] = 1.1
    try:
        resolve_train_config(invalid)
    except ValueError as error:
        assert "same_prob" in str(error)
    else:
        raise AssertionError("配置错误接受了 same_prob > 1")

    for section, key, value in (("optimizer", "lr", True), ("loss.gan", "weight", True), ("loss.gan", "weight", "0.5")):
        invalid = copy.deepcopy(raw)
        target = invalid
        for part in section.split("."):
            target = target[part]
        target[key] = value
        try:
            resolve_train_config(invalid)
        except TypeError:
            pass
        else:
            raise AssertionError(f"配置错误接受了非数值类型：{section}.{key}={value!r}")

    for section, key, value in (
        ("generator", "img_resolution", True),
        ("discriminator", "group_size", "4"),
    ):
        invalid = copy.deepcopy(raw)
        invalid.setdefault(section, {})[key] = value
        try:
            resolve_train_config(invalid)
        except TypeError:
            pass
        else:
            raise AssertionError(f"配置错误接受了非整数类型：{section}.{key}={value!r}")

    for key, value in (
        ("coarse_resolution", 96),
        ("coarse_latent_resolution", 48),
        ("hq_bottleneck_resolution", 24),
    ):
        invalid = copy.deepcopy(raw)
        invalid["generator"][key] = value
        try:
            resolve_train_config(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"配置错误接受了非法生成器尺度：{key}={value!r}")

    for removed_key, removed_value in (
        ("aad_skip_layers", []),
        ("coarse_bottleneck_resolution", 32),
        ("num_style_blocks", 6),
        ("hq_channel_hold_level", 2),
        ("norm_eps", 1e-8),
        ("leaky_relu_slope", 0.2),
    ):
        legacy = copy.deepcopy(raw)
        legacy["generator"][removed_key] = removed_value
        try:
            resolve_train_config(legacy)
        except ValueError:
            pass
        else:
            raise AssertionError(f"配置错误接受了已移除的 generator.{removed_key}")

    invalid = copy.deepcopy(raw)
    invalid.setdefault("discriminator", {})["group_size"] = 0
    try:
        resolve_train_config(invalid)
    except ValueError as error:
        assert "group_size" in str(error)
    else:
        raise AssertionError("配置错误接受了 discriminator.group_size=0")

    for key, value in (("pose_weight", float("nan")), ("eye_weight", float("inf")), ("occluded_geometry_weight", float("nan"))):
        invalid = copy.deepcopy(raw)
        invalid["loss"]["hrffa"][key] = value
        try:
            resolve_train_config(invalid)
        except ValueError as error:
            assert "有限" in str(error)
        else:
            raise AssertionError(f"配置错误接受了 HRFFA 非有限数值：{key}={value}")

    # 旧顶层 identity/dataloader/src/dst 协议必须直接拒绝，不提供 alias。
    for obsolete_top_level in (
        {"identity": {"generator_provider": "BLENDFACE"}},
        {"dataloader": {}},
        {"src": []},
        {"dst": []},
    ):
        obsolete = copy.deepcopy(raw)
        obsolete.update(obsolete_top_level)
        try:
            resolve_train_config(obsolete)
        except ValueError:
            pass
        else:
            raise AssertionError(f"配置错误接受了旧顶层字段：{next(iter(obsolete_top_level))}")

    print("PASS: canonical config hierarchy, strict schema and unknown-field rejection")


def _check_reconstruction_scope() -> None:
    rec_per_sample = torch.tensor([1.0, 10.0, 5.0])
    same_mask = torch.tensor([True, False, True])

    same = _reduce_reconstruction_loss(rec_per_sample, same_mask, "same")
    all_pairs = _reduce_reconstruction_loss(rec_per_sample, same_mask, "all")
    no_same = _reduce_reconstruction_loss(rec_per_sample, torch.zeros(3, dtype=torch.bool), "same")

    torch.testing.assert_close(same, torch.tensor(3.0))
    torch.testing.assert_close(all_pairs, torch.tensor(16.0 / 3.0))
    torch.testing.assert_close(no_same, torch.tensor(0.0))
    print("PASS: reconstruction loss scope same/all")


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


def _check_scaled_optimizer_step() -> None:
    class FakeScaler:
        def __init__(self, *, enabled: bool, overflow: bool = False):
            self.enabled = enabled
            self.overflow = overflow
            self.scale_value = 8.0

        def get_scale(self) -> float:
            return self.scale_value

        def is_enabled(self) -> bool:
            return self.enabled

        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            if not self.overflow:
                optimizer.step()

        def update(self) -> None:
            if self.overflow:
                self.scale_value *= 0.5

    successful_parameter = torch.nn.Parameter(torch.tensor(1.0))
    successful_optimizer = torch.optim.SGD([successful_parameter], lr=0.1)
    successful_loss = successful_parameter.square()
    assert _scaled_backward_step(
        successful_loss,
        successful_optimizer,
        FakeScaler(enabled=True),
        name="FP16",
        named_parameters=[("weight", successful_parameter)],
    )
    assert not torch.equal(successful_parameter.detach(), torch.tensor(1.0))

    skipped_parameter = torch.nn.Parameter(torch.tensor(1.0))
    skipped_optimizer = torch.optim.SGD([skipped_parameter], lr=0.1)
    skipped_loss = skipped_parameter.square()
    assert not _scaled_backward_step(
        skipped_loss,
        skipped_optimizer,
        FakeScaler(enabled=True, overflow=True),
        name="FP16",
        named_parameters=[("weight", skipped_parameter)],
    )
    torch.testing.assert_close(skipped_parameter.detach(), torch.tensor(1.0))

    unscaled_parameter = torch.nn.Parameter(torch.tensor(1.0))
    unscaled_optimizer = torch.optim.SGD([unscaled_parameter], lr=0.1)
    unscaled_loss = unscaled_parameter.square()
    assert _scaled_backward_step(
        unscaled_loss,
        unscaled_optimizer,
        FakeScaler(enabled=False),
        name="BF16",
        named_parameters=[("weight", unscaled_parameter)],
    )
    assert not torch.equal(unscaled_parameter.detach(), torch.tensor(1.0))

    class FiniteLossNaNGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value: torch.Tensor) -> torch.Tensor:
            ctx.shape = value.shape
            return value.new_zeros(())

        @staticmethod
        def backward(ctx, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor]:
            (grad_output,) = grad_outputs
            return (torch.full(ctx.shape, torch.nan, device=grad_output.device, dtype=grad_output.dtype),)

    guarded_parameter = torch.nn.Parameter(torch.tensor(1.0))
    guarded_optimizer = torch.optim.SGD([guarded_parameter], lr=0.1)
    guarded_loss = FiniteLossNaNGradient.apply(guarded_parameter)
    try:
        _scaled_backward_step(
            guarded_loss,
            guarded_optimizer,
            FakeScaler(enabled=False),
            name="BF16",
            named_parameters=[("weight", guarded_parameter)],
        )
    except FloatingPointError as error:
        message = str(error)
        assert "BF16 gradient 出现 NaN/Inf" in message
        assert "parameter=weight" in message
        assert "finite=0/1" in message
    else:
        raise AssertionError("BF16 非有限梯度必须在 optimizer.step 前终止")
    torch.testing.assert_close(guarded_parameter.detach(), torch.tensor(1.0))
    print("PASS: FP16 overflow remains recoverable; BF16/FP32 non-finite gradients are blocked before optimizer.step")


def main() -> None:
    _check_reconstruction_scope()
    _check_step_boundary()
    _check_scaled_optimizer_step()
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

    resolved = {"train": {"batch_size": 8}, "generator": {"coarse_resolution": 128}, "discriminator": {"base_ch": 64}, "identity": {"provider": "BLENDFACE"}}

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
        ema_g_state = {"marker": torch.tensor(1)}
        train_g_state = {"marker": torch.tensor(2)}
        d_state = {"marker": torch.tensor(3)}
        coarse_d_state = {"marker": torch.tensor(4)}
        torch.save(
            {
                "version": CHECKPOINT_VERSION,
                "step": 7500,
                "run": {"id": metadata["run_id"], "config_sha256": metadata["config_sha256"]},
                "training_config": {
                    "semantics_version": TRAINING_SEMANTICS_VERSION,
                    "precision": "fp16",
                    "optimizer": {"lr": 1e-4},
                    "scheduler": {"type": "none", "t_max": 20000, "min_lr_ratio": 0.1},
                },
                "net_g": {"network_cfg": resolved["generator"], "state_dict": ema_g_state},
                "net_d": {"network_cfg": resolved["discriminator"], "state_dict": d_state},
                "net_d_coarse": {
                    "network_cfg": {**resolved["discriminator"], "img_resolution": resolved["generator"]["coarse_resolution"]},
                    "state_dict": coarse_d_state,
                },
                "training_state": {"net_g": train_g_state},
            },
            standalone,
        )
        assert resolve_branch_target(standalone) == standalone
        loaded, standalone_parent = _load_branch_checkpoint(standalone, resolved, reset_discriminator=False)
        assert loaded["step"] == 7500
        assert _require_resume_training_config(loaded) == {
            "semantics_version": TRAINING_SEMANTICS_VERSION,
            "precision": "fp16",
        }
        assert standalone_parent == {
            "run_id": metadata["run_id"],
            "checkpoint": standalone.name,
            "step": 7500,
            "config_sha256": metadata["config_sha256"],
            "discriminator": "inherit",
        }
        branch_g_state, branch_d_state, branch_d_coarse_state = _branch_model_states(loaded, reset_discriminator=False)
        assert branch_g_state["marker"].item() == 2  # training G, not EMA G
        assert branch_d_state is not None and branch_d_state["marker"].item() == 3
        assert branch_d_coarse_state is not None and branch_d_coarse_state["marker"].item() == 4
        branch_g_state, branch_d_state, branch_d_coarse_state = _branch_model_states(loaded, reset_discriminator=True)
        assert branch_g_state["marker"].item() == 2 and branch_d_state is None and branch_d_coarse_state is None

        invalid_training_config = dict(loaded)
        invalid_training_config["training_config"] = {
            "precision": "fp16",
            "optimizer": {"lr": 1e-4},
            "scheduler": {"type": "none", "t_max": 20000, "min_lr_ratio": 0.1},
        }
        try:
            _require_resume_training_config(invalid_training_config)
        except ValueError:
            pass
        else:
            raise AssertionError("缺少 semantics_version 的旧训练 checkpoint 被错误接受")

        parent = dict(standalone_parent)
        branch = create_run(runs_root, source_config, resolved, name="branch", parent=parent)
        assert json.loads(branch.metadata.read_text(encoding="utf-8"))["parent"] == parent

        _assert_branch_generator_compatible(resolved, dict(resolved))

        changed_generator = copy.deepcopy(resolved)
        changed_generator["generator"]["coarse_resolution"] = 256
        try:
            _assert_branch_generator_compatible(resolved, changed_generator)
        except ValueError:
            pass
        else:
            raise AssertionError("branch 错误接受了 Generator 架构变更")

        changed_discriminator = copy.deepcopy(resolved)
        changed_discriminator["discriminator"]["base_ch"] = 128
        try:
            _load_branch_checkpoint(standalone, changed_discriminator, reset_discriminator=False)
        except ValueError:
            pass
        else:
            raise AssertionError("branch 默认错误接受了 Discriminator 架构变更")
        _, reset_parent = _load_branch_checkpoint(standalone, changed_discriminator, reset_discriminator=True)
        assert reset_parent["discriminator"] == "reset"


        changed_provider = copy.deepcopy(resolved)
        changed_provider["identity"]["provider"] = "MS1MV3_ARCFACE_R50_FP16"
        _assert_branch_generator_compatible(resolved, changed_provider)

        document = json.loads(first.resolved_config.read_text(encoding="utf-8"))
        document["config"]["train"]["batch_size"] = 16
        first.resolved_config.write_text(json.dumps(document), encoding="utf-8")
        try:
            load_resolved_config(first)
        except ValueError:
            pass
        else:
            raise AssertionError("被修改的 resolved config 未被拒绝")

    print("PASS: run layout, latest/resume, simple branch semantics and config digest")


if __name__ == "__main__":
    main()
