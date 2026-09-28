"""训练 run / branch / scheduler 的无 GPU 回归检查。"""

import ast
import copy
import inspect
import json
import tempfile
import textwrap
import tomllib
from pathlib import Path
from unittest.mock import patch

import torch

from models.networks import Generator
from swapface.config import DEFAULT_GENERATOR_CONFIG, DEFAULT_LOSS_CONFIG, load_train_config, resolve_train_config
from swapface.contracts import CHECKPOINT_VERSION
from swapface.experiment import RunLock, RunPaths, config_sha256, create_run, load_resolved_config, resolve_branch_target, resolve_resume_target, write_latest
from swapface.train import (
    TRAINING_SEMANTICS_VERSION,
    Trainer,
    _branch_generator_mode,
    _branch_model_states,
    _compile_training_callable,
    _load_branch_checkpoint,
    _load_run_config,
    _print_hq_rebuild_summary,
    _reduce_reconstruction_loss,
    _require_resume_training_config,
    _resolve_branch_start_step,
    _scaled_backward_step,
    _submodule_state_dict,
    _supports_compiled_bf16,
)


def _check_train_config(root: Path) -> None:
    source = root / "canonical.toml"
    source.write_text(
        '[train]\nbatch_size = 8\nprecision = "fp16"\ncompile_module = false\n'
        "[optimizer.generator]\nlr = 5e-5\n"
        "[optimizer.hq_discriminator]\nlr = 6e-5\n"
        "[optimizer.coarse_discriminator]\nlr = 7e-5\n"
        '[scheduler.generator]\ntype = "cosine"\nt_max = 1234\nmin_lr_ratio = 0.2\n'
        '[scheduler.hq_discriminator]\ntype = "cosine"\nt_max = 2345\nmin_lr_ratio = 0.3\n'
        '[scheduler.coarse_discriminator]\ntype = "none"\nt_max = 3456\nmin_lr_ratio = 0.4\n'
        "[generator]\ncoarse_num_latent = 7\n"
        "[discriminator.hq]\nbase_ch = 96\ngroup_size = 8\n"
        "[discriminator.coarse]\nbase_ch = 32\nmax_ch = 256\ngroup_size = 4\n"
        '[identity]\nprovider = "MS1MV3_ARCFACE_R50_FP16"\n'
        '[loss.coarse.reconstruction]\nscope = "same"\n'
        "[loss.coarse.gan]\nweight = 0.6\n"
        '[loss.coarse.identity]\nprovider = "BLENDFACE"\nweight = 7.0\n'
        "[loss.coarse.l1]\nenable = true\nweight = 6.0\n"
        "[loss.coarse.r1]\nenable = true\ninterval = 8\ngamma = 5.0\n"
        "[loss.coarse.gaze]\nenable = true\nweight = 0.4\ndistribution_weight = 0.05\nconfidence_weighted = true\n"
        "[loss.coarse.hrffa]\nenable = true\npose_weight = 0.4\neye_weight = 0.8\nmouth_weight = 1.1\ncontour_weight = 1.2\ncontour_shape_weight = 0.3\noccluded_geometry_weight = 0.2\n"
        "[loss.coarse.vgg.weights]\nrelu2_2 = 0.5\n"
        '[loss.hq.reconstruction]\nscope = "all"\n'
        "[loss.hq.gan]\nweight = 0.8\n"
        '[loss.hq.identity]\nprovider = "MS1MV3_ARCFACE_R50_FP16"\nweight = 9.0\n'
        "[loss.hq.l1]\nenable = true\nweight = 7.0\n"
        "[loss.hq.r1]\nenable = true\ninterval = 32\ngamma = 12.0\n"
        "[loss.hq.gaze]\nenable = true\nweight = 0.75\ndistribution_weight = 0.2\nconfidence_weighted = false\n"
        "[loss.hq.hrffa]\nenable = true\npose_weight = 0.5\neye_weight = 1.25\nmouth_weight = 1.5\ncontour_weight = 2.0\ncontour_shape_weight = 0.4\noccluded_geometry_weight = 0.1\n"
        "[loss.hq.facs]\nenable = true\nweight = 0.8\nbrow_weight = 1.1\neye_weight = 1.2\nnose_weight = 0.9\nmouth_weight = 1.3\nlower_face_weight = 0.7\nasymmetry_weight = 1.4\n"
        "[loss.hq.wfm.weights]\n2 = 0.25\n"
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
    for loss_name in ("gan", "identity", "r1", "gaze", "hrffa", "facs", "reconstruction", "l1", "vgg"):
        assert DEFAULT_LOSS_CONFIG["coarse"][loss_name] is not DEFAULT_LOSS_CONFIG["hq"][loss_name]
    loaded = load_train_config(source)
    assert loaded == resolved

    assert resolved["train"]["batch_size"] == 8
    assert resolved["train"]["stage"] == "joint"
    assert resolved["train"]["compile_module"] is False
    assert resolved["train"]["precision"] == "fp16"
    assert resolved["optimizer"] == {
        "generator": {"lr": 5e-5},
        "hq_discriminator": {"lr": 6e-5},
        "coarse_discriminator": {"lr": 7e-5},
    }
    assert resolved["scheduler"] == {
        "generator": {"type": "cosine", "t_max": 1234, "min_lr_ratio": 0.2},
        "hq_discriminator": {"type": "cosine", "t_max": 2345, "min_lr_ratio": 0.3},
        "coarse_discriminator": {"type": "none", "t_max": 3456, "min_lr_ratio": 0.4},
    }
    assert resolved["identity"]["provider"] == raw["identity"]["provider"]
    assert resolved["loss"]["coarse"]["gan"] == {"weight": 0.6}
    assert resolved["loss"]["coarse"]["identity"] == {"provider": "BLENDFACE", "weight": 7.0}
    assert resolved["loss"]["coarse"]["reconstruction"] == {"scope": "same"}
    assert resolved["loss"]["coarse"]["l1"] == {"enable": True, "weight": 6.0}
    assert resolved["loss"]["coarse"]["vgg"]["weights"] == {"relu2_2": 0.5}
    assert resolved["loss"]["coarse"]["r1"] == {"enable": True, "interval": 8, "gamma": 5.0}
    assert resolved["loss"]["hq"]["gan"] == {"weight": 0.8}
    assert resolved["loss"]["hq"]["identity"] == {"provider": "MS1MV3_ARCFACE_R50_FP16", "weight": 9.0}
    assert resolved["loss"]["hq"]["l1"] == {"enable": True, "weight": 7.0}
    assert resolved["loss"]["hq"]["r1"] == {"enable": True, "interval": 32, "gamma": 12.0}
    assert resolved["loss"]["hq"]["reconstruction"] == {"scope": "all"}
    assert resolved["data"]["augmentation"]["rotation_range"] == [-3.0, 3.0]
    assert abs(resolved["data"]["sampling"]["same_prob"] - 0.25) < 1e-12
    assert resolved["loss"]["coarse"]["gaze"] == {"enable": True, "weight": 0.4, "distribution_weight": 0.05, "confidence_weighted": True}
    assert resolved["loss"]["hq"]["gaze"] == {"enable": True, "weight": 0.75, "distribution_weight": 0.2, "confidence_weighted": False}
    assert resolved["loss"]["coarse"]["hrffa"] == {
        "enable": True,
        "pose_weight": 0.4,
        "eye_weight": 0.8,
        "mouth_weight": 1.1,
        "contour_weight": 1.2,
        "contour_shape_weight": 0.3,
        "occluded_geometry_weight": 0.2,
    }
    assert resolved["loss"]["hq"]["hrffa"] == {
        "enable": True,
        "pose_weight": 0.5,
        "eye_weight": 1.25,
        "mouth_weight": 1.5,
        "contour_weight": 2.0,
        "contour_shape_weight": 0.4,
        "occluded_geometry_weight": 0.1,
    }
    assert resolved["loss"]["hq"]["facs"] == {
        "enable": True,
        "weight": 0.8,
        "brow_weight": 1.1,
        "eye_weight": 1.2,
        "nose_weight": 0.9,
        "mouth_weight": 1.3,
        "lower_face_weight": 0.7,
        "asymmetry_weight": 1.4,
    }
    assert resolved["loss"]["hq"]["wfm"]["weights"] == {"2": 0.25}
    assert resolved["data"]["src"][0]["adjustment"] == 1 and resolved["data"]["dst"][0]["adjustment"] == -1

    assert resolved["discriminator"] == {
        "hq": {"img_resolution": 512, "img_channels": 3, "base_ch": 96, "max_ch": 512, "group_size": 8},
        "coarse": {"img_resolution": 128, "img_channels": 3, "base_ch": 32, "max_ch": 256, "group_size": 4},
    }

    assert resolved["generator"]["coarse_num_latent"] == 7
    default_generator = copy.deepcopy(raw)
    default_generator["generator"].pop("coarse_num_latent")
    default_generator_resolved = resolve_train_config(default_generator)
    assert default_generator_resolved["generator"]["coarse_num_latent"] == 8

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
    missing_scope["loss"]["hq"]["reconstruction"].pop("scope")
    invalid_frozen_configs.append(missing_scope)

    missing_coarse_scope = copy.deepcopy(resolved)
    missing_coarse_scope["loss"]["coarse"]["reconstruction"].pop("scope")
    invalid_frozen_configs.append(missing_coarse_scope)

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

    stage_config = copy.deepcopy(raw)
    stage_config["train"]["stage"] = "hq"
    assert resolve_train_config(stage_config)["train"]["stage"] == "hq"
    stage_config["train"]["stage"] = "coarse"
    assert resolve_train_config(stage_config)["train"]["stage"] == "coarse"
    stage_config["train"]["stage"] = "invalid"
    try:
        resolve_train_config(stage_config)
    except ValueError as error:
        assert "train.stage" in str(error)
    else:
        raise AssertionError("配置错误接受了未知 train.stage")

    legacy_optimizer = copy.deepcopy(raw)
    legacy_optimizer["optimizer"] = {"lr": 1e-4}
    try:
        resolve_train_config(legacy_optimizer)
    except ValueError:
        pass
    else:
        raise AssertionError("配置错误接受了旧扁平 optimizer schema")

    legacy_scheduler = copy.deepcopy(raw)
    legacy_scheduler["scheduler"] = {"type": "none", "t_max": 100, "min_lr_ratio": 0.1}
    try:
        resolve_train_config(legacy_scheduler)
    except ValueError:
        pass
    else:
        raise AssertionError("配置错误接受了旧扁平 scheduler schema")

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

    for stage in ("coarse", "hq"):
        invalid = copy.deepcopy(raw)
        invalid["loss"][stage]["reconstruction"]["scope"] = "cross"
        try:
            resolve_train_config(invalid)
        except ValueError as error:
            assert f"loss.{stage}.reconstruction.scope" in str(error)
        else:
            raise AssertionError(f"配置错误接受了未知 {stage} reconstruction scope")

    obsolete = copy.deepcopy(raw)
    obsolete["loss"]["gan"] = {"weight": 1.0}
    try:
        resolve_train_config(obsolete)
    except ValueError as error:
        assert "gan" in str(error)
    else:
        raise AssertionError("配置错误接受了已移除的共享 loss.gan")

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

    for section, key, value in (("optimizer.generator", "lr", True), ("loss.coarse.gan", "weight", True), ("loss.hq.gan", "weight", "0.5")):
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
        ("discriminator.hq", "group_size", "4"),
    ):
        invalid = copy.deepcopy(raw)
        target = invalid
        for part in section.split("."):
            target = target.setdefault(part, {})
        target[key] = value
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
        invalid["loss"]["hq"]["hrffa"][key] = value
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


def _check_discriminator_overflow_isolation() -> None:
    class OverflowOnceScaler:
        def __init__(self) -> None:
            self.scale_value = 8.0
            self.attempts = 0

        def get_scale(self) -> float:
            return self.scale_value

        def is_enabled(self) -> bool:
            return True

        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            self.attempts += 1
            if self.attempts > 1:
                optimizer.step()

        def update(self) -> None:
            if self.attempts == 1:
                self.scale_value *= 0.5

    trainer = Trainer.__new__(Trainer)
    hq_parameter = torch.nn.Parameter(torch.tensor(1.0))
    coarse_parameter = torch.nn.Parameter(torch.tensor(2.0))
    hq_optimizer = torch.optim.SGD([hq_parameter], lr=0.1)
    coarse_optimizer = torch.optim.SGD([coarse_parameter], lr=0.1)
    hq_scaler = OverflowOnceScaler()
    coarse_scaler = OverflowOnceScaler()

    trainer._discriminator_stage_loss = lambda *_args, **_kwargs: hq_parameter.square()
    trainer._update_discriminator_stage(
        "hq",
        torch.empty(0),
        torch.empty(0),
        torch.nn.Identity(),
        torch.nn.Identity(),
        hq_optimizer,
        hq_scaler,
        [("hq.weight", hq_parameter)],
        r1_enabled=False,
        r1_interval=16,
        r1_gamma=10.0,
    )

    assert hq_scaler.attempts == 2
    assert abs(hq_scaler.get_scale() - 4.0) < 1e-12
    torch.testing.assert_close(hq_parameter.detach(), torch.tensor(0.8))
    torch.testing.assert_close(coarse_parameter.detach(), torch.tensor(2.0))
    assert coarse_scaler.attempts == 0 and abs(coarse_scaler.get_scale() - 8.0) < 1e-12
    assert not coarse_optimizer.state
    print("PASS: HQ discriminator FP16 overflow retry does not touch Coarse discriminator state")


def _check_generator_responsibility_boundary() -> None:
    model = Generator(
        img_resolution=16,
        img_channels=3,
        id_dim=8,
        coarse_resolution=8,
        coarse_latent_resolution=4,
        coarse_num_latent=1,
        coarse_base_ch=2,
        coarse_max_ch=8,
        hq_bottleneck_resolution=4,
        hq_base_ch=2,
        hq_max_ch=8,
    )
    target = torch.randn(2, 3, 16, 16)
    identity = torch.randn(2, 8)

    coarse = model.coarse(target, identity)
    coarse_with_resize, resize_in = model.coarse(target, identity, return_resize_in=True)
    torch.testing.assert_close(coarse_with_resize, coarse)
    torch.testing.assert_close(resize_in, torch.nn.functional.interpolate(target, size=8, mode="bilinear", align_corners=False))

    fake = model.hq(target, coarse.detach())
    fake.square().mean().backward()

    coarse_has_gradient = any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0 for parameter in model.coarse.parameters())
    hq_has_gradient = any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0 for parameter in model.hq.parameters())
    assert not coarse_has_gradient and hq_has_gradient

    model.zero_grad(set_to_none=True)
    coarse = model.coarse(target, identity)
    coarse.square().mean().backward()
    coarse_has_gradient = any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0 for parameter in model.coarse.parameters())
    assert coarse_has_gradient

    train_tree = ast.parse(textwrap.dedent(inspect.getsource(Trainer.train)))
    detach_boundaries = 0
    for node in ast.walk(train_tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        is_hq_forward = isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self" and node.func.attr == "train_hq"
        if not is_hq_forward:
            continue
        coarse_arg = node.args[1]
        if (
            isinstance(coarse_arg, ast.Call)
            and isinstance(coarse_arg.func, ast.Attribute)
            and coarse_arg.func.attr == "detach"
            and isinstance(coarse_arg.func.value, ast.Name)
            and coarse_arg.func.value.id == "coarse"
            and not coarse_arg.args
            and not coarse_arg.keywords
        ):
            detach_boundaries += 1
    assert detach_boundaries == 2

    discriminator_stage_calls = [
        node
        for node in ast.walk(train_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self" and node.func.attr == "_update_discriminator_stage"
    ]
    assert len(discriminator_stage_calls) == 2
    calls_by_stage = {call.args[0].value: call for call in discriminator_stage_calls if isinstance(call.args[0], ast.Constant)}
    assert set(calls_by_stage) == {"hq", "coarse"}

    def positional_attribute(call: ast.Call, index: int) -> str:
        value = call.args[index]
        assert isinstance(value, ast.Attribute)
        assert isinstance(value.value, ast.Name) and value.value.id == "self"
        return value.attr

    def keyword_attribute(call: ast.Call, keyword_name: str) -> str:
        keyword = next(item for item in call.keywords if item.arg == keyword_name)
        assert isinstance(keyword.value, ast.Attribute)
        assert isinstance(keyword.value.value, ast.Name) and keyword.value.value.id == "self"
        return keyword.value.attr

    assert positional_attribute(calls_by_stage["hq"], 5) == "optim_d_hq"
    assert positional_attribute(calls_by_stage["hq"], 6) == "scaler_d_hq"
    assert positional_attribute(calls_by_stage["hq"], 7) == "_hq_d_named_parameters"
    assert positional_attribute(calls_by_stage["coarse"], 5) == "optim_d_coarse"
    assert positional_attribute(calls_by_stage["coarse"], 6) == "scaler_d_coarse"
    assert positional_attribute(calls_by_stage["coarse"], 7) == "_coarse_d_named_parameters"
    assert keyword_attribute(calls_by_stage["hq"], "r1_enabled") == "enable_hq_r1_loss"
    assert keyword_attribute(calls_by_stage["hq"], "r1_interval") == "hq_r1_reg_step"
    assert keyword_attribute(calls_by_stage["hq"], "r1_gamma") == "hq_r1_gamma"
    assert keyword_attribute(calls_by_stage["coarse"], "r1_enabled") == "enable_coarse_r1_loss"
    assert keyword_attribute(calls_by_stage["coarse"], "r1_interval") == "coarse_r1_reg_step"
    assert keyword_attribute(calls_by_stage["coarse"], "r1_gamma") == "coarse_r1_gamma"

    discriminator_update_tree = ast.parse(textwrap.dedent(inspect.getsource(Trainer._update_discriminator_stage)))
    update_self_calls = {
        node.func.attr for node in ast.walk(discriminator_update_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
    }
    assert "_discriminator_stage_loss" in update_self_calls

    discriminator_stage_tree = ast.parse(textwrap.dedent(inspect.getsource(Trainer._discriminator_stage_loss)))
    r1_branches = [node for node in ast.walk(discriminator_stage_tree) if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "r1_step"]
    assert len(r1_branches) == 1

    generator_stage_calls = [
        node.func.attr
        for node in ast.walk(train_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr in {"_coarse_generator_stage_loss", "_hq_generator_stage_loss"}
    ]
    assert generator_stage_calls.count("_coarse_generator_stage_loss") == 1
    assert generator_stage_calls.count("_hq_generator_stage_loss") == 1

    coarse_stage_call = next(
        node
        for node in ast.walk(train_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self" and node.func.attr == "_coarse_generator_stage_loss"
    )
    assert isinstance(coarse_stage_call.args[1], ast.Name) and coarse_stage_call.args[1].id == "coarse_resize_in"
    assert isinstance(coarse_stage_call.args[4], ast.Name) and coarse_stage_call.args[4].id == "same_mask"

    coarse_forward_calls = [node for node in ast.walk(train_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "net_coarse"]
    active_coarse_calls = [call for call in coarse_forward_calls if any(keyword.arg == "return_resize_in" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in call.keywords)]
    assert len(active_coarse_calls) == 2

    def guarded_calls(stage_attr: str) -> set[str]:
        calls = set()
        for node in ast.walk(train_tree):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not (isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name) and test.value.id == "self" and test.attr == stage_attr):
                continue
            for statement in node.body:
                for child in ast.walk(statement):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name) and child.func.value.id == "self":
                        calls.add(child.func.attr)
        return calls

    assert {"train_hq", "_hq_generator_stage_loss", "_update_discriminator_stage"}.issubset(guarded_calls("hq_stage_active"))
    assert {"_coarse_generator_stage_loss", "_update_discriminator_stage"}.issubset(guarded_calls("coarse_stage_active"))

    init_tree = ast.parse(textwrap.dedent(inspect.getsource(Trainer.__init__)))
    channel_mismatch_pairs = set()
    for node in ast.walk(init_tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.NotEq) or len(node.comparators) != 1:
            continue
        left = node.left
        right = node.comparators[0]
        if isinstance(left, ast.Name) and isinstance(right, ast.Name):
            channel_mismatch_pairs.add((left.id, right.id))
    assert ("hq_d_channels", "generator_channels") in channel_mismatch_pairs
    assert ("coarse_d_channels", "generator_channels") in channel_mismatch_pairs

    def self_calls(method: object) -> set[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        return {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"}

    coarse_calls = self_calls(Trainer._coarse_generator_stage_loss)
    hq_calls = self_calls(Trainer._hq_generator_stage_loss)
    assert {
        "train_d_coarse",
        "coarse_gan_loss",
        "coarse_identity_embeddings_forward",
        "coarse_id_loss",
        "coarse_vgg_loss_forward",
        "coarse_l1_loss",
    }.issubset(coarse_calls)
    assert {"train_d", "hq_gan_loss", "hq_identity_embeddings_forward", "hq_id_loss", "hq_vgg_loss_forward", "hq_l1_loss"}.issubset(hq_calls)
    assert not {"train_d", "hq_gan_loss", "hq_id_loss"} & coarse_calls
    assert not {"train_d_coarse", "coarse_gan_loss", "coarse_id_loss"} & hq_calls

    def forward_component_owners(method: object) -> set[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        owners = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "forward_components":
                continue
            owner = node.func.value
            if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) and owner.value.id == "self":
                owners.add(owner.attr)
        return owners

    assert forward_component_owners(Trainer._coarse_generator_stage_loss) == {"coarse_hrffa_loss", "coarse_facs_loss"}
    assert forward_component_owners(Trainer._hq_generator_stage_loss) == {"hq_hrffa_loss", "hq_facs_loss"}

    affine_grid_calls = [node for node in ast.walk(train_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "affine_grid"]
    assert len(affine_grid_calls) == 1
    size_keyword = next(item for item in affine_grid_calls[0].keywords if item.arg == "size")
    assert isinstance(size_keyword.value, ast.Call) and isinstance(size_keyword.value.func, ast.Name) and size_keyword.value.func.id == "list"
    size_arg = size_keyword.value.args[0]
    assert isinstance(size_arg, ast.Attribute) and isinstance(size_arg.value, ast.Name) and size_arg.value.id == "dst" and size_arg.attr == "shape"

    train_source_teacher_calls = [
        node
        for node in ast.walk(train_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr in {"hq_identity_embeddings_forward", "coarse_identity_embeddings_forward"}
    ]
    assert {call.func.attr for call in train_source_teacher_calls} == {"hq_identity_embeddings_forward", "coarse_identity_embeddings_forward"}
    assert all(isinstance(call.args[0], ast.Name) and call.args[0].id == "source_identity_faces" for call in train_source_teacher_calls)
    print("PASS: HQ/Coarse gradients, D optimizers, losses, teachers and R1 execution paths remain independent")


def _check_discriminator_training_state_split() -> None:
    save_tree = ast.parse(textwrap.dedent(inspect.getsource(Trainer.save_ckpt)))
    training_state_keys: set[str] | None = None
    for node in ast.walk(save_tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "training_state" or not isinstance(node.value, ast.Dict):
            continue
        training_state_keys = {key.value for key in node.value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
        break
    assert training_state_keys is not None
    assert {
        "optim_d_hq",
        "optim_d_coarse",
        "scaler_d_hq",
        "scaler_d_coarse",
        "lr_scheduler_d_hq",
        "lr_scheduler_d_coarse",
    }.issubset(training_state_keys)
    assert not {"optim_d", "scaler_d", "lr_scheduler_d"} & training_state_keys
    print("PASS: HQ/Coarse discriminator optimizer, scaler and scheduler checkpoint states are split")


def _check_sample_gradient_maps() -> None:
    sample_tree = ast.parse(textwrap.dedent(inspect.getsource(Trainer._save_sample)))
    self_calls = {node.func.attr for node in ast.walk(sample_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"}
    assert {
        "train_d_coarse",
        "coarse_gan_loss",
        "coarse_identity_embeddings_forward",
        "coarse_id_loss",
        "train_d",
        "hq_gan_loss",
        "hq_identity_embeddings_forward",
        "hq_id_loss",
        "loss_grad_map",
    }.issubset(self_calls)
    print("PASS: sample visualization keeps Coarse/HQ discriminator and identity gradient maps")


def main() -> None:
    _check_reconstruction_scope()
    _check_generator_responsibility_boundary()
    _check_discriminator_training_state_split()
    _check_sample_gradient_maps()
    _check_step_boundary()
    _check_scaled_optimizer_step()
    _check_discriminator_overflow_isolation()
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

    resolved = {
        "train": {"batch_size": 8, "stage": "joint"},
        "generator": dict(DEFAULT_GENERATOR_CONFIG),
        "discriminator": {
            "hq": {"base_ch": 64},
            "coarse": {"base_ch": 32},
        },
        "identity": {"provider": "BLENDFACE"},
    }

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
                "training_config": {"semantics_version": TRAINING_SEMANTICS_VERSION, "precision": "fp16", "stage": "joint"},
                "identity_encoders": {
                    "generator": resolved["identity"]["provider"],
                    "coarse_identity_loss": "BLENDFACE",
                    "hq_identity_loss": "BLENDFACE",
                },
                "net_g": {"network_cfg": resolved["generator"], "state_dict": ema_g_state},
                "net_d": {"network_cfg": resolved["discriminator"]["hq"], "state_dict": d_state},
                "net_d_coarse": {
                    "network_cfg": resolved["discriminator"]["coarse"],
                    "state_dict": coarse_d_state,
                },
                "training_state": {"net_g": train_g_state},
            },
            standalone,
        )
        assert resolve_branch_target(standalone) == standalone
        loaded, standalone_parent = _load_branch_checkpoint(
            standalone,
            resolved,
            reset_hq_discriminator=False,
            reset_coarse_discriminator=False,
        )
        assert loaded["step"] == 7500
        assert _require_resume_training_config(loaded) == {
            "semantics_version": TRAINING_SEMANTICS_VERSION,
            "precision": "fp16",
            "stage": "joint",
        }
        assert standalone_parent == {
            "run_id": metadata["run_id"],
            "checkpoint": standalone.name,
            "step": 7500,
            "config_sha256": metadata["config_sha256"],
            "branch_mode": "inherit",
            "generator": {"coarse": "inherit", "hq": "inherit"},
            "discriminator": {"hq": "inherit", "coarse": "inherit"},
        }
        assert _resolve_branch_start_step("inherit", 7500, None) == 7500
        assert _resolve_branch_start_step("inherit", 7500, 123) == 123
        assert _resolve_branch_start_step("hq_rebuild", 7500, None) == 0
        assert _resolve_branch_start_step("hq_rebuild", 7500, 0) == 0
        try:
            _resolve_branch_start_step("hq_rebuild", 7500, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("HQ rebuild 错误接受了非零起始 step")

        branch_g_state, branch_d_state, branch_d_coarse_state = _branch_model_states(
            loaded,
            reset_hq_discriminator=False,
            reset_coarse_discriminator=False,
        )
        assert branch_g_state["marker"].item() == 2  # training G, not EMA G
        assert branch_d_state is not None and branch_d_state["marker"].item() == 3
        assert branch_d_coarse_state is not None and branch_d_coarse_state["marker"].item() == 4

        _, branch_d_state, branch_d_coarse_state = _branch_model_states(
            loaded,
            reset_hq_discriminator=True,
            reset_coarse_discriminator=False,
        )
        assert branch_d_state is None and branch_d_coarse_state is not None
        _, branch_d_state, branch_d_coarse_state = _branch_model_states(
            loaded,
            reset_hq_discriminator=False,
            reset_coarse_discriminator=True,
        )
        assert branch_d_state is not None and branch_d_coarse_state is None
        _, branch_d_state, branch_d_coarse_state = _branch_model_states(
            loaded,
            reset_hq_discriminator=True,
            reset_coarse_discriminator=True,
        )
        assert branch_d_state is None and branch_d_coarse_state is None

        invalid_training_config = dict(loaded)
        invalid_training_config["training_config"] = {"precision": "fp16"}
        try:
            _require_resume_training_config(invalid_training_config)
        except ValueError:
            pass
        else:
            raise AssertionError("缺少 semantics_version 的 checkpoint 被错误接受")

        invalid_training_config["training_config"] = {
            "semantics_version": str(TRAINING_SEMANTICS_VERSION),
            "precision": "fp16",
            "stage": "joint",
        }
        try:
            _require_resume_training_config(invalid_training_config)
        except TypeError:
            pass
        else:
            raise AssertionError("错误类型的 semantics_version 被错误接受")

        invalid_training_config["training_config"] = {
            "semantics_version": TRAINING_SEMANTICS_VERSION,
            "precision": "fp16",
            "stage": "invalid",
        }
        try:
            _require_resume_training_config(invalid_training_config)
        except ValueError:
            pass
        else:
            raise AssertionError("无效 stage 的 training_config 被错误接受")

        invalid_training_config["training_config"] = {
            "semantics_version": TRAINING_SEMANTICS_VERSION,
            "precision": "fp16",
            "stage": "joint",
            "legacy": True,
        }
        try:
            _require_resume_training_config(invalid_training_config)
        except ValueError:
            pass
        else:
            raise AssertionError("含未知字段的 training_config 被错误接受")

        parent = dict(standalone_parent)
        branch = create_run(runs_root, source_config, resolved, name="branch", parent=parent)
        assert json.loads(branch.metadata.read_text(encoding="utf-8"))["parent"] == parent

        # HQ rebuild：允许 HQ 分辨率/结构变化，只继承训练态 Coarse；HQ/HQ D 从零开始，step 固定为 0。
        parent_generator_cfg = {
            "img_resolution": 16,
            "img_channels": 3,
            "id_dim": 8,
            "coarse_resolution": 8,
            "coarse_latent_resolution": 4,
            "coarse_num_latent": 1,
            "coarse_base_ch": 2,
            "coarse_max_ch": 8,
            "hq_bottleneck_resolution": 4,
            "hq_base_ch": 2,
            "hq_max_ch": 8,
        }
        rebuilt_generator_cfg = dict(parent_generator_cfg)
        rebuilt_generator_cfg["img_resolution"] = 32
        rebuilt_generator_cfg["hq_base_ch"] = 4
        rebuilt_generator_cfg["hq_max_ch"] = 16

        parent_generator_model = Generator(**parent_generator_cfg)
        parent_training_state = parent_generator_model.state_dict()
        rebuild_checkpoint_path = root / "step_000007501.pth"
        parent_hq_d_cfg = dict(resolved["discriminator"]["hq"])
        parent_hq_d_cfg["img_resolution"] = 16
        rebuild_resolved = copy.deepcopy(resolved)
        rebuild_resolved["train"]["stage"] = "hq"
        rebuild_resolved["generator"] = rebuilt_generator_cfg
        rebuild_resolved["discriminator"]["hq"]["img_resolution"] = 32
        rebuild_resolved["identity"]["provider"] = resolved["identity"]["provider"]
        torch.save(
            {
                "version": CHECKPOINT_VERSION,
                "step": 7501,
                "run": {"id": metadata["run_id"], "config_sha256": metadata["config_sha256"]},
                "identity_encoders": {
                    "generator": resolved["identity"]["provider"],
                    "coarse_identity_loss": "BLENDFACE",
                    "hq_identity_loss": "BLENDFACE",
                },
                "net_g": {"network_cfg": parent_generator_cfg, "state_dict": parent_training_state},
                "net_d": {"network_cfg": parent_hq_d_cfg, "state_dict": d_state},
                "net_d_coarse": {"network_cfg": rebuild_resolved["discriminator"]["coarse"], "state_dict": coarse_d_state},
                "training_state": {"net_g": parent_training_state},
            },
            rebuild_checkpoint_path,
        )
        rebuild_checkpoint, rebuild_parent = _load_branch_checkpoint(
            rebuild_checkpoint_path,
            rebuild_resolved,
            reset_hq_discriminator=False,
            reset_coarse_discriminator=False,
        )
        assert _branch_generator_mode(rebuild_checkpoint, rebuild_resolved) == "hq_rebuild"
        assert rebuild_parent["branch_mode"] == "hq_rebuild"
        assert rebuild_parent["generator"] == {"coarse": "inherit", "hq": "reset"}
        assert rebuild_parent["discriminator"] == {"hq": "reset", "coarse": "inherit"}
        assert _resolve_branch_start_step(rebuild_parent["branch_mode"], 7501, None) == 0
        with patch("builtins.print") as print_mock:
            _print_hq_rebuild_summary(rebuild_checkpoint, rebuild_resolved, 0)
        printed_lines = [call.args[0] for call in print_mock.call_args_list]
        assert printed_lines == [
            "HQ rebuild:",
            "  HQ 配置变化:",
            "    img_resolution            : 16 -> 32",
            "    hq_base_ch                : 2 -> 4",
            "    hq_max_ch                 : 8 -> 16",
            "  Coarse                  : inherit + frozen",
            "  HQ                      : reset",
            "  HQ Discriminator        : reset",
            f"  Generator identity      : {resolved['identity']['provider']}",
            "  起始 step                : 0",
        ]

        rebuild_g_state, rebuild_hq_d_state, rebuild_coarse_d_state = _branch_model_states(
            rebuild_checkpoint,
            reset_hq_discriminator=False,
            reset_coarse_discriminator=False,
            hq_rebuild=True,
        )
        assert rebuild_hq_d_state is None and rebuild_coarse_d_state is not None
        rebuilt_model = Generator(**rebuilt_generator_cfg)
        fresh_hq_state = {name: tensor.clone() for name, tensor in rebuilt_model.hq.state_dict().items()}
        rebuilt_model.coarse.load_state_dict(_submodule_state_dict(rebuild_g_state, "coarse"), strict=True)
        for name, tensor in parent_generator_model.coarse.state_dict().items():
            torch.testing.assert_close(rebuilt_model.coarse.state_dict()[name], tensor)
        for name, tensor in fresh_hq_state.items():
            torch.testing.assert_close(rebuilt_model.hq.state_dict()[name], tensor)

        incompatible_coarse = copy.deepcopy(rebuild_resolved)
        incompatible_coarse["generator"]["coarse_base_ch"] = 4
        try:
            _load_branch_checkpoint(
                rebuild_checkpoint_path,
                incompatible_coarse,
                reset_hq_discriminator=False,
                reset_coarse_discriminator=False,
            )
        except ValueError as error:
            assert "Coarse" in str(error)
        else:
            raise AssertionError("HQ rebuild 错误接受了 Coarse 架构变化")

        incompatible_provider = copy.deepcopy(rebuild_resolved)
        incompatible_provider["identity"]["provider"] = "BLENDFACE" if resolved["identity"]["provider"] != "BLENDFACE" else "MS1MV3_ARCFACE_R50_FP16"
        try:
            _load_branch_checkpoint(
                rebuild_checkpoint_path,
                incompatible_provider,
                reset_hq_discriminator=False,
                reset_coarse_discriminator=False,
            )
        except ValueError as error:
            assert "identity provider" in str(error)
        else:
            raise AssertionError("HQ rebuild 错误接受了 Generator identity provider 变化")

        joint_rebuild = copy.deepcopy(rebuild_resolved)
        joint_rebuild["train"]["stage"] = "joint"
        try:
            _load_branch_checkpoint(
                rebuild_checkpoint_path,
                joint_rebuild,
                reset_hq_discriminator=False,
                reset_coarse_discriminator=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("非 HQ-only branch 错误接受了 Generator 架构变化")

        changed_generator = copy.deepcopy(resolved)
        changed_generator["generator"]["coarse_resolution"] = 256
        try:
            _load_branch_checkpoint(
                standalone,
                changed_generator,
                reset_hq_discriminator=False,
                reset_coarse_discriminator=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("branch 错误接受了 Generator 架构变更")

        changed_hq_discriminator = copy.deepcopy(resolved)
        changed_hq_discriminator["discriminator"]["hq"]["base_ch"] = 128
        try:
            _load_branch_checkpoint(
                standalone,
                changed_hq_discriminator,
                reset_hq_discriminator=False,
                reset_coarse_discriminator=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("branch 错误继承了架构已变化的 HQ Discriminator")

        _, hq_reset_parent = _load_branch_checkpoint(
            standalone,
            changed_hq_discriminator,
            reset_hq_discriminator=True,
            reset_coarse_discriminator=False,
        )
        assert hq_reset_parent["discriminator"] == {"hq": "reset", "coarse": "inherit"}

        changed_coarse_discriminator = copy.deepcopy(resolved)
        changed_coarse_discriminator["discriminator"]["coarse"]["base_ch"] = 96
        try:
            _load_branch_checkpoint(
                standalone,
                changed_coarse_discriminator,
                reset_hq_discriminator=False,
                reset_coarse_discriminator=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("branch 错误继承了架构已变化的 Coarse Discriminator")

        _, coarse_reset_parent = _load_branch_checkpoint(
            standalone,
            changed_coarse_discriminator,
            reset_hq_discriminator=False,
            reset_coarse_discriminator=True,
        )
        assert coarse_reset_parent["discriminator"] == {"hq": "inherit", "coarse": "reset"}

        changed_both_discriminators = copy.deepcopy(resolved)
        changed_both_discriminators["discriminator"]["hq"]["base_ch"] = 128
        changed_both_discriminators["discriminator"]["coarse"]["base_ch"] = 96
        _, reset_parent = _load_branch_checkpoint(
            standalone,
            changed_both_discriminators,
            reset_hq_discriminator=True,
            reset_coarse_discriminator=True,
        )
        assert reset_parent["discriminator"] == {"hq": "reset", "coarse": "reset"}

        # 只 reset 一边不能掩盖另一边自己的不兼容变更。
        try:
            _load_branch_checkpoint(
                standalone,
                changed_both_discriminators,
                reset_hq_discriminator=True,
                reset_coarse_discriminator=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("reset HQ D 时错误忽略了 Coarse D 架构不兼容")
        try:
            _load_branch_checkpoint(
                standalone,
                changed_both_discriminators,
                reset_hq_discriminator=False,
                reset_coarse_discriminator=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("reset Coarse D 时错误忽略了 HQ D 架构不兼容")

        changed_provider = copy.deepcopy(resolved)
        changed_provider["identity"]["provider"] = "MS1MV3_ARCFACE_R50_FP16"
        _load_branch_checkpoint(
            standalone,
            changed_provider,
            reset_hq_discriminator=False,
            reset_coarse_discriminator=False,
        )

        changed_provider_hq_only = copy.deepcopy(changed_provider)
        changed_provider_hq_only["train"]["stage"] = "hq"
        try:
            _load_branch_checkpoint(
                standalone,
                changed_provider_hq_only,
                reset_hq_discriminator=False,
                reset_coarse_discriminator=False,
            )
        except ValueError as error:
            assert "identity provider" in str(error)
        else:
            raise AssertionError("HQ-only branch 错误接受了 Generator identity provider 变化")

        document = json.loads(first.resolved_config.read_text(encoding="utf-8"))
        document["config"]["train"]["batch_size"] = 16
        first.resolved_config.write_text(json.dumps(document), encoding="utf-8")
        try:
            load_resolved_config(first)
        except ValueError:
            pass
        else:
            raise AssertionError("被修改的 resolved config 未被拒绝")

    print("PASS: run layout, latest/resume, branch/HQ rebuild semantics and config digest")


if __name__ == "__main__":
    main()
