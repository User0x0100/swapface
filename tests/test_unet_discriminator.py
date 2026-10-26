"""BasicSR UNetDiscriminatorSN 的结构、训练接入和配置回归测试。"""

import copy
import re
import tomllib
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from losses import DiscriminatorAdversarialLoss, GeneratorAdversarialLoss, WeakFeatureMatchingLoss
from models.discriminator import DiscriminatorType, UNetDiscriminatorSN, build_discriminator
from swapface.config import resolve_train_config
from swapface.train import Trainer


def _raw_config() -> dict:
    return tomllib.loads((Path(__file__).parents[1] / "experiments/train.toml").read_text())


def _stub_trainer(model: UNetDiscriminatorSN) -> tuple[Trainer, dict[str, float]]:
    trainer = Trainer.__new__(Trainer)
    trainer._completed_step = 0
    trainer.net_d_coarse = model
    trainer.train_d_coarse = model
    trainer.coarse_gan_loss = GeneratorAdversarialLoss(weight=1.0, reduction="mean")
    trainer.d_loss = DiscriminatorAdversarialLoss(weight=1.0, reduction="mean")
    trainer.amp_enabled = False
    trainer.amp_dtype = None
    trainer.enable_coarse_wfm_loss = False
    trainer.enable_coarse_gaze_loss = False
    trainer.enable_coarse_hrffa_loss = False
    trainer.enable_coarse_facs_loss = False
    trainer.enable_coarse_vgg_loss = False
    trainer.enable_coarse_l1_loss = False
    trainer.coarse_reconstruction_scope = "all"
    trainer.coarse_identity_embeddings_forward = lambda x: x.mean(dim=(2, 3))
    trainer.coarse_id_loss = lambda a, b: (a - b).square().mean() * 0
    trainer.prepare_identity_encoder_faces = lambda x, theta_restore: x
    logs: dict[str, float] = {}
    trainer.log = lambda key, value, **kwargs: logs.__setitem__(key, float(value.detach()))
    return trainer, logs


class TestUNetDiscriminatorSN(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._threads)

    def test_config(self) -> None:
        raw = _raw_config()
        assert "spatial_weight" not in resolve_train_config(raw)["loss"]["coarse"]["gan"]
        raw["discriminator"]["coarse"]["type"] = DiscriminatorType.UNET.name
        cfg = resolve_train_config(raw)["discriminator"]["coarse"]
        assert cfg == {
            "type": "UNET", "img_resolution": 128, "img_channels": 3,
            "base_ch": 64, "skip_connection": True,
        }
        resolved = resolve_train_config(raw)
        assert resolved["loss"]["coarse"]["r1"]["enable"] is False
        assert resolved["loss"]["hq"]["r1"]["enable"] is True
        model = build_discriminator(cfg)
        assert isinstance(model, UNetDiscriminatorSN) and model.network_cfg == cfg
        assert model.conv3.out_channels == 8 * cfg["base_ch"]

        compact = copy.deepcopy(raw)
        compact["discriminator"]["coarse"]["base_ch"] = 16
        compact["discriminator"]["coarse"].pop("max_ch")
        compact_cfg = resolve_train_config(compact)["discriminator"]["coarse"]
        assert compact_cfg["base_ch"] == 16 and "max_ch" not in compact_cfg
        assert build_discriminator(compact_cfg).conv3.out_channels == 128
        raw["discriminator"]["coarse"]["skip_connection"] = False
        assert resolve_train_config(raw)["discriminator"]["coarse"]["skip_connection"] is False
        # 即使 raw 配置里保留别的 D 的 max_ch，UNET 也不消费该参数。
        ignored = copy.deepcopy(raw)
        ignored["discriminator"]["coarse"]["max_ch"] = 128
        assert "max_ch" not in resolve_train_config(ignored)["discriminator"]["coarse"]
        invalid = copy.deepcopy(raw)
        invalid["loss"]["coarse"]["gan"]["spatial_weight"] = 0.25
        with self.assertRaisesRegex(ValueError, "已删除"):
            resolve_train_config(invalid)
        # HQ 可以独立选择 UNET，默认全分辨率为 512，且只关闭 HQ R1。
        hq_only = _raw_config()
        hq_only["discriminator"]["hq"]["type"] = "UNET"
        hq_cfg = resolve_train_config(hq_only)
        assert hq_cfg["discriminator"]["hq"] == {
            "type": "UNET", "img_resolution": 512, "img_channels": 3,
            "base_ch": 64, "skip_connection": True,
        }
        assert hq_cfg["loss"]["hq"]["r1"]["enable"] is False
        assert hq_cfg["loss"]["coarse"]["r1"]["enable"] is True
        assert build_discriminator(hq_cfg["discriminator"]["hq"]).conv3.out_channels == 512
        # HQ 和 Coarse 同时启用，两个 R1 都关闭，默认其他损失不变。
        both = copy.deepcopy(raw)
        both["discriminator"]["hq"]["type"] = "UNET"
        resolved_both = resolve_train_config(both)
        assert all(resolved_both["loss"][stage]["r1"]["enable"] is False for stage in ("hq", "coarse"))
        assert all("max_ch" not in resolved_both["discriminator"][stage] for stage in ("hq", "coarse"))
        # HQ 分辨率失配必须按 generator.img_resolution 校验。
        invalid = copy.deepcopy(hq_only)
        invalid["discriminator"]["hq"]["img_resolution"] = 256
        with self.assertRaisesRegex(ValueError, "必须等于 generator.img_resolution"):
            resolve_train_config(invalid)
        # 非 128 Coarse 分辨率必须与生成器一致。
        for resolution in (64, 256):
            variable = copy.deepcopy(raw)
            variable["generator"]["coarse_resolution"] = resolution
            variable["discriminator"]["coarse"]["img_resolution"] = resolution
            checked = resolve_train_config(variable)
            assert checked["discriminator"]["coarse"]["img_resolution"] == resolution
            assert checked["loss"]["coarse"]["r1"]["enable"] is False
        invalid = copy.deepcopy(raw)
        invalid["generator"]["coarse_resolution"] = 64
        with self.assertRaisesRegex(ValueError, "必须等于 generator.coarse_resolution"):
            resolve_train_config(invalid)

    def test_basicsr_structure_and_gradients(self) -> None:
        model = UNetDiscriminatorSN(num_in_ch=3, num_feat=4)
        self.assertEqual([getattr(model, f"conv{i}").out_channels for i in range(10)],
                         [4, 8, 16, 32, 16, 8, 4, 4, 4, 1])
        for index in range(10):
            layer = getattr(model, f"conv{index}")
            self.assertEqual(hasattr(layer, "weight_orig"), 1 <= index <= 8)
        self.assertFalse(any("global" in key for key in model.state_dict()))
        x = torch.randn(2, 3, 128, 128, requires_grad=True)
        real = torch.randn(2, 3, 128, 128)
        logits, fake_feats = model(x, return_feats=True)
        self.assertEqual(logits.shape, (2, 1, 128, 128))
        self.assertEqual([f.shape[-2:] for f in fake_feats], [(128, 128), (64, 64), (32, 32), (16, 16)])
        with torch.no_grad():
            _, real_feats = model(real, return_feats=True)
        wfm = WeakFeatureMatchingLoss({0: 0.5, 1: 1.0, 2: 1.0, 3: 1.0})
        loss = GeneratorAdversarialLoss(weight=1, reduction="mean")(logits) + wfm(fake_feats, real_feats)
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert model.conv9.weight.grad is not None
        assert model.conv3.weight_orig.grad is not None

    def test_reference_forward_and_skip(self) -> None:
        model = UNetDiscriminatorSN(num_in_ch=3, num_feat=4).eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            # 根据 BasicSR 的 forward 独立展开，防止引入 global head、concat skip 等结构偏差。
            x0 = F.leaky_relu(model.conv0(x), 0.2)
            x1 = F.leaky_relu(model.conv1(x0), 0.2)
            x2 = F.leaky_relu(model.conv2(x1), 0.2)
            x3 = F.leaky_relu(model.conv3(x2), 0.2)
            up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
            x4 = F.leaky_relu(model.conv4(up(x3)), 0.2) + x2
            x5 = F.leaky_relu(model.conv5(up(x4)), 0.2) + x1
            x6 = F.leaky_relu(model.conv6(up(x5)), 0.2) + x0
            expected = model.conv9(F.leaky_relu(model.conv8(F.leaky_relu(model.conv7(x6), 0.2)), 0.2))
            torch.testing.assert_close(model(x), expected)
            model.skip_connection = False
            without_skip = model(x)
            assert without_skip.shape == expected.shape
            assert not torch.allclose(without_skip, expected)
        with self.assertRaisesRegex(ValueError, "8 整除"):
            model(torch.randn(1, 3, 127, 128))

    def test_variable_resolution_forward(self) -> None:
        model = UNetDiscriminatorSN(num_in_ch=3, num_feat=4).eval()
        with torch.no_grad():
            for resolution in (64, 128, 256):
                score, features = model(torch.randn(1, 3, resolution, resolution), return_feats=True)
                assert score.shape == (1, 1, resolution, resolution)
                assert [t.shape[-1] for t in features] == [resolution // (2 ** i) for i in range(4)]

    def test_paired_inputs(self) -> None:
        model = UNetDiscriminatorSN(num_in_ch=3, num_feat=4).eval()
        fake, real = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            separate_fake, separate_real = model(fake), model(real)
            paired = model(torch.cat((fake, real)), True)
        fake_pair, real_pair = paired.chunk(2, dim=0)
        torch.testing.assert_close(fake_pair, separate_fake)
        torch.testing.assert_close(real_pair, separate_real)

    def test_discriminator_forces_r1_off(self) -> None:
        model = UNetDiscriminatorSN(num_in_ch=3, num_feat=4)
        trainer, logs = _stub_trainer(model)
        fake, real = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)
        for step in (0, 1):
            trainer._completed_step = step
            logs.clear()
            loss = trainer._discriminator_stage_loss(
                "coarse", fake, real, model, model,
                r1_enabled=True, r1_interval=16, r1_gamma=10,
            )
            assert loss.isfinite() and logs["coarse_d_loss"] >= 0
            assert "coarse_r1_loss_raw" not in logs
            assert "coarse_r1_loss" not in logs
            loss.backward()
            assert model.conv9.weight.grad is not None and torch.isfinite(model.conv9.weight.grad).all()
            model.zero_grad(set_to_none=True)

    def test_generator_with_optional_wfm(self) -> None:
        model = UNetDiscriminatorSN(num_in_ch=3, num_feat=4)
        for wfm_enabled in (False, True):
            trainer, logs = _stub_trainer(model)
            trainer.enable_coarse_wfm_loss = wfm_enabled
            if wfm_enabled:
                trainer.coarse_wfm_loss = WeakFeatureMatchingLoss({0: 0.5, 1: 1.0})
            fake = torch.randn(2, 3, 128, 128, requires_grad=True)
            total = trainer._coarse_generator_stage_loss(
                fake, fake.detach(), None, None, None,
                torch.zeros(2, 3), None, None,
            )
            assert "coarse_gan_loss" in logs
            assert ("coarse_wfm_loss" in logs) == wfm_enabled
            total.backward()
            assert fake.grad is not None and torch.isfinite(fake.grad).all()

    def test_hq_generator_wfm_and_r1(self) -> None:
        # HQ 使用相同的单空间输出协议，连同 WFM 与显式 R1 防护一起覆盖。
        model = UNetDiscriminatorSN(num_in_ch=3, num_feat=4)
        trainer, logs = _stub_trainer(model)
        trainer.net_d = model
        trainer.train_d = model
        trainer.hq_gan_loss = GeneratorAdversarialLoss(weight=1.0, reduction="mean")
        trainer.hq_identity_embeddings_forward = lambda x: x.mean(dim=(2, 3))
        trainer.hq_id_loss = lambda a, b: (a - b).square().mean() * 0
        trainer.hq_reconstruction_scope = "all"
        trainer.enable_hq_gaze_loss = False
        trainer.enable_hq_hrffa_loss = False
        trainer.enable_hq_facs_loss = False
        trainer.enable_hq_vgg_loss = False
        trainer.enable_hq_l1_loss = False
        trainer.enable_hq_wfm_loss = True
        trainer.hq_wfm_loss = WeakFeatureMatchingLoss({0: 0.5, 1: 1.0, 2: 1.0, 3: 1.0})
        fake = torch.randn(2, 3, 128, 128, requires_grad=True)
        real = torch.randn_like(fake)
        loss = trainer._hq_generator_stage_loss(
            fake, real, None, None, None, torch.zeros(2, 3), None, None,
        )
        assert "hq_gan_loss" in logs and "hq_wfm_loss" in logs
        loss.backward()
        assert fake.grad is not None and torch.isfinite(fake.grad).all()
        model.zero_grad(set_to_none=True)
        logs.clear()
        trainer._completed_step = 0
        d_loss = trainer._discriminator_stage_loss(
            "hq", fake.detach(), real, model, model,
            r1_enabled=True, r1_interval=16, r1_gamma=10.0,
        )
        d_loss.backward()
        assert "hq_d_loss" in logs
        assert "hq_r1_loss" not in logs and "hq_r1_loss_raw" not in logs
        assert model.conv9.weight.grad is not None and torch.isfinite(model.conv9.weight.grad).all()

    def test_hq_512_forward(self) -> None:
        model = UNetDiscriminatorSN(num_in_ch=3, num_feat=4).eval()
        with torch.no_grad():
            scores, features = model(torch.randn(1, 3, 512, 512), return_feats=True)
        assert scores.shape == (1, 1, 512, 512)
        assert [feature.shape[-2:] for feature in features] == [
            (512, 512), (256, 256), (128, 128), (64, 64),
        ]

    def test_sample_gradient_rms(self) -> None:
        model = UNetDiscriminatorSN(num_in_ch=3, num_feat=4).eval()
        trainer, _ = _stub_trainer(model)
        generated = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            rows = trainer._sample_stage_grad_maps(
                "coarse", generated, generated, None, None, None,
                torch.zeros(2, 3), None, (128, 128), None,
            )
        assert len(rows) == 2
        assert rows[0][0].startswith("COARSE GAN GRAD")
        assert rows[1][0].startswith("COARSE ID GRAD")
        x = generated.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(trainer.coarse_gan_loss(model(x)), x)
        expected = f"{float(grad.square().mean().sqrt()):.2e}"
        assert re.search(r"\bRMS ([0-9.eE+-]+)", rows[0][0]).group(1) == expected


if __name__ == "__main__":
    unittest.main()
