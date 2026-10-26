"""IDLoss 语义区域过滤：不下载外部权重的 CPU 回归测试。"""

import copy
import tomllib
import unittest
from pathlib import Path

import torch
from torch import nn

from misc.models.face_mask import FaceMasker
from swapface.config import DEFAULT_LOSS_CONFIG, resolve_train_config
from swapface.occlusion import gate_identity_gradient
from swapface.train import Trainer


def _config() -> dict:
    return tomllib.loads((Path(__file__).resolve().parents[1] / "experiments/train.toml").read_text(encoding="utf-8"))


class _Dataset:
    def __init__(self, dst: torch.Tensor):
        self.dst = dst

    def next(self):
        dst = self.dst
        return dst, dst, dst, torch.zeros(dst.shape[0], 2, 3), torch.ones(dst.shape[0]), dst


class _Parser:
    def __init__(self, labels: torch.Tensor):
        self.labels = labels
        self.last_input = None

    def predict_labels(self, images):
        self.last_input = images.clone()
        assert not torch.is_grad_enabled()
        assert images.dtype == torch.float32
        return self.labels


class _Occlusion:
    def __init__(self, gate: torch.Tensor):
        self.gate = gate

    def apply(self, dst):
        return dst + 0.1, self.gate


class TestIdentityParsingGradient(unittest.TestCase):
    def test_config_normalization(self):
        raw = _config()
        raw["loss"]["coarse"]["identity"]["exclude_parsing_labels"] = [13, 5, 4, 12]
        raw["loss"]["hq"]["identity"]["exclude_parsing_labels"] = [11]
        resolved = resolve_train_config(raw)
        assert resolved["loss"]["coarse"]["identity"]["exclude_parsing_labels"] == [4, 5, 12, 13]
        assert resolved["loss"]["hq"]["identity"]["exclude_parsing_labels"] == [11]
        assert resolve_train_config(resolved) == resolved
        assert DEFAULT_LOSS_CONFIG["coarse"]["identity"]["exclude_parsing_labels"] is not DEFAULT_LOSS_CONFIG["hq"]["identity"]["exclude_parsing_labels"]

        for invalid in (-1, 19, True, 1.5, "4"):
            broken = copy.deepcopy(raw)
            broken["loss"]["coarse"]["identity"]["exclude_parsing_labels"] = [invalid]
            with self.assertRaises((ValueError, TypeError)):
                resolve_train_config(broken)
        for invalid in ([4, 4], (4, 5), "4,5", None):
            broken = copy.deepcopy(raw)
            broken["loss"]["coarse"]["identity"]["exclude_parsing_labels"] = invalid
            with self.assertRaises((ValueError, TypeError)):
                resolve_train_config(broken)

    def test_predict_labels_raw_parsing_not_binary(self):
        parser = FaceMasker.__new__(FaceMasker)
        nn.Module.__init__(parser)  # noqa: PLC2801 - 跳过下载训练权重的构造器
        parser.mode = "parsing"
        scores = torch.zeros(2, 19, 3, 4)
        scores[:, 4, 0, 1] = 10.0
        scores[:, 12, 2, 3] = 10.0
        parser._predict_scores = lambda images: scores
        labels = parser.predict_labels(torch.rand(2, 3, 3, 4))
        assert labels.shape == (2, 3, 4)
        assert labels.dtype == torch.int64
        assert labels[0, 0, 1] == 4
        assert labels[0, 2, 3] == 12
        assert labels[0, 1, 1] == 0
        parser.mode = "occlusion"
        with self.assertRaisesRegex(ValueError, "parsing"):
            parser.predict_labels(torch.rand(2, 3, 3, 4))

    def test_stage_masks_with_occlusion_and_clean_target(self):
        dst = torch.ones((2, 3, 8, 8))
        labels = torch.zeros((2, 8, 8), dtype=torch.long)
        labels[:, :4, :4] = 4
        labels[:, :4, 4:] = 12
        labels[:, 4:, 4:] = 11
        occlusion_gate = torch.ones((2, 1, 8, 8))
        occlusion_gate[:, :, 4:, :] = 0.25
        parser = _Parser(labels)
        trainer = Trainer.__new__(Trainer)
        trainer.dataset = _Dataset(dst)
        trainer.occlusion = _Occlusion(occlusion_gate)
        trainer.face_parser = parser
        trainer.coarse_excluded_parsing_labels = (4, 12)
        trainer.hq_excluded_parsing_labels = (11,)
        trainer.parsing_exclusion_indices = (torch.tensor([4, 12]), torch.tensor([11]))

        _, augmented_dst, _, _, _, _, (coarse_gate, hq_gate) = trainer._next_training_batch()
        assert torch.allclose(parser.last_input, dst)
        assert torch.allclose(augmented_dst, dst + 0.1)
        assert not coarse_gate.requires_grad and not hq_gate.requires_grad
        assert coarse_gate[:, :, :4, :].count_nonzero() == 0
        assert torch.allclose(coarse_gate[:, :, 4:, :], torch.full_like(coarse_gate[:, :, 4:, :], 0.25))
        assert torch.all(hq_gate[:, :, :4, :] == 1)
        assert torch.allclose(hq_gate[:, :, 4:, :4], torch.full_like(hq_gate[:, :, 4:, :4], 0.25))
        assert hq_gate[:, :, 4:, 4:].count_nonzero() == 0

    def test_only_id_gradient_is_filtered(self):
        x = torch.randn(1, 3, 4, 4, requires_grad=True)
        gate = torch.ones(1, 1, 4, 4)
        gate[:, :, 1:3, 1:3] = 0
        filtered = gate_identity_gradient(x, gate)
        assert torch.equal(filtered, x)
        id_loss = filtered.square().sum()
        rec_loss = 3 * x.sum()

        (id_grad,) = torch.autograd.grad(id_loss, x, retain_graph=True)
        assert torch.equal(id_grad[:, :, 1:3, 1:3], torch.zeros_like(id_grad[:, :, 1:3, 1:3]))
        assert torch.allclose(id_grad, 2 * x.detach() * gate)
        (rec_grad,) = torch.autograd.grad(rec_loss, x, retain_graph=True)
        assert torch.all(rec_grad == 3)
        (total_grad,) = torch.autograd.grad(id_loss + rec_loss, x)
        assert torch.allclose(total_grad, id_grad + rec_grad)
        assert torch.all(total_grad[:, :, 1:3, 1:3] == 3)

    def test_training_stage_wiring(self):
        # 完整穿过两个 Generator stage 的 IDLoss 计算入口，不只测试底层 gate。
        trainer = Trainer.__new__(Trainer)
        trainer.train_d_coarse = lambda x, *args: x.sum() * 0.0
        trainer.train_d = lambda x, *args: x.sum() * 0.0
        trainer.coarse_gan_loss = lambda score: score
        trainer.hq_gan_loss = lambda score: score
        trainer.coarse_identity_embeddings_forward = lambda x: x
        trainer.hq_identity_embeddings_forward = lambda x: x
        trainer.prepare_identity_encoder_faces = lambda x, transform: x
        trainer.coarse_id_loss = lambda generated, reference: generated.sum()
        trainer.hq_id_loss = lambda generated, reference: generated.sum()
        trainer.enable_coarse_wfm_loss = trainer.enable_hq_wfm_loss = False
        trainer.enable_coarse_gaze_loss = trainer.enable_hq_gaze_loss = False
        trainer.enable_coarse_hrffa_loss = trainer.enable_hq_hrffa_loss = False
        trainer.enable_coarse_facs_loss = trainer.enable_hq_facs_loss = False
        trainer.enable_coarse_vgg_loss = trainer.enable_hq_vgg_loss = False
        trainer.enable_coarse_l1_loss = trainer.enable_hq_l1_loss = False
        trainer.log = lambda *args, **kwargs: None

        for stage in ("coarse", "hq"):
            x = torch.ones((1, 3, 4, 4), requires_grad=True)
            gate = torch.ones((1, 1, 8, 8))
            gate[:, :, :4, :4] = 0
            if stage == "coarse":
                loss = trainer._coarse_generator_stage_loss(
                    x, x.detach(), x.detach(), None, None, x.detach(), None, gate
                )
            else:
                # HQ 的传入门控由 _next_training_batch 生成，分辨率与 HQ 生成图一致。
                hq_gate = torch.nn.functional.interpolate(gate, size=x.shape[-2:], mode="area")
                loss = trainer._hq_generator_stage_loss(
                    x, x.detach(), x.detach(), None, None, x.detach(), None, hq_gate
                )
            loss.backward()
            assert torch.equal(x.grad[:, :, :2, :2], torch.zeros_like(x.grad[:, :, :2, :2]))
            assert torch.all(x.grad[:, :, 2:, :] == 1)
            assert torch.all(x.grad[:, :, :2, 2:] == 1)

    def test_no_filter_preserves_behavior(self):
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        assert gate_identity_gradient(x, None) is x
        trainer = Trainer.__new__(Trainer)
        trainer.dataset = _Dataset(x.detach())
        trainer.occlusion = None
        trainer.face_parser = None
        trainer.coarse_excluded_parsing_labels = ()
        trainer.hq_excluded_parsing_labels = ()
        trainer.parsing_exclusion_indices = (None, None)
        assert trainer._next_training_batch()[-1] == (None, None)


if __name__ == "__main__":
    unittest.main()
