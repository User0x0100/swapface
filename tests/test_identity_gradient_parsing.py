"""Single-stage IDLoss semantic gate and clean-image parsing regression (no model downloads)."""

import copy
import tomllib
import unittest
from pathlib import Path

import torch
from torch import nn

from misc.models.face_mask import FaceMasker
from swapface.config import resolve_train_config
from swapface.occlusion import gate_identity_gradient
from swapface.train import Trainer


def config():
    return tomllib.loads((Path(__file__).parents[1] / "experiments/train.toml").read_text())


class FakeDataset:
    def __init__(self, dst): self.dst = dst
    def next(self):
        dst = self.dst
        return dst, dst, dst, torch.zeros(dst.shape[0], 2, 3), torch.ones(dst.shape[0]), dst


class FakeParser:
    def __init__(self, labels): self.labels = labels; self.last_input = None
    def predict_labels(self, images):
        self.last_input = images.clone()
        assert not torch.is_grad_enabled() and images.dtype == torch.float32
        return self.labels


class FakeOcclusion:
    def __init__(self, gate): self.gate = gate
    def apply(self, dst): return dst + .1, self.gate


class TestIdentityParsing(unittest.TestCase):
    def test_config_normalization(self):
        raw = config()
        raw["loss"]["identity"]["exclude_parsing_labels"] = [13, 5, 4, 12]
        resolved = resolve_train_config(raw)
        assert resolved["loss"]["identity"]["exclude_parsing_labels"] == [4, 5, 12, 13]
        assert resolve_train_config(resolved) == resolved
        for bad in (-1, 19, True, 1.5, "4", [4, 4], (4, 5), "4,5", None):
            modified = copy.deepcopy(raw)
            modified["loss"]["identity"]["exclude_parsing_labels"] = [bad] if not isinstance(bad, (list, tuple)) and bad is not None else bad
            with self.assertRaises((ValueError, TypeError)):
                resolve_train_config(modified)

    def test_parser_labels_are_not_binary(self):
        parser = FaceMasker.__new__(FaceMasker)
        nn.Module.__init__(parser)  # noqa: PLC2801 - Construct no-download label parser fixture
        parser.mode = "parsing"
        scores = torch.zeros(2, 19, 3, 4)
        scores[:, 4, 0, 1] = 10.
        scores[:, 12, 2, 3] = 10.
        parser._predict_scores = lambda images: scores
        labels = parser.predict_labels(torch.rand(2, 3, 3, 4))
        assert labels[0, 0, 1] == 4 and labels[0, 2, 3] == 12

    def test_clean_target_parsing_and_occlusion(self):
        dst = torch.ones(2, 3, 8, 8)
        labels = torch.zeros(2, 8, 8, dtype=torch.long)
        labels[:, :4, :4] = 4
        labels[:, :4, 4:] = 12
        gate = torch.ones(2, 1, 8, 8)
        gate[:, :, 4:, :] = .25
        parser = FakeParser(labels)
        trainer = Trainer.__new__(Trainer)
        trainer.dataset = FakeDataset(dst)
        trainer.occlusion = FakeOcclusion(gate)
        trainer.face_parser = parser
        trainer.parsing_exclusion_indices = torch.tensor([4, 12])
        trainer.enable_mask_loss = False
        _, augmented, _, _, _, _, result_gate, weights, teacher = trainer._next_training_batch()
        assert teacher is None
        torch.testing.assert_close(parser.last_input, dst)
        torch.testing.assert_close(augmented, dst + .1)
        torch.testing.assert_close(weights, gate.flatten(1).mean(dim=1))
        assert result_gate[:, :, :4, :].count_nonzero() == 0
        torch.testing.assert_close(result_gate[:, :, 4:, :], torch.full_like(result_gate[:, :, 4:, :], .25))
        assert not result_gate.requires_grad

    def test_identity_only_gate(self):
        x = torch.randn(1, 3, 4, 4, requires_grad=True)
        gate = torch.ones(1, 1, 4, 4)
        gate[:, :, 1:3, 1:3] = 0
        filtered = gate_identity_gradient(x, gate)
        assert torch.equal(filtered, x)
        id_loss = filtered.square().sum()
        rec_loss = 3 * x.sum()
        (id_grad,) = torch.autograd.grad(id_loss, x, retain_graph=True)
        assert torch.equal(id_grad[:, :, 1:3, 1:3], torch.zeros_like(id_grad[:, :, 1:3, 1:3]))
        (total_grad,) = torch.autograd.grad(id_loss + rec_loss, x)
        torch.testing.assert_close(total_grad, id_grad + 3)

    def test_generator_loss_applies_gate_only_to_identity(self):
        trainer = Trainer.__new__(Trainer)
        trainer.train_d = lambda x, *args: x.sum() * 0
        trainer.gan_loss = lambda value: value
        trainer.identity_embeddings_forward = lambda x: x
        trainer.prepare_identity_encoder_faces = lambda x, theta_restore=None: x
        trainer.id_loss = lambda generated, ref: generated.sum()
        trainer._log_identity_threshold_metrics = lambda *args: None
        trainer.enable_wfm_loss = trainer.enable_mask_loss = trainer.enable_gaze_loss = trainer.enable_smirk_loss = trainer.enable_vgg_loss = trainer.enable_l1_loss = False
        trainer.geometry_augmentation_enabled = False
        trainer.log = lambda *args: None
        fake = torch.ones(1, 3, 4, 4, requires_grad=True)
        gate = torch.ones(1, 1, 4, 4)
        gate[:, :, :2, :2] = 0
        loss = trainer._generator_loss(fake, fake.detach(), fake.detach(), torch.zeros(1, 2, 3),
                                       torch.ones(1), fake.detach(), None, gate, None, None)
        loss.backward()
        assert fake.grad[:, :, :2, :2].count_nonzero() == 0
        assert torch.all(fake.grad[:, :, 2:, :] == 1)
        assert torch.all(fake.grad[:, :, :2, 2:] == 1)

    def test_no_filter(self):
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        assert gate_identity_gradient(x, None) is x
        trainer = Trainer.__new__(Trainer)
        trainer.dataset = FakeDataset(x.detach())
        trainer.occlusion = trainer.face_parser = None
        trainer.parsing_exclusion_indices = None
        trainer.enable_mask_loss = False
        assert trainer._next_training_batch()[-3:] == (None, None, None)


if __name__ == "__main__":
    unittest.main()
