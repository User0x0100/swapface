"""Contract tests for the buffalo_l WebFace600K PyTorch identity encoder."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from .iresnet import IBasicBlock, iresnet50, iresnet100, w600k_iresnet50
from .model import IDEncoder, IDEncoderProvider, IDEncoderProviderConfig


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=(2, 3)))


class TestW600KIDEncoder(unittest.TestCase):
    def test_provider_uses_shared_repo_and_fused_backbone(self) -> None:
        provider = IDEncoderProvider.WEBFACE600K_ARCFACE_R50.value
        self.assertEqual(provider.weight_filename, "w600k_r50.pth")
        self.assertIs(provider.backbone, w600k_iresnet50)
        self.assertEqual(IDEncoderProvider.BLENDFACE.value.weight_filename, "blendface.pth")

    def test_fused_block_layout(self) -> None:
        model = w600k_iresnet50()
        block = model.layer2[0]
        self.assertIsInstance(block, IBasicBlock)
        self.assertIsNotNone(block.conv1.bias)
        self.assertIsNotNone(block.conv2.bias)
        self.assertIsInstance(block.downsample, nn.Conv2d)
        self.assertIsNotNone(block.downsample.bias)
        self.assertIsInstance(block.bn2, nn.Identity)
        self.assertIsInstance(block.bn3, nn.Identity)
        self.assertEqual(block(torch.randn(1, 64, 8, 8)).shape, (1, 128, 4, 4))
        self.assertIn("bn1.weight", model.state_dict())
        self.assertNotIn("bn2.weight", model.state_dict())
        self.assertEqual(len(model.state_dict()), 263)

    def test_standard_backbones_unchanged(self) -> None:
        standard = iresnet50()
        block = standard.layer2[0]
        self.assertIsInstance(block.bn2, nn.BatchNorm2d)
        self.assertIsInstance(block.bn3, nn.BatchNorm2d)
        self.assertIsInstance(block.downsample, nn.Sequential)
        self.assertIsNone(block.conv1.bias)
        self.assertIsNone(block.conv2.bias)
        self.assertIsNone(standard.conv1.bias)
        self.assertIn("bn2.weight", standard.state_dict())
        self.assertEqual(len(standard.state_dict()), 475)
        self.assertEqual(len(iresnet100().state_dict()), 925)

    def test_fused_rejects_unsupported_zero_residual_init(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero_init_residual"):
            iresnet50(fused_bn=True, zero_init_residual=True)

    def test_pth_loader_and_unit_norm(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            weights = Path(d) / "tiny.pth"
            torch.save(_TinyBackbone().state_dict(), weights)
            provider = IDEncoderProvider.WEBFACE600K_ARCFACE_R50
            with (
                patch("misc.models.id_encoder.model.hf_hub_download", return_value=str(weights)) as download,
                patch.object(IDEncoderProviderConfig, "backbone", _TinyBackbone),
            ):
                encoder = IDEncoder(provider)
            download.assert_called_once_with(repo_id="leehx/model_hub", filename="w600k_r50.pth")
            self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
            faces = torch.randn(2, 3, 112, 112)
            result = encoder(faces)
            self.assertEqual(tuple(result.shape), (2, 512))
            torch.testing.assert_close(result.norm(dim=1), torch.ones(2))
            # WebFace600K 输出原生骨干特征的 L2 归一化结果，不再执行 emap。
            self.assertFalse(hasattr(encoder, "inswapper_emap"))
            torch.testing.assert_close(result, torch.nn.functional.normalize(encoder.backbone(faces), p=2, dim=1))

    def test_strict_load_rejects_wrong_keys(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            weights = Path(d) / "tiny.pth"
            torch.save({"bogus.weight": torch.ones(1)}, weights)
            provider = IDEncoderProvider.WEBFACE600K_ARCFACE_R50
            with (
                patch("misc.models.id_encoder.model.hf_hub_download", return_value=str(weights)),
                patch.object(IDEncoderProviderConfig, "backbone", _TinyBackbone),
            ):
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    IDEncoder(provider)


if __name__ == "__main__":
    unittest.main()
