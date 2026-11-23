"""推理源身份裁剪必须匹配训练端的 FFHQ / ArcFace-112 几何约定。"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from misc.face_alignment import (
    make_alignment_grid_theta,
    make_arcface128_to_arcface112_grid,
    make_ffhq_to_arcface_112_grid,
    transform_sampling_grid,
)
from misc.models.id_encoder import IDEncoderProvider, get_alignment_template
from swapface import swapper
from swapface.inference import get_arcface_alignment_template, get_ffhq_alignment_template


class RecordingEncoder(nn.Module):
    def forward(self, faces):
        self.last_input = faces.clone()
        return faces.new_ones((faces.shape[0], 512))


class FakeDetector(nn.Module):
    def detect(self, images):
        return None, [images.shape[0]]


@pytest.mark.parametrize("alignment", ["ffhq", "arcface"])
@pytest.mark.parametrize("resolution", [128, 256, 512])
def test_source_identity_crop_matches_training_geometry(tmp_path, monkeypatch, alignment, resolution):
    device = torch.device("cpu")
    raw_h, raw_w = 480, 560
    y = torch.arange(raw_h)[:, None].expand(raw_h, raw_w)
    x = torch.arange(raw_w)[None, :].expand(raw_h, raw_w)
    image = torch.stack((x % 256, y % 256, (x + y) % 256)).to(torch.uint8)

    template = (
        get_ffhq_alignment_template(resolution, device)
        if alignment == "ffhq"
        else get_arcface_alignment_template(resolution, device)
    )
    landmarks = template[None] * (480.0 / resolution * 0.8) + torch.tensor([60.0, 30.0])

    class FakeGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.network_cfg = {"img_resolution": resolution}
            self.face_alignment = alignment

    monkeypatch.setattr(swapper, "load_generator", lambda _: (FakeGenerator(), IDEncoderProvider.BLENDFACE, 1))
    monkeypatch.setattr(swapper, "RetinaFace", lambda **_: FakeDetector())
    monkeypatch.setattr(swapper, "IDEncoder", lambda _: RecordingEncoder())
    monkeypatch.setattr(swapper, "FaceMasker", lambda **_: nn.Identity())
    monkeypatch.setattr(swapper, "decode_image", lambda *args, **kwargs: image)
    monkeypatch.setattr(swapper, "extract_landmarks", lambda _: landmarks)
    path = tmp_path / "model.pth"
    path.touch()
    engine = swapper.SwapFace(str(path), device="cpu")

    # 训练端的身份裁剪定义独立于被测实例，避免使用同一个错误网格得出假阳性。
    expected_grid = (
        make_ffhq_to_arcface_112_grid(resolution, 1, device)
        if alignment == "ffhq"
        else make_arcface128_to_arcface112_grid(resolution, 1, device)
    )
    torch.testing.assert_close(engine.identity_encoder_grid, expected_grid, atol=0, rtol=0)
    theta = make_alignment_grid_theta(landmarks, template, (raw_h, raw_w), resolution)
    identity_grid = transform_sampling_grid(expected_grid, theta)
    if alignment == "arcface":
        # 用独立的 ArcFace-112 五点对齐公式核对复合网格。
        direct_theta = make_alignment_grid_theta(
            landmarks, torch.as_tensor(get_alignment_template(112)), (raw_h, raw_w), 112
        )
        direct_grid = F.affine_grid(direct_theta, (1, 3, 112, 112), align_corners=False)
        torch.testing.assert_close(identity_grid, direct_grid, atol=2e-6, rtol=0)

    expected_faces = F.grid_sample(
        image.float().unsqueeze(0) / 127.5 - 1,
        identity_grid, mode="bilinear", padding_mode="border", align_corners=False,
    )
    with monkeypatch.context() as context:
        original_grid_sample = swapper.NF.grid_sample
        calls = []

        def record_grid_sample(*args, **kwargs):
            calls.append(1)
            return original_grid_sample(*args, **kwargs)

        context.setattr(swapper.NF, "grid_sample", record_grid_sample)
        engine.extract_identity_embedding(tmp_path / "unused.png")
    assert len(calls) == 1
    torch.testing.assert_close(engine.id_encoder.last_input, expected_faces, atol=0, rtol=0)
