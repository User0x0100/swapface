"""Regression coverage for worker RNG, color semantics, affine math and static identity paths."""

import gc
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from misc.face_alignment import ffhq_to_arcface_112, make_ffhq_to_arcface_112_grid
from swapface.config import DEFAULT_DATA_CONFIG
from swapface.dataloader import TrainingDataLoader
from swapface.dataloader_common import ImageDecoderBackend
from swapface.dataloader_native import _color_twist, make_affine_thetas


def config(**changes):
    return DEFAULT_DATA_CONFIG["loader"] | DEFAULT_DATA_CONFIG["augmentation"] | DEFAULT_DATA_CONFIG["sampling"] | {
        "decoder_backend": ImageDecoderBackend.CPU, "decoder_hw_load": 0.,
        "num_threads": 2, "py_num_workers": 0, "prefetch_queue_depth": 1, "reader_prefetch_queue_depth": 1,
        "brightness": 0., "contrast": 0., "saturation": 0., "flip_prob": 0., "same_prob": 0.,
        "rotation_range": (0., 0.), "scale_factor_range": (1., 1.), "tx_range": (0., 0.), "ty_range": (0., 0.),
    } | changes


def require_dali():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA unavailable")
    pytest.importorskip("nvidia.dali")


def test_native_color_center():
    x = torch.tensor([0., 64., 127.5, 255.]).view(1, 1, 1, 4).expand(1, 3, 1, 4)
    with patch("swapface.dataloader_native._uniform", side_effect=[torch.ones(1), torch.tensor([.8]), torch.ones(1)]):
        out = _color_twist(x, 0., .2, 0.)
    torch.testing.assert_close(out, (x - 127.5) * .8 + 127.5)


@pytest.mark.parametrize("size", [32, 128, 256, 512])
def test_affine_reference(size):
    # Independent homogeneous pixel-space reference, not the simplified formula.
    b = 32
    device = torch.device("cpu")
    ranges = ((-40., 40.), (.6, 1.5), (-.2, .2), (-.3, .3))
    torch.manual_seed(45)
    augment, restore = make_affine_thetas(b, size, *ranges, device)
    torch.manual_seed(45)
    angle = -torch.empty(b).uniform_(*ranges[0]) * (torch.pi / 180)
    scale, tx, ty = [torch.empty(b).uniform_(*r) for r in ranges[1:]]
    c, s = angle.cos() * scale, angle.sin() * scale
    matrix = torch.eye(3).repeat(b, 1, 1)
    matrix[:, 0, 0], matrix[:, 0, 1] = c, -s
    matrix[:, 1, 0], matrix[:, 1, 1] = s, c
    center = size / 2
    matrix[:, 0, 2] = center + tx * size - c * center + s * center
    matrix[:, 1, 2] = center + ty * size - s * center - c * center
    norm_to_pixel = torch.tensor([[center, 0., center-.5], [0., center, center-.5], [0., 0., 1.]])
    expected = torch.linalg.inv(norm_to_pixel) @ matrix @ norm_to_pixel
    torch.testing.assert_close(restore, expected[:, :2], atol=5e-7, rtol=0)
    torch.testing.assert_close(augment, torch.linalg.inv(expected)[:, :2], atol=8e-7, rtol=0)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["ffhq", "direct", "mixed"])
def test_identity_modes(tmp_path, backend, mode):
    if backend == "cuda":
        require_dali()
    ffhq, arc = tmp_path / "ffhq", tmp_path / "arc"
    ffhq.mkdir()
    arc.mkdir()
    Image.new("RGB", (112, 112), (40, 80, 120)).save(ffhq / "face.png")
    Image.new("RGB", (112, 112), (160, 180, 200)).save(arc / "face.png")
    arc_source = (arc, 0., "arcface")
    src = [ffhq] if mode == "ffhq" else [arc_source] if mode == "direct" else [ffhq, arc_source]
    device = torch.device(backend)
    loader = TrainingDataLoader(batch_size=8, device=device, img_resolution=128, src=src, dst=[ffhq], **config(flip_prob=1.))
    inner = loader._loader
    assert inner.identity_mode == mode
    if backend == "cuda":
        assert ("src_arcface_faces" in inner._output_map) == (mode != "ffhq")
        assert ("src_arcface_mask" in inner._output_map) == (mode == "mixed")
    else:
        dataset = inner.loader.dataset
        if mode == "ffhq":
            assert dataset.identity_placeholder.numel() == 0
        if mode == "direct":
            assert inner.source_ffhq_grid is None
    module = "swapface.dataloader_dali" if backend == "cuda" else "swapface.dataloader_native"
    if mode != "mixed":
        with patch(module + ".prepare_source_identity_faces", side_effect=AssertionError("unused mixed branch")):
            source, _, _, _, _, identity = loader.next()
    else:
        source, _, _, _, _, identity = loader.next()
    grid = make_ffhq_to_arcface_112_grid(128, 8, device)
    expected_ffhq = ffhq_to_arcface_112(source, grid)
    expected_direct = torch.tensor([160., 180., 200.], device=device).view(1, 3, 1, 1).div(127.5).sub(1).expand_as(identity)
    direct_mask = (source[:, 0, 64, 64] > 0)[:, None, None, None]
    torch.testing.assert_close(identity, torch.where(direct_mask, expected_direct, expected_ffhq), atol=2e-5, rtol=0)
    del loader
    gc.collect()


def test_dali_color_and_zero_workers(tmp_path):
    require_dali()
    data = np.empty((32, 32, 3), dtype=np.uint8)
    data[:, :10] = 48
    data[:, 10:22] = 128
    data[:, 22:] = 208
    Image.fromarray(data).save(tmp_path / "face.png")
    loader = TrainingDataLoader(batch_size=8, device=torch.device("cuda"), img_resolution=32,
                                src=[tmp_path], dst=[tmp_path], **config(contrast=.4))
    _, dst, canonical, _, _, _ = loader.next()
    torch.testing.assert_close(dst, canonical)
    pixels = (dst[:, 0, 16, [5, 16, 27]] + 1) * 127.5
    # A correct contrast transform preserves the center and has symmetric slopes.
    factor = (pixels[:, 2] - pixels[:, 0]) / 160.
    torch.testing.assert_close(pixels[:, 1], 127.5 + .5 * factor, atol=2e-4, rtol=0)
    assert ((factor >= .6 - 1e-5) & (factor <= 1.4 + 1e-5)).all()


@pytest.mark.parametrize("geometry", [False, True])
def test_dali_color_unit_range_equivalence(tmp_path, geometry):
    require_dali()
    from swapface import dataloader_dali as dali

    # Include dark/bright and highly saturated pixels, with values outside the
    # nominal range after color twist. Clamp must still occur after interpolation.
    rng = np.random.default_rng(31)
    pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "face.png")
    cfg = config(brightness=.2, contrast=.4, saturation=.3)
    if geometry:
        cfg.update(rotation_range=(13., 13.), scale_factor_range=(.83, .83),
                   tx_range=(.17, .17), ty_range=(-.11, -.11))

    color_twist = dali.fn.color_twist
    normalize = dali._normalize_chw

    def fixed_color(image, **kwargs):
        return color_twist(image, device=kwargs["device"], brightness=1.2, contrast=1.4, saturation=.7)

    def load():
        loader = TrainingDataLoader(batch_size=2, device=torch.device("cuda"), img_resolution=64,
                                    src=[tmp_path], dst=[tmp_path], **cfg)
        result = tuple(value.clone() for value in loader.next())
        del loader
        gc.collect()
        return result

    # Reference: the previous corrected pipeline multiplied by 255 immediately
    # after color twist and normalized both target branches using that scale.
    with (
        patch.object(dali.fn, "color_twist", side_effect=lambda image, **kw: fixed_color(image, **kw) * 255.),
        patch.object(dali, "_normalize_chw", side_effect=lambda image, pixel_max=255.: normalize(image, 255.)),
    ):
        reference = load()

    with patch.object(dali.fn, "color_twist", side_effect=fixed_color):
        actual = load()

    for current, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(current, expected, atol=2e-6, rtol=0)
    assert actual[2].min() == -1 and actual[2].max() == 1
    assert actual[1].min() >= -1 and actual[1].max() <= 1


def test_dali_fork_independent_rng(tmp_path):
    require_dali()
    for i in range(256):
        Image.new("RGB", (32, 32), (i, 0, 0)).save(tmp_path / f"{i}.png")
    # DALI fork must start before any CUDA context in the child. Fresh interpreter required.
    code = """
import sys
import torch
from swapface.dataloader import TrainingDataLoader
from swapface.config import DEFAULT_DATA_CONFIG
from swapface.dataloader_common import ImageDecoderBackend
cfg = DEFAULT_DATA_CONFIG['loader'] | DEFAULT_DATA_CONFIG['augmentation'] | DEFAULT_DATA_CONFIG['sampling']
cfg.update(py_num_workers=4, py_start_method='fork', num_threads=2, prefetch_queue_depth=1,
           reader_prefetch_queue_depth=1, decoder_backend=ImageDecoderBackend.CPU,
           decoder_hw_load=0., brightness=0., contrast=0., saturation=0., flip_prob=0., same_prob=1.,
           rotation_range=(0.,0.), scale_factor_range=(1.,1.), tx_range=(0.,0.), ty_range=(0.,0.))
loader = TrainingDataLoader(batch_size=8, device=torch.device('cuda:0'), img_resolution=32,
                            src=[sys.argv[1]], dst=[sys.argv[1]], **cfg)
batches = [((loader.next()[1][:,0,0,0]+1)*127.5).round().int().tolist() for _ in range(4)]
assert all(len(set(b)) > 2 for b in batches), batches
print(batches)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=Path(__file__).parents[1],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
