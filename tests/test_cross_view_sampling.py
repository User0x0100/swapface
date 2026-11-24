"""Cross-view reconstruction sampling and both real loader backends."""

import copy
import tomllib
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from swapface.config import DEFAULT_DATA_CONFIG, resolve_train_config
from swapface.dataloader import ImageDecoderBackend, TrainingDataLoader
from swapface.dataloader_common import ImagePairSampler, build_image_pools


@pytest.fixture
def sources(tmp_path):
    src, dst, cross = (tmp_path / name for name in ("src", "dst", "cross"))
    for folder in (src, dst, cross):
        folder.mkdir()

    def image(path, color):
        Image.new("RGB", (128, 128), color).save(path)

    image(src / "src.png", (200, 0, 0))
    image(dst / "dst.png", (210, 0, 0))
    for identity, count, red in (("a", 2, 30), ("b", 5, 60), ("singleton", 1, 90)):
        folder = cross / identity
        folder.mkdir()
        for i in range(count):
            image(folder / f"{i}.png", (red, i * 30, 0))
        (folder / "metadata.txt").write_text("ignored")
    image(cross / "ignored.png", (120, 0, 0))
    (cross / "empty").mkdir()
    return src, dst, cross


def identity_source(path, adjustment=0.0):
    return path, adjustment, "ffhq", "identity"


def test_sampling(sources):
    src, dst, cross = sources
    identity = identity_source(cross)
    pools, _, info = build_image_pools([identity, identity_source(cross, 1.)])
    assert info[0][1] == 8
    assert info[1][2] == pytest.approx(2 * info[0][2])
    assert pools[0].index is pools[1].index
    assert {Path(group.root).name for group in pools[0].index.groups} == {"a", "b", "singleton"}

    from swapface.dataloader_common import _scan_image_index
    with patch("swapface.dataloader_common._scan_image_index", wraps=_scan_image_index) as scan:
        sampler = ImagePairSampler([identity], [identity], .2, .75)
        assert scan.call_count == 1
    assert sampler.src_pools[0].index is sampler.dst_pools[0].index
    rng = np.random.default_rng(42)
    counts = Counter()
    identities = Counter()
    swap_identities = Counter()
    for _ in range(10000):
        source, target, same, arcface = sampler.sample(rng)
        assert not arcface
        assert Path(source).parent.parent == Path(target).parent.parent == cross
        if not same:
            assert Path(source).parent != Path(target).parent
            counts["swap"] += 1
            swap_identities[Path(source).parent.name] += 1
        elif source == target:
            counts["same"] += 1
        else:
            assert Path(source).parent == Path(target).parent
            counts["cross"] += 1
            identities[Path(source).parent.name] += 1
    assert counts["swap"] / 10000 == pytest.approx(.8, abs=.02)
    assert counts["same"] / 10000 == pytest.approx(.05, abs=.02)
    assert counts["cross"] / 10000 == pytest.approx(.15, abs=.02)
    assert identities["a"] / counts["cross"] == pytest.approx(.5, abs=.05)
    assert "singleton" not in identities
    for name in ("a", "b", "singleton"):
        assert swap_identities[name] / counts["swap"] == pytest.approx(1 / 3, abs=.025)

    for prob, cross_prob, expected in ((0., 1., False), (1., 0., True)):
        sampler = ImagePairSampler([src, identity], [dst, identity], prob, cross_prob)
        for _ in range(100):
            source, target, same, _ = sampler.sample(rng)
            assert same == expected and (source == target) == expected
            if not same and Path(source).parent.parent == Path(target).parent.parent == cross:
                assert Path(source).parent != Path(target).parent


def test_cross_view_skips_regular_sampling(sources):
    src, dst, cross = sources
    sampler = ImagePairSampler([src], [dst, identity_source(cross)], 1., 1.)
    rng = np.random.default_rng(123)
    with (
        patch.object(sampler, "_sample_group", side_effect=AssertionError("unexpected regular group sampling")),
        patch.object(sampler, "_sample_file", side_effect=AssertionError("unexpected regular file sampling")),
    ):
        for _ in range(100):
            source, target, same, arcface = sampler.sample(rng)
            assert same and not arcface and source != target
            assert Path(source).parent == Path(target).parent
            assert Path(source).parent.parent == cross


def test_effective_sampling_log(sources, tmp_path, capsys):
    src, dst, cross = sources
    identity = identity_source(cross)
    ImagePairSampler([src, identity], [dst, identity], .2, .75)
    log = capsys.readouterr().out
    assert "identities: 3 multi_view_identities: 2" in log
    assert "swap: 80.000%" in log
    assert "same_image: 5.000%" in log
    assert "cross_view: 15.000%" in log
    assert "falling back" not in log

    single = tmp_path / "single"
    group = single / "person"
    group.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(group / "one.png")
    ImagePairSampler([src], [identity_source(single)], .2, .75)
    log = capsys.readouterr().out
    assert "identities: 1 multi_view_identities: 0" in log
    assert "swap: 80.000%" in log
    assert "same_image: 20.000%" in log
    assert "cross_view: 0.000%" in log
    assert "falling back to same-image reconstruction" in log

    ImagePairSampler([src], [dst], .2)
    log = capsys.readouterr().out
    assert "same_image: 20.000%" in log
    assert "cross_view: 0.000%" in log
    assert "falling back" not in log

    ImagePairSampler([src], [identity], .2, 0.)
    log = capsys.readouterr().out
    assert "same_image: 20.000%" in log
    assert "cross_view: 0.000%" in log
    assert "falling back" not in log


def test_flat_sampling_unchanged(sources):
    src, dst, _ = sources
    for i in range(3):
        Image.new("RGB", (8, 8)).save(src / f"{i}.png")
    sampler = ImagePairSampler([src, (dst, 1., "arcface")], [dst, src], .2)
    rng = np.random.default_rng(123)
    reference_rng = np.random.default_rng(123)

    def old_sample(pools, cdf):
        pool = pools[min(int(np.searchsorted(cdf, reference_rng.random(), side="right")), len(pools) - 1)]
        group = pool.index.groups[0]
        path = str(Path(group.root) / group.file_names[int(reference_rng.integers(len(group.file_names)))])
        return path, pool.alignment == "arcface"

    for _ in range(500):
        target, _ = old_sample(sampler.dst_pools, sampler.dst_cdf)
        same = bool(reference_rng.random() < .2)
        source, arcface = (target, False) if same else old_sample(sampler.src_pools, sampler.src_cdf)
        assert sampler.sample(rng) == (source, target, same, arcface)


def test_mixed_and_singleton(sources, tmp_path):
    src, dst, cross = sources
    rng = np.random.default_rng(99)
    identity = identity_source(cross)
    sampler = ImagePairSampler([src, identity], [dst, identity], 0.)
    pairs = set()
    for _ in range(500):
        source, target, same, _ = sampler.sample(rng)
        assert not same
        pairs.add((Path(source).parent == src, Path(target).parent == dst))
    assert len(pairs) == 4

    # One identity in the whole source dataset: exclude that pool only for its matching target.
    single = tmp_path / "single"
    group = single / "person"
    group.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(group / "one.png")
    single_source = identity_source(single)
    sampler = ImagePairSampler([src, single_source], [single_source], 0.)
    for _ in range(30):
        source, target, same, _ = sampler.sample(rng)
        assert Path(source).parent == src and Path(target).parent == group and not same
    with pytest.raises(ValueError, match="不同身份"):
        ImagePairSampler([single_source], [single_source], 0.)
    # No eligible cross-view identity: use same-image reconstruction, even at cross_view_prob=1.
    sampler = ImagePairSampler([single_source], [single_source], 1., 1.)
    source, target, same, _ = sampler.sample(rng)
    assert source == target and same


def test_config(sources):
    _, _, cross = sources
    raw = tomllib.loads(Path("experiments/train.toml").read_text())
    legacy = resolve_train_config(raw)
    assert "cross_view_prob" not in legacy["data"]["sampling"]
    explicit = copy.deepcopy(raw)
    for section in ("src", "dst"):
        for entry in explicit["data"][section]:
            entry["layout"] = "flat"
    assert resolve_train_config(explicit) == legacy
    raw["data"]["src"].append({"path": str(cross), "layout": "identity"})
    assert "cross_view_prob" not in resolve_train_config(raw)["data"]["sampling"]
    raw["data"]["dst"].append({"path": str(cross), "layout": "identity"})
    resolved = resolve_train_config(raw)
    assert resolved["data"]["sampling"]["cross_view_prob"] == .75
    assert resolve_train_config(resolved) == resolved
    for probability in (-.1, 1.1):
        invalid = copy.deepcopy(raw)
        invalid["data"]["sampling"]["cross_view_prob"] = probability
        with pytest.raises(ValueError):
            resolve_train_config(invalid)
    invalid = copy.deepcopy(raw)
    invalid["data"]["dst"][-1]["layout"] = "unknown"
    with pytest.raises(ValueError):
        resolve_train_config(invalid)
    raw["data"]["dst"][-1]["alignment"] = "arcface"
    with pytest.raises(ValueError):
        resolve_train_config(raw)


@pytest.mark.parametrize("alignment", ["ffhq", "arcface"])
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("same_prob", [0., 1.])
def test_loader(sources, alignment, backend, same_prob):
    if backend == "cuda":
        if not torch.cuda.is_available() or torch.version.hip is not None:
            pytest.skip("NVIDIA CUDA unavailable")
        pytest.importorskip("nvidia.dali")
    src, dst, cross = sources
    config = DEFAULT_DATA_CONFIG["loader"] | DEFAULT_DATA_CONFIG["augmentation"] | DEFAULT_DATA_CONFIG["sampling"]
    config.update(
        decoder_backend=ImageDecoderBackend.CPU, decoder_hw_load=0.0,
        num_threads=2, py_num_workers=1, prefetch_queue_depth=1, reader_prefetch_queue_depth=1,
        brightness=0., contrast=0., saturation=0., flip_prob=0.,
        rotation_range=(0., 0.), scale_factor_range=(1., 1.), tx_range=(0., 0.), ty_range=(0., 0.),
        same_prob=same_prob, cross_view_prob=1.,
    )
    loader = TrainingDataLoader(batch_size=4, device=torch.device(backend), img_resolution=128,
                                src=[identity_source(cross)], dst=[identity_source(cross)], alignment=alignment, **config)
    source, target, canonical, _, same, identity = loader.next()
    assert bool(same.all()) == bool(same_prob)
    assert bool(same.any()) == bool(same_prob)
    torch.testing.assert_close(target, canonical)
    # R encodes identity, G encodes view. Identity teacher must come from source, never target.
    if same_prob:
        torch.testing.assert_close(source[:, 0], target[:, 0], atol=1e-4, rtol=0)
        assert (source[:, 1, 64, 64] - target[:, 1, 64, 64]).abs().min() > .1
    else:
        assert (source[:, 0, 64, 64] - target[:, 0, 64, 64]).abs().min() > .1
    torch.testing.assert_close(identity[:, :, 56, 56], source[:, :, 64, 64], atol=1e-4, rtol=0)
