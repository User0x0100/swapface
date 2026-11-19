"""Standalone discriminator compute statistics for each supported architecture."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(("kind", "specific", "output_shape"), [
    ("FIR_MINIBATCH_STD", "max_ch = 8\nminibatch_std_group_size = 4", "(1, 1)"),
    ("INSTANCE_NORM_RESIDUAL", "max_ch = 8", "(1, 1)"),
    ("UNET", "skip_connection = true", "(1, 1, 32, 32)"),
])
def test_discriminator_profile_reports_complexity(tmp_path: Path, kind: str, specific: str, output_shape: str) -> None:
    pytest.importorskip("fvcore")
    pytest.importorskip("torchinfo")
    config = tmp_path / "model.toml"
    config.write_text(
        f'[discriminator]\ntype = "{kind}"\nimg_resolution = 32\nbase_ch = 4\n{specific}\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "models.discriminator.network", "--config", str(config), "--device", "cpu"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=True, timeout=60,
    )
    assert "Trainable params:" in result.stdout
    assert "NetWork_Info:" in result.stdout
    assert f"output: {output_shape}" in result.stdout
    assert "模型总MACs" in result.stdout
    assert "约合FLOPs" in result.stdout
