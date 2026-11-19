"""The standalone Generator compute-statistics entrypoint remains executable."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("generator_config", [
    '''type = "inswapper"
img_resolution = 32
id_dim = 512
latent_resolution = 8
num_latent = 1
base_ch = 4
max_ch = 8
skip_layers = [0, 1]
''',
    '''type = "stylegan2"
img_resolution = 16
id_dim = 512
w_dim = 16
mapping_layers = 1
start_resolution = 4
channels = [8, 16]
target_inject_mode = "add"
''',
])
def test_network_profile_reports_complexity(tmp_path: Path, generator_config: str) -> None:
    pytest.importorskip("fvcore")
    pytest.importorskip("torchinfo")
    config = tmp_path / "model.toml"
    config.write_text("[generator]\n" + generator_config, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "models.networks", "--config", str(config), "--device", "cpu"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=True, timeout=60,
    )
    assert "Trainable params:" in result.stdout
    assert "NetWork_Info:" in result.stdout
    assert "模型总MACs" in result.stdout
    assert "约合FLOPs" in result.stdout
    assert "output: (1, 3," in result.stdout
