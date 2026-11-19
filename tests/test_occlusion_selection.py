"""Occlusion selection must not synchronize GPU to discover selected batch rows."""

import pytest
import torch

from swapface.occlusion import OcclusionAugmentor


@pytest.mark.parametrize("selected_rows", [(), (1, 3), (0, 1, 2, 3)])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_occlusion_selects_rows_on_cpu_and_only_runs_occ_for_selected(
    selected_rows: tuple[int, ...], device_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    batch = 4
    random_values = torch.tensor([0.1 if i in selected_rows else 0.9 for i in range(batch)])
    original_rand = torch.rand

    def controlled_rand(*args, **kwargs):
        if args == (batch,):
            assert kwargs.get("device", "cpu") == "cpu"  # Batch selection must be on CPU.
            return random_values
        return original_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", controlled_rand)
    processed_batch_sizes: list[int] = []

    def mask_model(images: torch.Tensor) -> torch.Tensor:
        processed_batch_sizes.append(images.shape[0])
        return torch.ones((images.shape[0], 1, 224, 224), device=images.device)

    device = torch.device(device_name)
    augmentor = OcclusionAugmentor(
        0.5, "", device, mask_model,
        size_range_pct=(80, 80),
        center_x_range_pct=(0, 0),
        center_y_range_pct=(0, 0),
        rotation_range_deg=(0, 0),
    )
    clean = torch.zeros((batch, 3, 32, 32), device=device)
    output, gate = augmentor.apply(clean)

    assert processed_batch_sizes == ([len(selected_rows)] if selected_rows else [])
    assert output.shape == clean.shape
    assert gate.shape == (batch, 1, 32, 32)
    assert gate.device == clean.device
    assert output.device == clean.device
    for i in range(batch):
        if i in selected_rows:
            assert not torch.equal(output[i], clean[i])
            assert torch.any(gate[i] < 1)
        else:
            torch.testing.assert_close(output[i], clean[i])
            assert torch.all(gate[i] == 1)
