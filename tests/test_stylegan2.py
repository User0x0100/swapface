import pytest
import torch

from models.stylegan2 import FIRBlur, Generator


def make_model(img_resolution: int = 32) -> Generator:
    return Generator(
        img_resolution=img_resolution,
        mapping_layers=1,
        channel_multiplier=2,
    )


@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_full_generator_output(size: int) -> None:
    model = make_model(img_resolution=size)
    target = torch.randn(1, 3, size, size)
    identity = torch.randn(1, 512)

    with torch.no_grad():
        result = model(target, identity)

    assert result.shape == target.shape
    assert torch.isfinite(result).all()
    assert not any("mask" in name for name, _ in model.named_parameters())
    assert "use_mask" not in model.network_cfg


@pytest.mark.parametrize("channel_multiplier", [1, 2])
def test_channels_and_synthesis_shapes_match_styleswap_concat(channel_multiplier: int) -> None:
    model = Generator(img_resolution=128, mapping_layers=1, channel_multiplier=channel_multiplier)
    assert model.channels == {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 256 * channel_multiplier,
        128: 128 * channel_multiplier,
        256: 64 * channel_multiplier,
        512: 32 * channel_multiplier,
        1024: 16 * channel_multiplier,
    }
    assert model.conv4.conv.in_ch == model.channels[4]
    assert model.conv4.bias.shape[1] == 2 * model.channels[4]
    assert model.to_rgb4.conv.in_ch == 2 * model.channels[4]
    for resolution, stage in zip([8, 16, 32, 64, 128], model.stages, strict=True):
        assert stage.conv0.conv.in_ch == 2 * model.channels[resolution // 2]
        assert stage.conv0.conv.out_ch == model.channels[resolution]
        assert stage.conv1.conv.in_ch == 2 * model.channels[resolution]
        assert stage.to_rgb.conv.in_ch == 2 * model.channels[resolution]
        assert stage.conv0.bias.shape[1] == 2 * model.channels[resolution]
        assert stage.conv1.bias.shape[1] == 2 * model.channels[resolution]


def test_styled_conv_concatenates_target_features_after_modulated_conv() -> None:
    from models.stylegan2 import StyledConv

    torch.manual_seed(1)
    layer = StyledConv(16, 8, 32)
    x = torch.randn(2, 16, 4, 4)
    w = torch.randn(2, 32)
    features = torch.randn(2, 8, 4, 4)

    with torch.no_grad():
        layer.inject_gain.fill_(0.25)
        expected = torch.nn.functional.leaky_relu(
            torch.cat((layer.conv(x, w), 0.25 * features), dim=1) + layer.bias,
            negative_slope=0.2,
        ) * (2**0.5)
        actual = layer(x, w, features)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 16, 4, 4)


def test_each_resolution_injects_at_both_styled_convs() -> None:
    model = make_model(img_resolution=32)
    calls: list[tuple[int, tuple[int, ...]]] = []
    handles = []

    def record_input(module, args):
        calls.append((module.conv.out_ch, tuple(args[2].shape)))

    for module in [model.conv4, *(conv for stage in model.stages for conv in (stage.conv0, stage.conv1))]:
        handles.append(module.register_forward_pre_hook(record_input))
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 32, 32), torch.randn(1, 512))
    finally:
        for handle in handles:
            handle.remove()

    # Official StyleSwap: one injection at 4x4 and two at every higher resolution.
    assert [shape[-1] for _, shape in calls] == [4, 8, 8, 16, 16, 32, 32]
    assert all(channels == shape[1] for channels, shape in calls)


def test_full_generator_training_gradients() -> None:
    model = make_model(img_resolution=8)
    # Nonzero injection gain lets gradients flow into the target encoder.
    for module in model.modules():
        if hasattr(module, "inject_gain"):
            module.inject_gain.data.fill_(0.5)

    target = torch.randn(1, 3, 8, 8, requires_grad=True)
    identity = torch.randn(1, 512, requires_grad=True)
    output = model(target, identity)
    output.square().mean().backward()

    assert target.grad is not None and target.grad.abs().sum() > 0
    assert identity.grad is not None and identity.grad.abs().sum() > 0
    assert model.encoder.from_rgb.weight.grad is not None
    assert model.encoder.from_rgb.weight.grad.abs().sum() > 0
    assert model.conv4.conv.weight.grad is not None


def test_full_generator_invalid_config_and_inputs() -> None:
    with pytest.raises(ValueError, match="power of two"):
        make_model(img_resolution=48)
    model = make_model()
    with pytest.raises(ValueError, match="target"):
        model(torch.randn(1, 3, 64, 64), torch.randn(1, 512))
    with pytest.raises(ValueError, match="id_feat"):
        model(torch.randn(1, 3, 32, 32), torch.randn(1, 256))


def test_stylegan2_blur_preserves_double_resolution() -> None:
    # Transposed convolution output is (2H + 1) before the FIR filter.
    filter_up = FIRBlur((1, 3, 3, 1), (1, 1), gain=4.0)
    result = filter_up(torch.ones(1, 1, 9, 9))
    assert result.shape == (1, 1, 8, 8)
    torch.testing.assert_close(
        result[..., 2:-2, 2:-2],
        torch.full_like(result[..., 2:-2, 2:-2], 4.0),
    )


def test_modulated_upsample_uses_stylegan2_gain() -> None:
    model = make_model()
    for stage in model.stages:
        assert stage.conv0.conv.blur.gain == pytest.approx(4.0)
        assert stage.to_rgb.upsample.gain == 4
    assert model.encoder.downs[0].blur.gain == pytest.approx(1.0)


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_cuda_autocast_forward_backward_with_fp32_fir_filters(
    amp_dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is not available")
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported by this GPU")

    model = make_model(img_resolution=8).cuda().train()
    target = torch.randn(1, 3, 8, 8, device="cuda", requires_grad=True)
    identity = torch.randn(1, 512, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        output = model(target, identity)
        loss = output.float().square().mean()

    loss.backward()

    assert torch.isfinite(output).all()
    assert identity.grad is not None and torch.isfinite(identity.grad).all()
    assert target.grad is not None and torch.isfinite(target.grad).all()
    assert model.conv4.conv.weight.grad is not None
    assert torch.isfinite(model.conv4.conv.weight.grad).all()

    # AMP does not mutate FIR filter storage dtype.
    for module in model.modules():
        if isinstance(module, FIRBlur):
            assert module.kernel.dtype == torch.float32
