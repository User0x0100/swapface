# InSwapper 512-live structural reconstruction

This directory is a documentation/evidence archive for the InSwapper 512-live
reverse-engineering work. It intentionally contains no executable model or test
implementation. The canonical project Generator now lives in
`/home/User0/swap/models/networks.py`.

## Current reconstructed architecture

The network topology is grounded in the recovered HWX/CoreML live graph. The
production PyTorch implementation in `../models/networks.py` is a training-facing adaptation: it keeps the
recovered convolutional/HQ topology, uses a unified RGB domain of `[-1, 1]`, and
replaces the official precomputed 6144-value style payload with learned per-block
style projections from a single `[B, 512]` vector. The following recovered
structural details remain directly confirmed or mechanically fingerprinted:

- `input_1`: logical 3-channel 512x512 float tensor.
- 512 -> 128: bilinear `UNALIGN_CORNERS` (`align_corners=False`).
- Coarse encoder: `3->32->64->128->256`, bottleneck `256@32x32`.
- Coarse encoder/decoder activations: LeakyReLU with **negative_slope=0.2**.
  All seven activation tasks reference the same H14 post-scale kernel and its
  FP16 slope field is `0x3266` (~0.2).
- Six residual style blocks, two 3x3 Conv/modulation sites each.
- Style normalization: spatial centering + RMS normalization with **eps=1e-8**.
- Official Core ML `input_2` is 6144 floats = `6 x 2 x 512`; each 512 is
  direct `[gamma_256 | beta_256]` modulation data.
- First modulation in each style block is followed by ReLU; second is followed
  directly by residual addition.
- All recovered semantic Conv layers carry source bias. The 43 official
  `__@conv` aliases match the 43 biased Conv layers in the production Generator.
- Coarse decoder resize 32->64->128: bilinear `align_corners=False`.
- Official coarse RGB head: 7x7 Conv -> Tanh -> `(x + 1) * 0.5`.
- Coarse RGB 128->512: bilinear `UNALIGN_CORNERS` / `align_corners=False`.
- HQ input concat order: `[coarse_512, target_512]` -> 6 channels.
- HQ encoder downsampling: **MaxPool2d(2,2)** at every stage (H14 PE PoolMode=2).
- HQ refiner: narrow U-Net with five encoder pools and five skip-concat decoder stages.
- HQ decoder upsample `x1_1/x1_5/x1_9/x1_13/x1_17`: bilinear
  **`align_corners=True`**. Its H14 lowering uses the distinctive
  `transpose_resize_xfm -> trans_resize_conv_xfm` path.
- Official final head: **1x1 Conv 8->3 -> Sigmoid -> x255**. Its H14 task
  timing and layer count exactly match the official Task322 fingerprint.

Current production PyTorch graph: **43 Conv layers + one fused Linear(512,6144) style-bank
projection / 11,618,026 parameters**. This single affine is mathematically
identical to 12 independent Linear(512,512) projections, but exports as one Gemm.
Its flat output preserves the official 12 contiguous 512-value sites but is
exported as one 24-way 256-value `[gamma,beta,...]` Split, avoiding both
reshape/index Gather nodes and per-site secondary Split nodes in ONNX. The projection is an intentional training
adaptation, so the parameter count is not expected to match the official live
compiled-weight container.

## RGB value convention

The production PyTorch implementation uses one consistent RGB image domain:

- target input: `[-1, 1]`
- coarse RGB output: `[-1, 1]`
- HQ concat inputs: `[coarse_512, target_512]`, both `[-1, 1]`
- final RGB output: `[-1, 1]`

This is an intentional training-facing adaptation of the recovered official
interface. The official graph maps coarse Tanh output to `[0, 1]` and its final
head is `Sigmoid -> x255`; the production Generator instead keeps the coarse
Tanh output directly and applies Tanh at the final 1x1 RGB head. No internal conversion back
to `[0, 1]` or `[0, 255]` is performed. Hidden features remain unconstrained.

The production PyTorch training interface uses `input_2: [B, 512]` directly. One
fused `Linear(512,6144)` produces the complete 12-site style bank in a single
GEMM. Its flat layout is still the official 12 x `[gamma_256 | beta_256]` bank, but
for export it is consumed through a single 24-way Split into `[B,256]` gamma/beta
vectors. This keeps all 12 site parameter sets independent while reducing the
conditioning subgraph to one Gemm + one Split for ONNX/ORT/TensorRT.

The official macOS/Core ML interface is different: it consumes a precomputed
6144-float payload with independently stored values for all 12 sites. The demo
ships 20 `template_N.txt` resources in that original format.

## Formula-derived architecture parameters

The model does not require explicit encoder/decoder channel tables. Construction
parameters describe boundary resolutions and capacity; depths and channels are
derived once in `__init__`, so the exported graph remains fully static.

Default constructor values reproduce the recovered architecture:

```python
Generator(
    img_resolution=512,
    img_channels=3,
    id_dim=512,
    coarse_resolution=128,
    coarse_bottleneck_resolution=32,
    coarse_base_ch=32,
    coarse_max_ch=256,
    num_style_blocks=6,
    hq_bottleneck_resolution=16,
    hq_base_ch=8,
    hq_max_ch=128,
    hq_channel_hold_level=2,
    norm_eps=1e-8,
    leaky_relu_slope=0.2,
)
```

Coarse depth is derived as
`log2(coarse_resolution / coarse_bottleneck_resolution)`. The coarse feature
schedule contains the recovered 7x7 stem, one same-resolution 3x3 stage, and one
stride-2 stage per required downsample. Channels follow
`min(coarse_max_ch, coarse_base_ch * 2**level)`. With the defaults this gives
`(32, 64, 128, 256)` and exactly two stride-2 stages; the decoder is generated
from that schedule in reverse, yielding `256->128->64->32` with two upsampling
stages and one final same-resolution Conv.

The style-bank width is also derived rather than configured:
`num_style_blocks * 2 sites * 2 affine components * bottleneck_channels`.
Defaults therefore produce `6*2*2*256 = 6144` values from the fused projector.

HQ depth is `log2(img_resolution / hq_bottleneck_resolution)`. Its channel
schedule doubles from `hq_base_ch`, is capped by `hq_max_ch`, and
`hq_channel_hold_level` repeats one scale before doubling resumes. The recovered
default `hold_level=2` gives `(8, 16, 16, 32, 64, 128)`. Every HQ decoder stage
is then derived from the corresponding skip: `concat_ch = decoder_ch + skip_ch`,
`mid_ch = concat_ch // 2`, and `out_ch = skip_ch`.

## Archive scope

This directory is intentionally non-executable. Historical reverse-engineering
notes, environment snapshots, recovered template evidence, and test-result records
remain here for reference. The maintained implementation is
`../models/networks.py`; project-level training/inference/export tests live under
`../swapface/` and `../tests/`.

## Remaining architecture unknowns

The large-scale topology, channels, kernels, pooling, bias presence, coarse
activation slope, style normalization epsilon, and all major resize modes are now
constrained by compiled-model evidence. Remaining uncertainty is comparatively
small:

- HQ decoder concat order is directly recovered from DRAM buffer reuse: each
  encoder skip occupies the concat buffer base and the upsampled decoder feature
  is copied after it, i.e. `[encoder_skip, upsampled_decoder]`.
- Some lower-resolution HQ ReLU sites are represented through GOC/pool/resize
  fusion rather than a Conv-local `NLMode=1`. The repeated semantic layer
  sequence, exact full-stage fingerprints, and direct `NLMode=1` evidence at
  neighboring high-resolution stages consistently support Conv->ReLU throughout.
- App-side fixed 3x3 channel mixing before Core ML is intentionally outside
  the production Generator; it is preprocessing, not a learned layer.

Training design is intentionally out of scope.
