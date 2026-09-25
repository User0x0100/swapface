# InSwapper 512-live reverse-engineering snapshot — 2026-09-26

This document freezes the architecture state immediately before the physical M2
machine used for Core ML / ANE work is shut down and reset.

## Scope

Goal: reconstruct the **official live inference generator architecture** closely
enough to implement a faithful PyTorch equivalent. Training losses, discriminator,
data pairing, identity encoder training, and optimization policy are intentionally
out of scope.

The current PyTorch reference implementation is:

- `/home/User0/swap/inswapper512-reconstruction/model.py`
- class: `ReconstructedInSwapper512`

## Official application / model identity

Official release used during the investigation:

- application: `inswapper-512-live_macOS_v0.1.2`
- release archive SHA256:
  `2ABF0F4678EF58F64FA8DBCF706A120980807DEF8EF4DACC209311CE0154F943`
- bundle id: `ai.picsi.inswapper-512-live`
- compiled model resource: `Contents/Resources/inswapper_live.mlmodelc`
- Core ML encryption UUID:
  `B6637E72-DF09-41D0-9371-F69F3CD0F568`

The original signed application, protected model key material, authentication
material, and original/derived complete weight-bearing executables are **not**
included in the backup archive. Their hashes/metadata are recorded instead.

## Physical reverse-engineering host

Primary runtime host:

- Apple Mac mini `Mac14,3`
- Apple M2
- 8 GB RAM
- macOS 26.6 build 25G72
- arm64
- SIP enabled
- Developer Mode enabled

The official signed app was preserved unmodified during runtime experiments.

## ANECompiler breakthrough

The encrypted `.mlmodelc` could be semantically parsed by private ANECompiler even
when whole-network ANE lowering initially failed.

Original H14G failure:

- layer: `input_1_to_fp16`
- error: `InvalidWidth`

Exact toy sweep on this M2/H14 lowering path:

- width 16384: succeeds
- width 16385: `InvalidWidth`

The official input had been lowered to a flat tensor with:

- element count = `786432 = 3 * 512 * 512`
- effective lowered shape included `W=786432`

A live LLDB patch at `ZinIrTensor::CreateTensor` reinterpreted the same linear
buffer as `H=48, W=16384`. The element count was unchanged. This allowed a full
H14G compilation and produced the structural evidence used below.

Important: the patch changes only the validator-facing shape of the external
flat buffer. Internal network tensor shapes recovered after that boundary are the
ones used for reconstruction.

## Recovered compiled graph inventory

Successful compiled target yielded:

- about 17 MB `model.hwx`
- `analytics.json`
- `model.hwx.status.plist`
- analytics buffer
- 177 ordered layer groups
- 313 ordered semantic layer names
- 2500+ HWX symbols

The complete `model.hwx` is intentionally excluded from this backup because it is
a weight-bearing executable derived from the protected official model. Analytics,
status, symbols, register dumps, logs, and structural fingerprints are retained.

## External interface

Confirmed live graph inputs / output:

- `input_1`: logical RGB image, 512 x 512
- `input_2`: 6144 Float32 values
- `output`: RGB, 512 x 512

`6144 = 6 * 2 * 512`.

The macOS demo contains 20 files:

- `template_0.txt ... template_19.txt`

Every file contains exactly 6144 floats. `ModelProcessFilter` creates an
`MLMultiArray(Float32)` and copies the selected template vector element-for-element
to `currentInput2` / Core ML `input_2`. There is no runtime mapping network in the
live Core ML graph.

Internal shape:

- `dlatents = [6, 2, 1, 512]`
- 12 independent style vectors of length 512

## Coarse swap core

### Input downsample

Official 512 -> 128 sampling mode is directly identified as Core ML
`UNALIGN_CORNERS`.

H14 mode-specific registers match `UNALIGN_CORNERS` and reject:

- ALIGN_CORNERS
- STRICT_ALIGN_CORNERS
- OFFSET_CORNERS

PyTorch equivalent:

```python
F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
```

### Encoder

Recovered semantic tensor/channel schedule:

```text
3   @ 128x128
32  @ 128x128
64  @ 128x128
128 @  64x64
256 @  32x32
```

Architecture:

```text
ReflectionPad(3)
Conv7x7 3->32, bias=True
LeakyReLU(0.2)

Conv3x3 32->64, stride1, same/zero padding, bias=True
LeakyReLU(0.2)

Conv3x3 64->128, stride2, same/zero padding, bias=True
LeakyReLU(0.2)

Conv3x3 128->256, stride2, same/zero padding, bias=True
LeakyReLU(0.2)
```

Only the first 7x7 and the style-block 3x3 convolutions use explicit reflection
padding in the HWX symbols. Ordinary encoder/decoder 3x3 convolutions do not show
`reflectivepad` lowering and are represented with ordinary same padding.

### LeakyReLU slope proof

All seven coarse activation tasks (four encoder, three decoder) use H14 nonlinear
mode 2 and the same post-scale kernel.

For every one of these tasks the post-scale kernel contains at offset `+0x4a`:

- raw FP16 bits: `0x3266`
- decoded FP16 value: approximately `0.199951171875`

Therefore the source semantic activation is LeakyReLU with `negative_slope=0.2`.

### Style bottleneck

Bottleneck is directly recovered as:

- channels: 256
- spatial: 32 x 32

There are exactly:

- 6 residual style blocks
- 2 style-conditioned convolutions per block
- 12 style sites total

Each block:

```text
residual = x

ReflectionPad(1)
Conv3x3 256->256, bias=True
spatial centered RMS normalization, eps=1e-8
style affine: gamma*x + beta
ReLU

ReflectionPad(1)
Conv3x3 256->256, bias=True
spatial centered RMS normalization, eps=1e-8
style affine: gamma*x + beta

x = x + residual
```

The 512-value style vector is split as:

```text
style[0:256]   -> gamma / multiplicative scale
style[256:512] -> beta / additive bias
```

The ordering is independently confirmed from the public InSwapper-128 graph:
first Gemm half is consumed by Mul, second half by Add.

Normalization epsilon is `1e-8`, matching the public InSwapper lineage and the
recovered reduce-mean / square / epsilon / sqrt / reciprocal chain.

### Coarse decoder

```text
256@32 -> bilinear x2 (align_corners=False) -> Conv 256->128 -> LeakyReLU(0.2)
128@64 -> bilinear x2 (align_corners=False) -> Conv 128->64  -> LeakyReLU(0.2)
64@128 -> Conv 64->32 -> LeakyReLU(0.2)
ReflectionPad(3)
Conv7x7 32->3
Tanh
(x + 1) * 0.5
```

Coarse output is RGB 128 x 128 in [0,1].

### Coarse RGB 128 -> 512

Direct H14 fingerprint identifies bilinear `UNALIGN_CORNERS` /
`align_corners=False`.

The official lowering chain includes the same distinctive:

```text
1comp_conv_xfm
-> lkss_s4_xfm
-> pix_shuf_xfm
```

as the UNALIGN_CORNERS candidate. The apparent `pix_shuf_xfm` is an ANE lowering
of the resize, not evidence of a learned PixelShuffle layer in the source graph.

## HQ refinement U-Net

HQ input concat order is directly recovered as:

```text
[coarse_512, target_512]
```

Total input channels: 6.

### Encoder channel schedule

```text
512:  6 ->  8 ->   8
256:  8 -> 16 ->  16
128: 16 -> 16 ->  16
 64: 16 -> 32 ->  32
 32: 32 -> 64 ->  64
 16: 64 ->128 -> 128
```

Each stage uses two 3x3 Conv layers with bias and ReLU.

Five downsampling operations are directly identified as:

```text
MaxPool2d(kernel_size=2, stride=2)
```

H14 PE `PoolMode=2` occurs on all five downsampling tasks. This resolves the
previous MaxPool-vs-AvgPool uncertainty.

### Decoder upsample mode

There are five upsample tensors:

- `x1_1`   16 -> 32
- `x1_5`   32 -> 64
- `x1_9`   64 -> 128
- `x1_13` 128 -> 256
- `x1_17` 256 -> 512

All five use the same H14 lowering fingerprint:

```text
transpose_resize_xfm
-> trans_resize_conv_xfm
```

Exact same-shape toy comparison shows this corresponds to bilinear
`align_corners=True`, not nearest and not bilinear `align_corners=False`.

PyTorch equivalent:

```python
F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
```

This is intentionally different from the coarse-network resize mode.

### Skip concat order

All five concat buffers reuse the encoder skip tensor allocation as the beginning
of the concat buffer and copy the upsampled decoder channels after it.

Therefore the semantic concat order is:

```text
[encoder_skip, upsampled_decoder]
```

not the reverse.

### Decoder channel schedule

```text
32:  concat 64+128=192 -> 96 -> 64
64:  concat 32+64 = 96 -> 48 -> 32
128: concat 16+32 = 48 -> 24 -> 16
256: concat 16+16 = 32 -> 16 -> 16
512: concat  8+16 = 24 -> 12 -> 8
```

Each decoder block is semantically Conv -> ReLU -> Conv -> ReLU. At high
resolutions H14 exposes ReLU directly as `NLMode=1`; at some lower resolutions
it is absorbed into GOC/pool/resize fusion in the complete graph.

### Final head

Direct fingerprint match:

```text
Conv1x1 8->3, bias=True
Sigmoid
multiply by 255
FP32 output
```

Official Task322 static timings:

```text
NE    = 21455
L2    = 20844
DRAM  = 107402
Total = 107402
```

The exact `Conv1x1 -> Sigmoid -> x255` toy matches all four values and the
semantic layer count.

## Convolution inventory and parameter sanity check

Official recovered semantic Conv aliases: 43.

Current PyTorch model:

- total Conv: 43
- all 43 have bias
- 40 x 3x3
- 2 x 7x7
- 1 x 1x1
- parameter count: 8,466,154

Raw FP16 parameter payload would be:

- 16,932,308 bytes

Official compiled `weight.bin` size:

- 16,941,056 bytes

Difference:

- 8,748 bytes (~0.052%)

The compiled container also includes alignment / constants / compiler metadata,
so this is not used as an exact parameter-count proof, but it strongly constrains
any missing learnable module to be very small.

## App-side input handling

The live graph consumes `MLMultiArray` inputs. App-side preprocessing contains a
fixed channel/color transform before `input_1`; it is intentionally kept outside
`model.py` because it is preprocessing rather than a learned model layer.

For a bit-exact official-app comparison, this preprocessing must be reproduced.
For architecture training, the model graph itself is kept separate.

## Current remaining architecture uncertainty

The architecture is now considered suitable as the first trial-training baseline.
Remaining uncertainty is local rather than structural:

1. At several lower-resolution HQ stages, ReLU is not exposed as a Conv-local
   `NLMode=1` task because ANE partitions/fuses output conversion differently.
   The semantic layer sequence, exact neighboring-stage fingerprints, and direct
   high-resolution NLMode evidence consistently support Conv->ReLU.
2. Some compiler-internal GOC/layout details are not source-model operations and
   must not be copied into PyTorch as extra normalization/affine layers.
3. The app-side channel preprocessing is outside the reconstructed neural graph.

No remaining evidence suggests a different macro-architecture, missing large
parameter block, additional normalization stack in HQ, or a learned PixelShuffle
super-resolution module.

## Current validation

At snapshot time:

- architecture unit tests: 8 / 8 PASS
- smoke forward: PASS
- coarse shape: `[1,3,128,128]`
- final shape: `[1,3,512,512]`
- current source parameter count: 8,466,154

## Important evidence files to preserve

Primary Mac paths before reset:

```text
~/ane-research/
~/coreml-inswapper-test/static2/
~/coreml-inswapper-test/static3/
/tmp/lldb-livepatch-swap-dump/swapdump_h14g/
/tmp/swap-hwx-symbols.txt
/tmp/h14-force-decode.txt
```

The backup archive also contains the custom fingerprint scripts, LLDB logs,
ANECompiler option strings/disassembly, modified `coreml_to_ane_hwx`, environment
manifests, and selected toy fingerprint evidence needed to reproduce the above
conclusions.
