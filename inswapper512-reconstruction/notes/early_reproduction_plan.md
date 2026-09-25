# INSwapper-512-live 复刻研究与第一阶段设计

> 状态：研究/规划，尚未接入训练主线。
>
> 目标不是声称逐层复原闭源的 `inswapper-512-live`，而是在公开证据约束下，设计一个与其能力边界相近、可验证、可迭代的 512×512 one-shot face swap 模型。

## 1. 目标与边界

第一目标是复刻 **INSwapper-512-live base 的能力形态**：

- 原生 512×512 输入/输出；
- one-shot source identity；
- target pose / expression / gaze / lighting 尽量保持；
- 单模型实时部署友好；
- Generator 预算约 10M 参数、12~13 GFLOPs；
- ONNX / ORT / TensorRT 友好，不依赖动态控制流和难部署算子。

第二目标是追求 Picsi/Discord 可见服务的高保真效果，但它不能与 512-live 的公开指标混为一谈。官方公开表中：

| Model | Params | GFLOPs | Similarity | Realism | Attributes |
| --- | ---: | ---: | ---: | ---: | ---: |
| INSwapper-128 | 138.3M | 174.7G | 86.9 | 63.3 | 78.8 |
| INSwapper-Dax | - | - | 92.5 | 86.0 | 81.5 |
| Dax + Optimizer | - | - | 93.4 | 90.2 | 87.7 |
| INSwapper-512-live-base | 9.5M | 12.1G | 87.0 | 73.7 | 80.1 |
| INSwapper-512-live-mini | 4.7M | 6.3G | - | - | - |

因此项目需要两个 benchmark：

1. **Live benchmark**：是否在约 10M / 12G 下达到高质量实时换脸。
2. **Quality benchmark**：不强约束 FLOPs，追求 Dax/Picsi 类高保真输出。

不要用 Picsi 服务的主观效果反推 512-live 本身必须达到 Dax/Evi 水平。

## 2. 公开信息：事实、强证据与推断

### 2.1 官方确认

InsightFace 官方公开材料确认：

- InSwapper 使用 ArcFace identity vector 作为条件；
- Generator 是 StyleGAN2-based encoder-decoder；
- identity / attribute feature 通过 AdaIN 类机制融合；
- target 负责 pose、expression、lighting 等 attribute；
- 512-live-base：512×512，9.5M params，12.1 GFLOPs；
- 512-live-mini：512×512，4.7M params，6.3 GFLOPs；
- 512-live 面向实时、端侧场景。

参考：

- https://github.com/deepinsight/inswapper-512-live
- https://www.insightface.ai/blog/the-evolution-of-neural-network-face-swapping-from-deepfakes-to-one-shot-innovation-with-insightface
- https://github.com/deepinsight/insightface/tree/master/examples/in_swapper

### 2.2 对旧 INSwapper-128 的高置信结构证据

公开的 ReSwapper 项目按 ONNX graph 复刻了 `inswapper_128`。在本仓库环境中重新测量：

```text
ReSwapper StyleTransferModel_128
params = 138.292739 M
FLOPs  = 174.625653 G
```

这与 InsightFace 官方公布的：

```text
138.3 M / 174.7 G
```

几乎精确一致。

因此它可以作为 **旧 InSwapper Generator 拓扑的高置信结构锚点**，但不能据此声称 512-live 逐层相同。

旧模型的关键结构：

```text
target 128
  │
  ├─ 7×7 Conv: 3 -> 128
  ├─ 3×3 Conv: 128 -> 256
  ├─ 3×3 s2:   256 -> 512     # 64×64
  ├─ 3×3 s2:   512 -> 1024    # 32×32
  │
  ├─ StyleBlock(1024) × 6      # 身份融合集中在 32×32
  │
  ├─ up -> 3×3 1024 -> 512    # 64×64
  ├─ up -> 3×3 512 -> 256     # 128×128
  ├─ 3×3 256 -> 128
  └─ 7×7 128 -> RGB
```

StyleBlock 核心：

```text
Conv
 -> per-channel spatial centering/RMS normalization
 -> source style affine(scale, bias)
 -> ReLU
 -> Conv
 -> normalization
 -> source style affine
 -> residual add
```

这与当前仓库的 `IDInject` 思路高度同源。

旧官方 Python inference 公开的 identity latent 路径为：

```text
ArcFace embedding
 -> L2 normalize
 -> EMAP linear projection
 -> L2 normalize
 -> swapper source latent
```

因此第一版复刻应优先研究 **identity projection + 32² style residual bottleneck**，而不是继续扩展 AAD/U-Net skip。

### 2.3 StyleSwap 能提供的训练线索

StyleSwap 不是 InSwapper，但它提供了与 style-based face swap 高度相关的公开训练设计：

- source identity 映射到 style latent；
- target attribute 用空间 feature 表示；
- adversarial / identity / weak feature matching 用于普通 swap；
- L1 + VGG reconstruction 只用于有 pixel-level ground truth 的 self/cross-view same-identity 样本；
- cross-view 同身份重建使用同一个人的不同视频帧；
- identity encoder 输入使用 color jitter，减少 illumination 泄漏；
- mask branch 是后续可插拔模块，不必成为第一版前提。

参考：

- https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740644.pdf

## 3. 当前仓库与目标的差异

当前 `models/networks.py::Generator` 已具备：

- target encoder；
- 低分辨率 latent blocks；
- AdaIN identity injection；
- learned W-space mapping；
- decoder；
- 可选 AAD skip。

因此不应重写整个工程。

但当前主线有几处与 512-live 目标不一致：

1. 当前示例配置是 256×256、`num_depth=4`，bottleneck 为 16×16；512×512 下 4 次下采样恰好得到 32×32。
2. 当前 `base_ch=64/max_ch=512`，高分辨率通道过宽。
3. `WSpaceMap` 每个 latent 有独立 private MLP，参数开销和自由度都明显高于旧 InSwapper 的简单 style affine。
4. 当前 `IDInject` 第二次 modulation 后仍有 SiLU；旧 InSwapper 图复刻更接近第二次 modulation 后直接 residual add。
5. 当前 `FromRGB/ToRGB` 在高分辨率上各做两次 dense convolution，不适合 12G 预算。
6. 当前 `data.sampling.same_prob` 只能产生“同一张图片”的 self reconstruction，没有 identity-aware cross-view pairing。
7. 当前 `loss.reconstruction.scope` 同时控制 L1 与 VGG perceptual reconstruction；默认 `same` 时 different-ID pair 不承受这两类 reconstruction 约束。
8. 当前 WFM 默认关闭；StyleSwap 中 weak feature matching 更适合作为 ordinary different-ID pair 上的弱 target-attribute 约束。

## 4. Live512-V1：首选结构假设

### 4.1 设计原则

第一版不引入：

- AAD long skip；
- cross-attention；
- Transformer；
- progressive mask；
- 多尺度 discriminator；
- temporal network；
- diffusion。

先验证最核心的 InSwapper 假设：

> **target spatial information 压缩到 32×32；source identity 只通过 style modulation 进入；绝大部分身份变换计算集中在 32×32；高分辨率路径只负责廉价还原。**

### 4.2 结构

Generator 对外仍接收 512D identity，内部增加小 projector：

```text
ArcFace-like 512D
     │
     ├─ L2 normalize
     ├─ Linear / EMAP: 512 -> 256
     └─ L2 normalize
              │
              ▼
          style 256D
```

target path：

```text
RGB 512×512
  │
  ├─ Conv3×3  3 -> 16                         512²
  │
  ├─ AvgPool2 + Conv1×1  16 -> 32             256²
  ├─ Conv3×3 s2          32 -> 64              128²
  ├─ Conv3×3 s2          64 -> 128              64²
  ├─ Conv3×3 s2         128 -> 256              32²
  │
  ├─ InSwapperStyleBlock(256, style=256) × 6   32²
  │
  ├─ bilinear up + Conv3×3 256 -> 128           64²
  ├─ bilinear up + Conv3×3 128 -> 64           128²
  ├─ bilinear up + Conv3×3  64 -> 32           256²
  ├─ bilinear up + Conv1×1  32 -> 16           512²
  │
  └─ Conv3×3 16 -> RGB + Tanh                  512²
```

激活第一版使用 LeakyReLU(0.2)，尽量贴近旧 InSwapper / StyleGAN 系实现。

不使用 encoder→decoder long skip。

### 4.3 StyleBlock

```python
residual = x

x = conv1(x)
x = spatial_rms_norm(x)
x = affine1(style) * x + bias1(style)
x = relu(x)

x = conv2(x)
x = spatial_rms_norm(x)
x = affine2(style) * x + bias2(style)

return residual + x
```

其中 `spatial_rms_norm`：

```python
x = x - x.mean((2, 3), keepdim=True)
x = x * rsqrt(x.square().mean((2, 3), keepdim=True) + 1e-8)
```

该实现与旧 InSwapper 图复刻更接近，同时 ONNX/TRT 友好。

### 4.4 实测预算

在当前 WSL/PyTorch/fvcore 环境中，用上述结构构造真实 graph 测量：

```text
Generator core:
  params = 9.436739 M
  FLOPs  = 12.236358 G @ 512×512
```

其中：

```text
6 × StyleBlock @ 32² : 7.2493 G
Decoder              : 3.8210 G
Encoder/down         : 0.9395 G
Stem                 : 0.1132 G
Output               : 0.1132 G
```

这与官方 512-live-base 的 `9.5M / 12.1G` 非常接近。

这 **不能证明官方就是此结构**，但说明“旧 InSwapper 核心拓扑 + 256ch bottleneck + 极窄 512 高分辨率路径”足以自然解释官方量级，无需假设复杂的 MobileNet/Transformer/扩散结构。

512→256 identity projector 额外约 0.13M 参数，FLOPs 可忽略。部署接口仍应保持：

```
target RGB + 512D identity -> swapped RGB
```

## 5. 为什么 V1 不采用当前 WSpaceMap / AAD

### 5.1 WSpaceMap

旧 InSwapper 的每个 StyleBlock 已经拥有独立 style affine，因此即使所有 block 共用一个 latent，不同 block 仍能学习不同 identity transformation。

第一版继续叠加 shared/private W MLP 的收益不明确，却会：

- 增加参数；
- 增加身份路径自由度；
- 扩大 template shortcut / identity overfit 的搜索空间；
- 降低与公开旧模型的结构可比性。

因此 V1 只保留：

```text
512D identity -> EMAP/projector -> 256D style
```

所有 StyleBlock 共用 style vector，但 block 内 affine 独立。

### 5.2 AAD skip

AAD / long skip 会高带宽地把 target identity 送回 decoder。

V1 首先验证 InSwapper 的 bottleneck-only 信息路由，因此：

```
aad_skip_layers = none
```

如果 benchmark 证明 attribute preservation 不足，再单独做受限 skip ablation。

## 6. 训练数据设计

当前随机目录采样不足以复刻高质量 one-shot swap，需要新增 **identity-aware sampler**。

### 6.1 样本类型

每个 batch 混合三类样本，建议初始比例：

```text
60% different-ID swap
30% same-ID cross-view reconstruction
10% exact self reconstruction
```

比例是实验起点，不是公开的官方参数。

#### A. different-ID

```text
source identity = person A
target          = person B
ground truth    = none
```

负责学习真正的 swap。

#### B. same-ID cross-view

```text
source identity = person A, frame/image 1
target          = person A, frame/image 2
ground truth    = target frame/image 2
```

这是最重要的强监督。它允许同时监督：

- target pose；
- expression；
- gaze；
- lighting；
- texture；
- occlusion；
- identity。

而不会产生“不同身份时 fake 必须像 target pixel”的逻辑冲突。

#### C. exact self

```text
source = target = same image
```

保留少量，用于稳定颜色/细节，但不应成为 reconstruction 主体。

### 6.2 数据 manifest

现有 `[[data.src]]/[[data.dst]] path` 目录协议不足以支持 cross-view。

建议增加可选 manifest：

```text
path
identity_id
video_id          # optional
frame_id          # optional
quality_score     # optional
yaw/pitch/roll    # optional
occlusion_score   # optional
```

第一版只强制：

```
path + identity_id
```

video/frame metadata 后续用于视频 hard mining。

### 6.3 数据池职责

高质量静态数据：

- 提供 512 texture / lighting / photographic quality；
- 可用于 different-ID；
- 有 identity label 时也可用于 cross-view。

identity/video 数据：

- 提供 same-ID 不同 pose/expression；
- 提供真正的 cross-view reconstruction；
- 提供 extreme pose / occlusion / motion blur。

不要依赖“单张 FFHQ 随机仿射”代替真实 cross-view；它只能教 equivariance，不能提供真实 expression/pose 变化。

## 7. Loss 应用矩阵

第一版训练设计最重要的改动之一：

| Loss | different-ID | same-ID cross-view | exact self |
| --- | :---: | :---: | :---: |
| GAN | yes | yes | yes |
| Identity | yes | yes | yes |
| Weak Feature Matching | yes | yes | yes |
| L1 reconstruction | **no** | yes | yes |
| VGG reconstruction | **no** | yes | yes |
| Gaze/HRFFA/FACS | baseline 中先只做评测 | baseline 中先只做评测 | baseline 中先只做评测 |

### 7.1 Identity loss

identity condition encoder 与 identity-loss teacher 应分离。

建议：

```text
Condition:
  ArcFace R50 family / buffalo_l-compatible embedding
  -> trainable EMAP/projector
  -> Generator

Loss/Evaluation teacher:
  使用不同训练集或不同结构的识别模型
  例如 TransFace / TopoFR / 另一 ArcFace checkpoint
```

避免 Generator 只利用一个 recognition network 的特定 feature shortcut。

Identity Loss 对 fake 使用当前已有的 restore-to-canonical 路径。

参考 StyleSwap，在送进 identity encoder 之前对 source/fake identity crop 做适度 color jitter，降低 illumination 与 identity feature 的耦合。

### 7.2 Reconstruction

```
L_rec = lambda_l1 * L1 + lambda_vgg * VGG
```

**只对有真实 pixel target 的 same-ID cross-view / self 使用。**

当前实现已由 `loss.reconstruction.scope` 同时控制 L1/VGG；默认 `same` 时 different-ID pair 不承受 reconstruction 约束。

### 7.3 Weak Feature Matching

different-ID 没有像素 ground truth，但 target attribute 仍需保持。

因此 ordinary swap 上不用 VGG(fake, target) 强拉像素，而使用 discriminator feature matching 作为更弱的 attribute constraint：

```
L_wfm = Σ ||D_k(fake) - stopgrad(D_k(target))||_1
```

第一版只使用中/深层，避免最浅层过度复制 target identity/纹理。

### 7.4 GAN

先保留当前经过验证的 discriminator 与 lazy R1，不同时重写 G/D。

建议起始：

```text
G lr = 1e-4
D lr = 1e-4
Adam betas = (0.0, 0.99)
R1 interval = 16
R1 gamma = 10
EMA = current implementation
```

先不引入新的复杂 scheduler。

### 7.5 暂不进入 baseline 的损失

以下全部作为后续 ablation：

- Gaze loss；
- HRFFA；
- FACS；
- affine equivariance double-forward loss；
- mask BCE；
- temporal loss；
- LPIPS/DSSIM；
- segmentation-specific GAN。

原因不是它们无效，而是 baseline 必须先回答：

> 正确的架构 + 正确的 pair supervision 是否已经足以解决 original-face fallback、template attraction 和 attribute preservation？

## 8. 训练阶段

### Phase 0：冻结评测集

在训练新模型前固定 benchmark，至少包含：

- frontal easy；
- ±30° / ±60° yaw；
- strong expression；
- gaze；
- glasses；
- hand/hair/object occlusion；
- beard/makeup；
- lighting mismatch；
- age/gender/domain gap；
- low-resolution / blur；
- video sequence。

必须保存固定 source-target pair manifest。

### Phase 1：Live512-V1 baseline

只使用：

- new Generator；
- current discriminator；
- GAN；
- ID；
- WFM；
- scoped L1/VGG；
- mixed different-ID/cross-view/self sampler。

不启用额外 attribute teacher。

### Phase 2：hard-data fine-tune

在 baseline 已经证明 identity transfer 正常后，提高：

- extreme pose；
- occlusion；
- motion blur；
- video frame；
- difficult morphology gap。

降低 LR 做短程 fine-tune。

### Phase 3：结构 ablation

只在具体 failure 被 benchmark 证明后增加：

1. progressive ToMask / face mask；
2. gated target feature injection；
3. sparse affine-equivariance pass；
4. temporal consistency。

## 9. 评测协议

不要只看 TensorBoard loss 和 sample grid。

### 9.1 Identity

使用 **训练 identity-loss teacher 之外** 的模型评估：

```
cos(E_eval(fake), E_eval(source))
```

按 overall / pose bucket / occlusion bucket / identity bucket 分别统计。

### 9.2 Target attributes

利用项目已有模型建立独立指标：

- pose：HRFFA / SynergyNet；
- gaze：L2CS；
- expression：FACS；
- landmark/contour：HRFFA / FAN。

比较：

```
attribute(fake) vs attribute(target)
```

不要拿训练 loss 自己当唯一评测。

### 9.3 Realism

至少保留：

- FID/KID 或 face-domain equivalent；
- 图像清晰度/频谱统计；
- 人工 blind A/B。

官方 Realism 是组合指标，无法完全复现，因此我们的数值不能直接声称与官方 73.7 等价。

### 9.4 视频稳定性

Live 模型额外测：

- per-frame identity variance；
- landmark jitter；
- static-camera pixel/feature jitter；
- 遮挡进入/离开时的结构跳变。

### 9.5 Runtime

每个 checkpoint 同时记录：

- params；
- fvcore FLOPs；
- ONNX graph；
- ORT CUDA latency；
- TensorRT FP16 latency；
- peak VRAM；
- batch=1。

目标不是只追一个 FLOPs 数字。

## 10. 第一轮实验矩阵

不要一次改十个变量。

### E0 — current architecture scaled to 512

目的：内部对照。

```text
resolution=512
depth=4
base=16
max=256
latent blocks=6
no AAD skip
```

当前实现实测：

```text
id_dim=256:
params ≈ 10.50M
FLOPs  ≈ 14.59G
```

### E1 — Live512-V1 / old-InSwapper style block

采用第 4 节结构。

实测 core：

```
9.44M / 12.24G
```

这是主实验。

### E2 — E1 + identity projector

```
512 -> 256 -> L2
```

验证 source encoder compatibility 与 identity transfer。

### E3 — E2 + cross-view training

这是训练设计的关键实验，不是可选小修。

### E4 — E3 + WFM

检查 target expression/pose 与 original-face fallback 的平衡。

只有 E1~E4 完成后，才讨论 mask/AAD/更多 teacher loss。

## 11. 对当前代码的预计改造点

### `models/networks.py`

不要破坏现有 `Generator`。

新增独立：

```
LiveGenerator
IdentityProjector
InSwapperStyleBlock
SpatialRMSNorm
```

随后再引入 generator factory。

### `swapface/config.py`

新增 generator architecture discriminator，例如：

```toml
[generator]
architecture = "live512_v1"
```

不同 architecture 使用各自 schema，避免给旧 Generator 塞大量无效字段。

### `swapface/train.py`

- 通过 factory 构造 Generator；
- reconstruction mask 同时控制 L1 和 VGG；
- 加入 cross-view sample type；
- WFM 继续沿用已有实现；
- checkpoint 保存 architecture。

### `swapface/dataloader_common.py` / DALI / native

新增 identity-aware pair protocol。

不要把 identity grouping 硬编码进 DALI graph；pair selection 放到 Python/source 层，DALI 只负责 decode/augmentation。

### `swapface/inference.py` / `export.py`

通过 generator factory 加载 architecture。

保持最终推理接口：

```
target image + 512D identity -> swapped RGB
```

即使内部把 512D 投影到 256D，也不要让部署端理解训练细节。

## 12. 当前最重要的假设与风险

### 高置信

- 32×32 是旧 InSwapper 的核心 identity transformation resolution。
- 旧模型使用 6 个强 style residual blocks。
- source identity 通过 ArcFace latent + affine modulation 进入 Generator。
- 不依赖 U-Net long skip 也能完成换脸。
- 512-live 的低计算量要求高分辨率通道极窄。

### 中等置信

- 512-live 很可能仍保留“低分辨率重身份变换 + 高分辨率轻 decoder”这一代际结构。
- bottleneck 约 256 channels、style latent 约 256D 是与公开预算高度一致的合理量级。
- 最外层 1×1 / lightweight conv 是达到 12G 的合理方式。

### 未知

- 512-live 的确切 channel table；
- 是否仍恰好 6 个 StyleBlock；
- 是否使用 depthwise/grouped conv；
- 是否存在 multi-scale target feature injection；
- 是否有 mask branch；
- 确切 GAN loss 与 loss weights；
- 训练数据组成；
- Picsi/Discord 当前请求实际路由到 Dax、Evi、512-live 或其他内部模型。

因此后续实验结果必须优先于“猜官方”。

## 13. 第一阶段结论

最值得实施的路线：

1. 保留现有工程训练/导出框架；
2. 新增一个与旧 InSwapper 拓扑同源的 `LiveGenerator`；
3. 原生 512，bottleneck 固定 32×32；
4. 6×256ch InSwapper-style residual modulation；
5. 极窄高分辨率 encoder/decoder；
6. identity 512→256 projection；
7. 不使用 AAD long skip；
8. 重做 pair sampler，使 same-ID cross-view 成为真正的 reconstruction supervision；
9. VGG/L1 不再约束 different-ID pair；
10. 用独立 benchmark 同时评估 identity / realism / attributes / temporal stability。

这条路线同时满足：

- 与公开 InSwapper 结构证据一致；
- 能自然解释 512-live 的 9.5M / 12.1G 资源量级；
- 每个关键假设都能单独做 ablation，而不是堆叠不可解释模块。
