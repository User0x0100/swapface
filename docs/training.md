# SwapFace 训练运行与结果目录

本文描述训练配置、run 生命周期、checkpoint 保存与中途恢复的约定。目标是让一次训练具备明确边界、可恢复、可审计，并避免运行时路径污染实验配置。

## 设计原则

训练系统区分两类状态：

- **实验配置**：决定“训练什么”，例如模型结构、数据源、损失、学习率与数据增强。由 TOML 描述，可进入 Git。
- **运行状态**：决定“这一次从哪里执行、结果写到哪里”，例如 run ID、checkpoint、输出目录。由 CLI 与 run 元数据管理，不写入 TOML。

因此 `ckpt`、`log_path` 不再是 `[train]` 配置项。真正的 resume 必须继续原 run，并使用该 run 在创建时冻结的配置。

## 安装 GPU 后端

项目使用同一个 `pyproject.toml` / `uv.lock` 管理 CUDA 与 ROCm，但两组 GPU 依赖互斥。CUDA 是默认 dependency group，因此 NVIDIA 环境保持原有用法：

```bash
uv sync
```

当前 CUDA group 使用 PyTorch cu126，并包含 DALI、TorchCodec、ONNX Runtime GPU、xFormers 与 TensorRT。

AMD ROCm 环境显式关闭默认 CUDA group 并启用 ROCm group：

```bash
uv sync --no-default-groups --group rocm
```

当前 ROCm group 使用 PyTorch ROCm 7.2，并包含 PyTorch 官方 ROCm Triton 包。ROCm 训练不安装 DALI、TorchCodec、ONNX Runtime GPU、xFormers 或 TensorRT；这些依赖对应的推理/导出功能需单独完成 AMD 兼容后再加入。`cuda` 与 `rocm` group 被声明为冲突，不能同时启用。

ROCm 环境完成同步后，运行训练命令使用 `uv run --no-sync ...`（或先激活 `.venv` 后直接运行 Python），避免裸 `uv run` 按项目默认 `cuda` group 重新同步依赖。

## 新建训练

使用默认配置：

```bash
uv run python -m swapface.train
```

显式指定配置：

```bash
uv run python -m swapface.train --config experiments/train.toml
```

添加便于识别的短标签：

```bash
uv run python -m swapface.train \
  --config experiments/train.toml \
  --name blendface-vgg
```

`--name` 只参与 run ID，不改变实验配置。默认结果根目录为 `experiments/runs`；如确有需要，可用 `--runs-root PATH` 覆盖。

`[train].compile_module` 控制训练路径的 `torch.compile`：启用时，由 Trainer 分别编译 Coarse、HQRefiner、Discriminator、Generator 身份编码与 Identity Loss embedding 提取，以及启用的 VGG 热路径；设为 `false` 时，训练路径全部保持 eager。

## 配置职责

训练配置只存在一套 canonical schema；`swapface/config.py` 负责默认值、严格校验和 canonical 化，`Trainer` 直接消费该结构，不再维护第二套扁平配置参数。顶层职责为：

- `[train]`：训练 stage、batch、precision、device、compile 与输出周期；
- `[optimizer.*]` / `[scheduler.*]`：Generator、HQ D、Coarse D 各自的学习率与调度策略；
- `[identity]`：Generator 的 source identity 条件编码器；
- `[loss.*]`：每一种训练损失及其共享 reconstruction 策略；
- `[data.loader]` / `[data.augmentation]` / `[data.sampling]`：数据执行、增强和配对采样；
- `[[data.src]]` / `[[data.dst]]`：训练数据源；
- `[generator]` / `[discriminator.hq]` / `[discriminator.coarse]`：Generator 与两个 Discriminator 的独立模型定义。

Generator 条件编码器仍位于 `[identity]`。Coarse 与 HQ 的训练损失分别位于 `[loss.coarse.*]` 和 `[loss.hq.*]`，两边的 GAN、Identity teacher/weight、R1、gaze、HRFFA、FACS、L1/VGG 与 reconstruction scope 均独立配置。`gan.weight` 只缩放对应阶段的 Generator adversarial loss；两个全局判别器的 adversarial loss 固定权重 1.0；UNET 的 Global Head 仍为 1.0，Spatial Head 使用独立的 spatial_weight，同时作用于 D 与 G。

### WebFace600K ArcFace R50（InSwapper-128 身份编码器）

身份编码器可选 `WEBFACE600K_ARCFACE_R50`。它从项目共用模型仓库 `leehx/model_hub` 加载 `w600k_r50.pth`（原始权重来自 `Icar/buffalo_l-torch`，与 InsightFace `buffalo_l/w600k_r50.onnx` 输出数值等价），复用项目原有 IResNet50 / IBasicBlock 的 `fused_bn=True` 结构（与普通非融合 R50 checkpoint 不可互换）。权重与其他身份编码器一样通过公共 `MODEL_REPOSITORY_ID` 下载到 Hugging Face Hub 缓存，不把模型文件加入 Git。

```toml
[identity]
provider = "WEBFACE600K_ARCFACE_R50"  # Generator 的 source embedding

[loss.coarse.identity]
provider = "MS1MV3_ARCFACE_R50_FP16"  # Identity teacher 可独立选择
```

与其他 Provider 一样，输入必须是 ArcFace 112 对齐的 RGB、`[-1,1]`，输出是 L2 归一化的 512 维 embedding。`[loss.hq.identity].provider` 也可以独立选择该模型。对应 CLI 的 `id-encoder --provider WEBFACE600K_ARCFACE_R50` 可用于导出 ONNX。这里仅接入 **身份编码器**；InSwapper 换脸模型内部的 `emap` 投影矩阵不属于该编码器，不会用于本项目的生成器。InsightFace 的预训练权重有非商业研究用途限制，不因以 `.pth` 格式分发而获得商业使用授权。


优化策略按训练实体分开配置：`[optimizer.generator]`、`[optimizer.hq_discriminator]`、`[optimizer.coarse_discriminator]` 分别设置 LR；对应的 `[scheduler.generator]`、`[scheduler.hq_discriminator]`、`[scheduler.coarse_discriminator]` 分别设置 `type / t_max / min_lr_ratio`。Adam 的 `betas=(0.0, 0.99)` 与 fused 实现仍固定在训练代码中。

两个 Discriminator 的架构也完全独立：`[discriminator.hq].img_resolution` 必须等于 `generator.img_resolution`，`[discriminator.coarse].img_resolution` 必须等于 `generator.coarse_resolution`；两者的 `img_channels` 都必须等于 `generator.img_channels`。`base_ch` 与 `max_ch` 可以分别设置。默认的 FIR_MINIBATCH_STD 是单 Global Discriminator：Encoder 通过 residual block 与 FIR 抗混叠逐级下采样到 4x4，再经 MinibatchStd、3x3 Conv 和两层 Linear 输出每图一个 logit。INSTANCE_NORM_RESIDUAL 保留旧的全局残差判别器。新增 Coarse 专用 UNET（仅 128x128）：共享 Encoder 下采样到 8x8，Global Head 输出 [B,1]，带 Encoder Skip 的 Decoder 上采样至 128x128，Spatial Head 输出 [B,1,128,128]。UNET 用双线性上采样 + Conv 残差，不使用 ConvTranspose2d、Attention、CutMix、CR、GP、SN 或归一化层；其默认 base_ch=32、max_ch=256。

### 可选 Coarse U-Net 判别器

在新 run 配置中选择（已有 run 需 branch 并 reset Coarse Discriminator，不能原地 resume 改架构）：

```toml
[discriminator.coarse]
type = "UNET"
img_resolution = 128
img_channels = 3
base_ch = 32
max_ch = 256

[loss.coarse.gan]
weight = 1.0
spatial_weight = 0.25

[loss.coarse.r1]
enable = true
interval = 16
gamma = 10.0
```

只有 Coarse 可以选择 UNET，固定 128x128；Global/Spatial 都采用标准 Hinge Loss。默认模型 forward 输出 (global_logits, spatial_logits)，return_feats=True 的 Encoder 特征保留弱特征匹配接口；设置 forward(..., return_spatial=False) 时只返回 [B,1] Global logits，完全跳过 Decoder/Spatial Head（同时兼容 return_feats=True）。实际训练中 spatial_weight=0 会自动切换 Global-only 前向，包括常规 D/G step、Lazy R1 和 WFM 路径。显式设置 base_ch/max_ch 始终覆盖判别器默认通道配置。

训练 TensorBoard 的 Loss/coarse_g_global_loss 和 Loss/coarse_g_spatial_loss 分别记录 Global GAN Loss 与已乘 spatial_weight 的 Spatial GAN Loss，二者之和为 Loss/coarse_gan_loss（Spatial 关闭时后者分量记录 0）；已有 Loss/coarse_d_global_loss、Loss/coarse_d_spatial_loss 保留，D 的总对抗损失使用相同 spatial_weight。样本梯度图将 Coarse GAN 拆为 GAN GLOBAL GRAD 和 GAN SPATIAL GRAD 两行，仍在各行标签中显示图像梯度的 RMS、STR、DIR、EFF（Spatial 已包含 spatial_weight）；仅随 sample_save_every 计算，不再向 TensorBoard 写入梯度 RMS 指标。Spatial 关闭时不绘制 Spatial 梯度图。

R1 日志另包含 coarse_r1_branch_spatial（0=Global，1=Spatial）。Spatial Head 的独立参数没有全局 R1 约束，因此采用随机分支 R1，不代表已经比较出最优 R1 方案；注意 Spatial 的均值 R1 并不惩罚每一个独立 logit 的 Jacobian。默认其他判别器的配置、loss 与 checkpoint 行为不改变。

## Generator 梯度职责

`[train].stage` 决定 active training stage：

- `joint`：Coarse 与 HQRefiner 在同一个 iteration 中同时训练；
- `coarse`：只训练 Coarse 与 Coarse Discriminator；训练 step 不执行 HQ forward、HQ D 或 HQ losses；
- `hq`：Coarse 作为冻结的 conditioner 在 `no_grad` 下前向，只训练 HQRefiner 与 HQ Discriminator。

Generator、HQ Discriminator、Coarse Discriminator 分别拥有独立的 optimizer/scheduler/GradScaler。inactive Discriminator 保持在 CPU，不创建 optimizer/scaler/scheduler，也不占用训练 GPU 显存；inactive Generator stage 参数没有 gradient。EMA 只更新 active Generator stage。`hq` 模式下 HQ 输入仍使用冻结 Coarse 的输出；`joint` 模式下 HQ 输入使用 `coarse.detach()`，因此 Final/HQ losses 不会反向修改 Coarse。

- Coarse：`[loss.coarse]` 独立控制 source identity、coarse GAN、R1、gaze / HRFFA / FACS，以及 reconstruction scope 下的 L1/VGG；
- HQ：`[loss.hq]` 独立控制 HQ identity、HQ GAN/R1、gaze / HRFFA / FACS，以及 reconstruction scope 下的 L1/VGG；
- Discriminator：HQ 与 Coarse 独立选择架构，并分别维护自己的 D optimizer、scheduler、GradScaler 与 R1 enable/interval/gamma。默认 FIR 判别器通过 FIR 抗混叠下采样到 4x4，使用 MinibatchStd；普通 D step 将 fake/real 合并为一次 forward，但 MinibatchStd 仍分开统计。Coarse UNET 使用同一 Encoder 的 Global/Spatial 双输出，分别计算 Hinge 并按 [loss.coarse.gan].spatial_weight（默认 0.25）加权；Generator 同样使用此权重。其 lazy R1 每次在 Global 与 Spatial 分支中伪随机选择一个，选中 Spatial 时先对 128x128 logits 逐样本求空间平均再计算 R1（不是完整空间 Jacobian 的惩罚）；随机选择仅依赖 completed_step，因此 overflow retry/resume 的选择稳定；spatial_weight=0 时只选 Global Head。与原版 GP 不同，R1 只基于 real 输入，在未编译的 FP32 判别器上计算，并将 raw R1 乘以 interval。joint 下两个 D 独立更新；一个 D 的 FP16 overflow 仅重试该 D。

Coarse 内部首先将真实 `dst` 双线性缩放到 `coarse_resolution`；训练时通过 `return_resize_in=True` 复用这一实际模型输入作为 Coarse Discriminator real，以及 L1/VGG reconstruction target，不在 Trainer 中重复 resize。Coarse/HQ 的 L1 reconstruction 由各自的 `[loss.coarse.l1].gaussian_blur`、`[loss.hq.l1].gaussian_blur` 控制：整数 `0`（默认）直接计算逐像素 L1；正整数 `N` 启用 Gaussian blur（`sigma = 当前阶段分辨率 / N`），例如 `384` 对应之前硬编码的模糊强度，`256` 则更强。开启时仅对 prediction-target 差分进行等价的滤波，不影响 VGG，VGG 始终使用原图。Coarse L1/VGG 由 `[loss.coarse.reconstruction].scope` 控制作用范围；默认 `same`，只对 self-reconstruction pair 生效。Coarse 的 target 属性监督则先将低分辨率输出双线性上采样到训练分辨率，再使用与 Final 相同的 `theta_restore` 恢复到 canonical 坐标系，与 `dst_canonical` 比较。

## 训练数据源

训练数据只支持本地图片目录，不再由训练进程访问 Hugging Face、ModelScope 或其他在线数据集。`[[data.src]]` / `[[data.dst]]` 的 schema 只包含本地路径、权重调整和可选对齐方式：

```toml
[[data.src]]
path = "/path/to/source_faces"
adjustment = 0.0
alignment = "ffhq"  # 可省略；src 也可设为 arcface

[[data.dst]]
path = "/path/to/target_faces"
adjustment = 0.0
```

`alignment = "arcface"` 仅允许用于 `data.src`。这类目录中的图片应已经是 canonical ArcFace 对齐；训练管线会直接保留其 112×112 身份分支，不再经过 FFHQ→ArcFace 裁剪。`data.dst` 固定使用 FFHQ 对齐。多个目录仍按 `sqrt(file_count) * 2**adjustment` 分配目录采样权重。`backend`、`repo_id`、`revision`、`path_prefix`、在线数据集 proxy/cache 等旧字段均不再接受。模型权重仍可由各模型模块通过 Hugging Face Hub 获取；这与训练数据源是两个独立职责。

FairFace 可直接作为额外的 source identity 池。项目固定下载 `HuggingFaceM4/FairFace` 的 `0.25` 配置，并将官方 `train` 与 `validation` 三个 Parquet 统一映射为一个本地 identity pool；这里不保留训练/验证语义，因为 FairFace 仅作为 source ID 素材使用。随后通过项目 RetinaFace 5 点检测重新对齐到 ArcFace 112，而不是简单缩放原始图像：

```bash
uv run --no-sync python tools/download_datasets.py fairface --root datasets/downloads
uv run --no-sync python tools/postprocess_dataset.py \
  datasets/downloads/fairface/*.parquet \
  --format fairface \
  --output datasets/fairface-arcface112
```

输出按粗粒度目录组织为 `asian/`、`indian/`、`black/`、`white/`、`middle_eastern/`、`latino/`。FairFace 原始 `East Asian` 与 `Southeast Asian` 归入 `asian/`；`Indian` 保留为独立 `indian/`，避免将南亚身份混入主要用于补充东亚/东南亚身份的 source pool。由于训练数据扫描不递归，使用这些目录时应分别配置为 `[[data.src]]`，并设置 `alignment = "arcface"`。

## Run 目录

每次 fresh training 都创建新的唯一 run：

```text
experiments/
├── train.toml
└── runs/
    └── 20260910-162600_blendface-vgg/
        ├── config.toml
        ├── config.resolved.json
        ├── metadata.json
        ├── latest.json
        ├── checkpoints/
        │   ├── step_000010000.pth
        │   └── step_000020000.pth
        ├── samples/
        │   ├── step_000001000.png
        │   └── step_000002000.png
        └── tensorboard/
            └── events.out.tfevents.*
```

各文件职责如下：

- `config.toml`：启动该 run 时用户提供的 TOML 原文快照，便于人工阅读。
- `config.resolved.json`：展开显式训练配置协议默认值后的规范配置；这是 run 的机器可读事实来源。
- `metadata.json`：只记录 run ID、状态、配置摘要和可选的 branch 父来源。
- `latest.json`：很小的最新 checkpoint 指针，不复制大模型文件，适合本地文件系统和对象存储。
- `checkpoints/`：完整训练状态。
- `samples/`：与 completed step 对齐的训练可视化；每一行最左侧标注该行含义。基础图包含 src、dst、Coarse、Final、canonical dst、Final Identity 输入；随后对 active stage 的 GAN/Identity 以及当前实际启用的 WFM/Gaze/HRFFA/FACS/VGG/L1 分别绘制原有 `loss_grad_map` 风格的梯度图。未启用的可选损失不占行，R1 不进入 sample 可视化。

每个 active stage 在 EMA 输出图像上分别计算各项**已加权** Generator loss 的图像梯度 `g_i = dL_i / dx`，梯度只求导一次，标注以下量：

- `RMS`：图像梯度的均方根幅度；`STR`：梯度 L2 范数占各损失范数之和的百分比。
- `DIR`：单项梯度与合成梯度 `g = sum(g_i)` 的余弦相似度，取值 [-1, 1]。
- `EFF`：`dot(g_i, g) / dot(g, g) * 100%`，表示沿合成梯度方向的有符号投影贡献。可以为负或大于 100%；当 `||g|| / sum(||g_i||) > 0.2` 时计算，其余高抵消率（80% 以上）场景显示 `N/A`，避免近零分母放大为不直观的极端百分比；仅在显示数值时各项之和为 100%。
- `STAGE CANCEL`：`(1 - ||g|| / sum(||g_i||)) * 100%`，仅在该 stage 第一行梯度标签标注。当所有梯度为 0 时为 `N/A`；合成梯度小于各项范数之和的 0.01% 时，`DIR` 和 `EFF` 为 `N/A`，避免近乎完全抵消时分母过小。

梯度图默认沿用**每项损失、每张图片独立归一化**的亮度表现；可将 `swapface/train.py` 中 `SAMPLE_GRADIENT_HEATMAP_SCALE` 改为 `"shared"`，让同一 stage 的不同损失对每张图使用共享亮度标尺。任何热力图归一化都不参与指标计算。上述指标位于**生成图像空间**，不等于生成器参数梯度、Adam 更新份额或实际训练每步的精确贡献；样本生成器采用 EMA 权重。
- `tensorboard/`：TensorBoard event 文件；一次 resume 可能新增 event 文件，这是正常现象。

`experiments/*.toml` 可以进入 Git；`experiments/*/` 属于运行生成物，默认忽略。

## Step 与文件命名

当前 checkpoint 只保存一个权威的 `step`，表示已经完整完成的训练更新次数。

文件名统一使用 **completed step**：

```text
step_000000001.pth
step_000010000.pth
step_000010000.png
```

例如 `step_000010000.pth` 表示已经完成 10,000 次训练更新。`train.checkpoint_save_every = 10000` 会在 completed step 10,000、20,000、30,000... 保存。

checkpoint 仍通过临时文件写入后原子 `replace`；只有 checkpoint 成功落盘后才原子更新 `latest.json`。

`completed step` 只在一轮完整的 Discriminator 更新、Generator 更新、scheduler（若启用）和 EMA 更新全部完成后推进。checkpoint 内部 `step` 与文件名中的 step 必须完全一致。因此从 `step_000050000.pth` 恢复后，在下一轮更新真正完成前，completed step 仍是 50,000，不会把正在执行的 partial step 标记成 50,001。

## 中途恢复训练

推荐直接指定 run：

```bash
uv run python -m swapface.train \
  --resume experiments/runs/20260910-162600_blendface-vgg
```

程序会从 `latest.json` 找到 checkpoint，读取该 run 冻结的 resolved config，然后直接恢复 Generator、Discriminator、EMA、optimizer、scheduler 与 step，继续写入原 run。模型/优化器/scheduler 的 state dict 是否可加载由 PyTorch 自己校验。

也可以显式指定同一 run 的 **latest checkpoint**：

```bash
uv run python -m swapface.train \
  --resume experiments/runs/20260910-162600_blendface-vgg/checkpoints/step_000070000.pth
```

显式文件必须位于标准 `run/checkpoints/` 目录中，并且必须与 `latest.json` 指向同一文件。`--resume` 不允许从历史 checkpoint 回滚后继续写入原 run，否则会产生分叉历史并覆盖同 step 的 checkpoint/sample。历史 checkpoint 应通过 branch 创建新的 run。

## Branch：从已有 checkpoint 派生新实验

Branch 是一条新的训练时间线，不是 resume。普通 branch 从父 checkpoint 的**训练态 Generator**继续，默认同时继承父 Discriminator 权重；optimizer、scheduler 和 GradScaler 重新初始化。新配置可以改变 `[train].stage`，因此从 joint checkpoint 派生 `hq` 或 `coarse` 单 stage run 应使用 branch，而不是 resume。普通 branch 的 `completed_step` 默认继承父 checkpoint，便于直接比较分支前后的 TensorBoard loss 趋势：

```bash
uv run python -m swapface.train \
  --branch-from experiments/runs/20260910-162600-blendface-vgg/checkpoints/step_000050000.pth \
  --config experiments/train.toml \
  --name new-experiment
```

`--branch-from` 必须显式提供 `--config`。新 run 的 `metadata.json.parent` 会记录父 `run_id`、checkpoint、父 step、配置摘要、branch mode，以及 Generator 的 Coarse/HQ 和两个 Discriminator 分别是 `inherit` 还是 `reset`。

默认继承父 checkpoint 的 step。如果希望从其他 step 开始，单独指定 `--step`：

```bash
uv run python -m swapface.train \
  --branch-from experiments/runs/<run_id>/checkpoints/step_000050000.pth \
  --config experiments/train.toml \
  --step 30000
```

`--step 0` 可以让新分支从 0 重新计数。这个值就是新 run 的实际 `completed_step` 起点，因此也会同时影响 EMA decay、sample/checkpoint 保存周期和文件名；它不是单独的 TensorBoard 显示偏移。

普通 branch 要求 `[generator]` 与父 checkpoint 完全一致。默认还要求被继承的 Discriminator 架构一致。HQ D 与 Coarse D 可以分别重建：

```bash
# 只重建 HQ D
uv run python -m swapface.train \
  --branch-from experiments/runs/<run_id>/checkpoints/step_000050000.pth \
  --config experiments/train.toml \
  --reset-hq-discriminator

# 只重建 Coarse D
uv run python -m swapface.train \
  --branch-from experiments/runs/<run_id>/checkpoints/step_000050000.pth \
  --config experiments/train.toml \
  --reset-coarse-discriminator
```

需要两个 D 都重建时同时传入两个开关。Branch metadata 会分别记录 HQ/Coarse D 的 `inherit` / `reset`。HQ D 与 Coarse D 的架构分别由 `[discriminator.hq]` 和 `[discriminator.coarse]` 控制，因此可以只修改并 reset 其中一个；任何选择 inherit 的 D 都只要求自己的架构与父 checkpoint 保持一致。无论是否恢复 D，Branch 都不会继承 G/D optimizer、scheduler 或 GradScaler；step 默认继承父 checkpoint，也可以由 `--step` 指定。

默认训练语义使用单 Global Discriminator：Encoder 下采样到 4x4，保留 FIR 抗混叠与 MinibatchStd，移除 SpectralNorm，并使用 real-only lazy R1。可选 Coarse UNET 使用本文前述双输出训练路径。Resume 与 Branch 都只接受当前训练语义版本的 checkpoint，不提供旧训练结果迁移兼容。

### HQ rebuild：继承 Coarse，用新分辨率从零训练 HQ

当新配置使用 `train.stage = "hq"` 且 Generator 架构与父 checkpoint 不同时，Branch 会进入 **HQ rebuild** 模式。该模式只允许 HQ 部分变化；Coarse 的结构参数必须与父 checkpoint 完全一致，并且 `[identity].provider` 必须保持一致，因为冻结 Coarse 仍依赖同一 identity embedding 空间。

HQ rebuild 的固定语义为：

- 从父 checkpoint 的 `training_state.net_g` 中只提取并严格加载 `coarse.*`；
- 新配置重新创建 HQ，旧 HQ 权重完全不加载；
- HQ Discriminator 自动重新初始化，不需要额外传 `--reset-hq-discriminator`；
- Coarse 冻结并保持 `eval()`，HQ 为唯一可训练 Generator stage；
- Generator optimizer 只包含 HQ 参数；HQ D optimizer/scheduler/GradScaler 全部重新初始化；
- 新 run 强制从 `step 0` 开始，省略 `--step` 即可；显式非零 `--step` 会被拒绝；
- EMA 从“继承后的 Coarse + 新初始化 HQ”复制得到，后续只更新 HQ EMA，因此训练态/EMA 的冻结 Coarse 始终一致。
- 启动时会打印所有实际发生变化的 HQ Generator 配置项（`old -> new`），以及 Coarse/HQ/HQ D 的 inherit/reset 状态、identity provider 和起始 step。

例如，从一个 HQ256/Coarse128 checkpoint 开始训练新的 HQ512：

```bash
uv run python -m swapface.train \
  --branch-from experiments/runs/<run_id>/checkpoints/step_000080000.pth \
  --config experiments/train-hq512.toml \
  --name hq512
```

新配置应保持 Coarse 参数不变，只修改 HQ/full-resolution 相关配置，例如：

```toml
[train]
stage = "hq"

[generator]
img_resolution = 512
coarse_resolution = 128
coarse_latent_resolution = 32
coarse_num_latent = 8
coarse_base_ch = 64
coarse_max_ch = 512
hq_bottleneck_resolution = 16
hq_base_ch = 8
hq_max_ch = 128

[discriminator.hq]
img_resolution = 512
```

`img_resolution`、`hq_bottleneck_resolution`、`hq_base_ch`、`hq_max_ch` 等 HQ 结构参数都可以改变，因为 HQ 会完整重建；`img_channels`、`id_dim` 和所有 `coarse_*` 结构参数属于 Coarse compatibility signature，不允许变化。

因此三种入口的语义为：

- **fresh**：新 Generator、新 Discriminator、新训练状态；
- **resume**：原 run、原冻结配置、latest checkpoint，完整恢复训练态 Generator、EMA、Discriminator、optimizer、scheduler、GradScaler 与 step；
- **branch**：新 run；普通 branch 继承父训练态 Generator，HQ rebuild 只继承 Coarse 并重建 HQ/HQ D；optimizer/scheduler/GradScaler 均重新初始化。普通 branch 的 step 默认继承父 checkpoint，HQ rebuild 固定从 0 开始。

Branch 不检查 `training_config.semantics_version`，因为它不恢复父训练状态；resume 仍严格检查该版本。

### 可选严格校验训练精度

`[train].precision` 必须显式配置，支持 `fp32`、`fp16`、`bf16`。FP16 使用 Generator、HQ Discriminator、Coarse Discriminator 三个独立 `GradScaler`；BF16 不使用 loss scaling。仿射恢复与 HRFFA 的数值敏感几何求解继续固定为 FP32。

FP16 中只有实际执行了 optimizer update 的阶段才推进对应 scheduler。D overflow 时当前 global step 不推进；D 已成功而 G overflow 时只重算并重试 G，直到 G 成功后才更新 EMA 和 `completed_step`。任一 active D stage 的 loss 或 `g_loss` 出现 NaN/Inf 时直接终止，避免把非有限值误当作可通过降低 loss scale 恢复的 overflow。BF16/FP32 在 backward 后、`optimizer.step()` 前额外检查参数梯度；若任一梯度出现 NaN/Inf，则报告首个异常参数并终止，避免污染模型参数和 optimizer state。FP16 的梯度 overflow 仍由 `GradScaler` 负责跳过 update 和动态调整 scale，不走该 fatal gradient guard。

默认 resume 不要求当前实际训练精度与 checkpoint 一致，因为设备和软件环境可能变化。需要严格复现时可显式使用：

```bash
uv run python -m swapface.train \
  --resume experiments/runs/<run_id> \
  --strict-precision
```

此时程序比较 checkpoint 中记录的实际 `training_config.precision` 与当前进程实际生效的精度；不一致时拒绝 resume。`--strict-precision` 是本次 resume 的运行时策略，不写入 TOML，也不能用于 fresh training。

训练 checkpoint 还记录 `training_config.semantics_version` 和实际 `stage`。Resume 与 Branch 都要求 checkpoint 的 `semantics_version` 与当前训练代码完全一致，不提供旧训练语义迁移；Resume 还要求 active stage 一致，因为它会恢复 optimizer/scheduler/scaler 状态。Branch 不恢复这些状态，因此在同一训练语义版本内仍允许通过新配置切换 stage。纯推理/导出仍只依赖模型 checkpoint 格式。

## 配置冻结与哈希

run 同时保存用户原始 TOML 与由 `swapface/config.py` 按显式训练配置协议展开的规范 JSON：

```text
config.toml           # 用户输入原文
config.resolved.json  # 完整、显式、可哈希的规范配置
```

`config.resolved.json` 会显式记录：

- `[train]` 的协议默认值；
- Generator / Discriminator 的协议默认值；
- dataloader 的协议默认值；
- Identity encoder 的协议默认值；
- VGG 的协议默认权重；
- 本地数据源 path/adjustment。

其规范 JSON 内容计算 SHA-256 并写入 `metadata.json`。新 checkpoint 也记录同一摘要与 run ID；resume 时若摘要或 run ID 不一致，会拒绝继续训练。

## Run 状态

`metadata.json` 的 `status` 会在生命周期中更新：

```text
created -> running -> interrupted
                   -> failed
                   -> completed
```

Ctrl+C 属于 `interrupted`，后续可直接 `--resume`。训练开始后第一次 Ctrl+C 会请求在当前完整 step 结束后退出，从而保存一致的 checkpoint；再次 Ctrl+C 会立即中断，如果此时不在完整 step 边界则拒绝保存 partial checkpoint，并保持已有 `latest.json` 不变。初始化或训练异常会记录错误类型与消息并标记为 `failed`。训练进程因 Ctrl+C 退出时返回非零状态码 130。

同一个 run 还会持有 `.run.lock` 的 Linux advisory lock。若另一个训练进程同时尝试 resume 同一 run，会立即拒绝启动；锁由内核绑定到进程/文件描述符，进程崩溃后会自动释放，因此不会出现“残留 lockfile 导致永久锁死”的问题。

## TensorBoard

查看全部 run：

```bash
uv run tensorboard --logdir experiments/runs
```

恢复同一个 run 后，TensorBoard 目录可能出现多个 `events.out.tfevents.*`。它们仍属于同一逻辑 run；恢复时会设置 `purge_step`，让 TensorBoard 隐藏上次进程在最后 checkpoint 之后写出的失效 future steps，避免回滚恢复后出现重复曲线。

## 在线遮挡增强（Phase 2）

可选 `[data.occlusion]`：`prob` 为样本遮挡概率（0～1），`assets_dir` 为裁剪成透明 RGBA 的 PNG/WebP 遮挡素材目录；不设置目录或设为空字符串时，使用程序化随机纹理遮挡（RandOcc，**真实感弱于 NatOcc 手/物体素材**）。关闭时不加载 OCC 模型、不额外执行推理。素材在初始化时预加载到 GPU，训练循环内仅用批量仿射采样和 Alpha 合成。

遮挡只施加于 `dst`，因此 Generator 输入、Discriminator real/fake、L1/VGG 的 real target 均同步包含遮挡；`src`、`source_identity_faces` 与几何教师使用的干净 `dst_canonical` 不变。新增 `ID` 专属可见区域梯度门控：针对本批被遮挡的样本，使用项目现有 `FaceMasker(mode="occlusion")` 对**遮挡前、已增强的 dst** 推理，并结合合成遮挡的 Alpha，只允许可见面部像素接收 ID 梯度。未遮挡样本不改变原有 ID 梯度。Coarse 门控缩放至对应输出分辨率。遮挡样本的 ID encoder 仍然观察生成的遮挡面部，梯度门控不等价于无损恢复身份特征。

当前 OCC checkpoint 在 `eval()` 模式下可能输出全零，因此跟随现有 `swapper.py` 使用训练模式并冻结参数、关闭梯度；模型内部 BatchNorm 仍依据选中样本的 batch 统计运行。**这并不是几何鲁棒性的完整实现**：Gaze/HRFFA/FACS 尚未增加遮挡置信度控制，建议结合模型梯度贡献与遮挡实验进行评估。OCC 需要在训练时额外执行一次小 batch 前向，当前未对吞吐做硬保证。

```toml
[data.occlusion]
prob = 0.2
assets_dir = "/path/to/rgba-occluders"
```

NatOcc 原始遮挡物若采用独立的图像目录与 PNG 掩码目录，应**离线转换一次**（训练时无需加载遮挡物 Mask 文件）：

```bash
uv run --no-sync python tools/prepare_occluders.py \
  --images /path/to/NatOcc/images \
  --masks /path/to/NatOcc/masks \
  --output /path/to/rgba-occluders
```
