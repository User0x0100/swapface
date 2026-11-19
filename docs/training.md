# SwapFace 训练协议（单阶段）

生成器统一直接输出 `img_resolution × img_resolution` 的 RGB 图像。不存在 Coarse/HQ 分阶段训练、HQRefiner、中间超分图像或第二个判别器。

## 生成器

`[generator].type` 有两个选项：

- `inswapper`：完整分辨率 U-Net。`FromRGB → Encoder → LatentBlock → Decoder → ToRGB`；Bottleneck 使用既有 `WSpaceMap + IDInject/AdaIN`。`latent_resolution` 控制瓶颈分辨率，`num_latent` 控制身份残差块数量，`base_ch/max_ch` 控制各分辨率通道。`skip_mode = "cat" | "add"`；`skip_layers` 按 Decoder 从瓶颈向外、从 0 开始编号。空数组禁用跨级连接，最后一层是 FromRGB 与最高分辨率 Decoder 输出的融合。`latent_resolution >= 2`（Bottleneck 的反射填充卷积要求空间尺寸至少为 2）；仅缓存实际启用的 Skip 特征。
- `stylegan2`：直接生成完整分辨率 RGB，维持 StyleSwap Target 编码和多分辨率注入。参数：`w_dim`、`mapping_layers`、`start_resolution`、`channels`、`target_inject_mode`、`target_inject_layers`。可选 `mask_enabled`、`mask_noise_blank_level`、`mask_guidance_epsilon`、`mask_guidance_interpolation`。

`cat` 在对应 Decoder 层的上采样完成后与同分辨率 Encoder 特征按通道拼接，后续层的 `in_channels` 自动匹配；`add` 进行逐元素相加。没有 AAD、额外权重门控或隐藏的通道投影。

两类生成器配置互斥；不接受其他类型专属字段。训练配置中的 `img_channels` 已删除，Generator 和 Discriminator 都默认使用 RGB 三通道；该字段不进入 canonical config 或检查点网络配置。`id_dim` 等其他参数仍按其原有设计配置。修改分辨率、跳跃层级或融合方式会改变模型架构，需新建 run。

## 训练

`[train]` 包含 `batch_size`、`precision` (`fp32/fp16/bf16`)、`device`、`compile_module`、`compile_mode` 和日志/样本/检查点间隔，不再有 `stage`。`compile_mode` 原样传递给 PyTorch `torch.compile(mode=...)`，不额外校验；`compile_module = false` 时不使用。默认 `max-autotune-no-cudagraphs`，常见模式为 `default`（标准优化）、`reduce-overhead`（尝试减少 Python/Kernel Launch 开销，通常使用 CUDA Graph）、`max-autotune`（更积极的内核调优）、`max-autotune-no-cudagraphs`（积极调优但禁用 CUDA Graph）。具体支持模式以安装的 PyTorch 版本为准。

- `[optimizer.generator]`、`[optimizer.discriminator]`：两组 Adam 的学习率，`betas=(0,0.99)`，融合优化器。
- `[scheduler.generator]`、`[scheduler.discriminator]`：各自的 `none/cosine` 调度。
- `[discriminator]`：只有一个全分辨率判别器，支持 `FIR_MINIBATCH_STD/INSTANCE_NORM_RESIDUAL/UNET`。UNET 的 R1 强制关闭。
- `[loss.*]`：只有一套 `gan/identity/r1/wfm/gaze/smirk/reconstruction/l1/vgg`。只有 StyleGAN2 能配置 `[loss.mask]`。`reconstruction.scope` 为 `same` 或 `all`；L1/VGG 使用此范围约束。
- `[identity]`：生成器 Source 身份编码器；`[loss.identity]` 指定独立损失教师，二者可选相同以重用特征。

训练一步：数据加载及增强 → Generator 身份特征 → 生成最终 RGB（可含 Mask Logits）→ 单判别器更新（lazy R1）→ 单组 Generator Loss / backward → EMA / scheduler → 日志/预览/检查点。SMIRK、Gaze、VGG、ID 梯度筛选等逻辑保留。

## Run / Checkpoint

新训练：`uv run python -m swapface.train --config experiments/train.toml [--name NAME]`

严格恢复：`uv run python -m swapface.train --resume experiments/runs/<run_id> [--strict-precision]`

派生新 run：`uv run python -m swapface.train --branch-from <model.pth> --config experiments/train.toml [--step N] [--reset-discriminator]`

Branch 直接尝试加载 Generator 权重（`load_state_dict(strict=True)`），不再比较来源检查点的模型配置、对齐方式、身份编码器、训练精度或训练语义版本。优先继承 `training_state.net_g`，没有训练权重时可以继承 `net_g.state_dict`（EMA/推理模型），也支持直接保存的原始 Generator `state_dict`。Generator 权重的键名或张量形状不兼容时会明确报错，不会悄悄进行部分加载。判别器权重如果能够完整加载就继承；否则自动重新初始化，并把实际结果写入新 Run 的 `metadata.json`。`--reset-discriminator` 可以强制重置判别器。对于无 step 元数据的模型，默认从 step 0 开始，也可用 `--step` 显式指定。Resume 仍要求原 Run 冻结配置及身份教师一致。不支持旧版本 Coarse/HQ 模型或训练状态迁移。新格式 `CHECKPOINT_VERSION=8`、`TRAINING_SEMANTICS_VERSION=25`。检查点包含 `net_g`（EMA）、`net_d` 和 `training_state`（训练权重、优化器、scaler、scheduler）；`face_alignment` 与 Generator identity provider 一并校验。Resume 必须继续原 run 的冻结 canonical config。

预览：`python -m swapface.swapper --model <step.pth> --video <video> --identity <image>`；直接运行完整 Generator，不再提供中间分辨率模式。


---

## 安装 GPU 后端

依赖组按用途与计算后端区分，默认不启用依赖组（`default-groups = []`）。锁文件只面向 Linux x86_64，统一记录公共 PyPI、PyTorch CUDA 12.6、PyTorch ROCm 7.2、NVIDIA DALI 的**官方制品来源**，避免其他运行环境在更新锁文件时访问阿里云内网。

| 组 | 用途 | 依赖来源 |
| --- | --- | --- |
| `nvidia-train` | NVIDIA 最小训练环境 | PyTorch CUDA 12.6 + NVIDIA DALI + PyPI |
| `nvidia` | NVIDIA 完整开发环境（含导出、预览、测试） | PyTorch CUDA 12.6 + NVIDIA DALI + PyPI |
| `rocm` | AMD ROCm 环境 | PyTorch ROCm 7.2 + PyPI |

NVIDIA 最小训练（包含 DALI、SMIRK、VGG、遮挡增强所需的 `segmentation-models-pytorch`、TensorBoard、自定义 CUDA 算子）：

```bash
uv sync --locked --only-group nvidia-train
uv run --no-sync python -m swapface.train --config experiments/train.toml
```

阿里云 PAI-DSW 如果需要**内网镜像**，使用单独的镜像安装入口：

```bash
bash scripts/sync-pai.sh
uv run --no-sync python -m swapface.train --config experiments/train.toml
```

`sync-pai.sh` 先从同一份 `uv.lock` 导出 NVIDIA 最小训练组的固定版本和 SHA-256 哈希，再将 PyPI 包和 CUDA 12.6 wheel 分别从 PAI 的阿里云内网镜像安装到项目 `.venv`。因此**无需另设 `nvidia-train-aliyun` 依赖组，也不会重新锁定 ROCm 依赖或改写共享 `uv.lock`**。内网镜像须完整包含锁文件要求的相同 wheel，否则安装应失败，而不是静默更换版本或校验哈希。可以通过 `PAI_PYPI_INDEX`、`PAI_CUDA_WHEELS` 覆盖镜像地址；脚本将额外参数透传给 `uv pip sync`（例如 `--dry-run`）。此入口仅用于阿里云 PAI 或能够访问同一内网的环境。

锁定变更后在公共网络执行 `uv lock --check`。切换依赖组用 `uv sync --locked ...`，避免安装时顺便修改锁文件。**`--only-group` 控制安装范围，不会缩小 uv 重新锁定时要解析的所有依赖组**，因此不可将内网专有 URL 放入项目共享 `tool.uv.index`。

`--only-group` 用于避免安装 `[project].dependencies` 中与训练无关的包。当前最小组不含 TensorRT、ONNX Runtime GPU、TorchCodec、xFormers；遮挡增强所需的 `segmentation-models-pytorch` 已包含在最小训练组中，无需另行安装。

NVIDIA 完整开发环境：

```bash
uv sync --locked --group nvidia
```

AMD ROCm 环境：

```bash
uv sync --locked --group rocm
```

ROCm 不安装 DALI、TorchCodec、ONNX Runtime GPU、xFormers 或 TensorRT。NVIDIA 与 ROCm 组互斥。运行训练时建议使用 `uv run --no-sync ...`，避免再次同步环境。

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

- `config.toml`：原始 TOML 配置快照。
- `config.resolved.json`：含全部默认值的 canonical 配置及格式版本；Resume 以此为权威来源。
- `metadata.json`：Run ID、状态、配置哈希及 Branch 父来源。
- `latest.json`：指向最近一次成功保存的 checkpoint；不复制模型数据。
- `checkpoints/`：模型、优化器、AMP scaler、调度器和训练状态。
- `samples/`：EMA 输出、RGB 对比、身份输入和启用损失的梯度热力图。
- `tensorboard/`：TensorBoard event 数据，Resume 可产生多个 event 文件。

每次训练完成一个完整 step 后才可保存检查点。检查点先写临时文件、原子替换，再更新 `latest.json`。
`step_000010000.pth` 表示完成 10,000 步，`step_000010000.png` 是该步的可视化。

训练 Run 使用 `.run.lock` 防止并发 Resume 同一目录；创建新 Run 时自动分配唯一名称。
梯度热力图的 `RMS/STR/DIR/EFF` 度量来自生成图像空间而非参数空间；R1 不参与图像梯度可视化。

## 恢复、派生与状态管理

- `--resume experiments/runs/<run-id>`：必须继续该 Run 的 `latest`，并读取冻结的 `config.resolved.json`。传入历史 checkpoint 会被拒绝。
- `--branch-from /path/to/model.pth --config experiments/train.toml`：创建新 Run，只要求来源 Generator 权重能严格加载，不限制文件名、模型配置元数据、损失教师及数据对齐方式；没有训练权重时尝试 EMA/推理权重。判别器权重不兼容则自动重置。
- 可选 `--step N` 只影响 Branch 的新起始 completed step，不重置权重；`--strict-precision` 仅对 Resume 生效。
- 收到一次 Ctrl+C 时请求在完整 step 边界保存并退出；进程故障不会将 partial step 误写为 latest。
- Run 状态写入 `metadata.json`：`created`、`running`、`interrupted`、`failed` 或 `completed`。

## TensorBoard

查看全部 run：

```bash
uv run tensorboard --logdir experiments/runs
```

恢复同一个 run 后，TensorBoard 目录可能出现多个 `events.out.tfevents.*`。它们仍属于同一逻辑 run；恢复时会设置 `purge_step`，让 TensorBoard 隐藏上次进程在最后 checkpoint 之后写出的失效 future steps，避免回滚恢复后出现重复曲线。

## 在线遮挡增强

通过 `[data.occlusion]` 启用在线合成遮挡；删除该表或令 `prob = 0.0` 可关闭。`prob` 为样本概率（0～1），`assets_dir` 是本地透明 RGBA PNG/WebP 素材目录；空目录配置可使用程序化遮挡。遮挡影响生成器输入 `dst`，但不改变 Source 身份或目标 canonical 几何参考。ID 梯度可按可见区域过滤，SMIRK 按遮挡的原始可见比例降权；天然遮挡无法由此可靠识别。相关额外教师和遮挡采样可能增加训练时间。

### 已预处理的 NatOcc 遮挡素材（2026-10-23）

已在 AMD 服务器将 [face-occlusion-generation](https://github.com/kennyvoo/face-occlusion-generation) 引用的第三方图像及掩码转为 **328 张独立 RGBA PNG**：COCO Objects 128 张、11k Hands 200 张（只使用与作者提供的 200 张手部 Mask 匹配的图片；没有把完整 11k Hands 当成可用遮挡素材）。每张图裁掉透明边界，保持长宽比，最长边不超过 512 像素。训练加载器会进一步缩放到 192×192、预载入 GPU，并执行批量仿射变换与 Alpha 合成。

- **Hugging Face 私有 Dataset**：[`leehx/swapface-occluders`](https://huggingface.co/datasets/leehx/swapface-occluders)。需使用具有访问权限的账号认证；**训练进程不在线下载素材**，应在训练启动前离线准备。
- **数据包**：`swapface-occluders-20261023.tar.gz`（90,361,739 字节，约 86.18 MiB）；解压后为 `swapface-occluders/rgba/*.png`，另有 `manifest.json`（来源、尺寸、单文件 SHA-256）、`preview.jpg` 与 `README.md`。
- **数据包 SHA-256**：`dd42f5188943cdf068f5cbb4200fd378291127dbf43faf8d66ae2442680fc14a`。此哈希已与 AMD 生成包、Windows 本地包及 Hugging Face LFS 核对一致。
- **Windows 本地备份**：`C:\Users\User0\Downloads\swapface-occluders-20261023.tar.gz`；**预处理服务器**：`/workspace/occlusion-prep-20261023/`（包含生成包和预处理产物）。路径是本次操作记录，未来重建的服务器不保证存在。
- **授权边界**：底层 COCO/11k Hands 与上游生成项目涉及不同来源和使用条款；私有仓库仅作为访问受限的工作副本，不代表取得公开再分发或商业使用许可。不得直接转为公开仓库。

在新训练服务器上先登录 Hugging Face（例如 `hf auth login`），再执行：

```bash
mkdir -p /workspace/datasets
hf download leehx/swapface-occluders \
  swapface-occluders-20261023.tar.gz \
  --repo-type dataset \
  --local-dir /workspace/datasets

cd /workspace/datasets
printf '%s  %s\n' \
  'dd42f5188943cdf068f5cbb4200fd378291127dbf43faf8d66ae2442680fc14a' \
  'swapface-occluders-20261023.tar.gz' | sha256sum -c -

tar -xzf swapface-occluders-20261023.tar.gz
find swapface-occluders/rgba -maxdepth 1 -type f -name '*.png' | wc -l
# 预期 328
```

在当前 SwapFace 训练 TOML 中启用：

```toml
[data.occlusion]
prob = 0.2
assets_dir = "/workspace/datasets/swapface-occluders/rgba"
size_range_pct = [40.0, 90.0]
center_x_range_pct = [-15.0, 15.0]
center_y_range_pct = [-15.0, 15.0]
rotation_range_deg = [-40.0, 40.0]
```

**空间参数统一以训练图像边长为基准，使用直观的百分比数值（例如 `50.0` 表示 50%，不是 `0.5`）**：

- `size_range_pct`：遮挡物 RGBA 画布的最长边占训练图像边长的百分比。默认 `[40, 90]`，比早期写死的 `[20, 50]` 明显更大；允许超过 100%，但超出图像的部分会被裁掉。对 128×128 训练图像，默认名义长边为 51.2～115.2 像素。
- `center_x_range_pct` / `center_y_range_pct`：遮挡物中心相对**训练图像中心**的偏移，正值分别向右、向下，默认均为 `[-15, 15]`；`25.0` 代表向相应方向移动图像边长的 25%，不是移到图像坐标的 25% 处。水平/垂直位置独立随机采样。
- `rotation_range_deg`：旋转角度范围（度），默认 `[-40, 40]`，每张图独立随机采样。可写 `[0, 0]` 固定不旋转。
- `prob`：每张图像是否添加遮挡的概率，**仍使用 0～1 的概率值**，不属于空间百分比；`assets_dir` 则为本地素材目录。

预处理后的素材形状不会被拉伸：加载时先按比例将**最长边缩放到 192 像素**（包括放大小图），再居中放入 192×192 透明 RGBA 画布，因此尺寸百分比独立于原素材的像素分辨率。实际遮挡面积还会受到物体轮廓/Alpha、旋转和超出边界的部分影响。旧实现仅通过 `thumbnail()` 缩小素材，会让小素材填不满缓存画布，从而使实际遮挡比设定值更小；现已修正。 缓存中 RGB 使用**预乘 Alpha**，避免几何插值在透明边界产生暗边，Alpha 仍保留原始覆盖语义。

注意：`assets_dir` **只接受本地目录**，当前不递归扫描子目录；不要填 Hugging Face 仓库 ID 或 `.tar.gz` 路径。启用后需把 328 张素材预加载到 GPU（以 192×192 RGBA FP16 计，纹理本体约 92.25 MiB，另有临时张量与 OCC 模型开销）。关闭可删除 `[data.occlusion]` 或设置 `prob = 0.0`。

### 从原始素材重新转换

NatOcc 原始遮挡物若采用独立的图像目录与 PNG 掩码目录，应**离线转换一次**（训练时无需加载遮挡物 Mask 文件）：

```bash
uv run --no-sync python tools/prepare_occluders.py \
  --images /path/to/NatOcc/images \
  --masks /path/to/NatOcc/masks \
  --output /path/to/rgba-occluders
```

### 生成对齐标准（FFHQ / ArcFace-128 模板体系）

在 TOML 的 `[data]` 节添加 `alignment = "arcface"` 可启用全局 ArcFace-128 模板体系的生成坐标系（始终按 128 基准五点模板缩放，不是 112 模板等比缩放）。
缺省或显式 `alignment = "ffhq"` 均使用 FFHQ 对齐，并在规范化配置中显式记录 `alignment`。

- `alignment = "arcface"`：对 FFHQ 对齐数据采用固定五点模板拟合的仿射近似变换，
  统一转换 Generator source、target、未几何增强的 canonical reference；
  Discriminator、REC/VGG、语义损失与采样预览自然共享该坐标系。
- `[[data.src]].alignment` 是独立的**原始数据池格式**，可为 `ffhq` 或 `arcface`；
  当前 ArcFace 来源按 112 模板数据处理：生成输入必须使用 ArcFace112→ArcFace128 固定变换；身份分支仍保留原始 112 裁剪。
- `[[data.dst]]` 目前仍要求原始文件为 FFHQ 对齐，模型输出规范通过全局模式选择，
  不将已有 ArcFace 数据冒充 FFHQ 数据。
- ArcFace 生成输出给身份 teacher 时使用 ArcFace128→112 固定裁剪。128 输出裁剪 x=[8,120)、y=[0,112)，而不是整图缩放到 112。
- 固定 FFHQ→ArcFace 是基于现有 FFHQ 五点模板的**近似统一映射**，不是重新对每张图做关键点检测；
  极端表情、姿态的精确五点对齐可能存在差异。
- Generator 的对齐类型必须写入 checkpoint，并包含在导出 ONNX 元数据中；缺失字段会拒绝加载；
  严格 resume 禁止切换模式，切换应新建 run（可使用 branch）。
- 切换对齐标准后的图像空间分布不同，不建议将新模型直接视作原 checkpoint 的无缝续训。

## SMIRK 面部动态监督

`[loss.smirk]` 提供 `enable`、`pose_weight`、`expression_weight`、`jaw_weight`、`eyelid_weight`。使用冻结的 SMIRK Encoder 监督生成图像与 Target 的头部姿态、表情、下颌和眼睑运动。几何增强开启时，先恢复至 Target canonical 坐标系，再使用 RGB `[0,1]`、224×224 的 FP32 输入。

Jaw/Eyelid 越界时仍通过 Straight-Through 估计保持训练梯度，Pose 旋转矩阵与损失归约也固定 FP32。Target 教师结果在一个训练 step 内复用，跳过非有限结果；日志分别记录 `smirk_pose_loss`、`smirk_expression_loss`、`smirk_jaw_loss`、`smirk_eyelid_loss`。该模型不直接约束完整嘴唇轮廓和眼球注视方向（后者由 Gaze Loss 负责）。

SMIRK 预训练模型由 Hugging Face `Skywork/SkyReels-A1` 仓库的 `extra_models/smirk/SMIRK_em1.pt` 自动下载，固定 revision `6cee256d5ea38dbbe87ff026566c95b6401d3124` 与 SHA-256 `26b234e3cc31a5de226bcba4321bb5d7343a2ad48234c028b57a1c2f8c2d22d0`。未获取到可信的权重时拒绝继续，而非使用随机教师。输出是否匹配真实训练人脸仍需单独验证。
