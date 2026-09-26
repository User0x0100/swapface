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

`[train].compile_module` 控制训练路径的 `torch.compile`：启用时，由 Trainer 统一编译 Generator、Discriminator、Generator 身份编码与 Identity Loss embedding 提取，以及启用的 VGG/WFM 热路径；设为 `false` 时，训练路径全部保持 eager。

## 配置职责

训练配置只存在一套 canonical schema；`swapface/config.py` 负责默认值、严格校验和 canonical 化，`Trainer` 直接消费该结构，不再维护第二套扁平配置参数。顶层职责为：

- `[train]`：batch、precision、device、compile 与输出周期；
- `[optimizer]` / `[scheduler]`：优化器学习率与调度策略；
- `[identity]`：Generator 的 source identity 条件编码器；
- `[loss.*]`：每一种训练损失及其共享 reconstruction 策略；
- `[data.loader]` / `[data.augmentation]` / `[data.sampling]`：数据执行、增强和配对采样；
- `[[data.src]]` / `[[data.dst]]`：训练数据源；
- `[generator]` / `[discriminator]`：模型定义。

Identity Loss teacher 明确位于 `[loss.identity]`，不与 Generator 条件编码器混在 `[identity]`。L1 和 VGG 共享 `[loss.reconstruction].scope`。R1 位于 `[loss.r1]`，不再混入 `[train]`。 `[loss.gan].weight` 只缩放 Generator adversarial loss；Discriminator adversarial loss 固定权重 1.0，避免改变其与 R1 的相对尺度。

## 训练数据源

训练数据只支持本地图片目录，不再由训练进程访问 Hugging Face、ModelScope 或其他在线数据集。`[[data.src]]` / `[[data.dst]]` 的 schema 只包含：

```toml
[[data.src]]
path = "/path/to/source_faces"
adjustment = 0.0

[[data.dst]]
path = "/path/to/target_faces"
adjustment = 0.0
```

多个目录仍按 `sqrt(file_count) * 2**adjustment` 分配目录采样权重。`backend`、`repo_id`、`revision`、`path_prefix`、在线数据集 proxy/cache 等旧字段均不再接受。模型权重仍可由各模型模块通过 Hugging Face Hub 获取；这与训练数据源是两个独立职责。

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
- `samples/`：与 completed step 对齐的训练可视化。
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

Branch 是一条新的训练时间线，不是 resume。它从父 checkpoint 的**训练态 Generator**继续，默认同时继承父 Discriminator 权重；optimizer、scheduler 和 GradScaler 重新初始化。`completed_step` 默认继承父 checkpoint，便于直接比较分支前后的 TensorBoard loss 趋势：

```bash
uv run python -m swapface.train \
  --branch-from experiments/runs/20260910-162600-blendface-vgg/checkpoints/step_000050000.pth \
  --config experiments/train.toml \
  --name new-experiment
```

`--branch-from` 必须显式提供 `--config`。新 run 的 `metadata.json.parent` 会记录父 `run_id`、checkpoint、父 step、配置摘要，以及 Discriminator 是 `inherit` 还是 `reset`。

默认继承父 checkpoint 的 step。如果希望从其他 step 开始，单独指定 `--step`：

```bash
uv run python -m swapface.train \
  --branch-from experiments/runs/<run_id>/checkpoints/step_000050000.pth \
  --config experiments/train.toml \
  --step 30000
```

`--step 0` 可以让新分支从 0 重新计数。这个值就是新 run 的实际 `completed_step` 起点，因此也会同时影响 R1 周期、EMA decay、sample/checkpoint 保存周期和文件名；它不是单独的 TensorBoard 显示偏移。

Branch 始终要求 `[generator]` 与父 checkpoint 完全一致。默认还要求 `[discriminator]` 一致，并加载父 D 权重。若需要重新初始化 D（例如修改 D 架构或主动打破旧 GAN 平衡），使用：

```bash
uv run python -m swapface.train \
  --branch-from experiments/runs/<run_id>/checkpoints/step_000050000.pth \
  --config experiments/train.toml \
  --reset-discriminator
```

设置 `--reset-discriminator` 后不会加载父 D 权重，并允许新配置修改 `[discriminator]`。无论是否恢复 D，Branch 都不会继承 G/D optimizer、scheduler 或 GradScaler；step 默认继承父 checkpoint，也可以由 `--step` 指定。

因此三种入口的语义为：

- **fresh**：新 Generator、新 Discriminator、新训练状态；
- **resume**：原 run、原冻结配置、latest checkpoint，完整恢复训练态 Generator、EMA、Discriminator、optimizer、scheduler、GradScaler 与 step；
- **branch**：新 run，继承父训练态 Generator，默认也继承 Discriminator 权重；optimizer/scheduler/GradScaler 重新初始化，step 默认继承父 checkpoint。

Branch 不检查 `training_config.semantics_version`，因为它不恢复父训练状态；resume 仍严格检查该版本。

### 可选严格校验训练精度

`[train].precision` 必须显式配置，支持 `fp32`、`fp16`、`bf16`。FP16 使用独立的 Generator/Discriminator `GradScaler`；BF16 不使用 loss scaling。R1、仿射恢复与 HRFFA 的数值敏感几何求解继续固定为 FP32。

FP16 中只有实际执行了 optimizer update 的阶段才推进对应 scheduler。D overflow 时当前 global step 不推进；D 已成功而 G overflow 时只重算并重试 G，直到 G 成功后才更新 EMA 和 `completed_step`。总 `d_loss` / `g_loss` 出现 NaN/Inf 时直接终止，避免把非有限值误当作可通过降低 loss scale 恢复的 overflow。BF16/FP32 在 backward 后、`optimizer.step()` 前额外检查参数梯度；若任一梯度出现 NaN/Inf，则报告首个异常参数并终止，避免污染模型参数和 optimizer state。FP16 的梯度 overflow 仍由 `GradScaler` 负责跳过 update 和动态调整 scale，不走该 fatal gradient guard。

默认 resume 不要求当前实际训练精度与 checkpoint 一致，因为设备和软件环境可能变化。需要严格复现时可显式使用：

```bash
uv run python -m swapface.train \
  --resume experiments/runs/<run_id> \
  --strict-precision
```

此时程序比较 checkpoint 中记录的实际 `training_config.precision` 与当前进程实际生效的精度；不一致时拒绝 resume。`--strict-precision` 是本次 resume 的运行时策略，不写入 TOML，也不能用于 fresh training。

训练 checkpoint 还记录 `training_config.semantics_version`。它只约束 resume，用于阻止训练代码语义已经改变时继续恢复旧 optimizer/scheduler/scaler 状态；Branch 不恢复这些状态，因此不检查该版本。纯推理/导出仍只依赖模型 checkpoint 格式。

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
- VGG / WFM 的协议默认权重；
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
