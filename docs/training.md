# FaceSwap 训练运行与结果目录

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
uv run python -m faceswap.train
```

显式指定配置：

```bash
uv run python -m faceswap.train --config experiments/train.toml
```

添加便于识别的短标签：

```bash
uv run python -m faceswap.train \
  --config experiments/train.toml \
  --name blendface-vgg
```

`--name` 只参与 run ID，不改变实验配置。默认结果根目录为 `experiments/runs`；如确有需要，可用 `--runs-root PATH` 覆盖。

## 训练数据源

训练数据只支持本地图片目录，不再由训练进程访问 Hugging Face、ModelScope 或其他在线数据集。`[[src]]` / `[[dst]]` 的 schema 只包含：

```toml
[[src]]
path = "/path/to/source_faces"
adjustment = 0.0

[[dst]]
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
- `config.resolved.json`：展开 Trainer、模型和 dataloader 默认值后的规范配置；这是 run 的机器可读事实来源。
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

例如 `step_000010000.pth` 表示已经完成 10,000 次训练更新。`weight_save_every = 10000` 会在 completed step 10,000、20,000、30,000... 保存，而不是在内部 `iter == 0` 时额外保存一个易误解的 `0.pth`。

checkpoint 仍通过临时文件写入后原子 `replace`；只有 checkpoint 成功落盘后才原子更新 `latest.json`。

`completed step` 只在一轮完整的 Discriminator 更新、Generator 更新、scheduler（若启用）和 EMA 更新全部完成后推进。checkpoint 内部 `step` 与文件名中的 step 必须完全一致。因此从 `step_000050000.pth` 恢复后，在下一轮更新真正完成前，completed step 仍是 50,000，不会把正在执行的 partial iteration 标记成 50,001。

## 中途恢复训练

推荐直接指定 run：

```bash
uv run python -m faceswap.train \
  --resume experiments/runs/20260910-162600_blendface-vgg
```

程序会从 `latest.json` 找到 checkpoint，读取该 run 冻结的 resolved config，然后直接恢复 Generator、Discriminator、EMA、optimizer、scheduler 与 step，继续写入原 run。模型/优化器/scheduler 的 state dict 是否可加载由 PyTorch 自己校验。

也可以显式指定同一 run 的 **latest checkpoint**：

```bash
uv run python -m faceswap.train \
  --resume experiments/runs/20260910-162600_blendface-vgg/checkpoints/step_000070000.pth
```

显式文件必须位于标准 `run/checkpoints/` 目录中，并且必须与 `latest.json` 指向同一文件。`--resume` 不允许从历史 checkpoint 回滚后继续写入原 run，否则会产生分叉历史并覆盖同 step 的 checkpoint/sample。历史 checkpoint 应通过 branch 创建新的 run。

## Branch：从已有 checkpoint 派生新实验

Branch 从已有完整训练状态创建一个新的 run。它不会修改父 run，可以从标准 run 的 latest、其中任意历史 checkpoint，或单独保存的 v3 checkpoint 文件分叉：

```bash
uv run python -m faceswap.train \
  --branch-from experiments/runs/20260910-162600-blendface-vgg/checkpoints/step_000050000.pth \
  --config experiments/train.toml \
  --name lower-id-loss
```

独立 checkpoint 不需要保留原 run 目录；只要文件名保持 `step_<step>.pth`，其内部 v3 元数据、step 与 Generator/Discriminator 架构有效即可：

```bash
uv run --no-sync python -m faceswap.train \
  --branch-from /mnt/workspace/step_000208939.pth \
  --config /mnt/workspace/train.toml \
  --name lower-id-loss
```

`--branch-from` 必须显式提供 `--config`。Branch 以 checkpoint 自身保存的 v3 元数据和 Generator/Discriminator `network_cfg` 为准，在创建新 run 前与新配置比较模型定义；通过后立即创建新 run，并在 `metadata.json.parent` 中记录父 `run_id`、checkpoint、step 和配置摘要。checkpoint 权重、optimizer 和 scheduler 是否真的可恢复，直接交给 PyTorch 的 `load_state_dict()`；失败时使用原始错误并将新 run 标记为 `failed`。

Branch 允许修改训练配置，例如：

- loss 类型、开关和权重；
- Generator Identity provider 与 Identity Loss provider；
- 学习率与 scheduler；
- batch size、R1、BF16/compile；
- 数据源、数据增强与数据管线参数（NVIDIA 使用 DALI，ROCm 使用原生 PyTorch 管线）；
- 日志、sample 和 checkpoint 间隔。

但 Branch 的目的仍是**继续训练同一个模型定义**，因此以下配置必须与父 run 完全一致，否则在创建新 run 前直接拒绝：

- 整个 `[generator]`；
- 整个 `[discriminator]`。

若要修改这些模型定义，应创建 fresh run，而不是 branch。Branch 不做 partial load。

Branch 继承 Generator、Discriminator、EMA 和 Adam moments/step。optimizer 的学习率等运行超参数以新配置为准；当 `lr` 与 `lr_scheduler_t_max` 都与父 checkpoint 一致时继承 scheduler 进度，否则 scheduler 按新配置重新初始化。

因此三种入口具有明确语义：

- **fresh**：新模型、新配置、新 run；
- **resume**：原 run、原冻结配置、latest checkpoint，严格线性继续；
- **branch**：新 run、父 checkpoint 的已训练模型状态、新训练配置，但模型定义不可改变。

### 可选严格校验 BF16

BF16 是否实际生效不仅由 TOML 的 `bf16 = true` 决定，还取决于当前 GPU 与 `torch.compile` 能力。ROCm 使用 PyTorch 报告的 BF16 能力；NVIDIA 在启用 `torch.compile` 时继续要求 SM80+。默认 resume **不要求**当前实际 BF16/FP32 模式与 checkpoint 一致，因为设备和软件环境不属于严格训练配置约束。

如果某次恢复训练需要把数值精度模式也视为严格条件，可显式使用：

```bash
uv run python -m faceswap.train \
  --resume experiments/runs/20260910-162600_blendface-vgg \
  --strict-bf16
```

此时程序比较 checkpoint 中记录的实际 `training_config.bf16` 与当前进程实际生效的 BF16 状态；不一致时拒绝 resume。`--strict-bf16` 是本次 resume 的运行时策略，不写入 TOML，也不能用于 fresh training。

## 配置冻结与哈希

run 同时保存用户原始 TOML 与展开默认值后的规范 JSON：

```text
config.toml           # 用户输入原文
config.resolved.json  # 完整、显式、可哈希的规范配置
```

`config.resolved.json` 会显式记录：

- `[train]` 的默认参数；
- Generator / Discriminator 构造器默认值；
- dataloader 默认值；
- Identity encoder 默认值；
- VGG / WFM 默认权重；
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
