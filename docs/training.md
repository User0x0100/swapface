# FaceSwap 训练运行与结果目录

本文描述训练配置、run 生命周期、checkpoint 保存与中途恢复的约定。目标是让一次训练具备明确边界、可恢复、可审计，并避免运行时路径污染实验配置。

## 设计原则

训练系统区分两类状态：

- **实验配置**：决定“训练什么”，例如模型结构、数据源、损失、学习率与数据增强。由 TOML 描述，可进入 Git。
- **运行状态**：决定“这一次从哪里执行、结果写到哪里”，例如 run ID、checkpoint、输出目录。由 CLI 与 run 元数据管理，不写入 TOML。

因此 `ckpt`、`log_path` 不再是 `[train]` 配置项。真正的 resume 必须继续原 run，并使用该 run 在创建时冻结的配置。

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
- `metadata.json`：run ID、状态、Git commit/dirty、Python/平台信息、PyTorch/CUDA 信息和 resume 历史。
- `latest.json`：很小的最新 checkpoint 指针，不复制大模型文件，适合本地文件系统和对象存储。
- `checkpoints/`：完整训练状态。
- `samples/`：与 completed step 对齐的训练可视化。
- `tensorboard/`：TensorBoard event 文件；一次 resume 可能新增 event 文件，这是正常现象。

`experiments/*.toml` 可以进入 Git；`experiments/*/` 属于运行生成物，默认忽略。

## Step 与文件命名

内部 checkpoint v2 继续保留 0-based `iter` / `next_iter` 字段，以避免仅因目录重构破坏推理兼容。

对用户可见的文件名统一使用 **completed step**：

```text
step_000000001.pth
step_000010000.pth
step_000010000.png
```

例如 `step_000010000.pth` 表示已经完成 10,000 次训练更新。`weight_save_every = 10000` 会在 completed step 10,000、20,000、30,000... 保存，而不是在内部 `iter == 0` 时额外保存一个易误解的 `0.pth`。

checkpoint 仍通过临时文件写入后原子 `replace`；只有 checkpoint 成功落盘后才原子更新 `latest.json`。

## 中途恢复训练

推荐直接指定 run：

```bash
uv run python -m faceswap.train \
  --resume experiments/runs/20260910-162600_blendface-vgg
```

程序会：

1. 校验该目录是标准 run；
2. 从 `latest.json` 找到最新 checkpoint；
3. 校验 `config.resolved.json` 与 `metadata.json` 中的 SHA-256；
4. 使用 run 冻结的 resolved config；
5. 恢复 Generator、Discriminator、EMA、optimizer、scheduler 与迭代状态；
6. 继续写入原 run 的 `checkpoints/`、`samples/` 和 `tensorboard/`。

也可以显式指定同一 run 的 **latest checkpoint**：

```bash
uv run python -m faceswap.train \
  --resume experiments/runs/20260910-162600_blendface-vgg/checkpoints/step_000070000.pth
```

显式文件必须位于标准 `run/checkpoints/` 目录中，并且必须与 `latest.json` 指向同一文件。`--resume` 不允许从历史 checkpoint 回滚后继续写入原 run，否则会产生分叉历史并覆盖同 step 的 checkpoint/sample。若确实要从历史 checkpoint 开新分支，应使用未来独立的 `--init-from` 语义并创建新 run。

## Resume 与修改配置

`--resume` 不能与 `--config`、`--name` 或 `--runs-root` 同时使用。这是刻意设计的约束：

- **resume**：同一个实验、同一个 run、同一个冻结配置继续执行。
- **修改 loss / 数据集 / 模型后继续训练**：语义上已经是新的实验分支，不应伪装成 resume。

如果以后需要“从旧模型权重初始化一个新实验”，应单独实现 `--init-from` 一类接口，并创建新 run、记录 lineage；不要复用 `--resume`。

### 可选严格校验 BF16

BF16 是否实际生效不仅由 TOML 的 `bf16 = true` 决定，还取决于当前 GPU 与 `torch.compile` 能力。默认 resume **不要求**当前实际 BF16/FP32 模式与 checkpoint 一致，因为设备和软件环境不属于严格训练配置约束。

如果某次恢复训练需要把数值精度模式也视为严格条件，可显式使用：

```bash
uv run python -m faceswap.train \
  --resume experiments/runs/20260910-162600_blendface-vgg \
  --strict-bf16
```

此时程序比较 checkpoint 中记录的实际 `training_config.bf16` 与当前进程实际生效的 BF16 状态；不一致时拒绝 resume。`--strict-bf16` 是本次 resume 的运行时策略，不写入 TOML，也不能用于 fresh training。

## 配置冻结与哈希

TOML 适合人写，但 TOML 没有 `null`，而 dataloader 中存在 `huggingface_proxy = None` 之类有效默认状态。因此 run 同时保存：

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
- 远程数据源默认 revision/path_prefix/adjustment。

其规范 JSON 内容计算 SHA-256 并写入 `metadata.json`。新 checkpoint 也记录同一摘要与 run ID；resume 时若摘要或 run ID 不一致，会拒绝继续训练。

## Run 状态

`metadata.json` 的 `status` 会在生命周期中更新：

```text
created -> running -> interrupted
                   -> failed
                   -> completed
```

Ctrl+C 属于 `interrupted`，后续可直接 `--resume`。初始化或训练异常会记录错误类型与消息并标记为 `failed`，避免留下无法判断来源的空目录。训练进程因 Ctrl+C 退出时返回非零状态码 130，便于 shell/调度系统正确识别中断。

同一个 run 还会持有 `.run.lock` 的 Linux advisory lock。若另一个训练进程同时尝试 resume 同一 run，会立即拒绝启动；锁由内核绑定到进程/文件描述符，进程崩溃后会自动释放，因此不会出现“残留 lockfile 导致永久锁死”的问题。

## TensorBoard

查看全部 run：

```bash
uv run tensorboard --logdir experiments/runs
```

恢复同一个 run 后，TensorBoard 目录可能出现多个 `events.out.tfevents.*`。它们仍属于同一逻辑 run；恢复时会设置 `purge_step`，让 TensorBoard 隐藏上次进程在最后 checkpoint 之后写出的失效 future steps，避免回滚恢复后出现重复曲线。

## 旧目录

旧实现使用：

```text
experiments/my_experiment/
├── ckpt/
├── sample/
└── tensorboard/
```

这些目录没有 `config.resolved.json`、`metadata.json` 和 `latest.json`，因此不会被新的 `--resume` 当成标准 run。保留旧文件用于人工迁移或推理即可；新的训练结果不要继续写入旧目录。
