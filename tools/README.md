# 数据集下载与后处理工具

本目录包含当前训练数据准备使用的两个脚本：

- `download_datasets.py`：从 Hugging Face 下载并固定到指定 revision，本地以符号链接组织到 `datasets/downloads/`。
- `postprocess_dataset.py`：将 ZIP 图片数据转换为统一 PNG，或将 FairFace 转换为按粗粒度人种分类的 ArcFace 112 source identity 数据。

建议从项目根目录执行本文中的命令。

## 环境

先安装当前硬件对应的项目依赖。例如 NVIDIA：

```bash
uv sync --group cuda
```

AMD ROCm：

```bash
uv sync --group rocm
```

FairFace 后处理依赖 `pyarrow`，已经包含在项目基础依赖中。

## 1. 下载数据集

### 查看支持的数据集

```bash
uv run --no-sync python tools/download_datasets.py --list
```

当前支持：

- `lpff`
- `ffhq`
- `fairface`
- `all`

不指定数据集时等价于 `all`。

### 下载 FFHQ

```bash
uv run --no-sync python tools/download_datasets.py ffhq
```

默认输出：

```text
datasets/downloads/ffhq/
├── FFHQ-1024-1.zip
└── FFHQ-1024-2.zip
```

### 下载 LPFF

```bash
uv run --no-sync python tools/download_datasets.py lpff
```

默认输出：

```text
datasets/downloads/lpff/
└── stylegan.zip
```

### 下载 FairFace

```bash
uv run --no-sync python tools/download_datasets.py fairface
```

项目使用 `HuggingFaceM4/FairFace` 的 `0.25` 配置。官方 train 与 validation 都会下载，并统一作为 source identity pool 使用，本地不再保留训练集/验证集语义：

```text
datasets/downloads/fairface/
├── part-00000.parquet   # 原 train shard 0
├── part-00001.parquet   # 原 train shard 1
└── part-00002.parquet   # 原 validation shard
```

总计 97,698 张图片。

### 指定下载目录

```bash
uv run --no-sync python tools/download_datasets.py fairface \
  --root /workspace/datasets/downloads
```

`--root` 是本地符号链接目录。实际文件仍由 Hugging Face Hub cache 管理，因此不会额外复制一份大文件。

### 指定 Hugging Face 源

官方源：

```bash
uv run --no-sync python tools/download_datasets.py fairface \
  --source direct
```

镜像源：

```bash
uv run --no-sync python tools/download_datasets.py fairface \
  --source mirror
```

不传 `--source` 时，脚本遵循当前 `HF_ENDPOINT`，否则使用 Hugging Face 官方默认值。

### 强制重新下载

```bash
uv run --no-sync python tools/download_datasets.py fairface --force
```

`--force` 会将 `force_download=True` 传给 Hugging Face Hub。

### 只查看将要下载的文件

```bash
uv run --no-sync python tools/download_datasets.py fairface --list
```

会打印 artifact 名、仓库、固定 revision、远端文件和本地映射路径，不执行下载。

## 2. ZIP 图片数据后处理

默认 `--format` 为 `zip`。

该模式会：

1. 扫描输入 ZIP 内支持的图片。
2. 要求输入图片为正方形。
3. 转换为 RGB。
4. 按指定尺寸进行 Lanczos resize。
5. 以 PNG 输出到一个扁平目录。
6. 使用原子写入和 manifest 支持断点续处理。

默认输出尺寸为 `256x256`。

### FFHQ 示例

```bash
uv run --no-sync python tools/postprocess_dataset.py \
  datasets/downloads/ffhq/FFHQ-1024-1.zip \
  datasets/downloads/ffhq/FFHQ-1024-2.zip \
  --output datasets/ffhq-256
```

指定输出分辨率：

```bash
uv run --no-sync python tools/postprocess_dataset.py \
  datasets/downloads/ffhq/FFHQ-1024-1.zip \
  datasets/downloads/ffhq/FFHQ-1024-2.zip \
  --output datasets/ffhq-512 \
  --size 512
```

指定 CPU worker 数：

```bash
uv run --no-sync python tools/postprocess_dataset.py \
  datasets/downloads/ffhq/FFHQ-1024-1.zip \
  datasets/downloads/ffhq/FFHQ-1024-2.zip \
  --output datasets/ffhq-256 \
  --workers 16
```

### LPFF 示例

```bash
uv run --no-sync python tools/postprocess_dataset.py \
  datasets/downloads/lpff/stylegan.zip \
  --output datasets/lpff-256
```

注意：ZIP 模式会将目录结构压平成文件名。如果不同 ZIP 或不同子目录中存在同名图片，脚本会直接报错，而不是静默覆盖。

## 3. FairFace -> ArcFace 112

FairFace 专门作为 `src` 身份池使用。

完整流程：

```bash
uv run --no-sync python tools/download_datasets.py fairface

uv run --no-sync python tools/postprocess_dataset.py \
  datasets/downloads/fairface/*.parquet \
  --format fairface \
  --output datasets/fairface-arcface112
```

FairFace 模式不会简单把原图 resize 到 112，而是：

1. 从 Parquet 流式读取图片和 `race` 标签。
2. 使用项目内 RetinaFace 检测人脸及 5 点 landmarks。
3. 使用项目统一的 ArcFace canonical template 做仿射对齐。
4. 输出真正的 `112x112 RGB` ArcFace 身份图。
5. 按粗粒度人种目录分类。

默认使用 RetinaFace ResNet-50。

### 输出目录

```text
datasets/fairface-arcface112/
├── asian/
├── indian/
├── black/
├── white/
├── middle_eastern/
├── latino/
├── .postprocess.json
└── .failed.txt
```

当前映射：

| FairFace 原始标签 | 输出目录 |
| --- | --- |
| East Asian | `asian/` |
| Southeast Asian | `asian/` |
| Indian | `indian/` |
| Black | `black/` |
| White | `white/` |
| Middle Eastern | `middle_eastern/` |
| Latino_Hispanic | `latino/` |

`train` 和 `validation` 在这里没有不同用途，三个 Parquet 会一起处理并进入相同的人种目录。

输出文件名使用本地 shard 名保证唯一，例如：

```text
part-00000_000000.png
part-00001_043371.png
part-00002_010953.png
```

### 调整 GPU batch size

默认：

```text
--batch-size 128
```

显存不足时降低：

```bash
uv run --no-sync python tools/postprocess_dataset.py \
  datasets/downloads/fairface/*.parquet \
  --format fairface \
  --output datasets/fairface-arcface112 \
  --batch-size 32
```

显存充足时可以提高，例如：

```bash
... --batch-size 256
```

是否更快取决于 GPU、RetinaFace 和 CPU 图片解码吞吐。

### 指定设备

默认：

```text
--device cuda
```

指定具体 GPU：

```bash
... --device cuda:0
```

PyTorch ROCm 环境同样使用 `cuda` / `cuda:0` 设备字符串。

### 调整检测置信度

默认：

```text
--confidence 0.9
```

例如：

```bash
... --confidence 0.85
```

不建议无理由降低阈值，因为 FairFace 最终用于身份编码，错误人脸或错误 landmarks 的影响通常比少量丢样本更大。

### 使用轻量 RetinaFace

默认使用 ResNet-50 RetinaFace。需要降低检测计算量时可以：

```bash
... --mobilenet
```

此时改用 MobileNet0.25 RetinaFace。用于正式构建身份数据集时，默认仍建议使用 ResNet-50。

### CPU 解码 / PNG 写入线程

```bash
... --workers 16
```

`--workers` 在 FairFace 模式下用于图片解码和输出 PNG 的线程池，不控制 RetinaFace GPU batch size。

## 4. 断点续处理

两种后处理模式都支持断点续处理。

再次执行完全相同的命令即可：

```bash
uv run --no-sync python tools/postprocess_dataset.py \
  datasets/downloads/fairface/*.parquet \
  --format fairface \
  --output datasets/fairface-arcface112
```

已经存在的完整 PNG 会被跳过。

输出目录包含：

```text
.postprocess.json
```

它记录输入文件、文件大小、mtime、处理版本和 FairFace 关键参数。若输入或参数与已有结果不一致，脚本会拒绝继续向同一个目录写入，避免把两套不同配置的数据混在一起。

如果要使用不同参数重新处理，建议直接指定新的输出目录，例如：

```bash
--output datasets/fairface-arcface112-conf085
```

### FairFace 无法检测到人脸

FairFace 模式中 RetinaFace 没有检测到人脸的样本会记录到：

```text
.failed.txt
```

再次运行时这些 key 会被跳过，避免永久重复尝试同一个失败样本。

## 5. 在训练配置中使用 FairFace

FairFace 输出已经是 ArcFace 112，因此必须作为 `data.src` 使用，并声明：

```toml
alignment = "arcface"
```

由于训练数据扫描当前不递归，每个人种目录分别配置一个 source：

```toml
[[data.src]]
path = "datasets/fairface-arcface112/asian"
adjustment = 0.0
alignment = "arcface"

[[data.src]]
path = "datasets/fairface-arcface112/indian"
adjustment = 0.0
alignment = "arcface"

[[data.src]]
path = "datasets/fairface-arcface112/black"
adjustment = 0.0
alignment = "arcface"

[[data.src]]
path = "datasets/fairface-arcface112/white"
adjustment = 0.0
alignment = "arcface"

[[data.src]]
path = "datasets/fairface-arcface112/middle_eastern"
adjustment = 0.0
alignment = "arcface"

[[data.src]]
path = "datasets/fairface-arcface112/latino"
adjustment = 0.0
alignment = "arcface"
```

如果当前目的主要是补充亚洲 ID，可以只添加：

```toml
[[data.src]]
path = "datasets/fairface-arcface112/asian"
adjustment = 0.0
alignment = "arcface"
```

`data.dst` 不接受 `alignment = "arcface"`，仍应使用 FFHQ 对齐图像。

## 6. 参数速查

### `download_datasets.py`

```text
dataset                 lpff / ffhq / fairface / all
--root PATH             本地数据入口目录，默认 datasets/downloads
--source direct         Hugging Face 官方源
--source mirror         hf-mirror.com
--force                 强制重新下载
--list                  只列出 artifact，不下载
```

### `postprocess_dataset.py`

通用：

```text
archives ...            一个或多个输入 ZIP / Parquet
--output PATH           输出目录，必填
--format zip|fairface   默认 zip
--workers N             CPU 并行数
```

ZIP 模式：

```text
--size N                输出边长，默认 256
```

FairFace 模式：

```text
--size 112              固定为 112，通常无需传
--batch-size N          RetinaFace GPU batch，默认 128
--device DEVICE         默认 cuda
--confidence FLOAT      RetinaFace 阈值，默认 0.9
--mobilenet             使用 MobileNet0.25 RetinaFace
```

## 7. 常见错误

### `output directory belongs to a different postprocess run`

已有输出目录的 `.postprocess.json` 与本次输入或参数不一致。

不要直接混用。删除旧输出，或者换一个新的 `--output`。

### `output directory is not empty but has no .postprocess.json`

脚本拒绝向未知来源的非空目录写数据，以避免覆盖已有文件。

请选择空目录或新的输出路径。

### `flat-name collision`

ZIP 模式检测到两个输入图片压平后拥有相同文件名。需要先处理源数据命名冲突。

### `CUDA/ROCm device requested but torch.cuda.is_available() is false`

当前 PyTorch 环境没有可用 CUDA/ROCm GPU。检查当前 `uv` dependency group、驱动和运行环境。

### FairFace 处理后图片少于 97,698 张

优先检查：

```text
datasets/fairface-arcface112/.failed.txt
```

其中记录 RetinaFace 没检测到人脸的样本。实际最终数量等于 97,698 减去失败样本数。
