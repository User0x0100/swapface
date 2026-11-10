# 身份条件拆分导出（InSwapper / StyleGAN2）

`split` 仅在 ONNX 导出阶段将身份映射计算与逐帧生成计算分离，不影响训练模型、EMA 检查点或常规 `swapface` / `id-encoder` / `all` 导出模式。

```bash
uv run python -m swapface.export split \
  --checkpoint /path/to/step_XXXXXX.pth \
  --device cuda \
  --output-dir onnx_export
```

所有本次导出的产物放在 `--output-dir` 下的一个独立文件夹内：

```text
onnx_export/
└── swapface-split-STEP-PAIR-TIME/
    ├── swapface-STEP-client.onnx  # 不含身份映射权重的生成器
    ├── swapface-STEP-server.onnx  # 身份编码器 + 身份映射权重
    └── manifest.json              # 双端配对信息与文件哈希
```

不会额外创建 `-private` 目录，也不要求指定服务端输出路径。**这个目录是完整的构建产物，并不是客户端发布包。客户端部署时只应选择客户端 ONNX，不能将包含服务端 ONNX 的整个目录分发给终端。**

导出使用单个临时目录：先完成两份 ONNX 导出、结构校验、TensorRT 不支持的序列算子检查，并写入 manifest，最后通过一次目录重命名发布。失败时不留下半成品发行目录。

`split` 默认 ONNX opset 18、`optimize=False`；使用固定偏移的 `Slice` 避免产生 TensorRT 不支持的 `SplitToSequence` / `SequenceAt`。

## 输入输出协议（`swapface.split.version = 2`）

- 服务端输入 `faces`：ArcFace 112 对齐的 RGB 图像、float32、值域 `[-1,1]`；只输出 `identity_condition`，形状 `[B,D]`。原始 512D embedding 不对外输出。
- 客户端输入 `faces`（根据检查点的 Generator 对齐方式和分辨率）和 `identity_condition` `[B,D]`；输出 `swapped_faces`。客户端 ONNX 不含身份编码器或身份映射权重。
- 默认图像布局 NHWC；`--nchw` 同时改变两侧的图像布局。身份条件始终 `[B,D]`。
- `B` 为导出时固定 batch（默认 1），`D` 为各身份注入层拼接后的宽度，由架构及检查点确定。
- InSwapper 拼接顺序：各 `latent_space.layers` 依次 `adain0`、`adain1`，每个先 gamma 后 beta。
- StyleGAN2 拼接顺序：`conv4`、`to_rgb4`，随后各 Stage 依次 `conv0`、`conv1`、`to_rgb`。
- 相同身份的 `identity_condition` 可以缓存并逐帧复用；切换身份或模型时应重新获取。

## 配对及完整性

唯一的 `manifest.json` 包含 `generator_id`（有效 Generator 权重标识）、`encoder_id`（有效编码器权重标识）、`pair_id`（模型与协议参数指纹）、`batch_size`、`layout`、`coarse_type`、`condition_width`，以及 `artifacts.client` 和 `artifacts.server` 的文件名称、SHA-256、输入输出名称、对齐方式。

两份 ONNX 各自的 metadata 均包含相同的配对信息，并通过 `swapface.split.role=client/server` 区分角色。所有 SHA-256 均仅用于一致性与缓存识别，不能替代签名、认证或授权。

## 安全边界

客户端仅接收由服务端计算的身份调制参数。服务端的 IDEncoder 及全部身份映射权重只在 `server.onnx` 中，因此最终分发客户端模型时不应包含它。已经获得的身份条件可以被客户端缓存并复用，这是设计允许的行为。身份登记、映射查询权限和 WSS 参数下发依然属于实际客户端/服务端实现，未由此导出工具强制控制。

InSwapper 依然使用仿射身份映射；如果攻击者掌握大量已知 `(embedding, condition)` 样本，仍可能恢复线性变换。本次目录简化与此风险无关。

## 回归测试

```bash
uv run python -m swapface.test_split_identity
uv run python -m swapface.test_coarse_architecture
uv run python -m swapface.test_inference
# 可选：安装 CUDA + TensorRT 后验证小模型实际 Engine 构建
SWAPFACE_TEST_TRT_ENGINE=1 uv run python -m swapface.test_split_identity
```

已在真实 `step_000632974_hq.pth` + R100 Encoder 上验证 InSwapper 双端 ONNX 与 PyTorch 数值等价，客户端 TensorRT 解析通过；两种小尺寸架构 TensorRT Engine 构建通过。生产尺寸 FP16 Engine 运行和 Rust/WSS 集成尚未完成。
