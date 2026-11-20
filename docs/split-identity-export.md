# 身份编码 / 推理拆分导出

单阶段 InSwapper 和 StyleGAN2 使用同一接口：服务端输入已对齐 Source 图像，输出身份调制条件；客户端输入 Target 图像与该条件，直接输出最终分辨率 RGB 图像。

- InSwapper 的服务端包含身份编码器、共享 W 的 WSpaceMap 和 Bottleneck 每个 AdaIN 的独立线性层。客户端包含 FromRGB、完整 U-Net（包括启用的 cat/add Skip）、预计算 AdaIN、ToRGB，不保留身份映射权重。
- StyleGAN2 的服务端包含身份编码器、ID Mapping、Style Mapping 与各调制线性层；客户端包含原 StyleSwap Target 编码、Synthesis、可选 Mask 分支。
- 两者的完整 Generator 与客户端+服务端组合在数学上等价，更新 U-Net 跳层必须同步更新 `PrecomputedInSwapper`。
- 仅发布 `client.onnx`；`server.onnx` 为服务端私有。导出结果的 manifest 使用 `generator_type` 字段及 `SPLIT_PROTOCOL_VERSION=3`。该协议不兼容旧版使用 `coarse_type` 的消费端，需要同步调整客户端协议解析。
- `--nhwc/--nchw` 以及固定 batch 的导出约定不变。使用 ONNX 标准 Slice 避免 TRT 不支持的序列算子。

不再导出、中继或实例化 HQRefiner；客户端 ONNX 只需一次生成器推理即可输出最终结果。

## 导出命令与产物

```bash
uv run --no-sync python -m swapface.export split \
  --checkpoint /path/to/step_XXXXXX.pth \
  --device cuda \
  --output-dir onnx_export
```

构建目录示例：

```text
onnx_export/
└── swapface-split-STEP-PAIR-TIME/
    ├── swapface-STEP-client.onnx
    ├── swapface-STEP-server.onnx
    └── manifest.json
```

先在目标文件系统的临时目录中生成两份 ONNX、验证图结构、计算 SHA-256 并写入 manifest；全部成功后再原子重命名为最终目录。仅分发客户端 ONNX，不要直接公开构建目录。

## 输入输出协议（`swapface.split.version = 3`）

- 服务端输入：`faces`，ArcFace112 RGB、float32、`[-1,1]`；输出：`identity_condition`，`[B,D]`，不含原始 512 维 embedding。
- 客户端输入：`faces`（目标分辨率与 checkpoint 的对齐方式一致）、`identity_condition`；输出：`swapped_faces`，最终单阶段 RGB 图像。
- 默认 NHWC，也可指定 `--nchw`。`B` 为导出时固定 batch，`D` 取决于生成器架构；身份条件始终 `[B,D]`，与图像布局无关。
- InSwapper 条件拼接：依次遍历 Bottleneck `latent_space.layers`，每层 `adain0` 后接 `adain1`，每个 AdaIN 先 gamma 后 beta。
- StyleGAN2 条件拼接：`conv_first`、`to_rgb_first`，然后每个 Stage 的 `conv0`、`conv1`、`to_rgb`，与 Target 注入方式无关。
- 对同一身份和生成器，`identity_condition` 可缓存并复用；切换身份或模型需重新生成。

## 身份一致性与安全边界

manifest 包含 `generator_id`、`encoder_id`、`pair_id`、`generator_type`、`condition_width`、`batch_size`、`layout` 及每个制品的 SHA-256、输入输出名称与对齐方式。ONNX 的 metadata 带有相同的配对字段和 `swapface.split.role=client/server`。哈希仅保证缓存与配对一致性，不替代签名、身份授权和服务端鉴权。

客户端不含身份编码器或身份映射权重，但已经下发的条件可被客户端缓存。InSwapper 的仿射身份映射还存在已知输入输出样本过多时被逆推的潜在风险；目录与协议版本变更不消除该风险。

## 回归测试

```bash
uv run --no-sync python -m swapface.test_split_identity
uv run --no-sync python -m swapface.test_inswapper_architecture
uv run --no-sync python -m swapface.test_inference
uv run --no-sync pytest -q tests/test_stylegan2_mask.py
# 可选：在安装 TensorRT 的环境中额外构建小模型 Engine
SWAPFACE_TEST_TRT_ENGINE=1 uv run --no-sync python -m swapface.test_split_identity
```

以上测试覆盖模型条件映射数值一致性、ONNX 导出、ONNX Runtime 推理与 Mask 分支。正式部署的 TensorRT Engine 性能、精度和 Rust/WSS 集成仍需针对生产尺寸验证。
