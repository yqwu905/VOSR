# Token 压缩实验：执行记录

日期：2026-10-08。VOSR 基线 `7f8576fa56a73e4ea711b1656ebd122a1fdea068`。
源机制与运行方法见 [grace_token_compression.md](grace_token_compression.md)。

## 环境

CPU，Python 3.12.14；没有可用 CUDA/NPU，没有 VOSR2/Qwen VAE/DINO 预训练权重，没有人工转写的真实文字评测集。

验证依赖：`torch==2.5.1+cpu`、`torchvision==0.20.1+cpu`、`timm==1.0.11`、`accelerate==1.1.0`、`safetensors==0.4.4`、`fairscale==0.4.13`、`numpy==1.26.4`、`einops==0.8.0`、`wandb==0.25.0`、`tensorboard==2.21.0`、`pytest==9.1.1`。
核心版本与仓库 requirements 对齐，PyTorch/torchvision 使用 CPU 构建；没有声称验证完整 CUDA requirements 环境。

## 代码与合成运行

最终命令（在已安装上述依赖的环境运行）：

```bash
TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 VOSR_TEST_DDP=0 WANDB_MODE=offline python -m pytest -q \
  tests/test_ablation_core.py tests/test_ablation_model.py tests/test_ablation_logging.py \
  tests/test_token_compression.py tests/test_token_compression_eval.py \
  tests/test_compression_training.py tests/test_benchmark_protocol.py
```

结果：**76 passed, 3 skipped, 6 warnings，57.68 秒**。6 条 warning 来自 W&B 的弃用提示。
3 项 skipped 是既有 2 项 DDP 加本次 1 项 DDP；最初显式启用 `VOSR_TEST_DDP=1` 实际尝试后，均在 Gloo socket 初始化阶段因 `Operation not permitted` 失败，尚未进入分布式训练。没有修改/绕过网络权限。
早期较新 W&B 0.30 的内部测试 API 不兼容；切回仓库锁定的 0.25 后既有日志测试通过。

| 检查 | 结果 | 适用边界 |
|---|---|---|
| 原始 wrapper/SDT 回归 | 通过 | 包含实际 tiny LightningDiT 和既有路由/日志测试 |
| 权重兼容 | 参数 key/shape 完全相同，严格加载通过 | 使用随机初始化的真实 LightningDiT state_dict；未加载真实 VOSR2 权重 |
| 关闭压缩 | enabled=false、factor=1、disable_compression、force_dense 与原始模型输出逐值一致 | 同一未微调 tiny 权重；不表示微调后权重可自动回到原权重 |
| 实际 Attention 长度 | SA 与 CA query 的 hook 均观测到完整/压缩区间切换 | 默认形状与动态网格；不是仅给 mask 或只少算 MLP |
| RoPE/细节旁路 | 原网格坐标验证、恒等更新/梯度测试通过 | 合并区域细节交互损失仍需实图测试 |
| 训练与导出 | fp32/bf16、checkpoint 重算、cosine/细节 KD、optimizer step、strict export reload 通过 | CPU tiny backbone |
| 真实训练器 | 单进程 3 步、梯度累积、稠密 warmup、compressed+dense KD、保存、从第2步恢复通过 | 真实 trainer/DiT，VAE/DINO/退化/数据为合成夹具 |
| 分块推理 | 非正方形图像、跨 tile 融合、原尺寸裁剪、关闭压缩通过 | 真实 tiny DiT + synthetic VAE；不是完整预训练 VAE |
| 评估记录/汇总 | 三变体运行、实际 Attention hook、CPU显存=null、配置/输入一致性、OCR差值、错误识别器拒绝通过 | 合成数据和合成预测；测试得到的数值不作为质量/速度结果 |
| OCR CLI | 合成“中文小字→中文错字”产生 CER=0.25，四个分组一致 | 验证评分算法和入口；没有运行真实 OCR 引擎 |
| 静态检查 | 9 个运行文件 AST 解析、git diff --check、CLI --help 通过 | 代码检查 |

## 解析预测与尚未实测

| 项目 | 原始 | SDT | 新压缩 | 状态 |
|---|---:|---:|---:|---|
| 512 tile 的每层 SA Token | 1024 | 1024 | 8层1024 + 28层256 | 配置推导；tiny hook 已验证同机制 |
| SA QK/AV FLOPs 比例 | 1 | 1 | 0.270833 | 解析公式，非 profiler 计数 |
| SA 投影 FLOPs 比例 | 1 | 1 | 0.416667 | 解析公式 |
| 端到端推理延迟 | 未实测 | 未实测 | 未实测 | 缺权重、设备、真实输入 |
| GPU/NPU峰值显存 | 未实测 | 未实测 | 未实测 | 无对应设备 |
| 小字/中文/密集文字 CER、准确率 | 未实测 | 未实测 | 未实测 | 无真实文字集和固定识别器 |
| 多卡 DDP / NCCL / HCCL | 未验证 | 未验证 | 未验证 | Gloo 在环境初始化阶段被阻止；无 CUDA/NPU |

代码可用于启动对照实验，不能据此断言压缩后保住文字能力、获得某个速度倍数或在 Ascend 上可直接部署。
