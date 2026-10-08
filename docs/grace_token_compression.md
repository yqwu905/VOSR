# GRACE 启发的 VOSR2 Token 压缩实验

核查日期：2026-10-08。基线为 `yqwu905/VOSR@7f8576fa56a73e4ea711b1656ebd122a1fdea068`。
这是**可关闭的中间层 Token 瓶颈实验**，不是 GRACE 复现，也没有“保住文字能力”的实测结论。

## 1. GRACE 的真实机制与代码边界

原论文：[GRACE: Generation-Aware Latent Compression for Efficient Video Generation](https://arxiv.org/html/2610.10524v1)，2026-10-07 v1，Jiyoung Kim 等；[作者项目页](https://cvlab-kaist.github.io/GRACE/)。
官方代码核查固定在 [cvlab-kaist/GRACE@914c7a8](https://github.com/cvlab-kaist/GRACE/tree/914c7a8121d98417a0e350b2dba427cf9225155c)。

| 环节 | 核查结论 | 一手证据 |
|---|---|---|
| 压缩位置 | 改造视频 VAE；空间、时间再各降采样 2 倍。不是在 DiT 内跳过 MLP，也不是简单删 Token | 论文 §4、附录 A；`src/grace_geoprior.py` 的 `Encoder3d` / `WanVAE_._encode_prior` |
| 双潜变量 | 冻结原编码器处理低分辨率输入产生 base；新增 residual 分支从全分辨率提取细节，在**通道轴**拼接 | `src/grace_video_vae.py:GRACEGeopriorVideoVAE.single_encode`；代码 `z_prior` 对应 base，`z_main` 对应 residual，避免按名字反解 |
| 生成感知训练 | Stage 1 在冻结 DiT 中对齐教师/压缩潜变量的中间特征，同时优化重建；Stage 2 适配 DiT | 论文 §4.2–4.3；完整 Stage 1/2 训练脚本仍列在 README 的未完成项，不能声称已审计到官方训练损失实现 |
| 时序与位置 | 双时间调度让 base 先去噪；RoPE 坐标随压缩比例缩放 | `src/dual_sched_core.py:async_ladders/dual_step`，`src/davae_head_dual.py`，`src/inference_t2v_geoprior.py`，仓库自带 DiffSynth 的 `rope_pos_scale` 消费端 |
| 已发布内容 | 有实际 VAE、双分支头、推理入口、权重下载入口和 benchmark；训练代码不完整 | [README](https://github.com/cvlab-kaist/GRACE/blob/914c7a8121d98417a0e350b2dba427cf9225155c/README.md)。本次没有下载或运行 GRACE 权重 |
| 文字风险 | 作者明确报告招牌文字、细线、小物体可能丢失或断裂；VBench 不等于 OCR 保真 | 论文附录 J。不能用视频生成总体分数证明文字超分兼容性 |

本次 VOSR2 实验没有时序维度，继续使用原来的单步 `t=1,r=0`、Qwen 2D VAE、DINO/CA 和输入/输出通道。
完整迁移 GRACE 需要重新训练该 VAE、扩展双潜变量输入输出、制定单步适配方案并重新验证 OCR，超出本次最小实验。

## 2. 实现与可回退性

`models/token_compression.py` 对块区间 `[start_block,end_block)` 使用：

```text
p = P(x)                         # 2×2 平均合并，N → N/4
y = compressed_blocks(p)        # SA、CA query、MLP 实际处理 N/4 个 Token
x_out = x + U(y - p)             # 提升更新量，保留全分辨率入口细节
```

`U` 是最近邻复制。压缩区间为恒等映射时，`x_out=x` 精确成立。该旁路保留**入口特征差异**，不能保证被跳过的高分辨率语义交互或文字笔画被恢复；其中也可能包含噪声。前后稠密层和蒸馏需要学习补偿。

- 新模块无参数，原有参数键名/形状全部不变；`load_backbone_state` 的严格检查没有放宽。
- 默认 `compression_config=None`，旧配置、旧导出仍可加载。配置 `enabled: false`、`factor: 1` 或推理 `--no-compression` 都走原始分辨率路径。
- `force_dense=True` 同时关闭压缩与 MLP 路由；`--dense-mlp` 仍只是 SDT 的稠密 masked MLP 实现，不等价于关闭路由。
- 粗网格 RoPE 使用原网格单元中心 `(factor*i+(factor-1)/2)`，再应用 VOSR 原有动态网格缩放；不把合并 Token 错当成连续细网格坐标。
- 不支持的网格直接报错，不静默补零/压缩。tile 必须被 `8*patch_size*factor` 整除；默认 `512` 合法。
- `--no-compression` 只关闭结构；**微调后的权重并不会变回预训练权重**。完整回退使用 `original_export` 或原始分支/检查点。

训练复用 `train_vosr_ablation.py`：冻结教师、相同 LQ/noise、输出 MSE KD；新增可选的教师中间特征 cosine KD、潜空间有限差分 KD，以及原有 dense sandwich KD。特征先借助同一细节旁路恢复到完整网格再比较。
这不是 GRACE Stage 1 的 VAE 对齐：本实验固定 VAE、优化学生 DiT，使用显式固定损失权重，没有照搬其自适应梯度权重，也没有实现 OCR loss。

## 3. 理论对比（非实测加速）

默认 512 像素 tile、VAE f8、patch=2，完整网格 `N=1024`。36 层中第 4–31 层（零起始）共 28 层合并；前后各 4 层保留完整网格。

| 对照 | Attention Token 数/层 | SA QK+AV 总 FLOPs 比例 | SA 投影总 FLOPs 比例 | MLP |
|---|---:|---:|---:|---|
| 原始 VOSR2 | 全部 1024 | 1 | 1 | 全部 Token |
| 当前 SDT | 全部 1024 | 1 | 1 | 推理只处理路由选中的 Token；训练仍为 dense masked |
| 新压缩 | 8 层 1024；28 层 256 | **13/48 ≈ 0.2708** | **5/12 ≈ 0.4167** | 同一压缩区间减少 Token |

SA QK+AV 采用 `4BN²D`，QKV+输出投影 `8BND²`，一次乘加记 2 FLOPs。
CA query 也减少，但 DINO 条件 K/V 不减少；评估入口分别计算 CA 矩阵乘与投影。
这些是**解析计算量**，不含 MLP、norm、softmax、VAE、DINO、合并、细节旁路、图像转换。不能推导“整网 4×/16× 加速”或显存同比下降，FlashAttention 也不必显式存储 N² 矩阵。
参数显存不减少；稠密入口旁路仍占内存。新增 KD 特征只在训练时收集，训练内存可能上升。

## 4. 可复现运行

在仓库根目录使用已有 VOSR2 环境。修改 `configs/ablations/base_vosr2.yml` 中本地教师、VAE、DINO 和 TXT 训练数据路径；所有对照固定相同数据、分辨率、随机种子与硬件。
基线导出时 `input_size` 必须与所用检查点训练网格一致；下面沿用现有消融配置的 512 tile，不能据此推断官方原始训练分辨率。

```bash
# 原始预训练权重，仅转为统一评估导出格式，不训练
python export_vosr_baseline.py --config configs/ablations/base_vosr2.yml --output exp_vosr/original_export --upscale 4

# 新压缩的小规模真实权重验证（10步；不是质量收敛实验）
python train_vosr_ablation.py --config configs/ablations/token_compression_smoke.yml

# 正式对照，使用相同训练预算；已有 SDT 导出可直接复用并记录训练步数
torchrun --standalone --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/dydit_sdt.yml
torchrun --standalone --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/token_compression.yml

# 断点恢复仍用同一配置；沿用原训练器的 RNG/数据迭代策略，不承诺逐步位级重现
torchrun --standalone --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/token_compression.yml --resume exp_vosr/token_compression/checkpoint-00001000

# 压缩与结构回退，分别输出便于肉眼比对
python inference_vosr_ablation.py --export exp_vosr/token_compression/export --input data/eval/lq --output artifacts/compressed
python inference_vosr_ablation.py --export exp_vosr/token_compression/export --input data/eval/lq --output artifacts/fallback --no-compression
```

对于训练是否影响文字能力，建议增加同预算稠密 SFT 对照（使用 base 配置训练），与原始权重一起保留；不要把“微调收益”和“压缩影响”混为一个结论。首版三组仍按原始、已有 SDT、新压缩执行。

## 5. 性能入口与测量口径

替换 `configs/eval/text_regions.example.jsonl` 的示例路径和人工转写，调整 suite 的 export/output 路径。示例不是可运行数据集。

```bash
python benchmark_token_compression.py --suite configs/eval/token_compression.yml --check-only
python benchmark_token_compression.py --suite configs/eval/token_compression.yml
```

每个变体在**新进程**内运行，避免教师/其他学生同时驻留。程序拒绝混用不同的 backbone、CA、VAE、DINO、precision、upscale 或 tile 设置；拒绝覆盖既有运行目录。

输出每个变体的 `performance.json` 和 `images/<id>.png`：

- 一次不计时的 hook 观测记录每层 SA/CA 的实际输入长度；解析 FLOPs 覆盖所有重叠 tile。
- 每张图片单独 warmup，固定 seed 重复调用 `restore()`；计时前后同步设备，保存原始耗时、p50/p95。
- end-to-end 范围包括 resize、VAE 编码、DINO、分块 DiT、融合、解码和 CPU 图像转换；不含模型加载、磁盘读写、哈希或 OCR。
- 记录每次峰值 allocated/reserved、调用前 allocated，以及二者差值；总峰值包含已加载的 DiT/VAE/DINO。CPU 设备显存字段为 `null`，不伪填零。
- 保存权重 SHA256、输入 SHA256、manifest SHA256、配置、git commit/dirty 状态、运行时及设备信息。预训练依赖 VAE/DINO 路径也记录在 pipeline；跨机器复现实验应另外固定并归档它们的 revision/哈希。

CUDA 和 CPU 入口已实现；NPU 仅提供显式 `device: npu:0` 的 backend 入口，Ascend 算子、同步、显存 API 和端到端性能尚未验证。原训练器的 NPU 适配状态不因本分支改变。

## 6. OCR 分层对照

manifest 每行描述一张完整图，`size=[SR宽,SR高]` 必须等于 LQ 尺寸×upscale；`bbox=[x0,y0,x1,y1]` 使用 **SR/HQ 像素坐标**，右下边界不包含。一个 region 是一个可识别文字行/词，人工核对 `text`；标签 `small/chinese/dense` 可重叠，分组不是互斥分区。

冻结评测集和标签后再训练；训练与评测按来源/页面去重。小字阈值应在清单制作时固定（例如按 LQ 字高定义并在实验记录中说明），不能按各模型的预测重新选样本。密集页面仍需逐行给框，默认 PSM=7。

支持任意固定 OCR 引擎的 JSONL：

```json
{"id":"example_001","regions":[{"id":"line_001","text":"OCR预测文字"}]}
```

必须覆盖全部图和全部框，空预测用空字符串，重复/漏框直接报错；不允许根据置信度删除识别失败的框。**GT 是人工转写，不是教师模型或 OCR 的输出。**

```bash
# 已有 PaddleOCR/自用识别器：导出上述 JSONL 后评分；三组 engine-id 完全相同
python evaluate_text_ocr.py --manifest data/eval/manifest.jsonl --predictions artifacts/predictions_original.jsonl --engine-id '固定识别器版本+权重sha256+预处理版本' --output artifacts/token_compression_comparison/original/ocr.json

# 可直接执行的本地 Tesseract 适配器（需预装同版本 chi_sim+eng 数据，不自动下载）
python evaluate_text_ocr.py --manifest data/eval/manifest.jsonl --images artifacts/token_compression_comparison/compressed/images --engine-id 'tesseract版本+traineddata哈希' --language chi_sim+eng --psm 7 --output artifacts/token_compression_comparison/compressed/ocr.json

# 原始 / sdt / compressed 各运行一次 OCR 后汇总；缺失 OCR 明确显示未实测
python summarize_token_compression.py --run artifacts/token_compression_comparison
```

不要在三组之间切换 OCR 引擎或文字规范化方式。默认 NFC、保留空格/标点/大小写；可显式统一 `--strip-whitespace`。Tesseract CLI 只移除终端输出结尾的换行。参考：[官方 CLI 文档](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html)。

主指标：`CER=Σ编辑距离/ΣGT字符数`（含替换/删除/插入、不截断），`character_accuracy=1-CER`（插入很多时可为负），以及区域 `exact_match`。
同时保留逐框预测与 errors，查看中文笔画替换、重复字符、跨 tile 断裂。该协议评估**固定 GT 框的识别准确率**，不包括检测召回率；可另在 HQ 上测同一识别器的误差基线。`comparison.json` 提供相对原始模型的分组 CER 差值，不自动假定可接受下降阈值。

## 7. 验证边界

本次执行记录见 [token_compression_validation.md](token_compression_validation.md)。
关闭一致性、真实 tiny DiT 的形状/反传/导出、合成训练与分块流程验证，均不等价于真实预训练模型的兼容性/质量验收。
只有真实数据、真实权重、目标 CUDA/NPU 环境跑完三组对照后，才能回答是否保住小字/中文/密集文字，以及能否降低端到端延迟和峰值显存。
