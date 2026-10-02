# VOSR2 数据消融实验记录（2026-10-02）

已完成 24 次原始单步 VOSR2 1.4B 全参数续训，保留 36 个 cross-attention 块，
共 21,000 个优化器步、84,000 次来源样本抽取。完整分析见
[实验报告](../../vosr2_data_study.md)。

## 最佳配比和配置

在本次实际比较的续训方案中，验证规则选出的数据配方为 **AnyWord:EasyText = 50:50**，
采用冻结 VOSR2 生成的教师伪 GT、完整文字裁剪和 OCR 最低置信度 ≥ 0.9 筛选。
50:50 是来源采样概率；选定来源后，均匀采样原图分组，再均匀采样组内裁剪。
通过筛选的训练候选为 AnyWord 868 个裁剪 / 523 个原图组，EasyText 999 个裁剪 / 813 个原图组。
该配方不叠加参考转录一致性或 MUSIQ 分位筛选。

| 设置 | 数值 |
| --- | --- |
| 初始化 / 架构 | 发布版 VOSR2；单步 1.4B；全部 cross-attention 可训练；无 DyDiT |
| 裁剪 | 长边 1,024 归一化；完整文字裁剪；等比适配及边缘填充至 512×512 |
| LQ 退化 | 4×；模糊 σ∈[0.2, 1.2]；噪声标准差∈[0, 5]；JPEG 质量∈[65, 95] |
| 优化器 | AdamW；学习率 2e-6；betas=(0.9, 0.999)；weight decay=0.01 |
| 批量 / 预热 | batch=2，梯度累积 2，有效 batch=4；线性预热 50 步 |
| 精度 / 梯度 | BF16 计算、FP32 主参数；梯度裁剪 1.0；梯度检查点 |
| 训练目标 | HQ latent MSE + 0.1 × 冻结初始模型 velocity MSE；t=1、r=0 |
| 冻结模块 / 导出 | VAE、DINO 和初始教师冻结；导出在线 DiT 权重，无 EMA |

必须区分两个运行：

| 运行 | 步数 / 种子 | 解释 |
| --- | --- | --- |
| [`teacher_ocr_s42`](../../../configs/experiments/vosr2_teacher_ocr.json) | 500 / 42 | 全部 24 个续训模型中测试 1-NED 点估计最高，0.8681549594；仅作描述性报告 |
| [`teacher_ocr_aw050_confirm_s42`](orchestration/configs/teacher_ocr_aw050_confirm_s42.json) | 2,000 / 42 | 按三种子验证均值选出的组合及预先固定的代表种子；测试 1-NED 0.8653245045 |

发布版测试 1-NED 为 **0.8712976982**，高于全部续训模型。按冻结的验证规则，
最终推荐仍是发布权重 `preset/ckpts/VOSR2`。测试集未用于重新选择配方、训练步数或代表种子。
“最优”仅限这批实际测试的来源子集、数据处理方案、训练目标和预算；不同图像指标的最优点不同。

## 记录索引

- [官方测试完整 CSV](orchestration/realce_test_all.csv)：24 个续训模型与发布模型、bicubic 两个基线的七项指标。
- [测试结果及权重摘要](orchestration/test_results.json)、[冻结的验证选型](orchestration/final_validation_selection.json)。
- [处理方式验证](orchestration/processing_validation.csv)、[来源配比验证](orchestration/mixture_validation.csv)、[三种子确认验证](orchestration/confirmation_validation.csv)。
- [三种子测试均值及样本标准差](orchestration/test_confirmation_seed_summary.json)、[全部测试配对比较](orchestration/test_comparisons.json)。
- [测试点估计及 Pareto 取舍](orchestration/test_metric_tradeoffs.json)、[最终测试产物核验](orchestration/final_test_artifact_audit.json)。
- [36 个 cross-attention 块及完整导出权重核验](trained_checkpoint_audit.json)。
- `training/<运行名>/`：每次运行完整的逐步 `metrics.jsonl`，以及 `provenance.json`、`complete.json`、`data_summary.json`、`weight_update.json`。
- `comparisons/`：报告使用的配对统计；`figures/`：实际结果图的 PNG、PDF 和输入摘要。
- [主环境包版本](environment-main.txt)、[隔离 OCR 环境包版本](environment-ocr.txt)。这是实际环境快照，不替代项目的安装说明。
- [归档清单与 SHA256](archive_manifest.json)：每个复制记录的原始位置、字节数和摘要。
- [提交前核验](pr_verification.json)：测试、记录完整性与文档链接检查。

本目录中的复制记录与原始实验产物逐字节一致。JSON 内的 `artifacts/...`、`exp_vosr/...`
及 `/content/VOSR/...` 路径保留运行时原值；它们是来源记录，不是仓库内的超链接。
原始 `artifacts/vosr2_data_ablation/` 下被收录的记录在此目录保留相对层级；训练日志单独归入 `training/`。
数据图片、逐图预测、逐框 OCR 原始记录和模型大权重保留在实验工作区，其摘要收录于上述记录。

## 使用配置

先按[数据管线说明](../../data_pipeline.md)准备来源数据、发布版 VOSR2、Qwen VAE 和 DINO 缓存，
并用 `experiments.build_data` 与 `experiments.score_data` 构造配置指定的训练 manifest。
伪 GT 数据标记为 `research_ablation_unreviewed`，不代表人工字形验收。
已准备这些输入时，可从仓库根目录运行测试点估计最高的 500 步配置：

```bash
OMP_NUM_THREADS=4 USE_TF=0 TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 \
  python train_vosr2_sft.py --config configs/experiments/vosr2_teacher_ocr.json
```

输出目录必须为空；已有运行仅能在保留 `training_state.pt` 且配置、数据及训练代码摘要一致时恢复。
确认模型的记录可用于复现实验配置；最终优化器状态按配置已丢弃。
筛选队列、选型冻结和完整评测步骤见[实验报告](../../vosr2_data_study.md)。
RealCE 采用官方 4× 配对、261 张图及 3,414 个文字框；OCR 固定为 PP-OCRv5_server_rec，
因此文字指标不能直接解释为论文 CRNN 指标的复现。
