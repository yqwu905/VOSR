# VOSR2 FFN 半宽：实现说明与实验判据

本文解释训练实现的目标与取舍。实际准备数据、并行运行、恢复任务和评测的命令见
[实验操作文档](ffn_halfwidth_experiments.md)。实验定义以
[matrix.yml](../configs/ffn_halfwidth/matrix.yml) 为准。

## 1. 固定的结构目标

默认教师是原始 Dense VOSR2。36 个 block 的残差维度仍为 1536，token 网格、Attention、DINO、VAE 和
单步噪声接口保持一致。SwiGLU 实际 hidden 从 4096 物理裁到 2048；打包输入矩阵从
`[8192,1536]` 变为 `[4096,1536]`，输出矩阵从 `[1536,4096]` 变为 `[1536,2048]`。
所有普通半宽实验每层都满足这一要求。Dense、k75 和 head-split 组在实验清单中明确标记，
不能混用“平均半宽”和“逐层半宽”的结论。

初始化沿用 `models/mlp_compression.py` 的 linked SwiGLU 通道裁剪：同一神经元的两个输入分支行及输出列
一起选择。`keep_ratio` 配置保留教师的原始模型配置，加载全宽权重时完成裁剪；阶段间加载窄权重时
直接恢复窄参数，不再次选择通道。全宽 fade checkpoint 仅用于恢复训练，最终导出才是物理窄模型。

512 像素 crop、F8 VAE、patch2 下，FFN 主矩阵 MAC 从约 695.8G 减为 347.9G；在现有仓库的 DiT
计数口径下，Dense+半宽约为原始 Dense DiT 的 78.7%。这不包含 DINO/VAE，也不等于实测加速。
端到端测量使用 `tools/benchmark_ffn_export.py`。

## 2. 三个恢复机制

### 小文字加权统计

`tools/mlp_neuron_stats.py` 支持全 token 等权、文字加权和分层质量分配三种统计分布。
分层权重按小字、其他字、背景分配；空分层将质量让给当前样本中的非空分层。
统计量以 weighted Welford 合并，避免大均值情况下用 `E[x²]-E[x]²` 累积消减误差。
协方差只在指定层组内常驻，其他层通过确定性重放输入计算。源 checkpoint 和数据清单的指纹随统计保存。

校准成本需要显式规划：36 层、`layer_group_size:2` 会重放 18 遍完整编码器和教师；group8 为 5 遍。
GPU 上可探索 `moment_device:model`、`moment_dtype:float32`、group8；仅 FP32 协方差累加器约
512 MiB，另有临时矩阵、编码器、模型与最终约 2.25 GiB CPU 协方差。`max_tokens_per_sample:256`
可减少统计矩阵乘法与传输，但不减少完整教师前向。默认 0 使用全部有效 token，正数采用每图无放回
均匀抽样并修正纳入概率；中心协方差仍为近似，极小字可能被漏采，因此保留全 token 对照。
`chunk_tokens` 只分块计算，不减少样本数。运行手册列有配置示例。

同一加权分布的均值/协方差同时服务通道评分和最小二乘补偿。保留通道 K、删除通道 R 时：

$$B=C_{RK}(C_{KK}+\lambda I)^{-1},\quad c=\mu_R-B\mu_K,$$
$$W'_{3,K}=W_{3,K}+W_{3,R}B,\quad b'_3=b_3+W_{3,R}c.$$

这里的线性补偿是初始化，后续仍训练 SwiGLU 的非线性输入投影。

### 教师输入下的逐层重建

`tools/reconstruct_ffn.py` 在教师真实前向的 MLP hook 中取得已经归一化、条件调制的输入。
学生仅运行对应窄 MLP，并匹配教师门控后的 FFN 残差更新：

$$L_l=\frac{\operatorname{mean}_q\|g_l(f_l^S(u_l^T)-f_l^T(u_l^T))\|^2}
{\max(\operatorname{mean}_q\|g_l f_l^T(u_l^T)\|^2,\tau)}.$$

默认训练每层 `w12` 与 `w3`，每次优化器更新覆盖全部选定层。每层局部图即时 backward 并释放；
无需保存所有层的大激活缓存。多卡下各 rank 使用不同确定性样本，显式按固定顺序平均局部参数梯度，
因为此阶段不存在一次完整学生 DDP forward。日志的 `updates_per_layer` 可核对每层预算。

`recon_w3_only`、`recon_ungated`、重建学习率和预算实验分别检验非线性重学、残差门控和训练量的贡献。

### 真实学生路径上的联合恢复

`train_vosr_ablation.py` 的新增参数默认关闭，原有实验无需迁移。主线 B 阶段只更新 FFN，C/D 放开 DiT。
学习率中 `learning_rate` 是其他 DiT 参数的基准，`ffn_lr_multiplier` 是相对倍率，不是绝对 FFN 学习率。

门控 FFN 更新通过模型 forward 的 `mlp_update_layers` 显式返回到 `stats['mlp_updates']`，
与 block checkpointing 和 DDP 的返回张量路径一致，不依赖 backward 重算时会被覆盖的可变 hook 缓存。

## 3. 损失与数据语义

建议主线的起点为输出 KD 1、FFN 更新 KD 0.1、block feature KD 0.05、逐行 Charbonnier 0.1、逐行
Sobel 0.05。像素项在 C 阶段渐入，内部 KD 可以线性降低到初值的一半。系数是实验起点，尚无全规模
训练结果证明其最优。

输出 token 权重由原生图像 mask 做 area pooling：`1 + 2*text + 4*small_text`，并按权重均值归一。
内部 loss 先做每层教师能量归一，再在层间平均；小字权重同样来自对应残差网格的 area pooling。
当前单步关系为 `sr = noise - prediction`，相同噪声下 velocity KD 与相应 x0 latent MSE 等价。

数据中的文字行高始终从原始四边形计算，裁掉半行不会让它变成“小一号”的字。逐行 GT/Sobel 在原生
输出像素网格计算，先在每行有效区域归一，再在行间平均。VAE 参数冻结，但学生解码保持梯度。
Charbonnier/Sobel 使用未硬裁剪的 VAE 输出，避免超出像素范围的预测失去回传梯度；同一次解码的
有界视图交给 LPIPS/OCR，推理解码仍保持原有裁剪行为。
`alignment_valid:false` 的 ROI 不参与 GT/Sobel/OCR。只有显式 `teacher_correct:false` 且 GT 有效、
对齐时才降低相应教师 KD 权重；未知正确性不被当作错误。

OCR 是独立消融。它复用现有可微 PyTorch PP-OCR 识别器，冻结其参数并保持 eval；学生图像通过透视
ROI 变换、识别网络获得梯度。截断文字行与超过 `ocr_max_width` 的长行跳过 OCR，计数可从
`TextROILoss.last_counts` 读取。解码图不会为 OCR 人为压成错误长宽比。PaddleOCR 的预测接口仅用于评测。

自然图 source 没有文字框时必须明确声明 `allow_missing_text:true`、`text_crop_prob:0`。训练、校准及
难例数据均为训练 split；验证和测试只用于评估。`image_names` 支持在原图目录上选择难例子集，无需复制图像。

## 4. 训练预算与对照

| 阶段 | full 默认更新数 | 参数范围 | FFN / 其他 LR |
|---|---:|---|---|
| A 局部重建 | 每层 2000 | FFN | 1e-4 / 冻结 |
| B 适应 | 4000 | FFN | 5e-5 / 冻结 |
| C 联合恢复 | 24000 | DiT | 2e-5 / 2e-6 |
| D 小字修复 | 6000 | DiT | 5e-6 / 5e-7 |

默认 global batch16。`smoke` 用极短更新检查启动；`pilot` 用较少预算筛方向；`full` 才采用上表。
局部重建的额外计算单独记录，不能把额外训练量全部归因于方法。fade 使用独立的单阶段、等总更新数
控制组，不混入已经重建过的 kept 权重。Fading FFN 的 AdamW group 强制 weight_decay0，避免对整张
打包参数的衰减改变已冻结 dropped 行。

Dense 原始 checkpoint 是实际压缩目标的参照；Dense 继续训练的控制组反映新数据/预算的收益；
同完整新配方的 Dense 控制组用于衡量相同任务损失下的容量损失。主线再用多个训练 seed 检查稳定性。

## 5. 评测判据

字号桶以 native SR 高度计，默认边界 24、32、48、64、96 px。重点单独报告 32–48、48–64 和合计
`h<64`。固定 GT quads、识别器、文字归一化、噪声 seed、tiling 与图像色彩处理。

建议的目标是总体 CER 增加不超过 0.5 个百分点、OCR-A 下降不超过 1 个百分点；小字 CER 差值的
点估计不超过 0.5 个百分点，并查看配对图级 bootstrap 的 95% 上界是否低于 1 个百分点。
这是一组预设验收目标，不是已经取得的成绩，也不以“差异不显著”替代非劣判断。

`tools/evaluate_text_bins.py` 对齐同图同框，严格核对标签、原生行高与识别器设置。重采样单位是整张
图片，多次推理种子先在图片内汇总。另报告教师正确小字的保留率；它不能替代完整 CER。
同一页面拆成多张图片时，应先合并页面 ID 或使用真正独立的图片测试集，不能把相关 crop 当独立样本。

固定视觉图集比较 LQ、原始 Dense、学生和 GT，检查粘连、断笔、形近字替换，同时检查普通区域质量。
最终窄模型通过结构检查后，才在目标设备上记录同步 warmup 后的 p50/p95 端到端延迟。

## 6. 恢复与验证边界

端到端 checkpoint 记录 optimizer、EMA、各 rank RNG、sampler epoch 与已消费 batch 数。
`deterministic_crops:true` 时裁剪不依赖 worker 的预取状态；恢复先重建数据位置，再恢复 RNG。
同阶段用 `--resume`，换阶段用 `--init-from` 并重建 optimizer。world size 改变时应新开实验。
`checkpoint.complete`、`export.complete` 用于区分完整产物与中断写入。
完成训练但导出中断时，可以从最终 checkpoint 只重试导出，fade/curriculum 的运行态会按完成步数恢复。
历史未命名 optimizer 参数组按旧参数顺序迁移，保留 Adam moments，并修正 fading FFN 的 weight decay；
历史 checkpoint 缺少逐 rank RNG，不能声称逐位复现。无法严格匹配的旧布局明确拒绝，需另开 `--init-from`。

本次实现的验证使用 CPU 小模型和合成数据。真实 VOSR2 权重、大规模数据训练、CUDA/NPU 多卡吞吐及
最终 OCR 接近程度仍需在训练机器执行。受限执行环境不允许 Unix/TCP 进程通信 socket，相关多 worker
和 Gloo 测试有明确的 skip/opt-in 条件。没有用单进程通过来声称多卡训练已经验证。

## 方法参考

- [仓库原始 FFN 剪枝实验记录](mlp_neuron_prune.md)：既有通道选择、补偿、fade 和 k50 观察。
- [TinyFusion](https://arxiv.org/html/2412.01199v1)：关注恢复后的质量，并谨慎处理内部表征蒸馏。
- [PPCL](https://arxiv.org/html/2511.16156v2)：教师输入下模块拟合和整体恢复的参考；其结构和任务与此不同，
  且附录明确存在小文字失败案例。

上述研究支持组成机制，不能直接证明 VOSR2 每层半宽后的文字质量。
