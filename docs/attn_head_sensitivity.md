# VOSR2：逐 Head 池化敏感度与受控预算实验

基于 PR #33 / `0c5064f4c06b3c1b0e554ac994343ad7d1a1c739`。本改动复用现有
Head-split、训练器、导出、VAE/DINO 与 `evaluate.py`，不重复 12k 步基线训练。

**当前状态：已实现分析/干预/改进/对照入口并做 CPU 轻量验证；当前执行环境无 GPU，也没有
VOSR2、VAE、DINO 和 Real-CE 的真实权重/数据。因此尚无 864 个头的真实敏感度排名、OCR 改善量或新权重。
单元测试中的随机微型网络、替身 VAE/DINO/OCR 仅验证代码契约，不能作为模型效果证据。**

## 1. 需要区分的三个问题

1. **局部性**：同一批数据上的 3×3 attention mass，描述注意力分布；不能直接解释恢复重要性。
2. **固有替换敏感度**：Dense teacher 同一层、同一输入、同一权重，单个头从 Dense 改为 Pool/Conv，损失多大？
3. **已训练模型的恢复收益**：在发布版 f2 上恢复某个头的 Dense 计算，能否降低相对教师的误差并改善 OCR？
   这个问题与第 2 项不同：发布版已适应压缩，上游输入分布和其他权重都已改变。

原报告已有“相同预算下恢复敏感整层没有收益”的负结果。本轮优先逐头、少量分组筛选；
整层替换用于诊断和交互检查，不把它预设为改进方案。

## 2. 新增入口

| 文件 | 作用 |
|---|---|
| `models/head_sensitivity.py` | 同权重逐头计算、局部性、可恢复的替换上下文、卷积按 Head ID 迁移、精确理论 MAC |
| `tools/attn_head_sensitivity.py` | `prepare` 固定样本；`profile` 全头筛查；`ablate` 因果替换、KD、可选解码/OCR |
| `tools/plan_head_budget.py` | 按预算生成高敏感头保留方案、等 MAC 随机对照、短续训配置和可加载的迁移导出 |
| `tools/head_ocr_compare.py` | 复用 `evaluate.py` 逐框记录，严格配对、多种字号、以源图为单位的 bootstrap |
| `configs/analysis/head_sensitivity.yml` | 继承已有 f2 配置，保留教师/模型/数据语义 |
| `tests/test_head_sensitivity.py` | CPU 数值/预算/迁移/干预/缓存到 OCR 的轻量测试 |

`models/attn_compression.py` 新增可选 `dense_heads`，每层一个 Head 编号列表（均从 **0** 开始）。
三路分别为：全分辨率 Dense、全分辨率局部 DWConv、2×2 Pool Attention。
全分辨率头使用原网格 RoPE，池化头使用质心 RoPE，输出投影按对应通道列相加，bias 只加一次。
`dense_heads` 始终做 full attention；`attn_type: sparse/local` 仅作用于池化分支。
未配置新字段时旧导出结构与行为保持兼容；显式 `local_heads` 与 `dense_heads` 重叠会报错。
若局部头来自 `head_stats` 阈值，显式保护头从自动局部集合中移除。

## 3. 数据与实验协议

从 Real-CE **训练部分**抽取 64 张不同图片，每张一个确定的文字中心裁剪、噪声种子 42/43；
再用互不重叠的训练图片做复核。`--names` 接受每行一个图片 stem，可固定校准/复核划分。
默认不按字号均匀采样，因此必须检查小字 ROI 的 `*_count`，必要时扩样本，不能把 4 张旧统计推广为结论。
**正式 261 图测试集只用于最终冻结方案的比较，不用于选择 Head 或预算。**

缓存保存实际 LQ latent、noise、DINO features、VAE 归一化、文字/小字 polygon mask，记录图片、裁剪坐标、
源图与缓存 SHA256、配置、精度与版本。相同缓存输入供所有实验臂复用。模型与分片权重在排名报告中记录 SHA256。
默认 FP32 参数 + BF16 autocast，禁用 TF32；需要与 FP32 对照时分别构建缓存，避免混用编码精度。

字号采用现有 `evaluate.text_size` 的透视裁剪/竖排旋转规则，单位为 **SR/HQ 像素**：
`<32`、`32–48`、`48–64`、`64–96`、`>=96`，另报告总体和 `<64`。小字 mask 经面积池化映射到 token/latent。
没有小字的样本记为 `null` 并从小字均值排除；不记作 0。

`boxes.json` 只有几何、没有转录，因此足够测 ROI 特征/KD，但不能用于 OCR。
`prepare --ann` 可额外接收 HQ 坐标的 `Label.txt`；只保留完整落入裁剪的文字框，防止被截断的标签污染 OCR。
训练集原始转录需单独整理为这个已有格式，不使用测试标签来补充训练校准。

从仓库根目录运行，先修改本地配置中的 `teacher_checkpoint`、`vae_path`、DINO、配对数据路径：

```bash
python tools/attn_head_sensitivity.py prepare \
  --config configs/analysis/head_sensitivity.yml \
  --samples 64 --seed 11 --noise-seeds 42 43 \
  --output artifacts/head_sensitivity/cache --device cuda --precision bf16
# 有训练校准集转录时，在以上命令增加 --ann /path/to/train/Label.txt
# 固定互不重叠的图片划分时增加 --names /path/to/calibration_stems.txt
```

默认 CUDA 不可用时明确退出，不偷偷把真实实验换成随机输入。CPU 路径可用，但完整 1.4B 逐头评估很慢。
不依赖 PaddleOCR 来运行 `prepare`（没有 --ann 也可）/`profile`/未开启 OCR 的 `ablate`。

## 4. 第一级：36 层 × 24 Head 的同输入筛查

令一个头的输出为 \(Z_h=A_hV_h\)，用 Pool 或 Conv 得到 \(\tilde Z_h\)，
该头的输出投影列为 \(W_h\)，AdaLN 残差门为 \(g\)。分别测：

- `head_mse`：\(\|\tilde Z_h-Z_h\|^2/(Nd)\)，另给出 `head_nmse`。
- `projected_mse`：\(\|(\tilde Z_h-Z_h)W_h^T\|^2/(NC)\)。
- `residual_mse`：\(\|g\odot((\tilde Z_h-Z_h)W_h^T)\|^2/(NC)\)，另给出 `residual_nmse`。
- `text_mse` / `small_mse`：相同门控误差在文字 / `<64px` mask 上的加权均值。
- `mass3x3`：同样本上的局部性，单独输出；不是排名目标。

排名优先看绝对的 `small_mse`，同时看 `residual_mse` 与样本方差。只看归一化误差容易把能量很小、
实际影响很小的头排到前面。报告附 `mass3x3` 与 `small_mse` 的 Spearman 系数（含并列秩）。

Pool 在 **QKV 投影之前**池化，QK norm 在池化之后，使用质心 RoPE；不会错误地把“先 Dense QK norm 再池化”
当作生产方案。逐头 W_o 投影指标用 FP32 累积；池化输出上采样后再投影在实数运算下与生产顺序等价，
但不声称 BF16 下逐位相同。真实数值路径造成的差异由第二级生产模块干预测量。
Conv 比较使用校准 pattern 的归一化 7×7 核；这项衡量初始化近似误差，**不是发布版训练后卷积误差**。

```bash
python tools/attn_head_sensitivity.py profile \
  --config configs/analysis/head_sensitivity.yml \
  --cache artifacts/head_sensitivity/cache \
  --output artifacts/head_sensitivity/profile \
  --head-chunk 4 --query-chunk 128 --shortlist 24
```

输出 `heads.csv`、`layers.csv`、`ranking.json`、`per_sample.jsonl`。
每种候选（Pool/Conv）各有 864 行，含均值、标准差、有效样本数与排名。
层级 `sum_head_*` **只是筛查分数**；各头投影会相互抵消/增强，不能当作该层实际替换误差。
`shortlist_pool.json` / `shortlist_conv.json` / `shortlist_dense.json` 自动选择发布方案当前模式下小字误差最高的 24 个头，
`shortlist_joint.json` 用于联合恢复验证。没有任何小字样本时 shortlist 为空，不能继续声称有小字排名。

内存按 Head/query 分块，局部性不会一次存储 `36×24×N×N` 注意力矩阵。
每层会额外计算 Dense/Pool/Conv 候选，整个筛查仍有成本；建议先 `--samples 4` 验证数据与依赖，再扩大到 64–128。

## 5. 第二级：逐层、逐 Head 与联合替换

**Dense-drop**：其余所有模块不变，仅替换选中的 Head；继续执行余下全部层，记录最终 velocity MSE。
**Release-rescue**：从已训练 f2 导出加载，选中的 Head 改为 Dense；计算对同一个 Dense teacher 的误差变化。
这一步更直接回答“哪些头值得花额外预算”。恢复后的 QKV/W_o 仍来自发布模型，不把教师权重混进去。

```bash
# 实际整层替换，不把一级分数求和当作结果。36 个实验臂。
python tools/attn_head_sensitivity.py ablate \
  --config configs/analysis/head_sensitivity.yml --cache artifacts/head_sensitivity/cache \
  --unit layer --mode pool --output artifacts/head_sensitivity/layers

# 候选逐头 Pool：只有该头压缩，其余保持 Dense。
python tools/attn_head_sensitivity.py ablate \
  --config configs/analysis/head_sensitivity.yml --cache artifacts/head_sensitivity/cache \
  --groups artifacts/head_sensitivity/profile/shortlist_pool.json \
  --output artifacts/head_sensitivity/heads_pool

# 候选逐头 Conv：验证固定局部卷积是否比池化更好。
python tools/attn_head_sensitivity.py ablate \
  --config configs/analysis/head_sensitivity.yml --cache artifacts/head_sensitivity/cache \
  --groups artifacts/head_sensitivity/profile/shortlist_conv.json \
  --output artifacts/head_sensitivity/heads_conv

# 已训练发布版上的逐头恢复，更适合决定保护哪些头。
python tools/attn_head_sensitivity.py ablate \
  --config configs/analysis/head_sensitivity.yml --cache artifacts/head_sensitivity/cache \
  --export weights/head_split_f2 \
  --groups artifacts/head_sensitivity/profile/shortlist_dense.json \
  --output artifacts/head_sensitivity/rescue
```

`--unit head` 不筛选时可对所有 864 头进行最终输出评估；`--layers 0 1 --heads 0 1` 可缩小调试范围，
这些编号只是语法示例，并非已发现的敏感头。任意分组文件格式为：

```json
[{"name":"example_group","mode":"dense","heads":[[0,0],[1,0]]}]
```

有带转录缓存时增加 `--decode --ocr`，复用原 VAE 解码与 `evaluate.py`，得到每个实验臂的小字
OCR-A/CER/1-NED 及 `ocr_deltas.json` 排名。仅 `--decode` 则输出图像，之后可使用相同评测器单独评测。
缓存裁剪诊断的 OCR 不与完整 261 图的发布分数直接横比；最终使用下文的完整推理协议。

`kd_mse` 是相对 Dense teacher 的最终 velocity MSE；`output_delta_mse` 是相对当前模型基线的输出变化；
`delta_kd_mse` / `delta_small_kd_mse` 为“替换后 − 替换前”。Dense-drop 越大越敏感；
Release-rescue 越负越有益。后者按从小到大排序，不把仅仅改变很大的 Head 错认作改善头。

OCR 严格配对 `(noise seed, image, box index)`，标签、几何或样本数不一致报错，不取交集美化结果。
bootstrap 按**源图片**抽样；同图多个框、两个噪声种子一起抽样。少于 10 张独立图片不输出置信区间。
同时记录多个 seed 下的差异，不能把重复 seed 当成新增独立图片。

最后对预算选出的整个集合做一次 `--groups` 联合替换，检查收益是否保持；
单头损失不具可加性，排名仅用于提名，不据此直接承诺组合收益。

## 6. 两项受控改进与对照

### A. 按敏感度保留少量全分辨率 Head

原全局头仍 Pool，原局部头仍 Conv；只把高敏感头升级为 Dense。每个头的 MAC 包括 QKV、QK/AV、W_o：

\[
M_{dense}=4NCd+2N^2d,\quad
M_{pool}=4(N/f^2)Cd+2(N/f^2)^2d,\quad
M_{conv}=2NCd+Nd k^2.
\]

选择按 `实测误差 / 新增 MAC` 排序，在硬上限内贪心加入（不声称全局最优，也不声称误差可相加）。
原始 `N=1024,C=1536,d=64,f=2,k=7` 下 f2 的理论 SA 比例为 **23.2562%**，与 PR #33 一致。
推荐仅试以下三个小预算：

| SA MAC 上限（相对 Dense） | 预计整个 DiT MAC 比例¹ | 可增加保护头的数量范围² |
|---|---:|---:|
| 25% | 约 78.7% | 18–24 |
| 27% | 约 79.3% | 40–52 |
| 30% | 约 80.2% | 73–94 |

¹ 按原报告 DiT/SA MAC 和其余模块不变推算，不是速度实测。² 取决于原 Pool/Conv 比例与正收益候选数量；
预算不强制填满。MAC 不包括 norm/bias/interpolation，新增小分支和 gather 也可能影响延迟；最终还应在目标卡上计时。

用一级分数生成候选；或者把 `--ranking` 换成 `rescue/ranking.json`、`--metric delta_small_kd_mse`，
只保护在已训练模型上测得正收益的候选头：

```bash
python tools/plan_head_budget.py build \
  --config configs/analysis/head_sensitivity.yml \
  --ranking artifacts/head_sensitivity/profile/ranking.json --metric small_mse \
  --source-export weights/head_split_f2 --ratios 0.25 0.27 0.30 \
  --random-seeds 11 --steps 1000 --output artifacts/head_sensitivity/budgets
```

输出每个预算的布局 JSON、短续训 YAML、**同 MAC 随机对照**和 `experiments.json`。
随机对照严格匹配从 Pool/Conv 升级的数量（两者成本不同），不只匹配保护头总数。
`baseline.yml` 是同一发布版再训练相同步数的对照，用来区分额外训练带来的收益。

### B. 用实际输出误差重新划分 Conv/Pool

在上面 `build` 加 `--reclassify`：同一个头比较 `small_mse(Conv)` 与 `small_mse(Pool)`，
采用较低误差模式，再在同一个 MAC 上限内选择保护头。预算不足时明确拒绝，不偷偷放宽预算。
如果不同校准/复核划分的决策不稳定，保持旧分类；不要直接上新分类。
这会改变局部卷积参数的排列或数量，必须通过下述显式迁移，而不是仅修改 model.json。

可使用已有 `tools/attn_head_stats.py --samples 64` 在训练校准集上重新计算 pattern，指定新文件给 `head_stats`。
这是对卷积初始化的额外消融；重新分头、重新估计 kernel、保护头应分开验证。
默认流程沿用已发布模式的 kernel 来源，保护旧的已学习卷积。未实现额外卷积架构，避免同时改变过多因素。

### 导出迁移与轻量恢复训练

```bash
# 例：27% 上限的保护方案。按 Head ID 搬运旧卷积，新增 Conv 使用校准 kernel。
python tools/plan_head_budget.py export \
  --config configs/analysis/head_sensitivity.yml --source weights/head_split_f2 \
  --spec artifacts/head_sensitivity/budgets/protect_0p2700.json \
  --output artifacts/head_sensitivity/budgets/protect_0p2700

# 直接复用现有训练器；生成 YAML 指向上述迁移后的权重。
python train_vosr_ablation.py --config artifacts/head_sensitivity/budgets/protect_0p2700.yml
# 对 random_0p2700_s11 也先执行相同 export 迁移，再运行其 YAML。
python train_vosr_ablation.py --config artifacts/head_sensitivity/budgets/baseline.yml
```

保持数据、optimizer、EMA、总 batch、随机种子与 1000 步长度一致。先比较无需训练的联合替换；
只给 1–2 个有希望的预算做 500–1000 步恢复，避免重跑已有 12k 实验。发布版权重不被修改。
每个实际训练导出位于原训练器日志/检查点路径下，以训练器输出为准。

## 7. 最终完整测试入口与选择规则

使用已有全图分块推理（512 tile、64 overlap、VAE tile 2048）与同样的噪声种子/PP-OCRv5 评估。
对 Dense 原模型、f2 发布模型、f2 等步续训、候选保护方案、等 MAC 随机保护方案分别运行：

```bash
python inference_vosr_ablation.py --export /path/to/final/export \
  --input preset/datasets/RealCE_x4/val/LR --output preset/results/arm_s42 \
  --tile-size 512 --tile-overlap 64 --vae-tile-size 2048 --seed 42
python evaluate.py --pred preset/results/arm_s42 \
  --ann preset/datasets/RealCE_x4/val/Label.txt --ann-ref pred \
  --ocr-model PP-OCRv5_server_rec --size-buckets 32 48 64 96 \
  --output preset/results/arm_s42.eval
# seed 43 同样运行后，严格配对两个 seed：
python tools/head_ocr_compare.py \
  --reference preset/results/baseline_s42.eval preset/results/baseline_s43.eval \
  --candidate preset/results/arm_s42.eval preset/results/arm_s43.eval \
  --output artifacts/head_sensitivity/final_ocr_delta.json
```

选择依据：`<64px` CER 下降、OCR-A/1-NED 同方向、全图不退化、多个 seed 与独立校准划分结论稳定、
SA MAC 处于给定上限且优于等 MAC 随机对照。完整报告填入实际数值和源图 bootstrap 区间；
没有足够证据时保留 f2 基线，不把 feature proxy 的降低等同于 OCR 改善。

## 8. 轻量验证

在仓库已有环境运行：

```bash
TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_head_sensitivity.py tests/test_ablation_head_split.py \
  tests/test_ablation_data.py tests/test_evaluate.py
```

新测试覆盖混合三路独立参考、无 QKV bias、保护头 RoPE、旧导出兼容、异常时模块恢复、Head ID 卷积迁移、
反向梯度、真实 36×24 形状的 meta-device FLOP 计数与公式一致、预算硬上限/等 MAC 随机对照、
空 ROI、严格 OCR 配对、图像级 bootstrap，以及缓存→逐头筛查→release-rescue→解码/OCR→预算配置→导出加载。
测试仅用 CPU 微型网络或 meta device，不下载真实模型、不消耗真实训练数据。

本次执行记录：Python 3.11、PyTorch 2.5.1+cpu、timm 1.0.11、diffusers 0.35.0、NumPy 1.26.4，
上述四个测试文件 **88 passed**（约 29 秒）。两个警告来自既有 paired crop 的 NumPy 只读数组转换和
既有测试中的 `torch.load` 默认参数，不是新模型效果结果。三个命令行 `--help` 与 `git diff --check` 也通过。
