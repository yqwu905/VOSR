# VOSR2 → 多步 Pixel PDM：路线 B

实现依据为 `vosr2-l2p-training-report.html`（2026-11-02）的第 5、7–10 节。
这是报告建议的工程实现，尚无完整模型训练、OCR 保持率或 CUDA/Ascend 吞吐结论，
也不声称复现未公开的 VOSR2 训练配方。

## 训练与模型约定

保留 VOSR2 的 36 层 DiT、DINO/CA、时间投影与 p16 对应的原 token 网格。
新建 RGB 输入投影和已有的 `MicroDiffusionModelTop3SCWide` 全图 U-Net；
输入为 `cat(已上采样 LQ, x_t)`，两者尺寸一致，图像范围 `[-1,1]`。
U-Net 始终直接读取 LQ/noisy RGB 和最后一层 DiT feature。
不加载学生 VAE，也不在这一步叠加 head-split、FFN 压缩或全分辨率 skip。

启用方式：`training.objective: pixel_flow` 和 `student.pixel.prediction_type: x`。
缺省仍是原有的一步目标；旧的 `configs/ablations/pixel_l2p*.yml` 行为不变。
`prediction_type: x` 直接返回 FP32 clean RGB，不经过 `noise - velocity` 抵消。

时间方向与报告一致：**t=0 干净，t=1 噪声**。每个样本独立采样时间，
每次重新采样像素高斯噪声，VOSR2 的辅助时间保持 `r=0`：

\[
\operatorname{logit}(t)\sim\mathcal N(\mu,\sigma^2),\qquad
x_t=(1-t)x_0+t\gamma\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
\]

\[
\hat x_0=S_\theta([y,x_t],t,C(y)),\quad d(t)=\max(t,0.05),\quad
\mathcal L_\mathrm{flow}=\mathbb E\frac{\|\hat x_0-x_0\|^2}{d(t)^2}.
\]

这等价于对 **prediction 和 target 同时使用 d(t)** 的 velocity MSE。
代码直接计算 clean residual / d(t)，避免相近速度相减；加噪、残差、除法、loss 与 ODE 更新
都在 FP32 中执行。训练输出不硬 clip。默认 `flow_weight=1`、`gt_weight=0`；
再开 `gt_weight` 会额外加入不加时间权重的 RGB MSE，不会替代 flow loss。
LPIPS/OCR 默认关闭，应在 pixel FM 下独立校准梯度后再开，不能直接沿用旧端点 SFT 权重。

`kd_weight`、`feature_distill_weight`、`kd_text_weight` 必须为 0。
路线 B 不使用 latent/RGB velocity KD，也不默认叠加路线 A 的端点 KD。
教师生成的 RGB 在这里是 **clean 数据样本**，其 flow 噪声独立重采样，
不需要与生成教师图时的 latent 噪声配对。

## 配置与阶段

所有步数均为 optimizer updates；每个阶段重新开始 optimizer、LR schedule 和 EMA。

| 配置（`configs/pixel_pdm/`） | 更新数 | 可训权重 | LR | clean 目标来源 |
| --- | ---: | --- | --- | --- |
| `warmup.yml` | 2,000 | 输入投影、U-Net | 1e-4 | teacher RGB : real GT = 4:1 |
| `adapt.yml` | 18,000 | 新接口 + 首尾各 2 层 + time 投影 | 接口 5e-5；首尾 2e-6；time 1e-6 | 1:1 |
| `warmup_gamma2.yml` / `adapt_gamma2.yml` | 2k + 18k | 同上 | 同上 | 同上，仅 γ 改为 2 |
| `refine.yml`（可选） | 50,000 | 新接口 + 首尾各 5 层 + time 投影 | 同 adapt | 1:1 |

F1（γ=1）与 F2（γ=2）是独立实验：crop 清单、初始化、数据、训练预算和验证协议应相同。
先比较两个 20k pilot 的 8/16-step 真实输出，再决定是否继续 50k–100k。
`timestep_mean: 0.8, timestep_std: 0.8` 是后续独立候选；默认先用 `(0,1)`。
若采用另一论文的相反时间约定，不能直接照搬其 logit 均值符号。

冻结只设置 `requires_grad=False`；学生主干前向仍保留激活反传，输入投影可以收到穿过
中间冻结层的梯度。DINO 与条件 projector 在这些阶段保持冻结；time 投影在适配阶段低 LR 更新。
原 SA/CA/FFN/norm/time 权重严格检查，只有第一阶段的新接口允许新建。
适配/精调配置的 `require_pixel_init: true` 防止忘记上一阶段权重而从随机接口重新训练。
导出目录中有 flow 元数据时，阶段转换还检查 γ 一致。

8 卡起点为 BF16、microbatch=1、accumulation=4（global batch=32），
gradient checkpointing 开启；AdamW `(0.9,0.95)`、eps=1e-8、矩阵 decay=0.01、
bias/norm 不 decay、clip=1、500-update LR warmup、阶段 cosine 到 0.1、EMA=0.999。
CUDA 多卡保留现有 ZeRO optimizer-state 分片；不是 FSDP2，不分片模型权重。
这是资源起点，不是 8×A30 或 8×910B 显存/性能保证；Ascend 继续使用现有 NPU 路径，需实机验证。

## 准备数据

先按页面/来源/场景划分 train/dev/test，同页不同 crop 不能跨 split。
不要将 Real-CE、ScreenSR、EasyText 的正式 test split 用于训练或调参。
教师数据与真实 GT 数据各有自己的固定 LQ，不能在读取缓存后重新退化 LQ：

```text
preset/datasets/pixel_pdm/
  teacher_lq/   # 已上采样的固定 LQ RGB
  teacher_rgb/  # 对应的 VOSR2 RGB 输出，与 teacher_lq 同名、同尺寸
  gt_lq/        # 真实/合成、可靠配准的 LQ，已上采样
  gt_hq/        # 对应 GT，同名、同尺寸
```

先完成裁剪、几何变换和退化，再为这些 **固定 LQ** 生成教师目标。
固定 VOSR2 checkpoint、VAE 逐通道归一化、posterior mode、DINO 预处理、seed、clip 和色彩后处理；
优先使用未经 wavelet/adain 修正的 RGB。将这些信息、source/group id、crop 坐标、
退化参数/seed、文字框与 OCR 质量标记写入自己的 cache manifest。
本实现消费已准备好的 RGB 缓存，不自动生成教师数据或判断教师错字。
若错误区需要真值修正，应离线选择/修正 clean target，或作为 real-GT source 采样。

`data.sources` 的正 `weight` 是来源抽样概率，与各目录图像数无关；每次先抽来源、
再在来源内均匀抽图和同坐标 crop（有放回）。`samples_per_epoch` 只控制一次 sampler epoch 长度，
`max_steps` 控制训练预算。随机性遵循 DataLoader worker seed，DDP 各 rank 可抽到相同样本。

修改 `warmup.yml` 和 `adapt.yml` 中的数据路径。可在每个 source 加上：

```yaml
text_boxes: path/to/train_boxes.json  # stem -> [[x1,y1,...,x4,y4], ...]
text_crop_prob: 0.6
```

保留原字高的小字，特别是 32–48px、<64px 与跨 p16 边界的样本。
当前文字裁剪沿用已有随机 box 中心采样，不自动进行字高分层或 OCR 过滤。
缺少教师缓存时，可将两个阶段的 `data.dataset_type` 改为 `txt` 并设置已有
`data.train_dataset_config`，先用 HQ + 在线 RealESRGAN 退化跑通；这是 GT-only 对照，
不代表已经验证教师分布迁移。

## 启动、阶段转换与续训

先设置官方 `teacher_checkpoint`（用作初始化；训练时不实例化教师）、
离线环境所需的 `dino.local_repo`/cache，以及数据目录。第一阶段不需要 VAE 文件。
从仓库根目录执行，依赖按项目 `uv sync` 安装：

```bash
# F1，gamma=1：阶段 1
torchrun --standalone --nproc_per_node=8 train_vosr_ablation.py \
  --config configs/pixel_pdm/warmup.yml --run-dir-file /tmp/pdm-f1-warmup-run.txt

# 阶段 2；--init-from 创建新阶段，载入严格 pixel 权重但不继承 optimizer/EMA
torchrun --standalone --nproc_per_node=8 train_vosr_ablation.py \
  --config configs/pixel_pdm/adapt.yml \
  --init-from "$(cat /tmp/pdm-f1-warmup-run.txt)/export" \
  --run-dir-file /tmp/pdm-f1-adapt-run.txt

# 同一阶段中断后：原配置 + 该阶段 checkpoint
torchrun --standalone --nproc_per_node=8 train_vosr_ablation.py \
  --config configs/pixel_pdm/adapt.yml \
  --resume path/to/adapt-run/checkpoint-00001000
```

F2 使用 `warmup_gamma2.yml` 与 `adapt_gamma2.yml`，初始化自 F2 warmup。
精调用 `refine.yml --init-from <选定-adapt-run>/export`；若选中 F2，必须先把精调配置的
`pdm.noise_scale` 改为 2。不要用 `--resume` 改冻结范围、LR、γ 或时间分布。
raw checkpoint、`ema/` 和最终 `export/` 都记录 `pixel_flow` 元数据；最终 export 默认是 EMA。
`--resume` 恢复 live 权重、optimizer、EMA 和 update step，不承诺 worker/RNG 的逐样本完全重放。

训练 JSONL 含 `flow`、`pixel_mse`、`timestep_mean`、各组 `lr/*` 与梯度范数。
固定训练样本预览在同一份像素噪声上比较 4/8/16 步，保存到 `previews/`。
这是真实 ODE 输出，但训练样本分数不能当作 held-out 质量结论。

## Held-out dev 验证

把下面配置替换 `validation: null`；dev LQ 也须预先上采样且精确对齐。
训练启动时检查 dev 目录，rank 0 在 step 1、每 `every` 步及最终步执行 4/8/16-step：

```yaml
validation:
  every: 1000
  num_images: 8
  seed: 4321
  data:
    dataset_type: paired
    upscale: 4
    lq_dir: path/to/dev/lq_upsampled
    hq_dir: path/to/dev/hq
```

固定 dev crop/噪声，记录 `validation/{4,8,16}_step_psnr_rgb`，
对比图保存在 `validation/step-XXXXXXXX/`。PSNR 对 clipped RGB 按图像取均值；
只用于开发，不替代文字评测。dev 验证使用 live 权重，最终 export 使用 EMA，报告时标明二者。

## 多步整图推理与文字评测

```bash
for steps in 4 8 16; do
  python inference_vosr_ablation.py --export path/to/adapt-run/export \
    --input path/to/dev/original_lq --output "preset/results/pdm_${steps}" \
    --upscale 4 --tile-size 512 --tile-overlap 64 --steps "$steps" --seed 42
done

python evaluate.py --pred preset/results/pdm_8 --gt path/to/dev/hq \
  --lq path/to/dev/original_lq --ann path/to/dev/Label.txt --ann-ref gt \
  --output preset/results/pdm_8_eval
```

若输入已经上采样，使用 `--upscale 1`。默认使用 export 的 `inference_steps=8` 和 γ。
同一 seed/输入/tile 配置下各步数共用同一初始噪声；每个网络调用的 t 都大于 0。
每次先对整图的当前 `x_t` 切 tile，融合所有 tile 的 x-pred，再更新整图状态：

\[
x_{t_{i+1}}=x_{t_i}+(t_{i+1}-t_i)\frac{x_{t_i}-\hat x_0}{t_i}.
\]

最后一次更新直接返回 x-pred；**采样分母不套用训练 loss floor**。
DINO 每个 LQ tile 只提取一次，缓存到 CPU；不会在每个 ODE step 重算。
主干继续用方形 tile，矩形图片在整图层面补边/切 tile，避免把 `sqrt(N)` 当作任意矩形 token 网格。
改变 tile 大小会改变上下文和 DINO 条件，需要单独评估接缝与小字。

至少记录 OCR-A/CER/1-NED、PSNR/SSIM/LPIPS，以及 HQ 字高 <32、32–48、48–64、≥64px 分组。
统计 teacher 错/student 对与 teacher 对/student 错，检查 p16 周期纹理、平坦区彩噪和接缝。
正式小字分层/paired bootstrap 仍需结合带框标注做额外统计；`evaluate.py` 的使用与依赖见
[evaluation.md](evaluation.md)。普通 FM 的一步 Euler 只在代数上等于 x-pred，
不代表具备训练好的一步生成质量。

## 实现验证

```bash
OMP_NUM_THREADS=1 TORCHDYNAMO_DISABLE=1 python -m pytest \
  tests/test_pixel_flow.py tests/test_pixel_dit.py tests/test_ablation_logging.py \
  tests/test_ablation_data.py -q
```

覆盖一致 clamp 的数学目标、γ/逐样本时间、ODE 方向与正时间端点、冻结主干输入梯度、
分组 LR/decay、教师/GT 来源抽样、整图状态共享与 DINO 缓存，以及小模型实际训练、EMA、
续训、阶段转换、dev 验证和导出。测试使用 CPU 小模型与替身 DINO/退化，不是完整模型效果验证。

在允许 Gloo sockets 的主机上，额外检查双进程 DDP、分组 ZeRO optimizer 和 rank-0 验证：

```bash
VOSR_TEST_DDP=1 OMP_NUM_THREADS=1 TORCHDYNAMO_DISABLE=1 python -m pytest \
  tests/test_pixel_flow.py::test_two_rank_flow_with_zero_groups_and_rank_zero_validation -q
```
