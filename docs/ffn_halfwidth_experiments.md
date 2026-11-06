# FFN 半宽训练与小文字实验运行手册

本套件把 **校准 → OBS/LS 初始化 → 教师输入下的非线性局部重建 → B/C/D 恢复 → 分字高评价**做成可执行实验图。默认原始 VOSR2 为 36 层 SwiGLU、每层 hidden=4096；目标 student 每层保留 2048 个神经元，成对删除 `w12` 两半的对应行及 `w3` 的对应列，导出普通稠密矩阵。`keep_ratio:0.75` 和全宽实验是明确标注的容量对照。

**当前仓库提供的是实现、配置、CPU 验证与可运行命令，不是已经训练完成的半宽模型，也没有“小文字已接近原模型”的测量结果。** 完整 VOSR2 的 NVIDIA/Ascend 显存、吞吐、OCR 和画质仍须在目标机器上测量。CPU 小模型的梯度/恢复测试不能替代真实设备训练。

## 1. 文件与公共接口

| 文件 | 用途 |
|---|---|
| `configs/ffn_halfwidth/matrix.yml` | 42 个具名实验、继承关系、比较对象、预算与种子 |
| `configs/ffn_halfwidth/base.yml` | 算法默认值；实际权重和数据路径由 site 文件填入 |
| `configs/ffn_halfwidth/site.example.yml` | 机器、训练数据、校准数据、困难样本和评价资产的模板 |
| `tools/ffn_experiments.py` | 列表、配置展开、预检、共享依赖、任务认领、启动、续跑、评价 |
| `tools/mlp_neuron_stats.py` | 与源结构匹配的加权协方差校准及来源指纹 |
| `tools/reconstruct_ffn.py` | 教师输入下的非线性局部重建；支持 CUDA/NPU/CPU 与 torchrun |
| `tools/evaluate_text_bins.py` | 严格配对的 OCR 指标、按图像 bootstrap、教师正确实例保留率 |
| `tools/mine_text_hard_cases.py` | 仅从显式训练名单挖掘教师正确/学生错误的小文字 |
| `tools/benchmark_ffn_export.py` | 检查物理宽度，并测量实际推理管线 p50/p95 |

运行时配置均展开为无 `_base_` 的完整 YAML。`manifest.json` 记录 DAG、比较对象、预算、配置 SHA256 和输入/代码指纹。每个节点使用独立目录；训练产生的时间戳 run 目录通过 `run_dir.txt` 记录，成功后用固定 `nodes/<node>/export` 符号链接供后续阶段读取。不会把未结束的 checkpoint 当成完成结果。

## 2. 准备真实路径与数据

使用仓库的 Python 3.10/3.11 和已锁定依赖环境：NVIDIA 执行 `uv sync`；Ascend 在加载匹配的 CANN
环境后执行 `uv sync --group npu`。后续命令使用该环境中的 `python`，例如先激活 `.venv/bin/activate`，
或为命令加 `uv run`。测试依赖由 `uv sync --group dev` 安装。环境细节见 [README](../README.md#dependencies-and-installation)。

先复制 `site.example.yml` 并填写真实路径。模板里的必填 `null` 会使展开或预检失败；工具不会下载缺失模型、生成假数据或自动用其他 checkpoint 顶替。

必填公共资产为原始 dense 教师权重、VAE 本地目录、DINO 本地源码和权重缓存、训练 LQ/HQ 目录及文字框。`group:headsplit` 另需真实 head-split 源 checkpoint 与 `head_stats`；`roi_ocr`、`main_ocr` 另需本地 OCR `config.json` 和 `model.safetensors`。只有开启相应能力的节点才依赖这些可选资产。

训练用 paired LQ **已经放大到 HQ 尺寸**。推理评价用 `eval.*.native_lq_dir` **保留原始低分辨率**，由推理管线执行 upscale；不要把这两个目录混用。训练 `text_boxes` 是本仓库 JSON；评价 `annotations` 是 `evaluate.py` 使用的 PaddleOCR `Label.txt` 格式。文字坐标和字高均须匹配实际 SR/HQ 像素，不要先将小字统一放大到识别器高度再分桶。

富标注的最小示例（坐标仅演示格式，须换成真实标注）：

```json
{
  "page001.png": {
    "alignment_valid": true,
    "lines": [
      {
        "quad": [[32, 40], [232, 40], [232, 80], [32, 80]],
        "transcript": "小文字示例",
        "valid": true,
        "alignment_valid": true
      }
    ]
  }
}
```

四点按左上、右上、右下、左下的连续边界顺序给出；支持旋转/透视四边形。`native_height` 由原始框
计算，不必手填。旧版矩形或裸四边形列表仍可读取。`teacher_correct` 可省略，表示未知；只有核实
教师与 GT 的文字内容后才设为 `false`，不要根据低置信度猜测。`gt_latent` 是全图 HQ 监督消融，
只适用于全部配对图像已核验对齐的数据；行级不对齐样本应使用主线的 ROI 过滤路径。

训练支持明确配比的混合源，例如真实文字、合成文字与自然图像。比例必须按手头实际数据填写，不预设存在某个数据集。无文字自然图像源必须显式写：

```yaml
name: natural
weight: 0.2
lq_dir: /实际路径/自然图像/LQ
hq_dir: /实际路径/自然图像/HQ
text_boxes: null
text_crop_prob: 0.0
allow_missing_text: true
alignment_valid: true
text_sampling:
  mode: image
```

`alignment_valid` 反映真实配对情况。未对齐或无效文字不得作为 GT ROI 监督。可用富标注 `teacher_correct:false` 下调已核实教师错误区域的 KD，默认 `teacher_error_weight:0.1`；不要把低 OCR 置信度自动等同于教师错误。源级 `seed`、`deterministic_crops`、`text_metadata` 必须与混合数据全局设置一致。

主线 `small_text` 采样使用原生 SR 字高边界 `[24,32,48,64,96]`，六桶权重 `[.05,.15,.35,.25,.10,.10]`，保留裁块位置抖动及可见面积限制。`obs_smallcrop`、`obs_text_kd` 和校准权重实验分别改变裁块、损失和统计，避免三种因素一起变化而无法解释收益。

建议提供 `split_manifests.train/validation/test`，每行一个**全局唯一的样本 ID**。只要设置其中之一，就必须全部提供；重复或跨 split 重叠会中止预检。训练名单应包含校准、常规训练和困难训练样本。文件名在不同数据源可能重复时，请使用具有来源前缀的全局 ID。未提供名单时只检查目录，不宣称已经证明样本内容没有泄漏。

## 3. 预算与优化器

| 阶段 | full | pilot | 参数范围与默认学习率 |
|---|---:|---:|---|
| 校准 | 1,024 个 crop，可配置 | 同一校准资产可共享 | FP32 前向；默认 CPU FP64 按层组累积统计 |
| A 局部重建 | 每个所选层 2,000 次更新 | 每层 250 次 | `w12` 与 `w3`，LR=1e-4；教师输入与门控 FFN 残差 |
| B 恢复 | 4,000 optimizer steps | 1,000 | 仅 `blocks.*.mlp.*`，LR=5e-5 |
| C 联合恢复 | 24,000 | 2,000 | 全参数；非 FFN=2e-6，FFN 倍率 10，即 2e-5 |
| D 小文字精修 | 6,000 | 1,000 | 全参数；非 FFN=5e-7，FFN 倍率 10，即 5e-6 |

`smoke` 将 A/B/C/D 各设为 2 步，并将校准样本数限制到至多 8 个，用于真实资产和目标设备上的梯度、布局、加载与保存连通性检查，不用于质量比较。完整模型 smoke 仍需完整模型显存。小模型 CPU smoke 见后文测试命令。

默认全局 batch=16，严格满足：

```text
global_batch = nproc_per_node * batch_size_per_gpu * gradient_accumulation_steps
```

不能整除时提前报错。改变世界大小必须重新展开配置，不能在同一 checkpoint 的续跑中悄悄改变有效 batch。B 的 GT、像素和 OCR 损失关闭；这些实验从 C 开始。主线使用 ROI GT 像素/边缘项，**不隐含开启潜空间 GT 项**，`gt_latent` 是独立实验。A 的额外计算量单独记录：E2E 更新次数相同不代表总 GPU 时间相同，更不代表 teacher 前向次数相同。

42 个实验覆盖以下比较；完整 ID 和 `compare_to` 用 `list` 查询：

| 组 | 关键实验 | 回答的问题 |
|---|---|---|
| 控制 | `dense_control`, `dense_recipe_control`, `k75_capacity`, `k75_recipe_control` | 额外训练、完整新配方与 75% 容量分别贡献什么 |
| 初始化 | `random_none/mean`, `flap_none/mean/lstsq`, `obs_uniform` | 通道选择和输出补偿各自的作用 |
| 校准 | `obs_uniform/text/balanced` | 同一源 crop 上的统计权重是否保护小字 |
| 数据/输出 KD | `obs_smallcrop`, `obs_text_kd` | 字高采样与文字损失加权分别是否有效 |
| 局部恢复 | `recon_uniform/balanced/w3_only/ungated/lr5e5/double_budget` | 非线性基底、gate、数据权重及额外 A 预算 |
| 内部 KD | `update_kd`, `feature_kd`, `internal_kd`, `feature_all_layers` | FFN 更新、残差特征、层位和显存成本 |
| GT/笔画/OCR | `gt_latent`, `roi_charbonnier`, `roi_edge`, `roi_ocr` | 哪种 GT/识别监督修复文字，是否伤害自然图像 |
| 主线消融 | `main_recipe`, `main_ocr`, `main_no_internal`, `main_no_smallcrop`, `main_unweighted_kd` | 关键组成是否确实必要 |
| 优化 | `main_lr_ratio5`, `main_lr_ratio20`, `main_b_all_params` | 先恢复 FFN 与差分学习率是否有效 |
| 渐消独立组 | `fade_direct_control`, `fade6k`, `fade12k` | 全宽起步渐消与直接剪枝，在同一单阶段预算内比较 |
| 二级压缩 | `headsplit_control`, `headsplit_direct`, `headsplit_main` | 在另一源结构上的额外压缩代价 |

Fade 组为独立单阶段 34,000 步对照，使用原始源权重，不接局部重建。6k/12k fade 不能塞进默认 4k 的 B 阶段。pilot 中按总预算同比缩放 fade 步数，展开的配置记录实际数值。Fade 训练期间保留全宽计算，不能当作半宽训练吞吐。

Dense 与 head-split 分别校准。Head-split 局部重建使用相同 head-split 源结构；全局输出教师始终是原始 dense 教师。必须同时报告 head-split 本身相对 dense 的差距，不能把已经下降的 head-split 基准叫作“原始效果”。

## 4. 展开、预检与试运行

以下变量应由操作者设置为实际文件和共享目录；所有机器必须能以相同绝对路径访问仓库、输入数据和运行根目录。

```bash
export VOSR_SITE=/实际共享路径/vosr-site.yml
export VOSR_RUN_ROOT=/实际共享路径/ffn-pilot

python tools/ffn_experiments.py list
python tools/ffn_experiments.py list --select group:core

python tools/ffn_experiments.py materialize \
  --site "$VOSR_SITE" --output "$VOSR_RUN_ROOT" --profile pilot \
  --select all

python tools/ffn_experiments.py preflight \
  --output "$VOSR_RUN_ROOT" --check-runtime

python tools/ffn_experiments.py run \
  --output "$VOSR_RUN_ROOT" --dry-run
```

`materialize --dry-run` 只打印计划数量，不写文件。`run --dry-run` 输出真实 argv 列表和依赖，不认领任务、不建 attempt、不启动 Python/torchrun。已有不同内容的 manifest/config 永不覆盖；修改算法、输入路径、预算或代码后使用新的输出根目录。

如果暂时没有 head-split/OCR 资产，可以显式选出可运行子集；缺失资产的实验不会伪装成已执行：

```bash
python tools/ffn_experiments.py preflight --output "$VOSR_RUN_ROOT" \
  --exclude group:headsplit roi_ocr main_ocr

python tools/ffn_experiments.py run --output "$VOSR_RUN_ROOT" \
  --select obs_balanced --max-jobs 1 --stop-on-error
```

预检检查输入路径、LFS 指针、模型资产、数据配对、自然图像声明、split 重叠、配置指纹和实际可见设备。`run` 先强制做运行时预检；公共资产失败即停止。可选能力缺失只阻塞对应节点和下游，其余独立节点可继续；退出码和最终状态会报告阻塞项。

展开时对 checkpoint/VAE/DINO、训练代码、标注及 split 名单保存 SHA256；图像保存路径/大小/mtime 清单，**不是逐图像内容哈希**。启动 worker 时重新核验强哈希；每个节点开始前还会核验输入清单。校准和重建另写自己的来源指纹。请使用只读的数据版本，不能通过保留 mtime 的方式修改图像后复用旧统计。

完成产物首次发布时记录 SHA256，并在后续调度中检查大小/mtime；需要重新逐字节验产物时运行：

```bash
python tools/ffn_experiments.py preflight --output "$VOSR_RUN_ROOT" --verify-artifacts
```

## 5. NVIDIA 与 Ascend 配置

下面是可配置的起点，不是已测得的性能配方。先通过实际设备 smoke，再启动长程任务。

```yaml
# 例：8 张 NVIDIA A30；完整 DiT + teacher + Adam 显存压力大，应启用 ZeRO。
runtime:
  device: cuda
  nproc_per_node: 8
  global_batch_size: 16
  batch_size_per_gpu: 1
  precision: bf16
  gradient_checkpointing: true
  zero_optimizer: true
```

```yaml
# 例：8 张大显存 Ascend；不要直接假定 CUDA ZeRO 在 NPU 上已通过验证。
runtime:
  device: npu
  nproc_per_node: 8
  global_batch_size: 16
  batch_size_per_gpu: 1
  precision: bf16
  gradient_checkpointing: true
  zero_optimizer: false
calibration:
  moment_device: cpu
  moment_dtype: float64
  layer_group_size: 2
```

NPU 上加速统计可单独消融 `moment_device:model, moment_dtype:float32`；不要请求 NPU FP64 矩阵累积。可在 NVIDIA/Ascend 显存试运行后考虑以下明确标记为近似的配置：

```yaml
calibration:
  moment_device: model
  moment_dtype: float32
  layer_group_size: 8
  max_tokens_per_sample: 256
```

`max_tokens_per_sample:0` 保留精确的全 token 路径；256 是在正权重 token 中无放回均匀抽样并按抽样比例调整权重，降低协方差计算量，**不增加独立图像数量，也不减少教师全网格前向**。抽样可能漏过极少数小笔画 token，需与全 token 参考比较。各层组使用相同抽样 seed；sidecar 记录选择数量和近似策略。`chunk_tokens` 只是计算分块，本身不抽样。

36 层按 group=2 需要 18 次数据重放，group=8 为 5 次；FP32 的 8 层协方差累积器约 512 MiB，此外还有模型与临时矩阵，不能据此保证显存足够。分层统计降低设备驻留内存，但完整协方差产物仍约 2.4 GB，磁盘和 CPU RAM 也要留余量。低显存机器可先减小 microbatch、增加梯度累积并保持 global batch；若 global batch 也改变，应建立新实验，不与原比较混算。

launcher 使用当前进程的完整环境，不重写 `CUDA_VISIBLE_DEVICES`、`ASCEND_RT_VISIBLE_DEVICES`、NCCL/HCCL 或厂商变量。`nproc_per_node>1` 通过 `python -m torch.distributed.run --standalone --nnodes=1` 启动**单机多卡**，没有声称实现跨机器的同一分布式作业。**同一 plan 的 worker 使用相同 global batch、nproc、设备与优化器配置。** CUDA 和 Ascend 使用各自 site/plan 做平台复现，不跨后端搬迁同一个优化器 checkpoint。多机器运行的是下面的共享实验队列。

## 6. 并行机器、认领与续跑

推荐各机器/独立设备组使用动态 ready 队列。所有 worker 读同一个 manifest，原子创建 claim；同一节点只有一个赢家。依赖未完成的节点不会堵住其他独立实验。不要让两个 worker 使用同一组 GPU/NPU。
共享队列要求共享文件系统支持独占创建和原子 rename；最直接的部署是所有 worker 使用同一共享仓库、
模型、数据和运行目录。输入锁也记录大小与 mtime，各自重新 clone 或复制后文件时间不同的本地目录不能
直接冒充同一快照。

```bash
# 机器 A / 设备组 A
python tools/ffn_experiments.py run --output "$VOSR_RUN_ROOT" \
  --worker-id machine-a --max-idle-seconds 300

# 机器 B / 独立设备组 B，使用相同共享目录与绝对路径
python tools/ffn_experiments.py run --output "$VOSR_RUN_ROOT" \
  --worker-id machine-b --max-idle-seconds 300
```

如必须静态分片，所有 worker 使用同一个 `--shards N` 和不同 `--shard-index`；哈希分片认领的是节点，不是把整条依赖链锁在某机器。动态队列通常更不易出现设备空等。超时无就绪任务时 worker 正常退出，之后可以重启；不会将尚在其他机器运行的任务判为失败。
上面的 300 秒用于短试运行；正式长任务可改为 `--max-idle-seconds 3600` 或更长，避免等待其他机器完成
校准时 worker 提前退出。后台会每隔数秒检查就绪任务。

```bash
python tools/ffn_experiments.py status --output "$VOSR_RUN_ROOT"
python tools/ffn_experiments.py run --output "$VOSR_RUN_ROOT" --resume
```

续跑只选择带 `checkpoint.complete` 且有完整训练状态的最新 checkpoint。B/C/D 用 trainer `--resume`；A 用重建器 `--resume`，两者状态格式不同。进入新阶段使用 `--init-from`，初始化新优化器。`export.complete`、模型/管线配置、权重和产物指纹都通过后才解锁下游。失败 attempt 的日志和文件保留；没有完整 checkpoint 时 `--resume` 建立新的 attempt 从该节点初始状态重做，明确记为 `fresh_initialization`，不覆盖旧输出。

不会按时间自动抢走“过期” claim：远程长时间计算不能与崩溃可靠区分。确认原 worker 及子进程都已停止后才手动释放：

```bash
python tools/ffn_experiments.py release --output "$VOSR_RUN_ROOT" \
  --node 实际节点ID --reason '已确认原机器作业终止'
# 远程机器的 claim 还需显式 --force-remote；本机 owner PID 存活时拒绝释放。
```

没有共享存储时，每台机器用自己的 site 和输出目录展开独立计划，并用 `--select` 分工，例如一台运行
`group:initialization group:calibration`，一台运行 `group:reconstruction group:internal_kd`，一台运行
`group:main group:main_ablation`。依赖会在各自计划内部完整展开，但跨机器不会自动复用校准产物；
保留 manifest 和固定验证集，最后集中比较结果即可。

## 7. 训练集困难样本与 D 阶段

`site.hard_data` 只能指向真实、复核过的训练样本源。未设置时，D 明确复用普通 `train_data`，不虚构“已挖掘困难集”。困难样本工具要求显式训练名单，并可重复提供验证/测试排除名单：

```bash
python tools/mine_text_hard_cases.py \
  --student-details /实际路径/train_student_eval \
  --teacher-details /实际路径/train_dense_teacher_eval \
  --train-names /实际路径/train_names.txt \
  --exclude-names /实际路径/validation_names.txt \
  --exclude-names /实际路径/test_names.txt \
  --max-height 64 --output /实际路径/新的训练困难集
```

输出 `boxes.json`、`image_names.txt`、`mining.json`，不复制或生成图像。将前两者作为 `hard_data.sources[*].text_boxes/image_names`，配合原训练 LQ/HQ 目录；其他文字/自然图像源可保留一定权重防止遗忘。**OCR 正确不证明 LQ/HQ 几何对齐**：默认挖掘标注不会自动开启 alignment，只有富 `--boxes` 已明确标记对齐，或操作者已核验配对并显式使用 `--assume-aligned`，才能作为 GT ROI 监督。旧版 OCR detail 不含原生 SR quad 时，须提供具有匹配文本的富 `--boxes`，不能只凭不可靠的行号对应原始框。

本版计划在展开时锁定所有输入，因此不能在已展开计划中途替换 hard_data。验证集可用于选择配方，
验证/测试样本及标签不能并入训练困难集，测试集也不用于选择训练配方。若要把某次 C 模型生成的训练困难样本加入后续轮次，应先完成并审核训练集挖掘，建立带新 hard_data 的独立计划；其“多一轮训练/新增数据”成本单独记录。也可在第一轮开始前，用训练集 probe 的困难样本固定 D 数据。

## 8. 评价与公平比较

先固定原始 dense 教师的推理结果和 OCR detail，再填写 `site.eval.validation/test`。所有比较使用相同推理 seed、VAE/tiles/upscale、图像列表、标注、识别器、大小写规范。原始 teacher checkpoint 未必已经包含 `pipeline.json`，不要将原始权重目录误当成推理 export，也不要用 `dense_control` 继续训练后的输出替代原始教师。

训练结束后可以调用实际推理、`evaluate.py` 和配对汇总的串行流程：

```bash
python tools/ffn_experiments.py evaluate --output "$VOSR_RUN_ROOT" \
  --arm main_recipe --seed 42 --split validation --tag v1 --dry-run

python tools/ffn_experiments.py evaluate --output "$VOSR_RUN_ROOT" \
  --arm main_recipe --seed 42 --split validation --tag v1
```

评价输出也不覆盖；失败后保留日志，用新 `--tag` 重做。验证集用于选择配方和 checkpoint；只对最终预先选定模型运行测试集。`evaluate.py` 保留全图 PSNR/SSIM/LPIPS/DISTS；本套件不把这些指标等同于文字正确性。

独立比较已有 OCR 结果：

```bash
python tools/evaluate_text_bins.py \
  --student-details /实际路径/student_eval \
  --teacher-details /实际路径/dense_teacher_eval \
  --height-buckets 24 32 48 64 96 \
  --bootstrap 2000 --seed 42 --output /实际路径/新的配对报告
```

输出每桶及全体的 pooled CER、OCR-A、1-NED、student−teacher 差值、教师正确实例保留率，附 95% percentile CI。CER 用总编辑数除以总标签字符数；不会平均每个框的 CER。1-NED 优先使用 `pred.norm_text`，旧文件从 `pred.text` 和已知 normalization 重建；缺少预测文本时返回 `null`，不伪造分母。

Bootstrap 单位为**整张图像**，同图所有框及 student/teacher 配对一起抽样。少于 10 张有该桶文字的图像不显示区间，空桶/没有教师正确实例的保留率返回 `null`。这只是最小显示门槛，不意味着 10 张图足够验证结论。多个推理 seed 可分别传多个 detail 文件，先在图像内部平均；该区间不包含训练 seed 的方差。最终需要匹配训练 seed 的独立实验，例如：

```bash
python tools/ffn_experiments.py materialize \
  --site "$VOSR_SITE" --output /实际共享路径/ffn-final-seeds --profile full \
  --select main_recipe dense_recipe_control obs_balanced recon_balanced \
  --seeds 11 42 2026
```

严格模式要求 image/index、标签、SR 字高完全对应，并校验可用的识别器设置；新格式还比较原生 SR `quad` 坐标（允许有限浮点误差）和 `sr_size` 输出尺寸，旧格式缺少几何时明确记录 `geometry_verified:false` 及未核验实例数，不声称已证明同一文字框。`--allow-intersection` 仅用于显式诊断不完整输出，报告每文件排除量；不要用它筛掉失败图片后宣称更好。近似原模型的验收容忍度应在测试前写下，使用小字 CER 差值区间、教师正确保留率和固定局部可视化共同判断；本套件不预置必然达标的结论。

物理宽度及实际管线基准：

```bash
python tools/benchmark_ffn_export.py \
  --export /实际路径/完成的半宽export \
  --input /实际路径/原生LQ图像.png \
  --device cuda --expected-width 2048 \
  --warmup 5 --iterations 20 --output /实际路径/新的benchmark.json
```

Ascend 使用 `--device npu`。同硬件、相同精度和输入尺寸同时测 dense/half-width；FFN MACs 减半不等于完整管线延迟减半，VAE、DINO、Attention 与内存访问仍存在。75%/dense 对照不要使用 `--expected-width 2048`。

## 9. 本地验证与边界

### 本次实现的验收记录

以仓库基线 `1822ec5e0b86dce0a9ce13e379efa9006c20a0b8` 为起点，受影响的 16 个测试文件合计：
**328 passed、6 skipped、1 deselected**。执行环境为 Python 3.12、PyTorch 2.5.1+cpu；这是源码 CPU
验证环境，不替代项目正式支持的 Python 3.10/3.11、CUDA 或 Ascend 环境验收。

6 项跳过为需要 Unix socket 的 DataLoader worker 测试与未开启的 Gloo 多进程测试；1 项未选中为需要
本地通信服务的 W&B tracker 测试。不能将这些项目计作已通过。其余验证覆盖真实 tiny DiT 梯度、半宽布局、
统计/重建、文字 ROI 饱和像素梯度、EMA、checkpoint 完整性、中途精确恢复、历史 optimizer 迁移、
最终 checkpoint 只重试导出、全部 42 个实验的配置接口以及严格配对评测。VAE/DINO 的集成测试使用小型
替身，不声称验证了完整预训练编码器质量。

复现本轮选择的测试：

```bash
OMP_NUM_THREADS=1 TORCHDYNAMO_DISABLE=1 python -m pytest -q \
  tests/test_ffn_distillation.py tests/test_ffn_utilities.py \
  tests/test_text_roi_training.py tests/test_ffn_reconstruction.py \
  tests/test_ffn_experiments.py tests/test_evaluate_text_bins.py \
  tests/test_ablation_data.py tests/test_ablation_core.py tests/test_ablation_model.py \
  tests/test_ablation_mlp_prune.py tests/test_ablation_head_split.py \
  tests/test_ablation_logging.py tests/test_ablation_res1024.py \
  tests/test_pixel_flow.py tests/test_pixel_losses.py tests/test_evaluate.py \
  -k 'not test_trackers_write_scalars_and_images_same_step'
```

### 分项检查与真实设备验收

基础 planner/evaluator 测试不加载 GPU 模型：

```bash
python -m pytest -q tests/test_ffn_experiments.py tests/test_evaluate_text_bins.py
```

梯度、门控更新、真实窄矩阵布局、小模型校准/局部重建、EMA 导出及精确续跑的 CPU 测试：

```bash
python -m pytest -q tests/test_ffn_distillation.py tests/test_ffn_reconstruction.py \
  tests/test_ablation_mlp_prune.py tests/test_text_roi_training.py tests/test_ffn_utilities.py
```

针对本机真实设备，另展开 `--profile smoke`，先 `preflight --check-runtime`，再选择一个主线 arm 完整走 A/B/C/D；检查梯度有限、FFN 参数确实更新、teacher 冻结、mask/ROI 不为空、末尾 export 物理宽度为 2048。

真实两 rank Gloo/NCCL/HCCL 和多 DataLoader worker 需要允许进程共享内存、锁与 socket 的环境。当前受限 CPU 沙盒不能运行这些 IPC/socket 路径，不能把跳过的多进程测试写成通过。支持环境下显式开启：

```bash
VOSR_TEST_DDP=1 python -m pytest -q tests/test_ffn_reconstruction.py
```

正式长程训练前仍需完成目标 GPU/NPU 的分布式 smoke、显存/吞吐记录和一次同 checkpoint 的续跑一致性检查。本实现没有执行未提供的真实权重/数据实验，没有发布模型，也没有记录任何虚构 CER 或加速结果。
