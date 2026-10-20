# VOSR2 no-DINO / no-CA, DyDiT-SDT and hourglass token-merging ablations

These are **opt-in, one-step teacher-preserving fine-tuning experiments**. The
upstream `LightningDiT`, original FM/RCGM trainers, and inference scripts are not
modified. This is not a reconstruction of the unpublished VOSR2 data pipeline or
original training recipe, and text-quality retention has not been demonstrated.

## Configurations

| Config in `configs/ablations/` | Student during training | Exported student |
| --- | --- | --- |
| `no_dino_no_ca.yml` | No DINO features, no CA modules or feature projector; retain LQ latent concatenation | Physically no CA/projector weights; inference never loads DINO |
| `dydit_sdt.yml` | DINO/CA retained; global MLP budget; deterministic learned-capacity top-k with STE | Same capacity/ranking policy; optional sparse MLP gather/scatter |
| `no_dino_fade.yml` | Optional transition: freeze CA/projector, scale CA residuals from 1 to 0 over 5,000 optimizer steps | Final export removes CA/projector weights |
| `hourglass.yml` | U-shaped 2x2 token merging: blocks 0/35 on the full grid with their DINO CA, 1-34 merged, full-grid bypass, student DINO at 224; merge curriculum and hidden-state KD | Same static structure; 26.2% of the teacher's DiT MACs |
| `hourglass_align.yml` | Optional stage 0: only the merge/unmerge layers train, at the final structure | Initializes `hourglass.yml` via `student_checkpoint` |
| `hourglass_2_32_2.yml`, `hourglass_bypass_only.yml`, `hourglass_p4.yml` | Variant and controls of the hourglass, see [Hourglass token merging](#hourglass-token-merging) | 29.7%, 22.7%, 22.7% |
| `hourglass_pyramid_4.yml`, `hourglass_pyramid_6.yml` | Two-level hourglass: 4 or 6 full-grid blocks, the middle 12 or 18 merged once more (64 tokens), see [Two-level (pyramid) variant](#two-level-pyramid-variant) | 24.9%, 25.0% |
| `hourglass_sft_min.yml` | Lowest-compression hourglass with the SFT objective of `loss_all.yml`: blocks 3-8 and 28-33 merged, the teacher's DINO input, see [Lowest-compression check](#lowest-compression-check) | 78.0% |
| `no_dino_sft.yml` | The objective of `loss_all.yml` on the no-DINO/CA student, initialized from a `no_dino_fade.yml` export (set `student_checkpoint`); dense control of `mixres_route.yml` | No CA/projector weights |
| `mixres_route.yml` | `no_dino_sft.yml` with learned mixed resolution: in blocks 2-33 a router keeps the top share of 2x2 groups on the full grid and merges the rest; the share is drawn from 25-100% every step, see [Mixed-resolution routing](#mixed-resolution-routing-mixres_routeyml) | Same weights for every share; `--keep-ratio` at inference (default 0.75: 82.1% of the dense no-DINO DiT) |
| `dualres_min.yml`, `dualres_2_34.yml` (`dualres_2_34_no_detail.yml`) | Hourglass plus a narrow full-grid detail stream next to the merged blocks; `dualres_2_34` also pools the DINO tokens 2x2 for its merged blocks (`coarse_cond_pool: 2`), `dualres_min` keeps all 1024, see [Dual resolution, LQ-detail routing and zero input noise](#dual-resolution-lq-detail-routing-and-zero-input-noise) | 79.4%, 28.8% (27.5%) |
| `mixres_route_lq.yml`, `mixres_f4_lq.yml` | `mixres_route.yml` ranked by LQ-latent detail instead of a learned router; the f4 version keeps 4x4 groups whole and merges the rest 4x4 (259 tokens at share 0.2) | Elastic like `mixres_route.yml`; 22.9% at share 0.2 |
| `allmerged_sft.yml`, `allmerged_sft_zero_noise.yml` | All 36 blocks on 2x2-merged tokens on the SFT objective, with and without input noise (`student.noise_input: zero`, x0 = -velocity) | 23.2% |
| `la_*.yml`, `calib/`, `probe/` | The re-plan after `hourglass_sft_min`: every run starts from the trained `loss_all.yml` export, with a step-matched dense control, calibrated KD / OCR weights and probes, see [Re-plan after `hourglass_sft_min`](#re-plan-after-hourglass_sft_min-the-la_-lineage) | 100% control; 79.4%, 78.0%; 23.2-26.6% |
| `base_sft.yml`, `loss_lpips.yml`, `loss_ocr_kl.yml`, `loss_all.yml` | Base student (dense, DINO/CA kept) trained by plain SFT without the teacher term: GT, plus LPIPS, KL-OCR or both, see [Loss ablation configs](#loss-ablation-configs) | Teacher architecture |
| `sft_sparse_attn_lpips.yml` | As `loss_lpips.yml` with block-sparse self-attention (`attn_type: sparse`), see [Block-sparse self-attention](#block-sparse-self-attention-studentattn_type) | Teacher architecture, sparse self-attention |
| `res1024.yml` and `res1024_*.yml` | Dense student on 1024 crops (one 1024 tile at inference), extrapolated RoPE, teacher tiled at 512; each `res1024_*.yml` changes one point, see [1024 resolution](#1024-resolution-res1024yml) | Teacher architecture |
| `head_split_f2.yml`, `head_split_f4.yml` | Self-attention token reduction by head split: local heads (3x3 attention mass > 0.5) become 7x7 depthwise convs, the other heads attend among 2x2 (4x4) pooled tokens; CA and MLP keep every token. KD from the dense teacher on Real-CE pairs (`data.dataset_type: paired`), cosine LR, EMA, see [docs/attn_head_split.md](attn_head_split.md) | Self-attention on 25% (6.25%) of the tokens: 78.2% (74.6%) of the teacher's DiT MACs; release export `weights/head_split_f2` |
| `pixel_l2p.yml`, `pixel_l2p_unet.yml` | `loss_all.yml` in pixel space after L2P: no VAE, 16 px patchify (or the decoder's own down path) in, the MicroDiffusionModelTop3SCWide U-Net out, DINO/CA kept, blocks 5-30 frozen, see [Pixel-space student](#pixel-space-student-l2p-pixel_l2pyml) | No VAE; 69.7% of the latent pipeline's MACs |

The full frozen VOSR2 **teacher still uses DINO during training**, including the
no-DINO student experiment. Removing the teacher encoder would change the target
and is not equivalent. The fade config is an extra option for gradual removal;
the strict no-CA config removes the branch from student construction onward.

## Preparation and launch

Use the upstream VOSR environment (Python 3.10+ and PyTorch 2.x recommended).
Set `teacher_checkpoint`, `vae_path`, and `data.train_dataset_config` in
`configs/ablations/base_vosr2.yml`. The checkpoint may be an exact clean
`.safetensors` file, or the VOSR2 directory containing `checkpoints/ema_model.safetensors`.
Do not point at an arbitrary multistep model and assume it is a one-step teacher.

`auxiliary_time_cond: auto` detects whether the checkpoint has `r_embedder`
weights. Other architecture fields must match the checkpoint. Loading rejects
unexpected keys, missing backbone weights, and shape mismatches; only regenerated
RoPE buffers, deliberately removed CA weights, and newly initialized routers are
allowed exceptions. No random backbone weights are silently accepted.

Use the same two-level TXT format as the upstream `dataset_type: txt` trainer.
The base config points to `configs/train_txt/train_dataset_txt.txt`; edit its
entries to reference HQ image-list TXT files and integer repeat counts:

```text
# image-list TXT, repeat count (defaults to 1 when omitted)
data/dataset_A.txt, 2
data/dataset_B.txt, 1
```

Each referenced list, for example `data/dataset_A.txt`, contains one HQ image path
per line (no JSON, LQ column, weights, comments, or blank lines inside image lists):

```text
/datasets/HQ/sample_0001.png
/datasets/HQ/sample_0002.jpg
```

Both list paths and image paths are absolute or relative to the **working
directory**, normally the repository root, just like upstream. They are not
relative to the YAML or TXT file. Repeat counts duplicate the entire list; they
are nonnegative integers, not normalized probabilities. Zero disables a list,
and the combined training dataset must be nonempty. The ablation trainer uses
`data.dataset_type: txt`; WebDataset/tar inputs are not supported by this trainer.

The existing `TxtPairDataset` loads HQ images, enlarges images smaller than the
crop size, and takes random crops without mirroring/flipping text. Its `[0, 1]`
HQ batches go through the existing `RealESRGAN_degradation` with
`dataloaders/params_realsr.yml`, exactly as in the upstream trainers. LQ is degraded
online and resized back to the HQ crop size (`resize_bak=True`); HQ and LQ are then
normalized to `[-1, 1]` before VAE encoding. Optional `training.gt_weight` supervision
uses that same HQ crop. The teacher and student still share identical LQ latents
and noise. `resolution: 512` remains an experiment setting, **not a claim about
VOSR2's original training resolution or unpublished training data**.

This replaces the previous `data.manifest` JSONL interface. Remove `data.manifest`
and `data.upscale` from custom configs and use `data.train_dataset_config` with
`data.dataset_type: txt`. The degradation YAML controls the downsampling scale
(currently 4); exports use that scale as the default inference upscale, which can
still be overridden with `--upscale`. Old JSONL-run checkpoints cannot be resumed
with a changed config; to reuse their model weights, set `student_checkpoint` and
start a new run/output directory.

```bash
# Two requested independent experiments (run sequentially, or allocate separate GPUs).
torchrun --nproc_per_node=8 train_vosr_ablation.py \
  --config configs/ablations/no_dino_no_ca.yml

torchrun --nproc_per_node=8 train_vosr_ablation.py \
  --config configs/ablations/dydit_sdt.yml

# Optional gradual removal from the pretrained model.
torchrun --nproc_per_node=8 train_vosr_ablation.py \
  --config configs/ablations/no_dino_fade.yml
```

For one device use `python` instead of `torchrun`. Effective batch size is
`world_size * batch_size_per_gpu * gradient_accumulation_steps` (default 8 on
8 GPUs). CUDA DDP uses optimizer-state sharding via `ZeroRedundancyOptimizer` by
default. Non-reentrant activation checkpointing is off by default; set
`training.gradient_checkpointing: true` when activations do not fit (it may be changed on
resume). It is not FSDP and does not shard model weights/gradients. Memory fit and throughput must be measured on your
hardware; teacher + DINO + VAE also occupy memory. There is no validated Ascend/NPU
path, fp16 training, EMA, or automatic validation set in this new trainer. On NPU the
trainer and `inference_vosr_ablation.py` turn TorchDynamo off, so the `@torch.compile`
functions in `models/` run eagerly: torch_npu 2.5.1 registers no inductor backend for
`npu`, so otherwise each compiled frame fails once per rank and falls back to eager anyway. LPIPS and
KL-OCR losses are opt-in, see [LPIPS and KL-OCR losses](#lpips-and-kl-ocr-losses). Losses and previews are logged through Accelerate to TensorBoard and
W&B, with local JSONL/PNG copies as described below.

An optional offline DINO repository can be set with `dino.local_repo`; its weights
must also be available in the configured torch.hub cache.

## Loss tracking and intermediate images

The trainer uses `accelerate.logging.get_logger` for rank-aware console logs and
Accelerate trackers for TensorBoard/W&B. Existing DDP, gradient accumulation,
precision and optimizer sharding are still managed by this trainer; tracker setup
does not wrap the model or shard the data loader a second time. Launch commands
remain the same (`python` or `torchrun`). Run `uv sync` to install the pinned
dependencies, including `tensorboard==2.21.0` and the existing Accelerate/W&B ones.

The inherited defaults are:

```yaml
training:
  report_to: [tensorboard, wandb]  # or tensorboard, wandb, none
  tracker_project_name: vosr-ablation
  run_name: null                 # defaults to the output directory name
  wandb_mode: offline            # override with WANDB_MODE=online
  log_every: 10
  preview_every: 500             # 0 disables image generation
  preview_num_images: 2
  preview_seed: 1234
```

`loss`, `kd`, `dense_kd`, `gt`, `feature_kd`, `lpips`, `ocr_kl`, `budget`, `mlp_keep`, target keep ratio,
CA scale, learning rate, router learning rate (when enabled), and pre-clipping
gradient norm are logged at optimizer steps. Loss components are unweighted;
`loss` is the weighted total. Disabled GT/dense/feature/LPIPS/KL-OCR terms are recorded as zero.
Loss/keep statistics average across accumulation micro-batches and ranks for the
current optimizer step, not across the last `log_every` steps. Scalars are also
recorded at step 1, preview steps, and the final step.

Routing experiments additionally log `expected_mlp_keep` (mean sigmoid probability),
`mlp_keep_gap` (actual minus expected), and `router/layer_XX/expected_keep` /
`router/layer_XX/actual_keep` for each block. `stochastic_mlp_keep` is retained only
for legacy Gumbel routing; new top-k runs use the neutral `mlp_keep` name because
their masks are deterministic. During dense warmup the actual keep is deliberately
1 even if predicted capacity is smaller. Afterwards capacity rounding bounds the
top-k actual/expected gap by 0.5/N, where N is the tokens per image (1024 at the
default crop/patch size). This bound does not imply the learned capacity has
already reached `target_keep`.

Previews run at step 1, each `preview_every` steps and the final step. The first
N training samples are cropped and degraded once using a fixed seed, cached on
CPU, then reused with fixed latent noise. Each PNG has four labeled columns:
**LQ | Student SR | Teacher SR | HQ**. Both models see identical LQ/noise.
Previews use deterministic evaluation routing (dense masked MLP execution), and
record `preview/deterministic_mlp_keep`, `preview/expected_mlp_keep`,
`preview/keep_gap`, and `preview/layer_XX/actual_keep`. The preview gap compares
actual and expected keep on the **same inputs**; comparing preview keeps to a
training batch also includes image/noise differences. They restore student train/eval mode,
sparse-eval settings and Python/NumPy/Torch RNG states afterwards. Only rank 0
renders/logs images; the other ranks wait for its completion. These are training
sample previews, not held-out validation scores. Image generation adds inference
and VAE decode time/memory; reduce `preview_num_images` or disable previews if needed.

Each fresh run writes into its own `<run>` = `YYYYmmdd-HHMMSS` subdirectory of
`output_dir` (see [Checkpoints, resume, and inference](#checkpoints-resume-and-inference)).
For the DyDiT config the files are:

- `exp_vosr/dydit_sdt/<run>/config.json`: the resolved config.
- `exp_vosr/dydit_sdt/<run>/metrics.jsonl`: scalar records, including the optimizer step.
- `exp_vosr/dydit_sdt/<run>/logs/vosr-ablation/`: TensorBoard events.
- `exp_vosr/dydit_sdt/<run>/logs/wandb/`: W&B run data (offline by default).
- `exp_vosr/dydit_sdt/<run>/previews/step-00000500/sample-00.png`: local comparison image.

```bash
# One TensorBoard for all runs of the experiment
tensorboard --logdir exp_vosr/dydit_sdt --port 6006

# To log online instead of offline:
wandb login
WANDB_MODE=online torchrun --standalone --nproc_per_node=8 \
  train_vosr_ablation.py --config configs/ablations/dydit_sdt.yml

# Or sync one completed offline run later:
wandb sync exp_vosr/dydit_sdt/<run>/logs/wandb/offline-run-<timestamp>-<id>
```

`report_to: none` disables both trackers but keeps console, JSONL and local PNG
output. `preview_every: 0` independently disables previews. On `--resume`, which
continues in the checkpoint's run directory, JSONL appends, TensorBoard removes
events after the restored checkpoint step, and W&B starts a new run using the
restored optimizer-step axis (also recorded in the run config). Existing preview
PNGs at repeated steps are replaced. Do not change the model/data/optimization
config when resuming. Logging and preview settings may change, so existing
compatible training checkpoints can enable the new trackers.

## Objective and routing scope

Training uses the deployment endpoint `t=1, r=0` only. Given identical LQ latent
and noise, the student matches the frozen one-step teacher's velocity:

```text
input = concat(LQ_latent, noise)
L_KD = MSE(student(input, 1, 0), teacher(input, 1, 0))  # weight training.kd_weight, default 1
SR_latent = noise - student_velocity
L_GT = MSE(SR_latent, GT_latent)                 # optional, default weight 0
SR = VAE_decode(SR_latent)                       # only when a pixel loss is enabled
L_LPIPS = LPIPS(SR, HQ)                          # optional, default weight 0
L_KL_OCR = KL(OCR(HQ) || OCR(SR))                # optional, default weight 0
L_budget = (mean_rank_layer(expected_keep) - target_keep)^2  # per micro-batch
```

This preserves the one-step task; it does not claim to train a valid arbitrary-time
flow or multistep/RCGM model. Only one-step inference is provided for these exports.

`kd_weight: 0` drops the teacher term (plain SFT on the HQ crop; at least one of the
GT, LPIPS or KL-OCR weights must then be positive). Without feature distillation or
routed dense distillation the teacher is then not loaded at all: `kd` is logged as zero
and previews show LQ, student and HQ only. `L_GT` is the upstream `loss_fm`
(`vosr.py`) sampled only at `t=1`, where the noise input carries no information about
HQ, so MSE alone pulls the one-step output toward the posterior mean and blurs;
upstream obtains its one-step models by distillation instead.

Routing is an independent adaptation of the **SDT component** of
[DyDiT](https://github.com/alibaba-damo-academy/DyDiT), informed by its
`DyDiT/models.py`, `DyDiT/dynamic_model.py`, and `DyDiT/loss.py`. It is **not the complete DyDiT**:
there is no timestep-dependent width/head routing (TDW). Attention retains all
spatial tokens and unchanged RoPE positions. Each block predicts a spatial MLP
mask from its residual stream, using the same Linear-ReLU-Linear router shape.

### Global budget and layer allocation

`budget_scope: global` averages token probabilities over the local batch and
layers, then uses an **autograd-aware all-reduce across ranks before squaring**.
For example, two images on different ranks keeping 0.55 and 0.95 satisfy a 0.75
budget with zero loss, even at `batch_size_per_gpu: 1`. Layers may also choose
different capacities. The trainer uses equal local batches and fixed crop/token
counts (`drop_last=True`), so averaging the rank means gives the global mean.

Previously the global scope only averaged layers: each rank squared its own
micro-batch error. At local batch size one, averaging those losses equals the
global mean's squared error **plus the variance of image keep rates**, penalizing
cross-image allocation. DDP gradient averaging and increasing gradient
accumulation alone do not remove that variance term. The autograd-aware reduce
also communicates in backward; combined with DDP averaging it has the same
gradient scale as a loss over the concatenated global micro-batch. No extra
world-size multiplier is needed. Every rank must participate, including during
`no_sync()` micro-batches; rank-0-only previews do not compute this budget loss.

The budget batch contains `world_size * batch_size_per_gpu` images. Gradient
accumulation averages **separate global micro-batch losses**, not a single squared
mean over the entire accumulation window. With one rank and local batch size one
the budget still pulls each image's mean toward the target; a larger actual batch
is required for cross-image allocation in that case. `budget_scope: layer`
preserves the legacy local per-layer penalty.

All blocks in this backbone have identical MLP dimensions and token counts, so
mean MLP keep is also the fraction of the MLP matmul work retained. DyDiT's
original DynamicLoss instead budgets attention and MLP FLOPs together, including
its TDW masks. We keep `target_keep_ratio` in **MLP token units**, rather than
silently changing an existing 0.75 setting into 0.75 total model FLOPs. For a
fixed input size and excluding routing overhead, a whole-model estimate would be
`(F_fixed + F_MLP * mean_keep) / (F_fixed + F_MLP)`, where self-attention,
cross-attention and other unrouted work remain fixed. No measured FLOP or latency
claim is made. A heterogeneous-width/depth extension would need MLP cost weights.

### Matching training and inference decisions

`student.router_config.routing_mode: capacity_topk` is the new experiment default:

1. Each layer/image computes scores and `p = sigmoid(logits)`.
2. Its own capacity is `k = floor(sum_tokens(p) + 0.5)`; this is learned from that
   layer's outputs, **not a fixed identical capacity for all layers**.
3. Keep the k highest scores, with stable token-order tie breaking. There is no
   competition between different images for slots, no routing noise, and no
   train/eval change in the capacity or ranking rule.
4. Training uses `hard_mask + (p - p.detach())` for a straight-through surrogate;
   the discrete top-k indices/count alone do not provide router task gradients.
   The budget also differentiates through p. The STE is a surrogate, not the
   exact derivative of hard top-k.

This fixes the `p=0.75` counterexample: with 1024 tokens, both training and
inference keep 768 rather than inference keeping all 1024. Counts can round to
zero or N; there is no per-layer minimum-keep penalty. `temperature=1` and
`threshold=0.5` are required in this mode; capacity is not set by a threshold.
Dense warmup and the optional dense teacher-matching pass intentionally bypass
the mask. After warmup, training and inference use the same routing function for
the same inputs/parameters/precision. Sparse versus dense MLP kernels can still
introduce floating-point differences in later layers.

The small router's Linear-ReLU-Linear network runs in FP32 with autocast disabled
and its input explicitly cast to FP32. The trainer and export loader retain FP32
router parameters; do not downcast those parameters when adapting deployment.
Sigmoid probabilities and ranking scores remain FP32; the returned mask follows
the residual input dtype. Casting Linear outputs to FP32 only after BF16 execution
loses small score differences near the initial bias (about 4.595 at keep=0.99),
creating ties that stable sorting resolves by spatial order. Genuine equal FP32
scores still use the deterministic tie-break. This fix does not recover precision
already lost in upstream features or guarantee whole-model FP32/BF16 equality.
The initial weight std, keep bias, and stable sorting policy are unchanged.

This is a deliberate change from DyDiT's Gumbel-sigmoid training / threshold
inference, whose train/eval keep mismatch is also possible in the original code.
It is **not an exact reproduction of the original SDT training algorithm** and
does not demonstrate that already-trained stochastic routers have useful top-k
rankings. Fine-tune using the new policy before evaluating its quality. Matching
routing removes the policy mismatch; it does not guarantee convergence to the
budget or preservation of text quality.

The example starts near all-keep, uses 500 dense warmup steps, then ramps the
expected MLP keep budget to 0.75 over 4,500 steps. An optional dense sandwich
teacher-matching pass has weight 0.1. Router initialization occurs after backbone
initialization, so pretrained weights and the keep bias are not overwritten.

**Training still executes dense MLPs.** In no-grad evaluation, selected tokens can
be gathered, processed by the MLP, and scattered back; unselected tokens retain the
residual stream. `--dense-mlp` instead computes a dense MLP and applies the same
mask. Neither option compresses attention tokens. A 0.75 expected training keep
budget is not a guarantee of 25% total FLOP reduction or wall-clock speedup.
Inference keep follows each learned capacity up to integer rounding, which may
still miss the requested target if optimization has not converged. Inspect
`deterministic_mlp_keep` in inference logs. Stable sorting and sparse
gather/scatter/nonzero require target-backend support; measure their overhead.

### Existing checkpoints and legacy reproduction

Old exports without `routing_mode` still use `gumbel`: stochastic masks during
training and deterministic thresholds during evaluation. Resolved training
configs without `budget_scope` still use the old per-layer loss. These fallbacks
preserve legacy policy/config compatibility; they do not opt into top-k/global
budgeting. The FP32 router fix also applies to legacy Gumbel routers.
New shipped YAMLs explicitly set both `capacity_topk` and `global`. Model exports
persist the router configuration, so reloading a new export preserves top-k.

For a controlled new experiment, use the updated `dydit_sdt.yml` with a fresh
output directory. To reuse old model weights, set `student_checkpoint` to the
old checkpoint's `model.safetensors`; parameter names/shapes are unchanged. This
starts a **new** optimizer and step schedule, including dense warmup. Switching
policy only at inference is not a substitute for training with that policy.
`--resume` rejects a changed routing/budget configuration. To continue a legacy
run with its saved policy/config, pass its resolved `config.json` as `--config`
together with `--resume`. No router learning-rate or budget-weight defaults are changed by this
fix; those remain independent optimization experiments.

Existing weights and optimizer states remain loadable. Resuming a pre-fix
`global` run now uses the corrected cross-rank objective, and BF16 runs use FP32
router computation; this deliberately changes their subsequent trajectory even
if the saved config is identical. For an old-behavior control, use the old code
revision. FP32 checkpoint storage alone never implied FP32 forward computation.

## LPIPS and KL-OCR losses

Both are off by default and cost nothing unless their weight is positive. They
decode the student's one-step prediction with the Qwen VAE (with gradients, under
the training autocast) and compare it with the same HQ crop, RGB in `[-1, 1]`.
The loss networks are frozen and run in fp32; only the student is trained.
The weights below are examples, not tuned values:

```yaml
training:
  lpips_weight: 0.1         # > 0 enables LPIPS
  lpips_net: vgg            # pyiqa LPIPS v0.1 trunk: vgg (default) | alex
  lpips_model_path: null    # optional local LPIPS linear-layer .pth
  ocr_kl_weight: 0.1        # > 0 enables KL-OCR with the top-level `ocr` recognizer
ocr:
  type: ppocr               # PP-OCRv5_server_rec, the only type
  checkpoint: PaddlePaddle/PP-OCRv5_server_rec_safetensors  # local dir/file or HF repo id
  strip_height: 64          # HQ pixels
  strip_stride: 32
  temperature: 1.0
```

**LPIPS** builds pyiqa's `LPIPS` network directly, the same network and weights
as pyiqa's `lpips-vgg` / `lpips` metrics (identical values on the same images). It
does not use `pyiqa.create_metric`, because that also switches cuDNN to
deterministic, non-benchmark mode for the whole process. The weights are
torchvision's `vgg16-397923af.pth` (or `alexnet-owt-7be5be79.pth`) in
`<hub dir>/checkpoints/` and pyiqa's `LPIPS_v0.1_vgg-a78928a0.pth` (or
`LPIPS_v0.1_alex-df73285e.pth`) in `<hub dir>/pyiqa/`. In this trainer the torch hub
dir is `dino.cache_dir` (`preset/ckpts/torch_cache`), set when DINO loads; on
offline machines copy the files there. Rank 0 downloads first, the other ranks wait.

**KL-OCR** runs one recognizer on the prediction and on HQ and minimizes
`KL(p_HQ || p_SR)` of its per-frame CTC distributions: summed over classes,
averaged over frames, times `temperature ** 2`. The HQ branch has no gradient.
Training crops carry no text boxes, so both images are cut into the same
overlapping full-width strips; any text line up to `strip_height - strip_stride`
pixels tall lies inside one strip. Each strip is resized to the recognizer's line
height. This is a heuristic; match `strip_height` to the text sizes in your data.

The recognizer (`ocr.type: ppocr`, the only type) is a PyTorch port of
PP-OCRv5_server_rec, the model evaluate.py runs through PaddleOCR
(`models/ocr_recognizers.py`). `ocr.checkpoint` is the
`PaddlePaddle/PP-OCRv5_server_rec_safetensors` repo id (downloaded with
`huggingface_hub`) or a local copy of its directory or `model.safetensors`.

The PP-OCR port feeds BGR, like PaddleOCR training and PaddleX's Paddle-inference
path. On CPU it matches transformers 5.19's port of the same weights (largest
probability difference 1.1e-6) and PaddleOCR 3.7 / Paddle 3.2 inference (largest
difference 9.2e-5, identical text on seven rendered Chinese and English lines).
Training does not import Paddle.

Measured on CPU at a 512x512 crop, batch 1: the VAE decode is 537 GMACs forward and
keeps 6.1 GiB of activations for backward in fp32 (3.6 GiB under bf16 autocast);
KL-OCR with PP-OCRv5_server_rec (15 strips of 48x384 per image) is 212 GMACs forward
for both images and keeps 2.7 GiB. Backward is extra. GPU/NPU memory and speed have
not been measured.

The shipped YAMLs leave these keys commented out so running jobs can still be
resumed. Enabling a loss changes the training configuration, so start a new output
directory (set `student_checkpoint` to continue from earlier weights) instead of
`--resume`.

### Loss ablation configs

`base_sft.yml` is plain SFT: it inherits `base_vosr2.yml`, sets `kd_weight: 0` (the
teacher is not loaded) and turns on `gt_weight: 1.0`. The loss configs inherit
`base_sft.yml` and change only loss weights and `output_dir`, and
`sft_sparse_attn_lpips.yml` also sets `student.attn_type: sparse`
(`tests/test_ablation_core.py` checks this). The student keeps the teacher's dense
architecture with DINO and CA, starts from the same VOSR2 weights and uses the same
data, schedule and optimizer. `base_sft.yml` is the control; `base_vosr2.yml` stays
the teacher-KD baseline.

| Config | `kd_weight` | `gt_weight` | `lpips_weight` | `ocr_kl_weight` | `attn_type` |
| --- | --- | --- | --- | --- | --- |
| `base_vosr2.yml` (teacher KD) | 1 | 0 | 0 | 0 | full |
| `base_sft.yml` (control) | 0 | 1.0 | 0 | 0 | full |
| `loss_lpips.yml` | 0 | 1.0 | 0.02 | 0 | full |
| `loss_ocr_kl.yml` | 0 | 1.0 | 0 | 0.0005 | full |
| `loss_all.yml` | 0 | 1.0 | 0.02 | 0.0005 | full |
| `sft_sparse_attn_lpips.yml` | 0 | 1.0 | 0.02 | 0 | sparse |

`loss_lpips.yml` is the full-attention control of `sft_sparse_attn_lpips.yml`.

The weights balance gradients, not loss values. Without `kd`, AdamW makes the step
size nearly independent of the overall loss scale; what the weights set is the
relative pull of each loss. The LPIPS and KL-OCR weights give the same step-0 gradient
norm on the student's velocity output as `gt_weight: 1.0`: the median ratio over 20
ScreenSR `HR_512` text images, with the training degradation and the real VOSR2, Qwen
VAE and DINOv2 weights (fp32, CPU), is 0.016 for LPIPS and 0.00054 for KL-OCR, with a
roughly tenfold spread across images. Matching loss values instead would make LPIPS
pull about 17 times and KL-OCR about 330 times harder than GT, because the OCR's CTC
distributions are sharp. The three gradients are nearly orthogonal (median |cos| 0.002
to 0.024 per pair), so `loss_all.yml` keeps each single-loss weight. These are starting
points, not tuned values.

```bash
for name in base_sft loss_lpips loss_ocr_kl loss_all sft_sparse_attn_lpips; do
  torchrun --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/$name.yml
done
```

Nothing holds the output near the teacher in these runs, and the GT MSE pulls it toward
the posterior mean (see [Objective and routing scope](#objective-and-routing-scope)):
expect blur in `base_sft.yml`, which LPIPS and KL-OCR partly counter, and more
forgetting outside the training data than with `base_vosr2.yml`. Setting `kd_weight`
to a small positive value (e.g. 0.1) keeps a weak teacher anchor.

To start from a finished run instead of the VOSR2 weights, set `student_checkpoint`
to one of its `checkpoint-*` directories in every config, and run `base_sft.yml` again
from that checkpoint in a new output directory as the control.

## Hourglass token merging

### Choosing a token-compression scheme

`student.token_compression` selects how the student reduces DiT tokens. The
hourglass configs inherit `base_vosr2.yml`, so every scheme shares the frozen
teacher, data, degradation, one-step KD objective, logging, checkpoints and the
inference entrypoint.

| Scheme | Student config | What is compressed |
| --- | --- | --- |
| Dense (teacher architecture) | `token_compression` and `router_config` absent or null | Nothing |
| SDT MLP routing (`dydit_sdt.yml`) | `router_config: {...}` | MLP tokens only; attention and CA keep every token, so the DiT stays above 68% of its MACs even at keep 0.25 |
| Hourglass (`hourglass*.yml`) | `token_compression: {type: hourglass, ...}` | Every block between the full-grid entry and exit runs on factor x factor merged tokens |
| Mixed-resolution routing (`mixres_route.yml`) | `token_compression: {type: hourglass, routing: {...}}` | As the hourglass, but the router's top share of groups stays on the full grid inside the merged span |

MLP routing (`router_config`) and token merging cannot be combined: their budgets are in different
units, so the model rejects that configuration instead of inventing a joint budget.

### Structure

```text
LQ latent + noise (f8, 32 ch) -> p2 stem -> 1024 tokens per 512^2 crop
 -> fine_in pretrained blocks on the full grid                                  -> h_fine
 -> merge: SpaceToDepth(2) + Linear(4D -> D), initialized as the 2x2 average    -> y_in (256 tokens)
 -> pretrained blocks on merged tokens (RoPE at group centroids, CA on the student's DINO tokens) -> y_out
 -> unmerge: h_fine + DepthToSpace(Linear(D -> 4D)(y_out - y_in)), Linear initialized as a copy
 -> fine_out pretrained blocks on the full grid -> p2 head -> velocity; x0 = noise - velocity
```

The VAE, p2 stem, p2 head, timestep conditioning and every kept block reuse the
teacher's weights; a block only changes the token grid it runs on. With the bypass,
each full-grid token keeps its own deviation from its group mean and receives the
update that the merged blocks applied to its group. All shapes are static: no
top-k, gather or scatter.

| `token_compression` key | Default | Meaning |
| --- | --- | --- |
| `type` | required | `hourglass` |
| `factor` | 2 | Merge factor x factor tokens; latent sides must be multiples of `patch_size * factor` (`factor^2` with `inner_depth`) |
| `fine_in`, `fine_out` | 1, 1 | Leading / trailing pretrained blocks kept on the full grid |
| `drop_blocks` | `[]` | Pretrained block indices removed entirely; their weights are discarded on load. `feature_distill_layers` must not list them (`hourglass.yml` lists 18), or training stops at startup |
| `bypass` | true | `false` uses `Up(y_out)` alone, collapsing each group to one value (large-patch control) |
| `rope` | `centroid` | Merged-token RoPE at the centroid of its sub-tokens; `corner` uses the top-left sub-token, as `_get_dynamic_rope` would |
| `fine_cross_attention` | true | Keep the pretrained DINO CA in the full-grid blocks. `false` discards those weights on load and saves 0.8% of the DiT MACs, but puts the step-0 student about 21 dB PSNR from the teacher (see below) |
| `cond_pool` | 1 | Average-pool DINO tokens inside the model before `mlp_ca` |
| `inner_depth` | 0 | Blocks in the middle of the merged span merged once more (64 tokens per 512^2 crop), with their own merge, unmerge and bypass; see [Two-level (pyramid) variant](#two-level-pyramid-variant) |
| `inner_cond_pool` | 1 | Average-pool the DINO tokens once more, after `mlp_ca`, for the inner blocks only |
| `fine_blocks` | `[]` | Pretrained blocks of the merged span kept on the full grid, with their pretrained DINO CA; each one splits the span, and every merged stretch reuses the same merge, unmerge and bypass (no new weights) |
| `routing` | absent | Learned mixed resolution inside the merged span; needs `inner_depth: 0` and no `fine_blocks`, see [Mixed-resolution routing](#mixed-resolution-routing-mixres_routeyml) |

DiT MACs of this implementation for one 512^2 tile, counted with
`torch.utils.flop_counter.FlopCounterMode` on the meta device (GMACs = GFLOPs / 2;
the student's DINO gives 256 tokens unless noted; DINO and VAE are not included):

| Structure | Config | GMACs | Of dense |
| --- | --- | --- | --- |
| Dense teacher, 1024 DINO tokens | `base_vosr2.yml` | 1635.9 | 100% |
| Blocks 0/35 full grid with CA, 1-34 merged | `hourglass.yml` | 428.5 | 26.2% |
| Same with `fine_cross_attention: false` | example only | 414.9 | 25.4% |
| Blocks 0-1/34-35 full grid, 2-33 merged | `hourglass_2_32_2.yml` | 486.5 | 29.7% |
| Blocks 0-1/34-35 full grid, 2-33 merged, 12-23 merged twice | `hourglass_pyramid_4.yml` | 407.6 | 24.9% |
| Blocks 0-2/33-35 full grid, 3-32 merged, 9-26 merged twice with 64 DINO tokens | `hourglass_pyramid_6.yml` | 408.5 | 25.0% |
| All 36 merged, bypass only | `hourglass_bypass_only.yml` | 370.6 | 22.7% |
| All 36 merged, no bypass (p4 equivalent) | `hourglass_p4.yml` | 370.6 | 22.7% |
| `hourglass.yml` with `drop_blocks: [17, 18]` | example only | 408.4 | 25.0% |
| `hourglass.yml` with 1024 DINO tokens | example only | 593.4 | 36.3% |

The last row is why the student's DINO input shrinks too: cross-attention K/V
projections scale with the DINO token count, not with the latent tokens. Which
blocks to drop for a strict 25% budget is not decided here; choose them from a
sensitivity or saliency measurement, not from the example indices. These are MAC
counts, not measured latency.

### Distance from the teacher at step 0

Each step-0 difference was switched on alone and in combination on the real
teacher weights (`CSWRY/VOSR` revision `f24b306`, Qwen VAE and DINOv2-L from the
same revision), fp32 on CPU, for six ScreenSR `LR_512` images with the same latent
and noise as the teacher. PSNR compares the decoded student and teacher outputs:

| Student at initialization | PSNR to teacher, mean (min-max) |
| --- | --- |
| 12 merged blocks only (curriculum start) | 33.4 dB (27.2-38.5) |
| Student DINO at 224 only | 33.9 dB (26.4-42.2) |
| Student DINO at 448 with `cond_pool: 2` only | 34.1 dB (28.7-42.4) |
| Blocks 0/35 without CA only | 20.8 dB (15.6-24.9) |
| `hourglass.yml` at step 0; `hourglass_2_32_2.yml` and `hourglass_bypass_only.yml` merge the same 12 blocks then | 31.8 dB (25.6-36.4) |
| Same with `fine_cross_attention: false` | 21.1 dB (15.1-26.3) |
| `hourglass_p4.yml` at step 0 (no bypass) | 13.3 dB (9.6-16.8) |
| All 34 blocks merged, untrained | 23.1 dB (18.6-27.2) |

Block 0's CA adds the largest update of any block (its output RMS is 26% of the
residual stream's), which is why the full-grid blocks keep CA by default. Early
`hourglass.yml` previews should look close to the teacher. The p4 control is
expected to look like colored noise at first: without the bypass, the input noise
inside each merged 2x2 token group is averaged away, so `x0 = noise - velocity`
cannot cancel it. That detail accounts for about 70% of the p4 control's step-0
velocity error (1% with the bypass); only training can restore it.

### Two-level (pyramid) variant

`inner_depth` adds a UNet-like second level: the middle `inner_depth` blocks of the
merged span run on tokens merged 2x2 once more, with a second merge, unmerge and bypass
(`token_merge_inner`, `token_unmerge_inner`, same initialization):

```text
full grid h_fine (1024) -> merge -> y_in (256) -> blocks -> y_mid -> merge_inner -> u_in (64)
 -> inner blocks -> u_out -> y_mid + Up_inner(u_out - u_in) (256) -> blocks -> y_out
 -> h_fine + Up(y_out - y_in) (1024)
```

A full-grid block costs 39.1 GMACs, a 256-token block 10.1 and a 64-token block 3.4
(2.4 with `inner_cond_pool: 2`), so the 25% budget can keep 4 or 6 pretrained blocks
on the full grid instead of 2. All 36 blocks are kept:

| Config | Full grid | 256 tokens | 64 tokens | GMACs |
| --- | --- | --- | --- | --- |
| `hourglass.yml` | 0, 35 | 1-34 | - | 428.5 (26.2%) |
| `hourglass_pyramid_4.yml` | 0-1, 34-35 | 2-11, 24-33 | 12-23 | 407.6 (24.9%) |
| `hourglass_pyramid_6.yml` | 0-2, 33-35 | 3-8, 27-32 | 9-26, 64 DINO tokens | 408.5 (25.0%) |

The merge curriculum gives the inner level the last `inner_depth` blocks of growth:
with `coarse_depth` merged blocks it holds `max(0, inner_depth - (max_coarse_depth -
coarse_depth))`, centered in the merged span. Both configs start from the same 12
merged blocks as `hourglass.yml`, and their step-0 outputs on the real weights are
pixel-identical to it (31.8 dB above; checked on two of the six images). In
`hourglass_pyramid_4.yml` the inner level starts once more than 20 blocks are merged;
in `hourglass_pyramid_6.yml` it grows with the merged span from the start, so 12 blocks
stay at 256 tokens throughout. While the inner level is empty its merge layers get no
gradient; multi-rank runs with a merge curriculum therefore build DDP with
`find_unused_parameters=True`. Untrained final structures on the same six images, with
the student's DINO at 224, measured with a scratch prototype that this implementation
reproduces pixel for pixel on the two images checked:

| Untrained student | PSNR to teacher, mean (min-max) |
| --- | --- |
| Blocks 12-23 at 64 tokens, the rest on the full grid | 31.0 dB (24.9-35.8) |
| `hourglass.yml` final structure | 23.3 dB (18.5-27.0) |
| `hourglass_pyramid_4.yml` final structure | 23.8 dB (19.0-27.3) |
| `hourglass_pyramid_6.yml` final structure | 23.5 dB (19.7-27.8) |

Step-0 distance does not rank these structures; whether more full-grid blocks help
small text needs training at the same budget and OCR by font size. Their
`feature_distill_layers` include the inner entry and exit. An alignment stage needs
`trainable_parameters: ['^token_(merge|unmerge)(_inner)?\.']`; the pattern in
`hourglass_align.yml` does not match the inner layers.

### Student DINO input

The frozen teacher always uses the top-level `dino` settings, so its targets do
not change. `student.dino` may override `size` and/or `layer` for the student only;
`hourglass.yml` uses `size: 224` (16x16 = 256 tokens). When the student's settings
differ from the teacher's, the trainer runs DINO a second time on the same LQ batch,
and the student's settings are written to `pipeline.json` so inference feeds the
export the same features. Alternatively keep `size: 448` and set `cond_pool: 2` to
average the 32x32 tokens inside the model. The two have not been compared for text.

### Training recipe

The objective is unchanged: one-step KD at `t=1, r=0`, plus the optional GT term.
New options:

- `student.merge_curriculum_steps` / `merge_curriculum_start`: the number of merged
  blocks grows linearly from `merge_curriculum_start`, centered in the merged span,
  to all of them. Blocks outside the current span run on the full grid with their
  pretrained CA. `max_steps` must cover the curriculum. Checkpoints store the current
  `coarse_depth` in `model.json`; the final export has the full span.
- `training.feature_distill_weight` / `feature_distill_layers`: hidden-state KD at
  the listed pretrained block indices. The student's state is the one handed to the
  next block (merged right after the merge, full grid right after the unmerge); on
  merged states the teacher's state is 2x2-averaged. Each layer uses the relative
  MSE `||s - t||^2 / ||t||^2`, then layers are averaged.
- `training.trainable_parameters`: a list of regular expressions; only matching
  student parameters train. The rest are frozen before the optimizer and DDP are built.
- `training.token_merge_lr_multiplier` (default 1): learning-rate multiplier of the new
  `token_merge*` / `token_unmerge*` layers, logged as `token_merge_learning_rate`. It needs
  `student.token_compression`; when only those layers train (an alignment stage), set
  `learning_rate` instead.
- `student_checkpoint` and `--resume` work as before. Initializing from the dense
  teacher reports the new `token_merge`/`token_unmerge` weights and the discarded
  weights of dropped blocks or full-grid CA; any other missing, unexpected or
  reshaped weight is still an error.

Two starting paths are provided; they have not been compared:

1. `hourglass.yml` directly from the teacher: merge curriculum from 12 to 34 merged
   blocks over 3,000 steps, hidden-state KD with weight 0.1 at blocks 0, 6, ..., 30, 34.
2. `hourglass_align.yml` first (only merge/unmerge train for 2,000 steps at learning
   rate 1e-4, KD at the trunk entrance 0 and exit 34), then `hourglass.yml` with
   `student_checkpoint` set to that export's `model.safetensors` and
   `merge_curriculum_steps: 0`.

Weights, step counts and learning rates are untuned starting points. Logging adds
`feature_kd` (zero when disabled) and, for hourglass students, `coarse_depth`. This
trainer does not include text-region or OCR losses, an x0 or noise-skip output
parameterization, or content-adaptive refinement of text tokens.

```bash
torchrun --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/hourglass.yml

# Optional alignment stage; then hourglass.yml with student_checkpoint and merge_curriculum_steps: 0
torchrun --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/hourglass_align.yml
```

Exports run with the same `inference_vosr_ablation.py`. Tile size and overlap must
be multiples of `8 * patch_size * factor` pixels, `factor^2` with `inner_depth` (32 for
the single-level configs, 64 for the pyramid ones; the default 512/64 qualifies). Token
merging uses PyTorch reshape/permute; no ONNX or NPU export has been attempted, and an
on-device graph should express these rearrangements as SpaceToDepth/DepthToSpace.

### Lowest-compression check

The trained 25-30% students lose too much text. `hourglass_sft_min.yml` asks first whether token merging
can keep text at all, at the lowest compression worth having: if this student cannot match `loss_all.yml`,
merging more blocks is not worth training. It changes only token merging relative to `loss_all.yml`:

- blocks 3-8 and 28-33 run on 2x2-merged tokens (12 blocks); 0-2, 9-27 and 34-35 stay on the full grid.
  Untrained, the input blocks 0-2 and the middle blocks 15 and 17 cost the most OCR when merged alone and
  the output blocks 31-34 almost nothing; the other blocks were not measured reliably;
- the objective of `loss_all.yml` (GT 1.0, LPIPS 0.02, KL-OCR 0.0005, `kd_weight: 0`): the teacher is not
  loaded and there is no hidden-state KD;
- the teacher's DINO input (448, 1024 tokens) in every block, unlike most other hourglass configs;
- `token_merge_lr_multiplier: 10` for the new merge/unmerge layers; no merge curriculum (12 merged blocks
  from step 0).

DiT cost for one 512^2 tile: 1275.9 GMACs, 78.0% of the teacher's; DINOv2 at 448 is not included. Train
and evaluate it exactly as `loss_all.yml` and compare their OCR (`evaluate.py --ann ...`), small text
especially. Compare image metrics against the HQ images or between the two students, not against the
official VOSR2 output: the SFT objective moves both away from it. Not trained yet.

### Mixed-resolution routing (`mixres_route.yml`)

Untrained probes on the official VOSR2 kept most of the text when the text-bearing 2x2 groups stayed on the
full grid and the others were merged (OCR CER 0.54, dense 0.50, every group merged 1.00), but a fixed share
overflowed on text-dense crops, and leaving the other tokens untouched (skipping blocks) instead of merging
them dropped the background to 22 dB against 35 dB. This mode learns which groups stay on the full grid and
trains several shares at once, so one run gives the whole quality-compute curve and the share is chosen at
inference.

```text
blocks 0-1 on the full grid (1024 tokens = 256 2x2 groups per 512^2 crop)
 -> router: SpaceToDepth(2), Linear(4D -> 384) + ReLU + Linear(384 -> 1) in FP32, one score per group
 -> the top k = round(share x 256) groups keep their 4 tokens; the others are merged (hourglass merge, Linear(4D -> D))
 -> blocks 2-33 on one sequence of 4k + (256 - k) tokens; RoPE: kept tokens at their own grid positions,
    merged tokens at their group centroids
 -> back to the grid: kept groups take their tokens' outputs, merged groups h + Up(y_out - y_in) (bypass)
 -> blocks 34-35 on the full grid
```

- The selection (argsort of the detached scores) has no gradient. The router learns through a gate that is
  exactly 1 in the forward pass: a kept group's update is multiplied by `1 + (p - sg(p))`, a merged group's
  by `1 - (p - sg(p))`, with `p = sigmoid(score)`. A kept group's score rises when a larger full-grid update
  would lower the loss, a merged group's when its merged update raises it. There is no budget loss: the share
  is fixed within a step.
- Every optimizer step draws the share uniformly from `keep_ratios`; the lowest share allowed falls linearly
  from 1 to `min(keep_ratios)` over `warmup_steps`. The draw depends only on `training.seed` and the step,
  so ranks and resumed runs agree. Logged as `fine_group_ratio`.
- Share 1 merges nothing and is exactly the dense student of `no_dino_sft.yml`.
- Unlike the plain hourglass, this mode uses argsort, gather and scatter, once at the merge and once at the
  unmerge. At a fixed share the shapes stay static. Not benchmarked on CUDA or Ascend.
- The model rejects `attn_type` other than `full`, and hidden-state distillation (`feature_distill_*`) with
  routing.

| `routing` key | Default | Meaning |
| --- | --- | --- |
| `keep_ratios` | `[0.25, 0.5, 0.75, 1.0]` | Shares of groups kept on the full grid, drawn during training |
| `eval_keep_ratio` | 0.75 | Share in eval mode: previews and exports; `inference_vosr_ablation.py --keep-ratio` overrides it |
| `warmup_steps` | 0 | Optimizer steps over which the lowest share allowed falls from 1 to `min(keep_ratios)` (4500 in `mixres_route.yml`) |

DiT cost of `mixres_route.yml` for one 512^2 tile (counted as above, no DINO; the router, merge and unmerge
included):

| Share | GMACs | Of the dense no-DINO DiT | Of the teacher's DiT |
| --- | --- | --- | --- |
| 1 | 1160.0 | 100% | 70.9% |
| 0.75 | 952.8 | 82.1% | 58.2% |
| 0.5 | 752.3 | 64.9% | 46.0% |
| 0.25 | 559.0 | 48.2% | 34.2% |
| 0 (not trained) | 373.0 | 32.2% | 22.8% |

`no_dino_sft.yml` changes only the student relative to `loss_all.yml`: no DINO/CA, initialized from a
`no_dino_fade.yml` export. Its `student_checkpoint` is a placeholder; set it to that run's
`export/model.safetensors`. `mixres_route.yml` adds only `token_compression` to it: blocks 0-1 and 34-35 on
the full grid, the shares 0.25/0.5/0.75/1, warmup 4,500 steps. The new `group_router.*` weights train at
`router_lr_multiplier` (10); merge and unmerge start as the 2x2 average and copy. The teacher is not loaded.

```bash
torchrun --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/mixres_route.yml
for share in 1 0.75 0.5 0.25; do
  torchrun --nproc_per_node=8 inference_vosr_ablation.py --export exp_vosr/mixres_route/<run>/export \
    --input preset/datasets/inp_data --output preset/results/mixres_route_$share --keep-ratio $share
done
```

With `train_and_infer.sh`, pass the share after `--` (`-- --keep-ratio 1`). Compare the OCR of share 0.75
against share 1 of the same export (`evaluate.py --ann ...`): more than 0.02 CER lost ends the
token-compression line, up to 0.01 passes. Share 1 is trained in the same run with the same weights; train
`no_dino_sft.yml` as well only if share 1 itself looks worse than expected. Not trained yet.

## Dual resolution, LQ-detail routing and zero input noise

Three opt-in additions aimed at the 25% budget (the token count of an f16 VAE with p2, 256 per 512^2 crop)
without the text loss of uniform merging. None changes an existing config: absent keys keep every spec,
`model.json` and `pipeline.json` as before.

| Key | Default | Meaning |
| --- | --- | --- |
| `token_compression.detail` | absent | Narrow full-grid detail stream next to the merged blocks: `width` (512), `depth` (6 blocks), `kernel` (7, odd), `mlp_ratio` (4). Needs `inner_depth: 0`, no `routing` and no merge curriculum; works with `fine_blocks` |
| `token_compression.coarse_cond_pool` | absent (1) | Merged blocks attend to the DINO tokens pooled `factor x factor` once more, after `mlp_ca`; full-grid blocks keep all of them. `cond_pool` pools for every block |
| `token_compression.routing.score` | absent (`learned`) | `lq_detail`: rank groups by the variance of the LQ latent over their pixels instead of the `GroupRouter` (no parameters, no gradient, nothing to learn) |
| `student.noise_input` | `gaussian` | `zero`: the noise half of `cat(lq_latent, noise)` is zero in training, previews and the export's inference (`pipeline.json` records it), so x0 = -velocity and the network predicts the clean latent |

### Detail stream

```text
at each merge:      d = enter(h_fine)                                    Linear(D -> width), full grid
after chosen merged blocks (depth of them, spread over the merged blocks, the last after the last one):
                    d = d + Up(read(y))                                  Linear(D -> width), copied to the 2x2 sub-tokens
                    d = d + MLP(LN(DWConv_kxk(d)))                       on the 32x32 token grid
                    y = y + write(SpaceToDepth(d))                       Linear(4 width -> D), starts at zero
at each unmerge:    h = h_fine + Up(y_out - y_in) + exit(d)              exit: Linear(width -> D), starts at zero
```

The merged blocks keep global attention at 256 tokens; the stream keeps a full-resolution state that each merged
stretch can read and write, instead of the frozen bypass alone (`h_fine + Up(y_out - y_in)` only broadcasts the
group update). Untrained it is the plain hourglass bit for bit. One detail block of width 512 costs about 3.2 GMACs
on a 512^2 tile (a full-grid pretrained block 32-45, with or without DINO), so six add 1.4% of the teacher's DiT.
Its weights, the merge and the unmerge are the new modules (`detail_stream.*`) trained at
`token_merge_lr_multiplier`. The stream's weights do not depend on where the merged blocks are, so
`dualres_min.yml`'s export can initialize `dualres_2_34.yml` (`student_checkpoint`) for a progressive schedule.

### Configs and what each one tests

All but the `mixres_*` configs inherit `loss_all.yml` (SFT: GT + LPIPS + KL-OCR, no teacher), so `loss_all.yml`
and `hourglass_sft_min.yml` are their controls. The `mixres_*` configs inherit `mixres_route.yml` (no DINO,
`student_checkpoint` from a `no_dino_fade.yml` export). DiT MACs for one 512^2 tile, as above:

| Config | `_base_` | Structure | GMACs | Question |
| --- | --- | --- | --- | --- |
| `dualres_min.yml` | `hourglass_sft_min.yml` | its 12 merged blocks + detail stream (after blocks 4, 6, 8, 29, 31, 33) | 1298.2 (79.4%) | Does a full-grid stream win back the text that merging the same blocks costs? |
| `dualres_2_34.yml` | `loss_all.yml` | blocks 0-1 full grid, 2-35 merged with `coarse_cond_pool: 2`, detail stream | 470.4 (28.8%) | Is the dual-resolution student usable near the 25% budget? |
| `dualres_2_34_no_detail.yml` | `dualres_2_34.yml` | the same without the detail stream (optional control) | 449.7 (27.5%) | What does the stream add at this budget? |
| `allmerged_sft.yml` | `loss_all.yml` | all 36 blocks merged, `coarse_cond_pool: 2` | 379.6 (23.2%) | Uniform f16-equivalent baseline on the SFT objective |
| `allmerged_sft_zero_noise.yml` | `allmerged_sft.yml` | `student.noise_input: zero` | same | Is reproducing the input noise through merged tokens part of the loss? |
| `mixres_route_lq.yml` | `mixres_route.yml` | `routing.score: lq_detail` | as `mixres_route.yml` (82.1% / 64.9% / 48.2% of the no-DINO DiT at 0.75 / 0.5 / 0.25) | Does a fixed LQ-detail ranking keep text as well as the learned router? |
| `mixres_f4_lq.yml` | `mixres_route_lq.yml` | `factor: 4`, shares 0.1-1, eval 0.2: 259 tokens in blocks 2-33 | 374.3 at 0.2 (22.9%; 32.3% of the no-DINO DiT) | At the f16-equivalent count, is full resolution for the detailed fifth plus 4x4 elsewhere better than 2x2 everywhere? |

Run them like the other configs, and compare OCR per text size against their controls
(`evaluate.py --ann ...`). `--noise-input zero` on `inference_vosr_ablation.py` feeds zeros to any export, so the
untrained effect of dropping the noise can be checked on an existing dense export first:

```bash
for name in dualres_min dualres_2_34 allmerged_sft allmerged_sft_zero_noise mixres_route_lq mixres_f4_lq; do
  bash train_and_infer.sh --config configs/ablations/$name.yml --testset screensr=/data/ScreenSR/LR --nproc 8
done
python inference_vosr_ablation.py --export exp_vosr/loss_all/<run>/export --input preset/datasets/inp_data \
  --output preset/results/loss_all_zero_noise --noise-input zero
```

Validation: `tests/test_ablation_dualres.py` (22 tests, CPU) checks option validation, detail-block placement on the
real configs, bit-exact equality of an untrained stream with the plain hourglass (with and without `fine_blocks`),
gradients to every parameter under checkpointing, export/reload, DINO pooling per level, the LQ-detail score and the
groups it keeps, 4x4 routing token counts, the trainer and tiled inference with both noise inputs, the one-point
config diffs and the MAC ratios above. Two-rank Gloo trainer runs of the detail stream and of LQ-detail routing (with
zero noise) also completed. Not trained on real data.

## Re-plan after `hourglass_sft_min`: the `la_*` lineage

Trained, `hourglass_sft_min.yml` (78%) keeps more text than `hourglass_2_32_2.yml` and a 12-12-12 layout (blocks 12-23
merged, same SFT recipe), but its text is still visibly worse than `loss_all.yml`'s. A diagnosis of the code, the
recipe, the representation and the literature, with every finding re-checked by a second reviewer, found no bug in the
merge path (layouts, initialization, RoPE positions, bf16, tiling) and these causes:

- **No per-token computation in the merged span.** A merged stretch gives the four sub-tokens of a group a linear
  function of one 1536-d group update, so detail inside a 32x32-pixel group (stroke placement, edge sharpness) cannot be
  refined there. Commit 197fd0b (branch `claude/project-thread-2koy77`, "Which blocks need the full grid") attributed
  about 85% of x0's within-group detail to blocks 27-35 (block 35 alone 32%); `hourglass_sft_min.yml` merges 28-33 with
  only 34-35 after them. In a scratch experiment with random-weight frozen blocks and synthetic text pages (not in the
  repository), the detail stream cut the within-group error about 3x against the trained linear merge/unmerge, while
  adapting the blocks removed 10-17% of it.
- **No reachable full-grid target, weak small-text signal.** Plain SFT has GT MSE (minimized by the posterior mean at
  t=1), LPIPS and KL-OCR on 64/32-pixel full-width strips, which hold lines up to 32 px at 0.75x and often several lines.
- **Initialization.** The student starts from VOSR2 like `loss_all.yml`, so it must absorb the SFT shift and repair the
  merged blocks in the same 10k steps.
- **Evaluation.** One noise seed per image, the training loss's recognizer, and no text-size buckets
  (`evaluate.py --size-buckets` is back, see [the evaluation doc](evaluation.md)).

Noise at t=1 was ruled out as the cause (with the bypass it is about 1% of the step-0 error, see [Distance from the
teacher at step 0](#distance-from-the-teacher-at-step-0), and merged background stays at 35 dB, see
[Mixed-resolution routing](#mixed-resolution-routing-mixres_routeyml)); so was choosing merged blocks by fine-tuned recoverability instead of the step-0 probe (the trained ranking
sft_min > 12-12-12 followed the probe).

### Configs

Every `la_*` config starts from the trained `loss_all.yml` export: `teacher_checkpoint` points to it (replace `<run>`),
which also initializes the student, and the KD runs use it as the frozen dense twin. `la_dense.yml` continues
`loss_all.yml` itself and is the step-matched control. `max_steps` is 20000 and the learning rate is constant after
the 100-step warmup, so `checkpoint-0000N000` of any run is an N-thousand-step run: read the 2k and 3k checkpoints, stop
the runs that fail their rule, and let the others continue in the same job (a new job from `student_checkpoint` would
reset AdamW). DiT MACs for one 512^2 tile:

| Config | `_base_` | Change | GMACs | Question |
| --- | --- | --- | --- | --- |
| `la_dense.yml` | `loss_all.yml` | start from the loss_all export | 1635.9 (100%) | Control of every `la_*` run |
| `la_sftmin.yml` | `la_dense.yml` | `hourglass_sft_min`'s merged blocks | 1275.9 (78.0%) | How much of sft_min's gap was the VOSR2 initialization? |
| `la_dualres_min.yml` | `la_sftmin.yml` | + detail stream (after 4, 6, 8, 29, 31, 33) | 1298.2 (79.4%) | Does per-token computation recover the text at 78%? |
| `la_allmerged.yml` | `la_dense.yml` | all 36 blocks merged, `coarse_cond_pool: 2` | 379.6 (23.2%) | Gap of uniform merging at 256 tokens |
| `la_allmerged_detail.yml` | `la_allmerged.yml` | + detail stream (after 5, 11, ..., 35) | 400.3 (24.5%) | Main 25% candidate: does the stream recover text at 256 tokens? |
| `la_allmerged_detail_kd.yml` | `la_allmerged_detail.yml` | output KD to the dense twin (`kd_weight` from calibration) | 400.3 (24.5%) | Is a reachable target the lever? |
| `la_allmerged_detail_ocr48.yml` | `la_allmerged_detail.yml` | KL-OCR strips 48/16 (`ocr_kl_weight` from calibration) | 400.3 (24.5%) | Does a small-text OCR signal recover <= 32 px lines? |
| `la_mixca_f4.yml` | `la_dense.yml` | blocks 0-1 full grid; 2-35 4x4 LQ-detail routing at share 0.15 (214 tokens), DINO pooled | 398.6 (24.4%) | At matched MACs, content-adaptive resolution or uniform merging + stream? |
| `la_allmerged_detail_twin.yml` | `la_allmerged_detail_kd.yml` | + hidden-state KD at block 35 (the full-grid state after the last unmerge) | 400.3 (24.5%) | Wave 2: does a hidden target add to output KD? |
| `la_dr1.yml` | `la_allmerged_detail.yml` | block 0 on the full grid | 435.3 (26.6%) | Wave 2: is the entry block worth its cost? |
| `la_dr0_fo1.yml` | `la_allmerged_detail.yml` | block 35 on the full grid | 435.3 (26.6%) | Wave 2: or the exit block, at the same cost? |
| `la_allmerged_detail_k1.yml` | `la_allmerged_detail.yml` | `detail.kernel: 1` (per-token MLP, no spatial mixing) | 400.2 (24.5%) | Wave 2: is the 7x7 depthwise conv needed (simpler on-device graph)? |

The first eight are wave 1. `la_mixca_f4.yml` is gated by its probe below; `la_allmerged_detail_kd.yml` and
`la_allmerged_detail_ocr48.yml` take their loss weights from the calibration below.

### Free checks, calibration and probes

`tools/probe_helpers.sh` defines, after `source tools/probe_helpers.sh` with `LQ`, `ANN`, `ANNREF` (the image the
Label.txt boxes are drawn on) and optionally `HQ` set:

- `ev EXPORT NAME SEED [inference args]`: restore with `inference_vosr_ablation.py` (8 ranks) and evaluate with
  `--size-buckets 16 24 32 48` for PP-OCRv5_server_rec and PP-OCRv4_server_rec (a recognizer the loss does not train against);
- `mk SRC DST 'PYTHON'`: copy an export with an edited `model.json` `token_compression` (bound to `t`), weights symlinked.
  Like `coarse_depth`, the detail blocks of an edited detail export are spread again over its new merged blocks, as a
  trained config with that layout would place them;
- `ko SRC DST`: copy with the detail stream's exit and write layers zeroed (knock-out: was the stream used?).

`tools/cer_gap.py REF_DIR ... -- ARM_DIR ...` pools CER per text size over several noise seeds and gives a 95% paired
image-bootstrap interval of the difference.

1. Noise floor: `ev` the loss_all and hourglass_sft_min exports at seeds 42, 43 and 44 and compare seeds with
   `cer_gap.py`. The single-seed spread sets the smallest effect worth reading (at least 0.02 CER on lines < 32 px).
2. Where the gap sits: per bucket, for the trained sft_min, hourglass_2_32_2 and 12-12-12 exports.
3. Budget: `ev` the `checkpoint-0000N000` directories (1k-10k) of hourglass_sft_min and loss_all. A gap still closing at
   10k means a step or learning-rate limit; a flat gap means the structure or the target.
4. Per-stretch attribution of the trained sft_min export: `mk $T p/early 't["fine_blocks"]=list(range(9, 34))'` keeps only
   3-8 merged, `range(3, 28)` only 28-33.
5. Untrained structures on loss_all weights: train `probe/probe_la_allspan.yml` (one step at learning rate 0) and edit
   copies of its export with `mk` (for example only blocks 3-8 and 28-33 merged, or `coarse_cond_pool: 2`).
6. Routing gate for `la_mixca_f4.yml`: train `probe/probe_la_mixca_f4.yml` (export `$P`). The routed path must be exact:
   on a copy without the DINO pooling, `mk $P p/nopool 't.pop("coarse_cond_pool")'`, `--keep-ratio 0.999` (every group
   kept, through the routed gather, RoPE and scatter) must match `--keep-ratio 1` (the dense model) up to bf16 rounding.
   On `$P` itself the two differ by the pooled DINO, because share 1 bypasses `coarse_cond_pool`; report that difference,
   it is not a failure. Then share 0.15 on `$P` must beat, on small text, the untrained all-merged structure with the same
   pooling (`mk $U p/u_am 't.update(coarse_cond_pool=2)'` on the export `$U` of check 5), as on VOSR2 (CER 0.54 vs 1.00).
7. `--noise-input zero` on the loss_all export: what the zero-noise input costs the dense model before any training.

Loss weights are calibrated like the LPIPS and KL-OCR weights above, on the target structure: each
`calib/calib_am_*.yml` turns on one objective on `la_allmerged_detail.yml` for 20 steps at learning rate 0 (no weight
changes), then `tools/calib_am.py` prints `kd_weight` and `feature_distill_weight` as the median ratio of the SFT
objective's gradient norm to theirs, and the 48/16 `ocr_kl_weight` that keeps loss_all's OCR pull. Until these values
are filled in, `la_allmerged_detail_kd.yml`, `la_allmerged_detail_twin.yml` and `la_allmerged_detail_ocr48.yml` hold the
sentinel `CALIBRATE`, which stops the trainer at startup:

```bash
for c in sft kd fkd ocr64 ocr48; do
  torchrun --standalone --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/calib/calib_am_$c.yml || break
done
python tools/calib_am.py   # then replace CALIBRATE in the three la_* configs
```

### Reading the runs

Read CER on lines below 32 px (`CER_<32`, one 2x2-merged token covers 32x32 output pixels) and on lines of 32 px or more,
against `la_dense.yml` at the same step, averaged over two noise seeds (three if the seed spread exceeds 0.02). Count an
effect only if it is at least the noise floor of check 1, its bootstrap interval excludes 0, PP-OCRv4 agrees in sign and
it holds at 2k and 3k. For an arm against its control, the recovered share is
`(gap_control - gap_arm) / gap_control`: 0.5 or more adopts the change, 0.25-0.5 runs on to 5k, below 0.25 stops it at
3k. `la_sftmin.yml` and `la_allmerged.yml` are measurements, read as their gap to `la_dense.yml` (`la_sftmin.yml` also
against `hourglass_sft_min`'s gap at the same step). `la_dense.yml` and every config named as an arm's control keep
running until the last of their arms stops; the stop rule does not apply to them. Before calling a detail-stream run negative, `ko` its 3k checkpoint: if zeroing the stream changes CER by less than
0.005 the stream was not used, and the run should be repeated with `token_merge_lr_multiplier: 30`. A 25% student passes
when, at the same step as `la_dense.yml` and for two training seeds, all-text CER is within 0.01, `CER_<32` within 0.02,
PSNR within 0.3 dB and LPIPS within 0.01.

`tests/test_ablation_dualres.py` checks the one-point diffs and MACs of these configs, the calibration and probe configs,
trainer runs of the dense-twin recipe on a detail student and of LQ-detail routing with DINO and KD (also completed on
two Gloo ranks), and the three tools. None of them has been trained on real data. Runs with a teacher build it with a
separate random generator, so a KD run and its no-teacher control initialize the student's new modules identically, and
`student.noise_input: zero` is rejected together with a teacher term (the frozen teacher expects Gaussian noise).

## Block-sparse self-attention (`student.attn_type`)

`student.attn_type` selects the student's self-attention; the teacher always uses
full attention, so the key is rejected under `model:`. Cross-attention is unchanged.

| `attn_type` | Self-attention |
| --- | --- |
| `full` (default) | Dense SDPA, the original model |
| `sparse` | Tokens grouped into 8x8 windows (64 tokens); each query window attends to the 8 key windows with the highest mean-q . mean-k score, per head |
| `local` | Each query window attends to its 3x3 neighbourhood of windows (fewer at the border) |

`SparseProcessAttnAigc` (sparse) and `SparseProcessAttnAigc_Local0408` (local) in
`models/sparse_attention.py` are copied verbatim from an on-device implementation; do not
edit them. VOSR2 applies RoPE first, then `windowed_attention` reorders the row-major
tokens into window order, calls the copied module and restores the order. The token
grid must be square with sides divisible by 8 (a 512 px tile gives 32x32 tokens = 16
windows). `sparse` keeps exactly 8 windows, so with 8 or fewer windows (hourglass
merged spans) it is computed as full attention. The masks are dense boolean tensors
passed to SDPA: training simulates the sparsity but does not save time.

Override the type at inference without retraining (`model.json` records the trained one):

```bash
python inference_vosr_ablation.py --export exp_vosr/ablation_base/<run>/export \
  --input preset/datasets/inp_data --output preset/results/base_sparse --upscale 4 --attn-type sparse

# The official checkpoint through the upstream script
python inference_vosr_onestep.py -c preset/ckpts/VOSR2 -i preset/datasets/inp_data -o preset/results/vosr2_local \
  --tile_size 512 --attn_type local
```

## 1024 resolution (`res1024*.yml`)

In the base configs "512" is the crop, the model's `input_size` (64 latents, a 32x32 token
grid, the RoPE grid) and the default inference tile at once. The `res1024` configs keep the
teacher's reference grid (`data.resolution: 512`, so `input_size` stays 64) and train on 1024
crops through these opt-in keys. Absent keys keep the previous behaviour, so no existing config
changes meaning.

| Key | Values | Effect |
| --- | --- | --- |
| `model.rope_mode` | `interpolate` (default), `extrapolate` | RoPE on a token grid other than `input_size / patch_size`. `interpolate` squeezes the positions into the trained range (upstream `_get_dynamic_rope`; spacing 0.5 on a 1024 tile). `extrapolate` keeps integer positions: neighbours stay one position apart, distances reach 63. The hourglass merged grid follows the same rule. On the reference grid both use the trained RoPE buffer. Recorded in `model.json`; `inference_vosr_ablation.py --rope-mode` overrides it |
| `data.crop_size` | pixels, default `data.resolution` | HQ crop size of `TxtPairDataset` (which bicubic-upscales images whose short side is smaller). The export's `pipeline.json` `resolution` (the default inference tile) is this size |
| `dino.tile`, `student.dino.tile` | pixels or `null` | A crop larger than the tile is split into tiles that each get their own `dino.size` input; the token grids are laid out as one row-major grid (CA has no positions; hourglass `cond_pool` pools true neighbours). A 1024 crop with tile 512 at 448 gives 4 x 1024 tokens. Training, previews and inference all go through `dino_features`, so a larger inference tile must be a multiple of `tile` |
| `training.teacher_tile`, `training.teacher_tile_overlap` | pixels, default overlap 64 | The frozen teacher runs a larger crop as overlapping tiles, each with the DINO of its own LQ crop (top-level `dino`), and Gaussian-blends the velocities, as `inference_vosr_ablation.py --tile-size 512` does. Previews use the same target. The top-level DINO of the whole crop is then not computed unless the student uses the same settings. Rejected with feature distillation when a crop exceeds the tile |
| `training.vae_tile_size` | pixels, `0` = whole image, default `data.crop_size` | VAE encode/decode in training, previews and pixel losses. Below the crop size it uses Gaussian-blended tiles with 64 px overlap (`min(64, max(8, tile / 8))`), like inference. The default is one pass per crop, as before. The export's `pipeline.json` records it as the default `--vae-tile-size`. The Qwen VAE has a global self-attention in its encoder and decoder mid blocks, so tiles and the whole image give different latents |

| Config | `_base_` | Ablation point |
| --- | --- | --- |
| `res1024.yml` | `base_vosr2.yml` | Stage 1: extrapolated RoPE, 1024 crops only, DINO per 512 px (teacher and student), teacher tiled at 512 with 64 px overlap |
| `res1024_rope_interp.yml` | `res1024.yml` | `rope_mode: interpolate` |
| `res1024_dino_whole.yml` | `res1024.yml` | Student DINO on the whole crop at 448 (1024 tokens, the 512 cost) |
| `res1024_loss_all.yml` | `res1024.yml` | Plain SFT, no teacher term (the teacher is not loaded): GT 1.0, LPIPS 0.02 and KL-OCR 0.0005 (the `loss_all.yml` losses) on the whole decoded 1024 crop |

Zero-shot evidence for the defaults (official weights, no training, six rendered 1024^2 text
pages and two DIV2K crops; OCR CER on 21-28 / >=29 px lines): 512 tiling 0.608 / 0.111; one 1024
tile with interpolated RoPE 0.994 / 0.998 (striped garbage); extrapolated RoPE with whole-crop
DINO 0.682 / 0.231; with DINO per 512 px 0.650 / 0.139. Whole-image vs 512-tiled VAE on the same
text pages (fixed noise): the LQ latent differs by 0.6% (relative L2), the outputs by 49-58 dB PSNR,
and CER by at most 0.002 for both the one-tile and the 512-tiled DiT.

Cost (FlopCounterMode on meta tensors, not measured on hardware): one 1024 step of `res1024.yml`
is about 57 TMACs (student forward + backward + recomputation 37.3, tiled teacher 14.7, DINO 3.5,
VAE encode 1.4), 6.5x a 512 step, counting gradient checkpointing; without it the recomputation (one student forward, 9.3) drops out, about 48 TMACs.
Memory per device at batch 1 on 8 ranks (CPU measurement of the real block sizes plus dtype arithmetic, not a GPU
or NPU measurement): about 21 GB that do not depend on the crop (fp32 student weights, gradients and DDP buckets
5.2 GB each, ZeRO-sharded AdamW 1.3, bf16 teacher 2.6, DINOv2-L 1.1, VAE 0.2), plus student activations of
9.3 GB at 512 and 29.7 GB at 1024 without checkpointing (0.4 and 1.2 GB with
`training.gradient_checkpointing: true`). Images
smaller than 1024 are upscaled into blurry HQ crops; count them before training:

```bash
python -c "
from PIL import Image
paths = [line.strip() for line in open('path/to/image_list.txt') if line.strip()]
print(sum(min(Image.open(p).size) >= 1024 for p in paths), 'of', len(paths), 'images have a short side >= 1024')"
```

Train `res1024.yml` and the ablations independently, then restore with an export:

```bash
torchrun --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/res1024.yml

# Default tiles = the training crop: DiT 1024, VAE 1024 (one pass on a 1024 output); --tile-size 512 also works
python inference_vosr_ablation.py --export exp_vosr/res1024/<run>/export \
  --input preset/datasets/inp_data --output preset/results/res1024 --upscale 4
```

## Pixel-space student (L2P, `pixel_l2p*.yml`)

L2P ([arXiv 2605.12013](https://arxiv.org/abs/2605.12013), code `TencentYoutuResearch/T2I-L2P`)
turns a pretrained latent DiT into a pixel-space model: the VAE is dropped, a pixel patch
embedding replaces the latent one, a small convolutional decoder replaces the final layer and the
VAE decoder, and only the outer transformer layers are fine-tuned. `student.pixel` applies this to
VOSR2 (`models/pixel_dit.py`, `PixelLightningDiT`); without the key nothing changes.

**Token size.** A VOSR2 token covers 16 x 16 HQ pixels: the Qwen-Image VAE downsamples by
`2 ** len(temperal_downsample)` = 8 (`spatial_compression_ratio` in `models/qwenimage_vae2d.py`)
and the DiT patch is 2. A 16-pixel patch therefore keeps the 32 x 32 token grid of a 512 crop, the
RoPE positions (`input_size` becomes the crop in pixels, so `input_size / patch_size` stays 32) and
the DiT cost. `patch_size: 32` quarters the tokens and turns on the decoder's 2x `c_upsample`.

| Part | `pixel_l2p.yml` |
| --- | --- |
| Input | `cat(LQ, x_t)`, RGB in `[-1, 1]` (6 channels, LQ first as in the latent model) |
| Encoder | `encoder: patch`: 16 x 16 patchify + Linear (`x_embedder`, a stride-16 conv 6 -> 1536), fresh init |
| Transformer | The 36 pretrained blocks with their DINO CA, unchanged. The DINO branch reads the LQ image and gives 1024 tokens (448 input) whatever the image tokens are |
| Decoder | `MicroDiffusionModelTop3SCWide` as supplied: conv U-Net 48/96/192/384 channels down to 1/16, `cat(p4, c)` -> 1x1 bottleneck, additive skips at 1/8, 1/4, 1/2, none at full resolution; 5.0 M parameters. Its image input is the model input; `c` is the last block's hidden state as a 1536 x H/16 x W/16 map. `c_upsample` exists only for `patch_size` 32 (the supplied nearest-initialized 2x transposed conv) |
| Output | The decoder predicts the clean image; `forward` returns `(x_t - x0) / t` in fp32, so the trainer and inference keep computing the image as `noise - velocity` |

Choices that differ from a literal port, and why:

- **x-prediction.** At the one-step `t = 1`, `x_t` is pure noise, so a velocity target
  `noise - x0` would make the decoder copy per-pixel noise through a network whose last skip is at
  half resolution. JiT ([arXiv 2511.13720](https://arxiv.org/abs/2511.13720)) finds x-prediction
  necessary for large pixel patches. The loss value is unchanged: at `t = 1` the x0 error is the
  velocity error.
- **Decoder image input.** L2P's decoder reads the noisy image; here it reads `cat(LQ, x_t)`,
  because at `t = 1` the noisy image carries nothing for the skips while the LQ carries the image.
- **Decoder feature input.** As L2P: the last block's hidden state, the final layer (norm, adaLN,
  projection) removed. VOSR2's last hidden state is already of order one per token (RMS median
  1.45, range 1.44-1.46 on three ScreenSR text crops with real weights and DINO, through the HF
  `facebook/dinov2-large` port; largest |x| 39.5, in channels 1527 and 1505), so no normalization
  is added.

**Loading.** From a latent checkpoint, new-module weights whose shape differs keep their fresh init
and same-shaped ones load, like L2P's loader: `x_embedder.proj.weight` is new, `x_embedder.proj.bias`
loads; `final_layer.*` is discarded, `decoder.*` is new. `--resume` loads strictly. Checked with the
real VOSR2 EMA weights on CPU: 841 tensors load and 33 are new, and a 256^2 forward runs. Without CA
(475 tensors load), a 512^2 forward/backward also runs in fp32 (gradient checkpointing on) with no
gradient in the frozen blocks.

**Training recipe** (`pixel_l2p.yml`, `_base_: loss_all.yml`, so its control is the latent
`loss_all.yml` with the same data, schedule and initialization):

| Item | Setting | Source |
| --- | --- | --- |
| Initialization | The official VOSR2 weights (`teacher_checkpoint`, `student_checkpoint: null`), DINO/CA included | As `loss_all.yml` |
| Trainable | Blocks 0-4 and 31-35 and every module outside the blocks (patch embedding, time embedding, `t_block`, DINO projector `layer_norm` and `mlp_ca`, decoder; the CA of blocks 5-30 stays frozen): 413.9 M of 1396.3 M parameters, via `training.trainable_parameters` | L2P freezes layers `5 .. n - 5` (`freeze_except` in `base_pipeline_L2P.py`) |
| Optimizer | AdamW, learning rate 5e-5 (warmup 100), weight decay 0.01, batch 1 x 8 ranks, 10,000 steps | L2P's training script: AdamW 5e-5, batch 8; steps as `loss_all.yml` |
| Losses | GT 1.0 (pixel MSE of the image vs the HQ crop), LPIPS-vgg 0.013, KL-OCR 0.0008 on the image itself, no VAE decode | `loss_all.yml` terms; weights below |
| Teacher | None. The latent teacher's velocity has no pixel counterpart; `kd_weight` or `feature_distill_weight` above 0 is rejected | |

The loss weights follow the rule of `loss_all.yml` (same step-0 gradient norm as `gt_weight: 1.0`),
now on the predicted image. The pixel student's own step-0 output is no reference point (its decoder
starts from scratch), so VOSR2's one-step output on the same crop stands in for a trained one: over 12
ScreenSR text crops (fp32, CPU, pyiqa LPIPS-vgg and the PP-OCRv5 port, through `lpips_loss` and
`build_ocr_kl`) the median ratio is 0.0127 for LPIPS (0.004-0.027) and 0.00081 for KL-OCR
(0.00024-0.011); taken at the LQ image instead, 0.0093 and 0.00071. `pixel_l2p.yml` uses 0.013 and
0.0008. These are starting points, not tuned values.

**Alternative encoder** (`pixel_l2p_unet.yml`, `_base_: pixel_l2p.yml`, only `encoder: unet`): the
pretrained blocks were trained on tokens of VAE latents, a deep non-linear encoding of the pixels,
which one linear map of raw 16 x 16 patches cannot imitate; with `patch` the first five blocks must
absorb that change. The decoder's down path already computes 384-channel features at 1/16 of the
image, so `unet` embeds those (1x1 conv 384 -> 1536) as the tokens: a non-linear, multi-scale stem
at no extra cost, trained from both the DiT and the decoder side. L2P, DiP and JiT all use a linear
patchify; this variant is untested.

Cost per 512^2 tile, batch 1 (FlopCounterMode on meta tensors, GMACs):

| Pipeline | DiT incl. embedding and CA | DINOv2 | Decoder | VAE encode + decode | Total | Of latent |
| --- | --- | --- | --- | --- | --- | --- |
| `loss_all.yml` (latent) | 1635.9 | 271.5 | - | 317.8 + 550.1 | 2775.4 | 100% |
| `pixel_l2p.yml` | 1638.0 | 271.5 | 23.8 | - | 1933.3 | 69.7% |
| `pixel_l2p_unet.yml` | 1636.2 | 271.5 | 23.8 | - | 1931.5 | 69.6% |
| `pixel_l2p.yml` with `patch_size: 32` (example only) | 529.1 | 271.5 | 26.2 | - | 826.8 | 29.8% |

DINOv2 is the first 18 layers of DINOv2-L at 448 input. At patch 16 the DINO branch (DINOv2 plus CA
and its projector, 747.5) is the same in both pipelines, so the saving is the VAE. At patch 32 the
decoder has 14.4 M parameters, 9.4 M of them in `c_upsample`.

```bash
torchrun --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/pixel_l2p.yml
python inference_vosr_ablation.py --export exp_vosr/pixel_l2p/<run>/export \
  --input preset/datasets/inp_data --output preset/results/pixel_l2p --upscale 4
```

Exports record `pixel` in `model.json` and `vae_path: null` in `pipeline.json`; inference loads no
VAE and tiles in pixels (tile and overlap multiples of `patch_size`); each tile gets the DINO of its
own LQ crop, as in the latent pipeline. `student.pixel` does not combine with `router_config` or
`token_compression`. Compare `pixel_l2p.yml` and `pixel_l2p_unet.yml` with `loss_all.yml` at the same
steps (CER and OCR accuracy first, then PSNR/LPIPS); the patch embedding and decoder start from
scratch, so look at the trend over checkpoints, not only the last one.

## Checkpoints, resume, and inference

`output_dir` names the experiment. Each fresh run creates `output_dir/<run>/`, where
`<run>` is rank 0's local start time as `YYYYmmdd-HHMMSS` (a run started in the same
second as an existing one gets `-1`, `-2`, ...), and writes everything there: `config.json`,
logs, previews, `checkpoint-XXXXXXXX/` and the final `export/`. All ranks use rank 0's
directory. Fresh runs therefore never fail on, or overwrite, an earlier run's output.

Each `checkpoint-XXXXXXXX/` includes model weights, model/pipeline JSON, and
`training_state.pt` containing optimizer state, step and resolved config. Resume
with the same model/data/optimization config (logging/preview options may change).
`--resume` creates no new directory: the run continues in the directory that holds
the checkpoint (`exp_vosr/dydit_sdt/20261012-085251/` below), so its checkpoints,
logs and export stay together. Checkpoints written before run directories existed
resume in their old `output_dir` the same way.

```bash
torchrun --nproc_per_node=8 train_vosr_ablation.py \
  --config configs/ablations/dydit_sdt.yml \
  --resume exp_vosr/dydit_sdt/20261012-085251/checkpoint-00001000
```

Resume restores optimization state and step-based schedules, **not exact data
loader position or worker RNG**; it is not bitwise replay. Intermediate fade
checkpoints retain CA parameters and their current scale. Only the final `export/`
strips inactive conditioning; use a training checkpoint, not the inference-only
export, for resume.

```bash
python inference_vosr_ablation.py \
  --export exp_vosr/no_dino_no_ca/<run>/export \
  --input preset/datasets/inp_data --output preset/results/no_dino --upscale 4

python inference_vosr_ablation.py \
  --export exp_vosr/dydit_sdt/<run>/export \
  --input preset/datasets/inp_data --output preset/results/dydit_sdt --upscale 4

# Multi-NPU/GPU data-parallel inference: each rank restores a disjoint shard of images
torchrun --nproc_per_node=8 inference_vosr_ablation.py \
  --export exp_vosr/dydit_sdt/<run>/export \
  --input preset/datasets/inp_data --output preset/results/dydit_sdt --upscale 4
```

The new inference entrypoint reads `model.json`; do not load these exports with
the unchanged upstream inference script. It supports Qwen 2D VAE, per-tile
conditioning, globally shared noise, Gaussian latent-velocity blending, and
optional tiled VAE encode/decode. `--vae-tile-size` (`0` = whole image) defaults to the
export's `pipeline.json` `vae_tile_size`, the VAE tile used in training (the training crop
unless `training.vae_tile_size` is set). Exports without the field keep the old default 512. Non-square images are padded/tiled into square
DiT inputs, then cropped to the requested size. There is no automatic color fix.
Use `--vae-path` to relocate VAE assets, and `--dense-mlp` for a backend without
sparse gather/scatter. No-DINO exports do not instantiate or fetch the encoder.
Under `torchrun`, images are split round-robin across ranks (no inter-rank
communication); the RNG is reseeded with `--seed` per image, so outputs are
identical regardless of the number of devices.

### Train, then restore test sets

`train_and_infer.sh` runs the trainer and, once it finishes, restores each test set
with the final `export/` of that same run:

```bash
bash train_and_infer.sh --config configs/ablations/hourglass.yml \
  --testset preset/datasets/inp_data --testset screensr=/data/ScreenSR/LR \
  --nproc 8 -- --upscale 4
```

- `--testset [NAME=]PATH` (repeatable): an LQ folder or a single image. Outputs go to
  `<run>/inference/NAME/`; NAME defaults to the folder name (two folders both called
  `LR` need explicit names).
- `--nproc N` (default 8): `torchrun --standalone --nproc_per_node` for training and
  inference; a test set with fewer images uses one process per image.
- `--resume CHECKPOINT` is passed to the trainer, so the run and its inference results
  stay in the checkpoint's run directory.
- Options after `--` go to `inference_vosr_ablation.py` for every test set; without
  them, tiling, VAE tiling and upscale come from the export as described above.
- `torchrun` must be on `PATH` (activate `.venv`, or run `uv run bash train_and_infer.sh ...`);
  `TORCHRUN=/path/to/torchrun` selects another launcher executable.

Test sets are checked before training starts (path exists, folder has images, names
are unique). The script learns the run directory from the trainer's `--run-dir-file`,
so concurrent runs of the same config cannot be mixed up. A failed test set does not
stop the others; the script exits non-zero and lists it at the end.

## Validation

```bash
python -m compileall -q models/sdt_router.py models/lightningdit_ablation.py models/token_hourglass.py \
  ablation_utils.py train_vosr_ablation.py inference_vosr_ablation.py
TORCHDYNAMO_DISABLE=1 python -m pytest -q tests/test_ablation_core.py tests/test_ablation_data.py tests/test_ablation_logging.py tests/test_ablation_model.py tests/test_sdt_distributed.py tests/test_ablation_hourglass.py tests/test_ablation_res1024.py
```

The core suite tests STE gradients, deterministic evaluation, sparse/dense MLP
agreement (none/all/mixed selections), routing validation, full-length attention
inputs, dense warmup gradient edges, inactive CA, unchanged residual equations and
state names, RNG-preserving checkpoint recomputation, config inheritance and strict
checkpoint checks. The data suite checks upstream TXT list repetition,
working-directory path handling, HQ crops, invalid/empty configs, and degradation
input/output normalization.
It uses the real upstream TXT dataset and requires `torchvision` and `webdataset`;
the normalization test substitutes a deterministic degradation object.
Full-backbone tests additionally compare with upstream LightningDiT and exercise
stripped export/reload and routed backward.
Those tests require the upstream timm/triton environment.

Previous TXT-migration validation: **34 CPU tests passed; the full-backbone test module was
skipped because timm was unavailable**. Python compilation and the trainer's
`--help` check also passed. No real VOSR2 checkpoint, full RealESRGAN degradation,
end-to-end image inference, CUDA distributed training, throughput benchmark, or
TextSR quality evaluation was run in this environment. Before long training,
run the full suite in the VOSR environment, then a short real-checkpoint smoke run
and fixed-seed OCR/text-crop comparisons against the full teacher.


Logging extension validation: **39 tests passed, 2 skipped** (full backbone lacks
`timm`; two-rank Gloo is opt-in). Real TensorBoard events and offline W&B history
were checked for scalar/image records at the same step. A CPU integration test
runs the actual trainer with tiny model/VAE/degradation substitutes, saves a
checkpoint, resumes with updated logging options, and checks the output events.
Preview determinism, RNG preservation and restoration after errors are covered.
The opt-in two-rank Gloo test requires a host that permits communication sockets:

```bash
VOSR_TEST_DDP=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_ablation_logging.py tests/test_sdt_distributed.py
```

The implementation environment disallowed Gloo socket creation, so that test
could not run here. Real VOSR2 weights and CUDA/NPU training remain unverified.

Global-budget/top-k update validation: **61 CPU tests passed, 2 opt-in Gloo tests
skipped**. This run includes the real small LightningDiT backbone tests (timm and
fairscale installed), train/eval and sparse/dense agreement, exported policy
roundtrip, FP32/BF16 router checks, the p=0.75 regression, independent per-image
capacities, STE task gradients, and budget-only optimization of the real router.
The actual trainer integration fixture also covers routed dense warmup, global
budget, accumulation, save/resume, and the new TensorBoard/preview diagnostics.
Compilation, CLI help, and diff whitespace checks passed. These tests do not use
real VOSR2 weights, validate TextSR quality, benchmark latency, or validate the
stable-sort/sparse operators on CUDA/Ascend.

Cross-rank budget / FP32 router fix validation: **68 CPU tests passed, 3 opt-in
Gloo tests skipped**, using PyTorch 2.5.1+cpu and the repository-pinned
Accelerate/timm/W&B versions. Added regressions compare BF16-autocast routing and
STE gradients with FP32 at dim=1536, 1024 tokens and input scales 1/4/16; BF16
residual inputs are also covered. A socket-free simulated SUM collective checks
cross-rank budget values and DDP-averaged gradients against a concatenated-batch
reference. The separate two-rank test checks real communication, mixed task/budget
loss scaling, legacy layer scope and `no_sync()` accumulation. Its explicit run
was blocked during Gloo initialization with `Operation not permitted`, so real
distributed execution remains unverified here. Compilation, trainer CLI help and
diff whitespace checks passed. No CUDA/Ascend or real-checkpoint OCR validation
was performed for this fix.

Hourglass token-merging validation: **96 tests passed** with `VOSR_TEST_DDP=1`
(92 passed and 4 opt-in Gloo tests skipped without it), on PyTorch 2.14.1+cpu with
timm 1.0.30, fairscale 0.4.13, Accelerate 1.1.0, TensorBoard 2.21.0 and W&B 0.25.0.
`tests/test_ablation_hourglass.py` checks the 2x2 average/copy initialization, exact
equality with the dense student at zero merged blocks, bit-exact bypass with identity
merged blocks (and the lossy no-bypass control), per-block token counts and the
curriculum span, centroid/corner RoPE against the upstream RoPE, distillation
targets, strict loading from a dense teacher, in-model DINO pooling, gradients to
every parameter under checkpointing, export/reload, and the MAC ratios above at the
real VOSR2 size. A CPU integration fixture runs the actual trainer with a small real
backbone (curriculum, feature KD, student DINO override, previews, save/resume,
one and two Gloo ranks), a merge-only alignment run, tiled inference of the
export, and the startup error for a distillation layer listed in `drop_blocks`.
The real VOSR2 checkpoint was used only for the step-0 forward comparison above
(CPU, no training). No CUDA/Ascend training, latency benchmark or TextSR/OCR
evaluation was run for this change.

Two-level hourglass validation: **108 ablation tests passed** with `VOSR_TEST_DDP=1`
in the same environment. The new tests cover option validation, per-block token counts
and the order in which the curriculum fills the two levels, the inner RoPE, distillation
targets against 4x4 teacher averages, gradients under checkpointing, export/reload and
the MAC ratios of both pyramid configs. The trainer fixture runs one and two Gloo ranks
with the inner level empty at the first step; the two-rank run fails without
`find_unused_parameters`. The real VOSR2 checkpoint was used only for the step-0 and
final-structure comparisons above (CPU, no training).

Mixed-resolution routing validation: **158 ablation tests passed** with `VOSR_TEST_DDP=1` on PyTorch
2.5.1+cpu. The new tests cover option validation, exact equality with the dense student at share 1,
agreement with the dense student when every group is kept through the routed path (reference and other
grids, both `rope_mode`s), agreement with the plain hourglass at share 0, per-block token counts, a
forward-neutral gate with gradients to the router, export/reload of the eval share, the seeded share draw
and its warmup, the MAC ratios above at the real VOSR2 size, and the trainer with one and two Gloo ranks
(steps that keep every group leave the router unused), followed by tiled inference of the export. A
separate two-rank run with `zero_optimizer: true` also completed. No real checkpoint, CUDA/Ascend
training, latency benchmark or OCR evaluation was run for this change.

Head-split self-attention validation (`student.attn_compression`, see [docs/attn_head_split.md](attn_head_split.md#validation)):
**65 tests in `tests/test_ablation_head_split.py` and `tests/test_ablation_data.py` passed**, and the existing
ablation suites (core, model, hourglass, sparse attention, res1024, dual resolution, logging, pixel, SDT) **218
passed, 7 skipped** on PyTorch 2.11 (CUDA host, tests on CPU). The release export restored through
`inference_vosr_ablation.py` scores Real-CE x4 OCR-A 0.656 / CER 0.125 / 1-NED 0.849 (PaddleOCR, `evaluate.py`).
