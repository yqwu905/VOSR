# VOSR2 fine-tuning ablations

These are **opt-in, one-step teacher-preserving fine-tuning experiments**. The
upstream `LightningDiT`, original FM/RCGM trainers, and inference scripts are not
modified. This is not a reconstruction of the unpublished VOSR2 data pipeline or
original training recipe, and text-quality retention has not been demonstrated.

For the opt-in **arbitrary-time, multistep pixel PDM (route B)**, use
[`configs/pixel_pdm/`](../configs/pixel_pdm/) and [pixel_pdm_training.md](pixel_pdm_training.md).
That objective, staged LR schedule and ODE sampler are separate from the endpoint ablations below.

## Configurations

| Config in `configs/ablations/` | Student during training | Exported student |
| --- | --- | --- |
| `no_dino/no_dino_no_ca.yml` | No DINO features, no CA modules or feature projector; retain LQ latent concatenation | Physically no CA/projector weights; inference never loads DINO |
| `sdt/dydit_sdt.yml` | DINO/CA retained; global MLP budget; deterministic learned-capacity top-k with STE | Same capacity/ranking policy; optional sparse MLP gather/scatter |
| `no_dino/no_dino_fade.yml` | Optional transition: freeze CA/projector, scale CA residuals from 1 to 0 over 5,000 optimizer steps | Final export removes CA/projector weights |
| `no_dino/no_dino_sft.yml` | The objective of `loss/loss_all.yml` on the no-DINO/CA student, initialized from a `no_dino/no_dino_fade.yml` export (set `student_checkpoint`) | No CA/projector weights |
| `base/base_sft.yml`, `loss/loss_lpips.yml`, `loss/loss_ocr_kl.yml`, `loss/loss_all.yml` | Base student (dense, DINO/CA kept) trained by plain SFT without the teacher term: GT, plus LPIPS, KL-OCR or both, see [Loss ablation configs](#loss-ablation-configs) | Teacher architecture |
| `sparse/sft_sparse_attn_lpips.yml` | As `loss/loss_lpips.yml` with block-sparse self-attention (`attn_type: sparse`), see [Block-sparse self-attention](#block-sparse-self-attention-studentattn_type) | Teacher architecture, sparse self-attention |
| `res1024/res1024.yml` and `res1024_*.yml` | Dense student on 1024 crops (one 1024 tile at inference), extrapolated RoPE, teacher tiled at 512; each `res1024_*.yml` changes one point, see [1024 resolution](#1024-resolution-res1024yml) | Teacher architecture |
| `head_split/head_split_f2.yml`, `head_split/head_split_f4.yml` | Self-attention token reduction by head split: local heads (3x3 attention mass > 0.5) become 7x7 depthwise convs, the other heads attend among 2x2 (4x4) pooled tokens; CA and MLP keep every token. KD from the dense teacher on Real-CE pairs (`data.dataset_type: paired`), cosine LR, EMA, see [docs/attn_head_split.md](attn_head_split.md) | Self-attention on 25% (6.25%) of the tokens: 78.2% (74.6%) of the teacher's DiT MACs; release export `weights/head_split_f2` |
| `ffn_spatial/ffn_spatial_f*_c*.yml` (random and `_calibrated` variants) | Split each SwiGLU's hidden units across full and 2x2-pooled token grids; both branches read all channels, the coarse branch projects before upsampling. Five tiers, explicit calibrated index files, checkpoint conversion and training from `head_split_f2`; see [docs/ffn_mixed_resolution.md](ffn_mixed_resolution.md) | FFN projection MACs: 100%, 25%, 43.75%, 34.375%, 25% for 4096+0, 0+4096, 1024+3072, 512+3584, 512+2048; not end-to-end speed claims |
| `pixel/pixel_l2p.yml`, `pixel/pixel_l2p_unet.yml` | `loss/loss_all.yml` in pixel space after L2P: no VAE, 16 px patchify (or the decoder's own down path) in, the MicroDiffusionModelTop3SCWide U-Net out, DINO/CA kept, blocks 5-30 frozen, see [Pixel-space student](#pixel-space-student-l2p-pixel_l2pyml) | No VAE; 69.7% of the latent pipeline's MACs |

The full frozen VOSR2 **teacher still uses DINO during training**, including the
no-DINO student experiment. Removing the teacher encoder would change the target
and is not equivalent. The fade config is an extra option for gradual removal;
the strict no-CA config removes the branch from student construction onward.

## Preparation and launch

Use the upstream VOSR environment (Python 3.10+ and PyTorch 2.x recommended).
Set `teacher_checkpoint`, `vae_path`, and `data.train_dataset_config` in
`configs/ablations/base/base_vosr2.yml`. The checkpoint may be an exact clean
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
  --config configs/ablations/no_dino/no_dino_no_ca.yml

torchrun --nproc_per_node=8 train_vosr_ablation.py \
  --config configs/ablations/sdt/dydit_sdt.yml

# Optional gradual removal from the pretrained model.
torchrun --nproc_per_node=8 train_vosr_ablation.py \
  --config configs/ablations/no_dino/no_dino_fade.yml
```

For one device use `python` instead of `torchrun`. Effective batch size is
`world_size * batch_size_per_gpu * gradient_accumulation_steps` (default 8 on
8 GPUs). CUDA DDP uses optimizer-state sharding via `ZeroRedundancyOptimizer` by
default. Non-reentrant activation checkpointing is off by default; set
`training.gradient_checkpointing: true` when activations do not fit (it may be changed on
resume). It is not FSDP and does not shard model weights/gradients. Memory fit and throughput must be measured on your
hardware; teacher + DINO + VAE also occupy memory. There is no validated Ascend/NPU
path or fp16 training in this trainer. EMA is opt-in; route B also supports an explicit
held-out paired dev set (see its guide). On NPU the
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
  train_vosr_ablation.py --config configs/ablations/sdt/dydit_sdt.yml

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
flow or multistep/RCGM model. These endpoint exports use one-step inference; route B
pixel-flow exports use the separate multistep path described in [its guide](pixel_pdm_training.md).

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

For a controlled new experiment, use the updated `sdt/dydit_sdt.yml` with a fresh
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

`base/base_sft.yml` is plain SFT: it inherits `base/base_vosr2.yml`, sets `kd_weight: 0` (the
teacher is not loaded) and turns on `gt_weight: 1.0`. The loss configs inherit
`base/base_sft.yml` and change only loss weights and `output_dir`, and
`sparse/sft_sparse_attn_lpips.yml` also sets `student.attn_type: sparse`
(`tests/test_ablation_core.py` checks this). The student keeps the teacher's dense
architecture with DINO and CA, starts from the same VOSR2 weights and uses the same
data, schedule and optimizer. `base/base_sft.yml` is the control; `base/base_vosr2.yml` stays
the teacher-KD baseline.

| Config | `kd_weight` | `gt_weight` | `lpips_weight` | `ocr_kl_weight` | `attn_type` |
| --- | --- | --- | --- | --- | --- |
| `base/base_vosr2.yml` (teacher KD) | 1 | 0 | 0 | 0 | full |
| `base/base_sft.yml` (control) | 0 | 1.0 | 0 | 0 | full |
| `loss/loss_lpips.yml` | 0 | 1.0 | 0.02 | 0 | full |
| `loss/loss_ocr_kl.yml` | 0 | 1.0 | 0 | 0.0005 | full |
| `loss/loss_all.yml` | 0 | 1.0 | 0.02 | 0.0005 | full |
| `sparse/sft_sparse_attn_lpips.yml` | 0 | 1.0 | 0.02 | 0 | sparse |

`loss/loss_lpips.yml` is the full-attention control of `sparse/sft_sparse_attn_lpips.yml`.

The weights balance gradients, not loss values. Without `kd`, AdamW makes the step
size nearly independent of the overall loss scale; what the weights set is the
relative pull of each loss. The LPIPS and KL-OCR weights give the same step-0 gradient
norm on the student's velocity output as `gt_weight: 1.0`: the median ratio over 20
ScreenSR `HR_512` text images, with the training degradation and the real VOSR2, Qwen
VAE and DINOv2 weights (fp32, CPU), is 0.016 for LPIPS and 0.00054 for KL-OCR, with a
roughly tenfold spread across images. Matching loss values instead would make LPIPS
pull about 17 times and KL-OCR about 330 times harder than GT, because the OCR's CTC
distributions are sharp. The three gradients are nearly orthogonal (median |cos| 0.002
to 0.024 per pair), so `loss/loss_all.yml` keeps each single-loss weight. These are starting
points, not tuned values.

```bash
for name in base/base_sft loss/loss_lpips loss/loss_ocr_kl loss/loss_all sparse/sft_sparse_attn_lpips; do
  torchrun --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/$name.yml
done
```

Nothing holds the output near the teacher in these runs, and the GT MSE pulls it toward
the posterior mean (see [Objective and routing scope](#objective-and-routing-scope)):
expect blur in `base/base_sft.yml`, which LPIPS and KL-OCR partly counter, and more
forgetting outside the training data than with `base/base_vosr2.yml`. Setting `kd_weight`
to a small positive value (e.g. 0.1) keeps a weak teacher anchor.

To start from a finished run instead of the VOSR2 weights, set `student_checkpoint`
to one of its `checkpoint-*` directories in every config, and run `base/base_sft.yml` again
from that checkpoint in a new output directory as the control.

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
windows). `sparse` keeps exactly 8 windows, so with 8 or fewer windows it is
computed as full attention. The masks are dense boolean tensors
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
| `data.crop_size` | pixels, default `data.resolution` | HQ crop size of `TxtPairDataset` (which bicubic-upscales images whose short side is smaller). The export's `pipeline.json` `resolution` (the default inference tile) is this size |
| `training.teacher_tile`, `training.teacher_tile_overlap` | pixels, default overlap 64 | The frozen teacher runs a larger crop as overlapping tiles, each with the DINO of its own LQ crop (top-level `dino`), and Gaussian-blends the velocities, as `inference_vosr_ablation.py --tile-size 512` does. Previews use the same target. The top-level DINO of the whole crop is then not computed unless the student uses the same settings. Rejected with feature distillation when a crop exceeds the tile |
| `training.vae_tile_size` | pixels, `0` = whole image, default `data.crop_size` | VAE encode/decode in training, previews and pixel losses. Below the crop size it uses Gaussian-blended tiles with 64 px overlap (`min(64, max(8, tile / 8))`), like inference. The default is one pass per crop, as before. The export's `pipeline.json` records it as the default `--vae-tile-size`. The Qwen VAE has a global self-attention in its encoder and decoder mid blocks, so tiles and the whole image give different latents |

| Config | `_base_` | Ablation point |
| --- | --- | --- |
| `res1024/res1024.yml` | `base/base_vosr2.yml` | Stage 1: extrapolated RoPE, 1024 crops only, DINO per 512 px (teacher and student), teacher tiled at 512 with 64 px overlap |
| `res1024/res1024_rope_interp.yml` | `res1024/res1024.yml` | `rope_mode: interpolate` |
| `res1024/res1024_dino_whole.yml` | `res1024/res1024.yml` | Student DINO on the whole crop at 448 (1024 tokens, the 512 cost) |
| `res1024/res1024_loss_all.yml` | `res1024/res1024.yml` | Plain SFT, no teacher term (the teacher is not loaded): GT 1.0, LPIPS 0.02 and KL-OCR 0.0005 (the `loss/loss_all.yml` losses) on the whole decoded 1024 crop |

Zero-shot evidence for the defaults (official weights, no training, six rendered 1024^2 text
pages and two DIV2K crops; OCR CER on 21-28 / >=29 px lines): 512 tiling 0.608 / 0.111; one 1024
tile with interpolated RoPE 0.994 / 0.998 (striped garbage); extrapolated RoPE with whole-crop
DINO 0.682 / 0.231; with DINO per 512 px 0.650 / 0.139. Whole-image vs 512-tiled VAE on the same
text pages (fixed noise): the LQ latent differs by 0.6% (relative L2), the outputs by 49-58 dB PSNR,
and CER by at most 0.002 for both the one-tile and the 512-tiled DiT.

Cost (FlopCounterMode on meta tensors, not measured on hardware): one 1024 step of `res1024/res1024.yml`
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

Train `res1024/res1024.yml` and the ablations independently, then restore with an export:

```bash
torchrun --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/res1024/res1024.yml

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

| Part | `pixel/pixel_l2p.yml` |
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

**Training recipe** (`pixel/pixel_l2p.yml`, `_base_: loss_all.yml`, so its control is the latent
`loss/loss_all.yml` with the same data, schedule and initialization):

| Item | Setting | Source |
| --- | --- | --- |
| Initialization | The official VOSR2 weights (`teacher_checkpoint`, `student_checkpoint: null`), DINO/CA included | As `loss/loss_all.yml` |
| Trainable | Blocks 0-4 and 31-35 and every module outside the blocks (patch embedding, time embedding, `t_block`, DINO projector `layer_norm` and `mlp_ca`, decoder; the CA of blocks 5-30 stays frozen): 413.9 M of 1396.3 M parameters, via `training.trainable_parameters` | L2P freezes layers `5 .. n - 5` (`freeze_except` in `base_pipeline_L2P.py`) |
| Optimizer | AdamW, learning rate 5e-5 (warmup 100), weight decay 0.01, batch 1 x 8 ranks, 10,000 steps | L2P's training script: AdamW 5e-5, batch 8; steps as `loss/loss_all.yml` |
| Losses | GT 1.0 (pixel MSE of the image vs the HQ crop), LPIPS-vgg 0.013, KL-OCR 0.0008 on the image itself, no VAE decode | `loss/loss_all.yml` terms; weights below |
| Teacher | None. The latent teacher's velocity has no pixel counterpart; `kd_weight` or `feature_distill_weight` above 0 is rejected | |

The loss weights follow the rule of `loss/loss_all.yml` (same step-0 gradient norm as `gt_weight: 1.0`),
now on the predicted image. The pixel student's own step-0 output is no reference point (its decoder
starts from scratch), so VOSR2's one-step output on the same crop stands in for a trained one: over 12
ScreenSR text crops (fp32, CPU, pyiqa LPIPS-vgg and the PP-OCRv5 port, through `lpips_loss` and
`build_ocr_kl`) the median ratio is 0.0127 for LPIPS (0.004-0.027) and 0.00081 for KL-OCR
(0.00024-0.011); taken at the LQ image instead, 0.0093 and 0.00071. `pixel/pixel_l2p.yml` uses 0.013 and
0.0008. These are starting points, not tuned values.

**Alternative encoder** (`pixel/pixel_l2p_unet.yml`, `_base_: pixel_l2p.yml`, only `encoder: unet`): the
pretrained blocks were trained on tokens of VAE latents, a deep non-linear encoding of the pixels,
which one linear map of raw 16 x 16 patches cannot imitate; with `patch` the first five blocks must
absorb that change. The decoder's down path already computes 384-channel features at 1/16 of the
image, so `unet` embeds those (1x1 conv 384 -> 1536) as the tokens: a non-linear, multi-scale stem
at no extra cost, trained from both the DiT and the decoder side. L2P, DiP and JiT all use a linear
patchify; this variant is untested.

Cost per 512^2 tile, batch 1 (FlopCounterMode on meta tensors, GMACs):

| Pipeline | DiT incl. embedding and CA | DINOv2 | Decoder | VAE encode + decode | Total | Of latent |
| --- | --- | --- | --- | --- | --- | --- |
| `loss/loss_all.yml` (latent) | 1635.9 | 271.5 | - | 317.8 + 550.1 | 2775.4 | 100% |
| `pixel/pixel_l2p.yml` | 1638.0 | 271.5 | 23.8 | - | 1933.3 | 69.7% |
| `pixel/pixel_l2p_unet.yml` | 1636.2 | 271.5 | 23.8 | - | 1931.5 | 69.6% |
| `pixel/pixel_l2p.yml` with `patch_size: 32` (example only) | 529.1 | 271.5 | 26.2 | - | 826.8 | 29.8% |

DINOv2 is the first 18 layers of DINOv2-L at 448 input. At patch 16 the DINO branch (DINOv2 plus CA
and its projector, 747.5) is the same in both pipelines, so the saving is the VAE. At patch 32 the
decoder has 14.4 M parameters, 9.4 M of them in `c_upsample`.

```bash
torchrun --nproc_per_node=8 train_vosr_ablation.py --config configs/ablations/pixel/pixel_l2p.yml
python inference_vosr_ablation.py --export exp_vosr/pixel_l2p/<run>/export \
  --input preset/datasets/inp_data --output preset/results/pixel_l2p --upscale 4
```

Exports record `pixel` in `model.json` and `vae_path: null` in `pipeline.json`; inference loads no
VAE and tiles in pixels (tile and overlap multiples of `patch_size`); each tile gets the DINO of its
own LQ crop, as in the latent pipeline. `student.pixel` does not combine with `router_config`.
Compare `pixel/pixel_l2p.yml` and `pixel/pixel_l2p_unet.yml` with `loss/loss_all.yml` at the same
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
  --config configs/ablations/sdt/dydit_sdt.yml \
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
bash train_and_infer.sh --config configs/ablations/head_split/head_split_f2.yml \
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

Run the fast regression suite from the repository root:

```bash
python -m pytest -q
```

Trainer subprocesses, real tracker I/O, export CLIs and external OCR checks are marked `integration`:

```bash
python -m pytest -q -m integration
python -m pytest -q -m ""  # all unit and integration tests
VOSR_TEST_DDP=1 python -m pytest -q -m "" tests/test_ablation_logging.py tests/test_sdt_distributed.py
```

Tests use tiny CPU models and local fixture data; no model downloads are needed.
CPU tensor threads default to one to avoid overhead on small tensors; set `VOSR_TEST_THREADS` to override.
The two-rank Gloo checks are opt-in through `VOSR_TEST_DDP=1`. These tests do not measure
CUDA/NPU throughput or image quality with real release checkpoints.
