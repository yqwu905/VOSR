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
default, plus non-reentrant activation checkpointing. It is not FSDP and does not
shard model weights/gradients. Memory fit and throughput must be measured on your
hardware; teacher + DINO + VAE also occupy memory. There is no validated Ascend/NPU
path, fp16 training, EMA, or automatic validation set in this new trainer. LPIPS and
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

For the DyDiT config the files are:

- `exp_vosr/dydit_sdt/metrics.jsonl`: scalar records, including the optimizer step.
- `exp_vosr/dydit_sdt/logs/vosr-ablation/`: TensorBoard events.
- `exp_vosr/dydit_sdt/logs/wandb/`: W&B run data (offline by default).
- `exp_vosr/dydit_sdt/previews/step-00000500/sample-00.png`: local comparison image.

```bash
tensorboard --logdir exp_vosr/dydit_sdt/logs --port 6006

# To log online instead of offline:
wandb login
WANDB_MODE=online torchrun --standalone --nproc_per_node=8 \
  train_vosr_ablation.py --config configs/ablations/dydit_sdt.yml

# Or sync one completed offline run later:
wandb sync exp_vosr/dydit_sdt/logs/wandb/offline-run-<timestamp>-<id>
```

`report_to: none` disables both trackers but keeps console, JSONL and local PNG
output. `preview_every: 0` independently disables previews. On `--resume`, JSONL
appends, TensorBoard removes events after the restored checkpoint step, and W&B
starts a new run using the restored optimizer-step axis (also recorded in the run
config). Existing preview PNGs at repeated steps are replaced. Do not change the
model/data/optimization config when resuming. Logging and preview settings may
change, so existing compatible training checkpoints can enable the new trackers.

## Objective and routing scope

Training uses the deployment endpoint `t=1, r=0` only. Given identical LQ latent
and noise, the student matches the frozen one-step teacher's velocity:

```text
input = concat(LQ_latent, noise)
L_KD = MSE(student(input, 1, 0), teacher(input, 1, 0))
SR_latent = noise - student_velocity
L_GT = MSE(SR_latent, GT_latent)                 # optional, default weight 0
SR = VAE_decode(SR_latent)                       # only when a pixel loss is enabled
L_LPIPS = LPIPS(SR, HQ)                          # optional, default weight 0
L_KL_OCR = KL(OCR(HQ) || OCR(SR))                # optional, default weight 0
L_budget = (mean_rank_layer(expected_keep) - target_keep)^2  # per micro-batch
```

This preserves the one-step task; it does not claim to train a valid arbitrary-time
flow or multistep/RCGM model. Only one-step inference is provided for these exports.

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

Routing and token merging cannot be combined: their budgets are in different
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
| `factor` | 2 | Merge factor x factor tokens; latent sides must be multiples of `patch_size * factor` |
| `fine_in`, `fine_out` | 1, 1 | Leading / trailing pretrained blocks kept on the full grid |
| `drop_blocks` | `[]` | Pretrained block indices removed entirely; their weights are discarded on load. `feature_distill_layers` must not list them (`hourglass.yml` lists 18), or training stops at startup |
| `bypass` | true | `false` uses `Up(y_out)` alone, collapsing each group to one value (large-patch control) |
| `rope` | `centroid` | Merged-token RoPE at the centroid of its sub-tokens; `corner` uses the top-left sub-token, as `_get_dynamic_rope` would |
| `fine_cross_attention` | true | Keep the pretrained DINO CA in the full-grid blocks. `false` discards those weights on load and saves 0.8% of the DiT MACs, but puts the step-0 student about 21 dB PSNR from the teacher (see below) |
| `cond_pool` | 1 | Average-pool DINO tokens inside the model before `mlp_ca` |

DiT MACs of this implementation for one 512^2 tile, counted with
`torch.utils.flop_counter.FlopCounterMode` on the meta device (GMACs = GFLOPs / 2;
the student's DINO gives 256 tokens unless noted; DINO and VAE are not included):

| Structure | Config | GMACs | Of dense |
| --- | --- | --- | --- |
| Dense teacher, 1024 DINO tokens | `base_vosr2.yml` | 1635.9 | 100% |
| Blocks 0/35 full grid with CA, 1-34 merged | `hourglass.yml` | 428.5 | 26.2% |
| Same with `fine_cross_attention: false` | example only | 414.9 | 25.4% |
| Blocks 0-1/34-35 full grid, 2-33 merged | `hourglass_2_32_2.yml` | 486.5 | 29.7% |
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
be multiples of `8 * patch_size * factor` pixels (32 for the shipped configs; the
default 512/64 qualifies). Token merging uses PyTorch reshape/permute; no ONNX or
NPU export has been attempted, and an on-device graph should express these
rearrangements as SpaceToDepth/DepthToSpace.

## Checkpoints, resume, and inference

Each `checkpoint-XXXXXXXX/` includes model weights, model/pipeline JSON, and
`training_state.pt` containing optimizer state, step and resolved config. Resume
with the same model/data/optimization config (logging/preview options may change):

```bash
torchrun --nproc_per_node=8 train_vosr_ablation.py \
  --config configs/ablations/dydit_sdt.yml \
  --resume exp_vosr/dydit_sdt/checkpoint-00001000
```

Resume restores optimization state and step-based schedules, **not exact data
loader position or worker RNG**; it is not bitwise replay. Fresh runs refuse a
nonempty output directory. Intermediate fade checkpoints retain CA parameters
and their current scale. Only the final `export/` strips inactive conditioning;
use a training checkpoint, not the inference-only export, for resume.

```bash
python inference_vosr_ablation.py \
  --export exp_vosr/no_dino_no_ca/export \
  --input preset/datasets/inp_data --output preset/results/no_dino --upscale 4

python inference_vosr_ablation.py \
  --export exp_vosr/dydit_sdt/export \
  --input preset/datasets/inp_data --output preset/results/dydit_sdt --upscale 4

# Multi-NPU/GPU data-parallel inference: each rank restores a disjoint shard of images
torchrun --nproc_per_node=8 inference_vosr_ablation.py \
  --export exp_vosr/dydit_sdt/export \
  --input preset/datasets/inp_data --output preset/results/dydit_sdt --upscale 4
```

The new inference entrypoint reads `model.json`; do not load these exports with
the unchanged upstream inference script. It supports Qwen 2D VAE, per-tile
conditioning, globally shared noise, Gaussian latent-velocity blending, and
optional tiled VAE encode/decode. Non-square images are padded/tiled into square
DiT inputs, then cropped to the requested size. There is no automatic color fix.
Use `--vae-path` to relocate VAE assets, and `--dense-mlp` for a backend without
sparse gather/scatter. No-DINO exports do not instantiate or fetch the encoder.
Under `torchrun`, images are split round-robin across ranks (no inter-rank
communication); the RNG is reseeded with `--seed` per image, so outputs are
identical regardless of the number of devices.

## Validation

```bash
python -m compileall -q models/sdt_router.py models/lightningdit_ablation.py models/token_hourglass.py \
  ablation_utils.py train_vosr_ablation.py inference_vosr_ablation.py
TORCHDYNAMO_DISABLE=1 python -m pytest -q tests/test_ablation_core.py tests/test_ablation_data.py tests/test_ablation_logging.py tests/test_ablation_model.py tests/test_sdt_distributed.py tests/test_ablation_hourglass.py
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
