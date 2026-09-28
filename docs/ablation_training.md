# VOSR2 no-DINO / no-CA and DyDiT-SDT ablations

These are **opt-in, one-step teacher-preserving fine-tuning experiments**. The
upstream `LightningDiT`, original FM/RCGM trainers, and inference scripts are not
modified. This is not a reconstruction of the unpublished VOSR2 data pipeline or
original training recipe, and text-quality retention has not been demonstrated.

## Configurations

| Config in `configs/ablations/` | Student during training | Exported student |
| --- | --- | --- |
| `no_dino_no_ca.yml` | No DINO features, no CA modules or feature projector; retain LQ latent concatenation | Physically no CA/projector weights; inference never loads DINO |
| `dydit_sdt.yml` | DINO/CA retained to isolate MLP spatial routing | Deterministic router; optional sparse MLP gather/scatter |
| `no_dino_fade.yml` | Optional transition: freeze CA/projector, scale CA residuals from 1 to 0 over 5,000 optimizer steps | Final export removes CA/projector weights |

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
path, fp16 training, EMA, automatic validation set, or automatic OCR loss in this
new trainer. Logging is local JSONL; no external experiment-tracking service is used.

An optional offline DINO repository can be set with `dino.local_repo`; its weights
must also be available in the configured torch.hub cache.

## Objective and routing scope

Training uses the deployment endpoint `t=1, r=0` only. Given identical LQ latent
and noise, the student matches the frozen one-step teacher's velocity:

```text
input = concat(LQ_latent, noise)
L_KD = MSE(student(input, 1, 0), teacher(input, 1, 0))
SR_latent = noise - student_velocity
L_GT = MSE(SR_latent, GT_latent)                 # optional, default weight 0
L_budget = mean_layer((expected_keep - target_keep)^2)
```

This preserves the one-step task; it does not claim to train a valid arbitrary-time
flow or multistep/RCGM model. Only one-step inference is provided for these exports.

Routing is an independent adaptation of the **SDT component** of
[DyDiT](https://github.com/alibaba-damo-academy/DyDiT), informed by its
`DyDiT/models.py` and `DyDiT/dynamic_model.py`. It is **not the complete DyDiT**:
there is no timestep-dependent width/head routing (TDW). Attention retains all
spatial tokens and unchanged RoPE positions. Each block predicts a spatial MLP
mask from its residual stream. Training uses logistic/Gumbel-sigmoid noise and a
hard straight-through mask. Evaluation thresholds deterministic logits.

The example starts near all-keep, uses 500 dense warmup steps, then ramps the
expected MLP keep budget to 0.75 over 4,500 steps. An optional dense sandwich
teacher-matching pass has weight 0.1. Router initialization occurs after backbone
initialization, so pretrained weights and the keep bias are not overwritten.

**Training still executes dense MLPs.** In no-grad evaluation, selected tokens can
be gathered, processed by the MLP, and scattered back; unselected tokens retain the
residual stream. `--dense-mlp` instead computes a dense MLP and applies the same
mask. Neither option compresses attention tokens. A 0.75 expected training keep
budget is not a guarantee of 25% total FLOP reduction, wall-clock speedup, or even
exactly 0.75 deterministic inference keep. Inspect `deterministic_mlp_keep` in
inference logs. Gather/scatter/nonzero require target-backend support.

## Checkpoints, resume, and inference

Each `checkpoint-XXXXXXXX/` includes model weights, model/pipeline JSON, and
`training_state.pt` containing optimizer state, step and resolved config. Resume
with the same config:

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
```

The new inference entrypoint reads `model.json`; do not load these exports with
the unchanged upstream inference script. It supports Qwen 2D VAE, per-tile
conditioning, globally shared noise, Gaussian latent-velocity blending, and
optional tiled VAE encode/decode. Non-square images are padded/tiled into square
DiT inputs, then cropped to the requested size. There is no automatic color fix.
Use `--vae-path` to relocate VAE assets, and `--dense-mlp` for a backend without
sparse gather/scatter. No-DINO exports do not instantiate or fetch the encoder.

## Validation

```bash
python -m compileall -q models/sdt_router.py models/lightningdit_ablation.py \
  ablation_utils.py train_vosr_ablation.py inference_vosr_ablation.py
TORCHDYNAMO_DISABLE=1 python -m pytest -q tests/test_ablation_core.py tests/test_ablation_data.py tests/test_ablation_model.py
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

TXT-migration validation: **34 CPU tests passed; the full-backbone test module was
skipped because timm was unavailable**. Python compilation and the trainer's
`--help` check also passed. No real VOSR2 checkpoint, full RealESRGAN degradation,
end-to-end image inference, CUDA distributed training, throughput benchmark, or
TextSR quality evaluation was run in this environment. Before long training,
run the full suite in the VOSR environment, then a short real-checkpoint smoke run
and fixed-seed OCR/text-crop comparisons against the full teacher.
