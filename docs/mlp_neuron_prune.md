# FFN neuron pruning on top of the head split (`student.mlp_compression`)

Goal: cut the MLP compute of [`head_split_f2`](attn_head_split.md) while keeping its text super-resolution quality
on Real-CE. The head split moved self-attention onto 25% of the tokens, but every block MLP still runs on all 1024
tokens of a 512 px tile: the MLPs are 54% of the head-split DiT (695.8 of 1279.9 GMACs). This series asks whether
the MLPs can be made narrower, how much narrower, and which part of the method carries the result.

The series starts from the release export [`weights/head_split_f2`](../weights/head_split_f2) (download the shards
of release `head-split-f2-v1` first, see its README) and repeats its training recipe, with the pruned student
distilled from the frozen dense VOSR2.

## Method

Every block MLP of VOSR2 is a SwiGLU, `w3(silu(x1) * x2)` with `x1, x2 = w12(h)` and 4096 hidden neurons.
`models/mlp_compression.py` keeps `keep_ratio` of each block's neurons. A kept neuron j keeps its rows j and
j + 4096 of `w12` and its column j of `w3`, so a pruned block is an ordinary, narrower `SwiGLUFFN`: the same weight
names, dense matmuls, and widths that are multiples of 128 (3072 or 2048). Nothing else in the network changes.

- **Ranking** (`score: fluctuation`): neuron j is kept by its fluctuation Var(a_j) * |g2 * w3[:, j]|^2, where a_j is
  its activation (the input of `w3`) and g2 is the block's MLP gate. This is the token-varying energy that the
  neuron writes into the residual stream; it is FLAP's fluctuation metric with VOSR2's adaLN gate, which is a
  constant of the one-step model (t=1, r=0). `score: random` keeps a random subset per block (`seed`) as the
  control.
- **Mean compensation** (`compensation: mean`): the pruned neurons' mean output `w3[:, drop] @ mean(a_drop)` is
  moved into the `w3` bias. A pruned neuron that never varied therefore changes nothing, and the error at step 0 is
  only the dropped neurons' variation. `compensation: none` drops them outright.
- **Whole MLPs** (`keep_ratio: 0` for a block): the block's MLP becomes `ConstantMLP`, one learned vector (initially
  the MLP's mean output) added to every token. Its `norm2` is removed as well, because a constant ignores its input
  (otherwise the norm weight gets no gradient and DDP fails).

The pruning is applied when full-width weights are loaded into the student. `load_backbone_state` calls the model's
`prepare_pretrained_state`; the source is the head-split export or, for `_joint`, the dense VOSR2. Checkpoints and
exports of the pruned student are already narrow and load unchanged. model.json stores the resolved structure (the
hidden width of every block), so resuming, exporting and `inference_vosr_ablation.py` do not need the statistics.

### Config keys

| Key | Meaning |
| --- | --- |
| `student.mlp_compression.type` | `neuron_prune` (the only type) |
| `keep_ratio` | share of the 4096 neurons each block keeps: one number for every block, or one per block (0 removes the block's MLP, 1 keeps it whole) |
| `score` | `fluctuation` (default) or `random` |
| `compensation` | `mean` (default) or `none` |
| `neuron_stats` | the `.safetensors` from `tools/mlp_neuron_stats.py`; needed for `fluctuation` and for `mean` (only `random` with `none` needs none) |
| `seed` | permutation seed of `score: random` (default 0) |

`mlp_compression` cannot be combined with `token_compression` or `router_config`, and it needs the SwiGLU MLPs
(`use_swiglu: true`). It combines with `attn_compression` and `attn_type`. Configs without the key are unchanged.

### Neuron statistics

`tools/mlp_neuron_stats.py` measures the network that the pruned student is initialized from. This is the config's
student without `mlp_compression`, loaded from `student_checkpoint` (or from the teacher when that is null), exactly
as the trainer loads it. For every block and neuron it stores the mean and variance of a_j over all tokens, the
output energy |g2 * w3[:, j]|^2, and |w3|_F^2 per block.

The measurement uses 64 text-centred crops of the config's paired training data (Real-CE x4 train) at the one-step
input: Gaussian noise, t=1, r=0, and the student's DINO features, all in FP32. The trainer compares |w3|_F^2 with
the checkpoint it prunes, and refuses statistics measured on another checkpoint.

```bash
# The release export, ranked once for every config that starts from it (k75, k50, random, nocomp, depth)
python tools/mlp_neuron_stats.py --config configs/ablations/ffn_prune_k75.yml \
  --output configs/ablations/ffn_prune/head_split_f2_neuron_stats.safetensors --device cuda
# ffn_prune_k75_joint starts from the dense VOSR2 with the untrained head split: its own statistics
python tools/mlp_neuron_stats.py --config configs/ablations/ffn_prune_k75_joint.yml \
  --output configs/ablations/ffn_prune/vosr2_neuron_stats.safetensors --device cuda
```

A config whose statistics file is missing stops at startup and names this tool.

## Compute

Per 512 px tile, one step, measured with `FlopCounterMode` at the real VOSR2 size (checked by
`tests/test_ablation_mlp_prune.py`). One full MLP costs 19.3 GMACs per block (1024 tokens, 18.9 M MACs per token).

| Student | MLP neurons per block | MLP MACs | DiT MACs vs dense VOSR2 | vs `head_split_f2` | MLP parameters |
| --- | --- | --- | --- | --- | --- |
| Dense VOSR2 | 4096 | 695.8 G | 1635.9 G (100%) | - | 679.8 M |
| `head_split_f2` / `ffn_prune_ctrl` | 4096 | 695.8 G | 1279.9 G (78.2%) | 100% | 679.8 M |
| `ffn_prune_k75` (`_random`, `_nocomp`, `_joint`) | 3072 | 521.8 G | 1106.0 G (67.6%) | 86.4% | 509.9 M |
| `ffn_prune_k75_depth` | 4096 in 27 blocks, 0 in 9 | 521.8 G | 1106.0 G (67.6%) | 86.4% | 509.9 M |
| `ffn_prune_k50` | 2048 | 347.9 G | 932.0 G (57.0%) | 72.8% | 339.9 M |

Latency has not been measured. Narrower dense matmuls usually come close to their MAC share on GPUs and NPUs, but
check this on the target device.

## Training-free probe

These numbers were measured on `weights/head_split_f2` before any training, with an exploration harness that is not
part of the repository. The harness ran 6 crops of rendered text, either downscaled 4x with bicubic ("mild") or
heavily degraded, and read the output with the PP-OCRv5 server recognizer. They decide the shape of the series, not its outcome.

The fluctuation is concentrated. On average the top 25% of a block's neurons hold 63% of the fluctuation energy
and the top 50% hold 82%. Blocks 0, 15, 18 and 25 are the most concentrated (top 25% above 83%).

Each variant below was applied without training (mild set, CER of the OCR read; "x0 err" is the relative error of
the predicted clean latent against the unpruned model):

| Variant | MLP MACs | DiT vs `head_split_f2` | CER | x0 err |
| --- | --- | --- | --- | --- |
| Unpruned `head_split_f2` | 100% | 100% | 0.198 | 0 |
| Fluctuation pruning, keep 0.88 | 88% | 93.5% | 0.242 | 0.043 |
| **Fluctuation pruning, keep 0.75** | **75%** | **86.4%** | **0.299** | **0.081** |
| Fluctuation pruning, keep 0.5 | 50% | 72.8% | 0.591 | 0.170 |
| 9 whole MLPs replaced by their mean, same MACs as keep 0.75 | 75% | 86.4% | 0.506 | 0.171 |
| Smooth neurons on 2x2-pooled tokens | 62.5% | 79.6% | 0.400 | 0.123 |
| Low rank in and out, same MACs as keep 0.5 | 50% | 72.8% | 0.749 | 0.261 |
| Every MLP on 2x2-pooled tokens | 25% | 59.2% | 1.000 | 0.459 |

On the heavily degraded set (unpruned CER 0.843, too low to rank by CER), random pruning at keep 0.5 doubles the
latent error of fluctuation pruning (x0 err 0.340 vs 0.208). Whole-MLP removal at keep 0.75's MACs again doubles
it (0.182 vs 0.092).

Replacing one block's MLP by its mean output, one block at a time (x0 err, heavy set, 2 crops), shows how
sensitive each block is:

| Blocks | x0 err |
| --- | --- |
| 5, 7, 6, 3, 2, 4 | 0.022 - 0.031 (least sensitive) |
| 30, 8, 29 | 0.035 - 0.038 |
| 28, 31, 20, 27, 21, 16, 19 | 0.042 - 0.048 |
| 1, 9-15, 17, 18, 22-26 | 0.050 - 0.068 |
| 32, 33, 34, 0, 35 | 0.070 - 0.099 (most sensitive) |

`ffn_prune_k75_depth` removes the nine least sensitive MLPs: blocks 2-8, 29 and 30.

Untrained, width pruning is the best way to cut the MLPs; depth, token pooling and low rank all lose more at the
same MACs. Training must close the gap from CER 0.198 to 0.299 on the mild set. The series tests whether 12k KD
steps do that, and whether the ranking and the compensation still matter once the student has trained.

## Ablation matrix

Every config inherits `head_split_f2.yml`'s recipe unchanged. That recipe is one-step velocity KD from the frozen
dense VOSR2 on text-centred Real-CE x4 crops, batch 4, lr 1e-5 with cosine decay to 0.1x, EMA 0.999, and 12k
steps. Each config changes one point of its `_base_`, which `tests/test_ablation_mlp_prune.py` checks:

| Config | `_base_` | Ablation point | Starts from | Steps | DiT vs `head_split_f2` | Question |
| --- | --- | --- | --- | --- | --- | --- |
| `ffn_prune_ctrl.yml` | `head_split_f2.yml` | `student_checkpoint: weights/head_split_f2`, no pruning | release export | 12k | 100% | How good does the export get with the same extra 12k steps? This is the reference for every other run. |
| `ffn_prune_k75.yml` | `ffn_prune_ctrl.yml` | `mlp_compression`: keep 0.75, fluctuation, mean | release export | 12k | 86.4% | Does 25% neuron pruning keep the text? (main run) |
| `ffn_prune_k75_random.yml` | `ffn_prune_k75.yml` | `score: random` | release export | 12k | 86.4% | Does the ranking matter after training? |
| `ffn_prune_k75_nocomp.yml` | `ffn_prune_k75.yml` | `compensation: none` | release export | 12k | 86.4% | Does mean compensation matter after training? |
| `ffn_prune_k75_depth.yml` | `ffn_prune_k75.yml` | `keep_ratio` per block: MLPs of blocks 2-8, 29, 30 removed, the rest whole | release export | 12k | 86.4% | Width or depth at equal MACs? |
| `ffn_prune_k50.yml` | `ffn_prune_k75.yml` | `keep_ratio: 0.5` | release export | 12k | 72.8% | The next compression level |
| `ffn_prune_k75_joint.yml` | `ffn_prune_k75.yml` | `student_checkpoint: null`, own `neuron_stats`, 24k steps | dense VOSR2 | 24k | 86.4% | One stage (head split and pruning together) or two stages, at the same 24k total steps? |

`ffn_prune_ctrl` is needed because every pruned run trains 12k steps more than the release export. Compare each
run with `ffn_prune_ctrl`, not with the release export's CER 0.125, so the difference is the pruning and not the
extra training.

## Run order and decision rules

Run the series in stages; each stage needs the results of the one before.

1. **Stage 1: `ffn_prune_ctrl` and `ffn_prune_k75`.** Is 25% neuron pruning viable at all?
2. **Stage 2 (attribution): `_random`, `_nocomp`, `_depth`.** Run these when Stage 1 passes, if you want to know
   which part of the method carries the result, or right away when Stage 1 is borderline (below).
3. **Stage 3: `ffn_prune_k50`, and `ffn_prune_k75_joint`.** Run `k50` only after `k75` passes. `_joint` is
   optional: it tells whether future compressions can skip the intermediate head-split training.

### Decision rules

Evaluate Real-CE x4 (261 images) with `evaluate.py` at noise seeds 42 and 43, and take the mean. Let
ΔCER = CER(run) - CER(`ffn_prune_ctrl`). For reference, the head split itself cost +0.015 CER (dense VOSR2 0.110,
release export 0.125).

| ΔCER of `ffn_prune_k75` | Verdict | Next |
| --- | --- | --- |
| ≤ +0.005 | pass | Stage 3: `ffn_prune_k50` under the same rule |
| +0.005 to +0.015 | borderline | Stage 2. Adopt a variant that reaches ≤ +0.005; otherwise also check the lines under 64 px (`--size-buckets 32 48 64 96`), where `head_split_f2` already loses most, and treat a larger gap there as a fail |
| ≥ +0.015 | fail | Narrow MLPs do not hold the text at 75%. Stop the width line. A gentler point (e.g. `keep_ratio: 0.875`) is one key away, but its saving is small (93.5% of `head_split_f2`) |

How to read the Stage 2 and Stage 3 runs:

| Run | Compare with | Read |
| --- | --- | --- |
| `_random` | `k75` | Within 0.005: training recovers the ranking, and the ranking is not worth carrying into later stages. Worse: keep `fluctuation` |
| `_nocomp` | `k75` | Within 0.005: the compensation only helps at step 0. Worse: keep `mean` |
| `_depth` | `k75` | Better or equal: removing whole MLPs is simpler to deploy (fewer kernels), so prefer it. Worse: the untrained probe's ordering holds after training |
| `_joint` | `k75` | Better or equal: one stage from dense VOSR2 is enough, and no intermediate export is needed |
| `k50` | `ffn_prune_ctrl` | Same rule as `k75`. A pass sets the next point; a fail makes `k75` the operating point |

## Commands

```bash
gh release download head-split-f2-v1 --repo yqwu905/VOSR --dir weights/head_split_f2 --pattern "*.safetensors"
python tools/prepare_realce.py --realce /path/to/RealCE --output preset/datasets/RealCE_x4   # if not done yet
python tools/mlp_neuron_stats.py --config configs/ablations/ffn_prune_k75.yml \
  --output configs/ablations/ffn_prune/head_split_f2_neuron_stats.safetensors --device cuda

# Stage 1 (one GPU, batch 4, as head_split_f2; keep world size x batch at 4)
bash train_and_infer.sh --config configs/ablations/ffn_prune_ctrl.yml --nproc 1 \
  --testset realce_x4=preset/datasets/RealCE_x4/val/LR
bash train_and_infer.sh --config configs/ablations/ffn_prune_k75.yml --nproc 1 \
  --testset realce_x4=preset/datasets/RealCE_x4/val/LR

python evaluate.py --pred <run>/inference/realce_x4 --ann preset/datasets/RealCE_x4/val/Label.txt \
  --ann-ref pred --size-buckets 32 48 64 96 --output <run>/inference/realce_x4.eval
```

The export of a pruned run restores like any other: `python inference_vosr_ablation.py --export <run>/export ...`.
model.json carries `attn_compression` and the per-block MLP widths, so no statistics file is needed for inference.

## Validation

`tests/test_ablation_mlp_prune.py` (CPU, 34 tests; the whole suite: 382 passed, 5 skipped) covers:

- option validation and spec resolution;
- keep_ratio 1 equal to the unpruned model;
- fluctuation and random selection, and the mean compensation against a reference in which the dropped
  activations are held at their mean;
- whole-MLP removal and the norm2 it no longer needs;
- the statistics fingerprint and errors for missing or malformed files;
- the export round trip and resume without the statistics;
- composition with the head split, and backward reaching every parameter;
- rejection of `router_config` and of non-SwiGLU MLPs;
- the MAC ratios above at the real VOSR2 size;
- each shipped config changing only its ablation point.

An end-to-end test runs `tools/mlp_neuron_stats.py` on a head-split export, trains the pruned student with the
trainer (pruning at load), resumes with the statistics file deleted, and restores the export through
`inference_vosr_ablation.py`. No real checkpoint, GPU training or OCR evaluation was run for this change.
