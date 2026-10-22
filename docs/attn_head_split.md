# Head-split self-attention token reduction (`student.attn_compression`)

Goal: cut the tokens that VOSR2's **self-attention** processes to 25% (or fewer) while keeping text
super-resolution quality on Real-CE. Cross-attention to DINO and the MLP are left untouched; the block-sparse
attention of `student.attn_type` is a separate, already validated option and still applies on top (see below).

Release weights: [`weights/head_split_f2/`](../weights/head_split_f2) (the two shards are attached to the GitHub
release `head-split-f2-v1`, see its README; LFS uploads to this public fork are refused), trained with
[`configs/ablations/head_split_f2.yml`](../configs/ablations/head_split_f2.yml).

## Method

VOSR2 (36 blocks x 24 heads) has two kinds of self-attention heads. Most attend globally, but 143 of 864 heads
put more than half of their attention on the 3x3 token neighbourhood of the query, many of them almost entirely on
the query itself or on one fixed neighbour. Attention among pooled tokens cannot represent such a head: after
2x2 average pooling "attend to yourself" becomes "attend to your 2x2 group", which blurs strokes. This is why plain
token pooling of self-attention loses most of the OCR gain (table below).

`models/attn_compression.py` therefore splits the heads of every block:

- **local heads** (dense 3x3 attention mass > `local_threshold`, 0.5) become a `kernel` x `kernel` (7x7) depthwise
  conv over their full-grid values: no queries, keys or softmax. Each conv starts from the head's measured mean
  attention weight at every offset around the query (renormalized to sum 1), i.e. the conv is initialized to the
  head's average attention pattern;
- **global heads** attend among the `factor` x `factor` average-pooled tokens (factor 2: 256 of 1024 tokens per
  512 px tile; RoPE at the group centroids). Their output projection runs on the pooled grid and is upsampled
  bilinearly to the full grid.

Both parts use their heads' columns of the pretrained output projection and add up, so all pretrained weights
keep their names and meaning; the only new parameters are the local convs (`blocks.N.attn.local_conv.weight`).
With `factor: 1` and no local heads the module is exactly the original attention.

The per-head statistics come from `tools/attn_head_stats.py` (dense teacher, 4 text-centred Real-CE crops); the
shipped result is [`configs/ablations/head_split/vosr2_head_stats.json`](../configs/ablations/head_split/vosr2_head_stats.json).

Compute per 512 px tile (one step, `FlopCounterMode`, checked by `tests/test_ablation_head_split.py`):

| Student | Self-attention tokens | Self-attention MACs | DiT MACs |
| --- | --- | --- | --- |
| Dense VOSR2 | 1024 (100%) | 463.9 G (100%) | 1635.9 G (100%) |
| `head_split_f2.yml` | 256 (25%) | 107.9 G (23.3%) | 1279.9 G (78.2%) |
| `head_split_f4.yml` | 64 (6.25%) | 47.8 G (10.3%) | 1219.8 G (74.6%) |

## Training recipe

One-step distillation from the frozen dense VOSR2: MSE between student and teacher velocities at t=1, r=0, with
the same LQ latent, noise and DINO features. The student starts from VOSR2 (all weights trainable), the local convs
from the head patterns. Data: Real-CE x4 train pairs (576 images, disjoint from the test set), 512 px crops, 80%
centred on an annotated text box (`data.dataset_type: paired`). AdamW (0.9, 0.95), lr 1e-5 (local convs 10x),
100 warmup steps, cosine decay to 0.1x over 12k steps, weight decay 0, batch 4 on one GPU, EMA 0.999 (the export
holds the EMA weights).

New trainer options (all optional; absent keys keep existing configs unchanged):

| Key | Meaning |
| --- | --- |
| `student.attn_compression` | `type: head_split`, `factor`, `kernel`, `upsample`, `blocks`, and `head_stats` + `local_threshold` (or explicit `local_heads`) |
| `data.dataset_type: paired` | real LQ/HQ pairs: `lq_dir` (LQ already upsampled to the HQ size), `hq_dir`, optional `text_boxes` / `text_crop_prob`, `upscale` |
| `training.lr_schedule`, `training.min_lr_ratio` | `constant` (default) or `cosine` after warmup |
| `training.ema_decay` | EMA of the trainable weights; checkpoints get `ema/`, `export/` holds the EMA |
| `training.token_merge_lr_multiplier` | now also applies to the local convs |

```bash
# Real-CE x4 (official protocol) for training and evaluation
python tools/prepare_realce.py --realce /path/to/RealCE --output preset/datasets/RealCE_x4
# Optional: recompute the head statistics (on CPU this reproduces the shipped file exactly)
python tools/attn_head_stats.py --config configs/ablations/head_split_f2.yml --output /tmp/head_stats.json
python train_vosr_ablation.py --config configs/ablations/head_split_f2.yml
```

## Inference and evaluation

```bash
gh release download head-split-f2-v1 --repo yqwu905/VOSR --dir weights/head_split_f2 --pattern "*.safetensors"
python inference_vosr_ablation.py --export weights/head_split_f2 \
  --input preset/datasets/RealCE_x4/val/LR --output preset/results/head_split_f2
python evaluate.py --pred preset/results/head_split_f2 --ann preset/datasets/RealCE_x4/val/Label.txt \
  --ann-ref pred --size-buckets 32 48 64 96 --output preset/results/head_split_f2.eval
```

The export stores Linear/Conv weights in bf16 and everything else in FP32, in two shards below GitHub's 2 GB LFS
file limit (`tools/pack_export.py`). Under the default bf16 autocast this is bit-identical to the FP32 weights.

## Results on Real-CE x4 (261 images, 3414 text lines, PP-OCRv5 server recognition)

OCR-A = exact line accuracy, CER = character error rate, 1-NED = 1 - normalized edit distance. Means of noise seeds
42 and 43 where two seeds are listed; "gain kept" is the share of the dense model's OCR-A gain over bicubic.

| Model | SA tokens | Steps | OCR-A | CER | 1-NED | Gain kept |
| --- | --- | --- | --- | --- | --- | --- |
| Bicubic x4 | - | - | 0.562 | 0.165 | 0.803 | 0% |
| Dense VOSR2 | 100% | - | 0.676 | 0.110 | 0.871 | 100% |
| **Head split f2 (release)** | **25%** | 12k + EMA | **0.655** | **0.125** | **0.849** | **81%** |
| Head split f4 | 6.25% | 12k + EMA | 0.632 | 0.134 | 0.840 | 61% |
| Head split f2 | 25% | 3k | 0.644 | 0.131 | 0.841 | 72% |
| Head split f4 | 6.25% | 3k | 0.609 | 0.145 | 0.828 | 41% |
| Pooled K/V only, all queries | 25% K/V | 3k | 0.616 | 0.147 | 0.822 | 47% |
| Pooled Q/K/V (plain token pooling) | 25% | 3k | 0.580 | 0.161 | 0.805 | 15% |
| Whole block merged (attention + MLP) | 25% | 3k | 0.469 | 0.230 | 0.746 | < 0 |

- The remaining gap is in small text: lines under 64 px tall (under 4 tokens) lose most; at 64 px and above the
  25% student is close to the dense model.
- Spot checks: no block artifacts or tile seams; failures are stroke-level (tiny text rewritten as wrong
  characters, thin strokes merging around 40 px).
- Not helpful in the exploration: text-weighted KD, full attention in the most sensitive blocks at the same total
  budget, higher learning rate (3e-5), full-resolution queries for the global heads.
- Merging the whole block (MLP included, `token_compression`) is worse than bicubic even after training: reduce
  tokens in self-attention only.

These numbers were measured with the exploration harness (the same model and tiling as
`inference_vosr_ablation.py`, with tiles batched). The released export reproduces that model bit-exactly under bf16
autocast; restored through `inference_vosr_ablation.py` (seed 42, the commands above) it scores OCR-A 0.656,
CER 0.125, 1-NED 0.849 on the same test set.

## Composition with block-sparse attention

`student.attn_type` / `--attn-type sparse|local` applies to the global heads on the pooled grid. A 512 px tile
pools to 16x16 tokens (4 windows of 8x8), where `sparse` keeps all windows, so it equals full attention there; on
larger tiles the block-sparse pattern applies to the pooled tokens.

## Validation

- `tests/test_ablation_head_split.py` (CPU): spec resolution and errors, per-head reference implementation (pooled
  global heads, local heads as fixed relative-offset attention), identity at factor 1, conv initialization from the
  head statistics, centroid RoPE on reference and other grids, composition with `attn_type`, strict loading from the
  dense teacher, gradients under checkpointing, export round trip without the statistics file, sharded bf16 packing,
  the MAC ratios above at the real VOSR2 size, and the trainer on paired data with cosine LR, EMA, resume and tiled
  inference of the export. Paired-data contract tests are in `tests/test_ablation_data.py`.
- With the real weights (GPU): the integrated model equals the exploration model that trained the release
  bit-exactly under bf16 autocast; the packed bf16 shards equal the FP32 export bit-exactly; a short run of
  `head_split_f2.yml` exercised checkpoints, EMA export and resume; `tools/prepare_realce.py` regenerates the
  exact Real-CE data of the experiments and `tools/attn_head_stats.py` the exact head statistics.

## Per-head sensitivity and budgeted full-resolution heads

See [attn_head_sensitivity.md](attn_head_sensitivity.md) for the opt-in analysis pipeline, individual/group
interventions, small-text OCR comparisons, and short recovery controls. `attn_compression.dense_heads` can protect
selected heads at full resolution while retaining the existing pooled and local-convolution branches. The new
analysis infrastructure has CPU contract tests; no additional real-model accuracy claim is made without running
the documented GPU experiments.
