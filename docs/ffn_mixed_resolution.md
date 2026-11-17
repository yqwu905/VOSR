# Full-grid and pooled-grid SwiGLU hidden units

This experiment splits **hidden units**, not input channels or attention heads.
Each branch reads the entire 1536-channel FFN input. It extends the existing
`train_vosr_ablation.py` one-step distillation trainer and starts from the released
`weights/head_split_f2` student with the frozen dense VOSR2 teacher. Attention,
cross-attention, normalization, modulation and the residual stream retain their
existing behavior. All 36 block FFNs use the chosen tier; `mlp_ca` is unchanged.

## Five tiers

Every tier has a random config and a `_calibrated.yml` variant under
`configs/ablations/`. The table lists config stems (append `.yml`).

| Config | Full width | Coarse width | FFN projection MACs / original |
| --- | ---: | ---: | ---: |
| `ffn_spatial_f4096_c0` | 4096 | 0 | 100% |
| `ffn_spatial_f0_c4096` | 0 | 4096 | 25% |
| `ffn_spatial_f1024_c3072` | 1024 | 3072 | 43.75% |
| `ffn_spatial_f512_c3584` | 512 | 3584 | 34.375% |
| `ffn_spatial_f512_c2048` | 512 | 2048 | 25% |

For `N` tokens, input/output width `D`, full width `Mf`, coarse width `Mc`, and
pooling factor `s`, the three projections cost `3ND(Mf + Mc/s²)` MACs, versus
`3ND × 4096` in the source. A FLOP convention counting a multiply and an addition
separately doubles both. The table assumes `s=2`. It excludes pooling,
interpolation, activations, normalization and all other model work; these are
**not measured latency or end-to-end compute ratios**. The first four tiers keep
the original FFN parameter count; the last retains 2560/4096 = 62.5% of its
hidden units, with the original output-bias width.

## Computation and initialization

Let `z` be the input **after** `norm2`, scale and shift, and `P` be 2×2 spatial
average pooling. For each group `q`:

```text
h_q(z) = silu(z W_gate,q + b_gate,q) * (z W_value,q + b_value,q)
FFN(z) = h_full(z) W_down,full + U(h_coarse(P(z)) W_down,coarse) + b_down
```

`U` is bilinear upsampling (`align_corners=False`) by default; nearest is also
supported. The coarse branch projects back to `D` **before** upsampling, so all
three of its matrix multiplications operate on `N/4` tokens. The two `D`-channel
contributions are added, not concatenated. Each group has its own trainable
packed gate/value projection and output projection; there is only one shared
output bias. The existing block gate `g2` and residual addition follow this sum.
Empty branches allocate no parameters and do no work.

For original hidden unit `j`, the conversion copies rows `j` and `j+4096` of
`w12`, both matching biases, and column `j` of `w3` together. It copies `w3.bias`
once. The `4096+0` control keeps the original order. Factor 1 with all units
retained recovers the dense FFN algebra (floating-point summation may differ).
Pooling is an architectural approximation requiring fine-tuning; moving it
through SiLU or normalization would not preserve the calculation.

The `512+2048` tier discards the other 1536 units with **no bias compensation**.
Both branches are trained normally with the inherited recipe (12k steps,
velocity KD, cosine LR, EMA); no learned router or budget penalty is introduced.
Set the existing `training.ffn_lr_multiplier` or trainable-parameter filters if a
separate FFN learning rate or an FFN-only alignment stage is needed.

## Random grouping

The random configs use `grouping: random, seed: 0`. Each block uses a dedicated
CPU permutation with seed `seed * 1000003 + block_index`, independent of global
training RNG state and the accelerator. The first `Mf` units go to the full
branch, the next `Mc` to the coarse branch; the rest are dropped. The same seed
gives nested selections across widths, including the identical 512 full units
in the two `f512` tiers. Changing the grouping seed requires a fresh run.

Run from the repository root, after setting the checkpoint, VAE and paired-data
paths inherited from `head_split_f2.yml` / `base_vosr2.yml`. Download the release
shards as described in [weights/head_split_f2/README.md](../weights/head_split_f2/README.md).

```bash
python train_vosr_ablation.py --config configs/ablations/ffn_spatial_f1024_c3072.yml
```

Replace the config name with any row in the table to run that tier. For multiple
GPUs, use the existing `torchrun --nproc_per_node=... train_vosr_ablation.py`
launcher; keep the effective batch matched when comparing tiers.

## Calibrated grouping, including explicit manual assignments

Calibrate the **full-width source checkpoint**, before the spatial split. The
tool reconstructs the source using the config's attention settings, then loads
`student_checkpoint` (or the teacher when absent). It ignores the requested FFN
split during this source pass. Use representative **training** crops, not the
held-out evaluation split.

```bash
python tools/calibrate_ffn_spatial.py \
  --config configs/ablations/ffn_spatial_f1024_c3072.yml \
  --output-dir artifacts/ffn_spatial_calibration \
  --samples 64 --seed 11 --noise-seed 0 --weighting balanced --device cuda

python train_vosr_ablation.py \
  --config configs/ablations/ffn_spatial_f1024_c3072_calibrated.yml
```

The output directory must be new. One calibration writes all five group files,
`scores.safetensors`, and `calibration.json` with source checkpoint hashes, data
manifest provenance, sample/noise seeds, model settings and device. It supports
the existing single-source and `data.sources` paired-data paths. The default
balanced weighting allocates per-image mass 0.5 / 0.3 / 0.2 to small text / other
text / background, with the existing empty-stratum fallback. Use `--weighting
uniform` for an unweighted control. All spatial tokens are used; `--chunk-hidden
256` bounds temporary activation memory without subsampling neurons or tokens.
Statistics are measured in FP32 with CPU FP64 accumulation, at `t=1, r=0`.

For each source neuron `j`, on the actual normalized/modulated source input:

```text
pool_error[j] = E_w[(h_j(z) - U(h_j(P(z))))²] * ||g2 * W_down[:,j]||²
drop_error[j] = E_w[h_j(z)²]                  * ||g2 * W_down[:,j]||²
```

These are the squared errors of that neuron's individual gated output
contribution. `E_w` uses the configured token weights. Linearity lets the
calibrator upsample the scalar hidden activation when scoring it, while the
actual model upsamples the projected output. Scores ignore cross-neuron
correlations; they are a selection heuristic, not a guarantee about final
quality or the jointly optimal partition.

Selection first retains the `Mf+Mc` largest `drop_error` units. Among those, it
places the `Mf` largest `pool_error` units in the full branch and the others in
the coarse branch. This protects units most sensitive to spatial pooling. When
all 4096 are kept, the drop ranking has no effect. Stable sorting breaks ties
by source neuron index. No assumption is made that an index represents a
particular attention head or semantic feature.

To specify your own groups, copy/edit the generated JSON and point the config
to it:

```yaml
student:
  mlp_compression:
    type: mixed_resolution
    full_hidden: 512
    coarse_hidden: 2048
    factor: 2
    upsample: bilinear
    grouping: calibrated
    groups_file: artifacts/my_groups_f512_c2048.json
```

The manifest has `version: 1`, `source_hidden: 4096`, `factor`, `upsample`,
`source_mlp_sha256`, `full_indices` and `coarse_indices`. The last three are
lists with one entry per block. Each index is a **source** neuron ID in
`[0,4095]`. Edit the two index lists, preserving the source hashes. Their lengths
must match the configured widths; indices must be unique and disjoint within
each block. Omitted units are dropped. Every source FFN's packed weights and
biases are hashed in canonical FP32 and checked before conversion. Different
weights, pooling settings, malformed indices and width mismatches fail loudly.
This checks FFN weight identity; use the provenance to also match the attention,
data and preprocessing used for calibration. Never edit hashes to bypass a
mismatch; recalibrate the intended source.

The saved scores can also be loaded with `safetensors.torch.load_file` and passed
to `select_groups` in `tools/calibrate_ffn_spatial.py` for other valid width
choices, using the source fingerprints from `calibration.json`.

## Checkpoints, exports and inference

Conversion happens only when dense-source `mlp.w12` / `mlp.w3` weights are loaded.
Split weights are stored as `mlp.full.*`, `mlp.coarse.*`, and `mlp.bias` and are
strictly shape-checked on reload. `model.json` records resolved widths, pooling
factor and interpolation; it contains no calibration-file dependency. Resume
and inference never reselect units or re-read a group file. Preserve calibration
artifacts separately if you need to reconstruct the initial partition.

```bash
python train_vosr_ablation.py \
  --config configs/ablations/ffn_spatial_f1024_c3072_calibrated.yml \
  --resume exp_vosr/ffn_spatial_f1024_c3072_calibrated/<run>/checkpoint-00003000

python inference_vosr_ablation.py --export <run>/export --input <image.png> --output <sr-output>
python tools/pack_export.py --export <run>/export --output <packed-export>
python tools/benchmark_ffn_export.py --export <run>/export --input <image.png> \
  --output <new-benchmark.json> --device cuda --expected-width 4096
```

For `512+2048`, use `--expected-width 2560`. The benchmark reports both branch
widths and measures the existing complete SR pipeline. EMA exports, sharded BF16
packing, tiled inference and gradient checkpointing use the existing paths.
Square token grids must be divisible by the FFN factor; the model's
`latent_multiple` combines the attention and FFN factors using their LCM, so
inference padding/tile validation accounts for both. Token merging and dynamic
FFN token routing are rejected with this option because they do not preserve
the square spatial grid required here. Existing neuron-pruning configs remain
unchanged; select only one `mlp_compression.type`.

## Validation

```bash
OMP_NUM_THREADS=1 TORCHDYNAMO_DISABLE=1 VOSR_TEST_DDP=1 python -m pytest -q tests/test_ffn_mixed_resolution.py
```

Tests check the two branches against explicit per-neuron forward/gradient
references, actual matrix-operation counts for all five full-size tiers on the
meta device, seed reproducibility, calibration scores and edited indices,
source mismatch rejection, empty branches, model grid alignment, checkpointed
backpropagation, export round trips, and real trainer KD/EMA/resume with a small
DiT (only VAE/DINO are fixtures), plus two-rank Gloo training with sharded Adam
states. Omit `VOSR_TEST_DDP=1` on hosts that disallow local Gloo sockets.
GPU/NPU speed, full-data convergence and small
text quality still need to be measured on the target hardware and evaluation set.
