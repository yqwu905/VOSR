# DyDiT FFN token routing on the head split (`router_config.routing_mode: dydit`)

Goal: on top of [`head_split_f2`](attn_head_split.md), which already cut self-attention to 23.3%, let every block's
FFN skip the tokens that do not need it, with a target skip ratio that is a config value. The FFN budget is 25% of
the FFN token work (skip 75%), the token count of an F16 VAE. Config:
[`configs/ablations/ffn_route_k25.yml`](../configs/ablations/ffn_route_k25.yml).

## Method

This is the spatial-wise dynamic token (SDT) part of DyDiT, read from the paper (arXiv 2410.03456, Sec. 3.3 and
3.4, Table 6) and the official code ([alibaba-damo-academy/DyDiT](https://github.com/alibaba-damo-academy/DyDiT):
`DyDiT/models.py` `DiTBlock`, `DyDiT/dynamic_model.py` `TokenSelect` and `_gumbel_sigmoid`, `DyDiT/loss.py`
`DynamicLoss`, `DyDiT/train.py` and `train0.5.sh`). The timestep-wise dynamic width (TDW) part is not used: VOSR2
runs one step at t=1, where a timestep-dependent width is just a fixed narrower MLP, i.e. the
[neuron pruning](mlp_neuron_prune.md) series.

| | DyDiT (official code) | Here |
| --- | --- | --- |
| Router | per block, not shared: Linear(C, C/16), ReLU, Linear(C/16, 1) | same (`SDTRouter`, run in FP32) |
| Router input | residual stream after attention, before the MLP norm | after self- and cross-attention, before `norm2` |
| Training mask | Gumbel-Sigmoid, tau 5, hard at 0.5, straight-through; multiplies the gated MLP output | same: `temperature: 5.0`, `threshold: 0.5` |
| Inference | sigmoid(logit) > 0.5 (no noise); the paper gathers the kept tokens through the MLP and scatters them back | same; `inference_vosr_ablation.py` gathers by default (`--dense-mlp` multiplies a dense MLP by the same mask) |
| Budget | 2.0 x (mean over the batch of dynamic / static FLOPs - lambda)^2, on the straight-through masks | `budget_weight: 2.0` x (mean over layers, images and tokens of the straight-through masks - `target_keep_ratio`)^2 |
| Target schedule | 1.0 for 100 steps, then 1.0 -> lambda over 1000 steps | 0.99 -> 0.25 over `budget_warmup_steps: 3000` |
| Router init | xavier weights, last bias 35 (every token kept) | last layer N(0, 1e-3), bias logit(`init_keep_prob: 0.9999`) |
| Optimizer | AdamW lr 1e-4 for all weights, weight decay 0, batch 256 | head_split_f2's recipe (lr 1e-5 cosine, batch 4, 12k steps); routers x10, i.e. 1e-4 at the peak |
| Complete-model loss | only DiT-XL's first stage (30k of 200k steps; none for S and B) | not used (`dense_distill_weight: 0`) |

Notes on the differences:

- **Budget units.** DyDiT budgets attention and MLP FLOPs together. Here attention is fixed by the head split, so the
  budget counts MLP tokens only; with every MLP the same size this is DyDiT's loss with the attention FLOPs held out
  (`tests/test_ablation_ffn_route.py` checks value and router gradients against a transcription of `loss.py`).
  Skip ratio = 1 - `target_keep_ratio`. The mean runs over layers and images, so layers and images may keep
  different shares; only the average is held to the target.
- **Why a new mode.** The legacy `gumbel` mode routes the same way but budgets the expected keep sigmoid(logit),
  whose gradient vanishes once the logits saturate. `dydit` budgets the realized hard mask through the
  Gumbel-Sigmoid surrogate (gradient sigmoid'((logit + noise) / tau) / tau), as `loss.py` does. Inference is
  identical. Existing configs and exports keep their modes.
- **Init.** DyDiT's bias 35 only guarantees "keep everything" when its xavier last layer stays far below 35 on
  every token, which depends on the residual stream's scale. A small last layer with a high keep bias starts from
  the same all-keep state on any scale.
- **Schedule.** DyDiT ramps over 1000 steps with a backbone learning rate of 1e-4 and batch 256. The backbone here
  learns at 1e-5 with batch 4, so the ramp is longer (a quarter of the 12k steps), leaving 9k steps at the target.

### Inference keep can differ from the trained keep

Training holds the realized keep under Gumbel noise, whose expectation is the mean of sigmoid(logit). Inference keeps
the tokens with logit > 0. The two agree once the logits are saturated, not before (DyDiT has the same gap). The
previews log both: `preview/deterministic_mlp_keep` (what inference will run) and `preview/expected_mlp_keep`.
`inference_vosr_ablation.py` prints `deterministic_mlp_keep` per image. If the deterministic keep stays above the
target, the compute target is missed: raise `budget_weight` or train longer.

The kept-token count varies per image and per layer (that is the point: text keeps more, flat background less), so a
deployment needs dynamic shapes for the gather. A fixed-share per-layer top-k would give static shapes; it is not
part of this config.

## Config

`ffn_route_k25.yml` inherits `ffn_prune_ctrl.yml` (head_split_f2's recipe from its release export, the step-matched
control of the FFN series) and changes only these keys:

| Key | Value | Meaning |
| --- | --- | --- |
| `student.router_config.routing_mode` | `dydit` | DyDiT SDT on every block's MLP |
| `student.router_config.temperature` | 5.0 | Gumbel-Sigmoid tau |
| `student.router_config.threshold` | 0.5 | hard mask threshold |
| `student.router_config.init_keep_prob` | 0.9999 | initial keep probability of every token |
| `training.target_keep_ratio` | 0.25 | FFN keep target; skip ratio 0.75. Any share in [0, 1] |
| `training.budget_weight` | 2.0 | DyDiT's loss coefficient |
| `training.budget_warmup_steps` | 3000 | linear ramp of the target from 0.99 |
| `training.dense_warmup_steps` | 0 | route from the first step (the base config has 500) |
| `training.dense_distill_weight` | 0.0 | no extra dense pass (the base config has 0.1) |

`budget_scope: global` and `router_lr_multiplier: 10` are stated explicitly but equal the inherited values. Another
skip ratio is one key (`target_keep_ratio`) plus `output_dir`.

## Compute

MACs per 512 px tile (1024 tokens), `FlopCounterMode` for the backbone, the routers (two small Linear layers per
block on every token) counted directly:

| Student | FFN tokens kept | DiT MACs | vs `head_split_f2` | vs dense VOSR2 |
| --- | --- | --- | --- | --- |
| Dense VOSR2 | 100% | 1635.9 G | | 100% |
| `head_split_f2` | 100% | 1279.9 G | 100% | 78.2% |
| + routing, keep 0.5 | 50% | 937.5 G | 73.2% | 57.3% |
| + routing, keep 0.25 (`ffn_route_k25`) | 25% | 763.5 G | 59.7% | 46.7% |

The routers cost 5.4 G. The gather/scatter is not counted, and the real keep at inference is data-dependent (see
above).

## Run and judge

```bash
gh release download head-split-f2-v1 --repo yqwu905/VOSR --dir weights/head_split_f2 --pattern "*.safetensors"
bash train_and_infer.sh --config configs/ablations/ffn_route_k25.yml --nproc 1 \
  --testset realce_x4=preset/datasets/RealCE_x4/val/LR
python evaluate.py --pred <run>/inference/realce_x4 --ann preset/datasets/RealCE_x4/val/Label.txt \
  --ann-ref pred --size-buckets 32 48 64 96 --output <run>/inference/realce_x4.eval
```

During training, `mlp_keep` (the realized keep) should follow `target_keep`; `router/layer_XX/actual_keep` shows
which blocks skip. Compare with `ffn_prune_ctrl` as in the [neuron pruning series](mlp_neuron_prune.md#decision-rules)
(Real-CE x4, noise seeds 42 and 43): ΔCER ≤ +0.005 passes, ≥ +0.015 fails, in between check the lines under 64 px.
A result only counts when the inference `deterministic_mlp_keep` averages at most 0.26 over the test set. On a fail,
`target_keep_ratio: 0.5` (one key) tells whether the method or the 75% skip loses the text.

## Validation

`tests/test_ablation_ffn_route.py` (CPU) covers:

- `dydit` routing equal to `gumbel` for the same noise, and inference equal to logit > 0;
- the block's keep carrying the straight-through gradient in `dydit` and not in the other modes;
- the budget equal in value and router gradients to DyDiT's `loss.py` (attention FLOPs held out);
- the head split with routed FFNs: backward reaching the routers and local convs, sparse gather equal to the dense
  masked MLP, and the export round trip (`router_config` and `attn_compression` in model.json);
- the shipped config changing only the routing keys against `ffn_prune_ctrl`;
- an end-to-end run of the trainer (target ramp, budget, router learning rate, logs, resume) and a restore through
  `inference_vosr_ablation.py`.

No GPU training or OCR evaluation was run for this change.
