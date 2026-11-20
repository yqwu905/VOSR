# Ablation configurations

Run commands from the repository root. `_base_` paths are relative to each YAML file; data,
checkpoint and output paths remain relative to the repository root.

| Directory | Experiments |
| --- | --- |
| `base/` | Dense KD and SFT baselines: [base_sft.yml](base/base_sft.yml), [base_vosr2.yml](base/base_vosr2.yml) |
| `no_dino/` | Remove or fade DINO cross-attention: [no_dino_fade.yml](no_dino/no_dino_fade.yml), [no_dino_no_ca.yml](no_dino/no_dino_no_ca.yml), [no_dino_sft.yml](no_dino/no_dino_sft.yml) |
| `sdt/` | Spatial MLP routing: [dydit_sdt.yml](sdt/dydit_sdt.yml) |
| `loss/` | LPIPS, OCR and per-character TextVerifier supervision. [loss_all.yml](loss/loss_all.yml), [loss_lpips.yml](loss/loss_lpips.yml), [loss_ocr_kl.yml](loss/loss_ocr_kl.yml), [glyph_sft.yml](loss/glyph_sft.yml) |
| `head_split/` | Pooled/local attention heads and learned head selection: [head_split_f2.yml](head_split/head_split_f2.yml), [head_split_f2_learned.yml](head_split/head_split_f2_learned.yml), [head_split_f2_search.yml](head_split/head_split_f2_search.yml), [head_split_f4.yml](head_split/head_split_f4.yml) |
| `pixel/` | Pixel-space students: [pixel_l2p.yml](pixel/pixel_l2p.yml), [pixel_l2p_unet.yml](pixel/pixel_l2p_unet.yml) |
| `res1024/` | 1024-pixel crops, RoPE and DINO tiling: [res1024.yml](res1024/res1024.yml), [res1024_dino_whole.yml](res1024/res1024_dino_whole.yml), [res1024_loss_all.yml](res1024/res1024_loss_all.yml), [res1024_rope_interp.yml](res1024/res1024_rope_interp.yml) |
| `sparse/` | Block-sparse attention: [sft_sparse_attn_lpips.yml](sparse/sft_sparse_attn_lpips.yml) |
| `ffn_spatial/` | Full/coarse FFN branches and calibrated groups: [ffn_spatial_f0_c4096.yml](ffn_spatial/ffn_spatial_f0_c4096.yml), [ffn_spatial_f0_c4096_calibrated.yml](ffn_spatial/ffn_spatial_f0_c4096_calibrated.yml), [ffn_spatial_f1024_c3072.yml](ffn_spatial/ffn_spatial_f1024_c3072.yml), [ffn_spatial_f1024_c3072_calibrated.yml](ffn_spatial/ffn_spatial_f1024_c3072_calibrated.yml), [ffn_spatial_f4096_c0.yml](ffn_spatial/ffn_spatial_f4096_c0.yml), [ffn_spatial_f4096_c0_calibrated.yml](ffn_spatial/ffn_spatial_f4096_c0_calibrated.yml), [ffn_spatial_f512_c2048.yml](ffn_spatial/ffn_spatial_f512_c2048.yml), [ffn_spatial_f512_c2048_calibrated.yml](ffn_spatial/ffn_spatial_f512_c2048_calibrated.yml), [ffn_spatial_f512_c3584.yml](ffn_spatial/ffn_spatial_f512_c3584.yml), [ffn_spatial_f512_c3584_calibrated.yml](ffn_spatial/ffn_spatial_f512_c3584_calibrated.yml) |

```bash
python train_vosr_ablation.py --config configs/ablations/head_split/head_split_f2.yml
```

Set local checkpoint and dataset paths before training. See [the training guide](../../docs/ablation_training.md)
and [head-split details](../../docs/attn_head_split.md). New attention modules use
`training.new_module_lr_multiplier` (local convolutions and head-search logits).
Existing configs and resume checkpoints using `token_merge_lr_multiplier` remain compatible.

Multistep pixel PDM has its own series directory: [`configs/pixel_pdm/`](../pixel_pdm/).
