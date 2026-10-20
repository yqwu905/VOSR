# head_split_f2: VOSR2 with head-split self-attention on 25% of the tokens

Export of `configs/ablations/head_split_f2.yml` (see `docs/attn_head_split.md`): Real-CE x4 OCR-A 0.655,
CER 0.125, 1-NED 0.849 (dense VOSR2: 0.676 / 0.110 / 0.871; bicubic: 0.562 / 0.165 / 0.803).

GitHub does not accept Git LFS uploads to this public fork, so the two weight shards are attached to the
release `head-split-f2-v1`. Download them into this directory next to `model.safetensors.index.json`:

```bash
gh release download head-split-f2-v1 --repo yqwu905/VOSR --dir weights/head_split_f2 --pattern "*.safetensors"
# or: https://github.com/yqwu905/VOSR/releases/tag/head-split-f2-v1
```

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `model-00001-of-00002.safetensors` | 1609528488 | `71e3fd2a72d590e8e020f331adbcbc83cc5fe639f4d161bff41d72c47fe8502c` |
| `model-00002-of-00002.safetensors` | 1182408408 | `d13989700ff98db0e76090359d62539b0fbc748b4d118adcf878f234997bfb7f` |

Linear/Conv weights are bf16, all other tensors FP32 (`tools/pack_export.py`); under the default bf16 autocast this
is bit-identical to the FP32 weights. Then run:

```bash
python inference_vosr_ablation.py --export weights/head_split_f2 --input <LR folder> --output <output folder>
```
