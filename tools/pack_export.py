"""Pack an export of train_vosr_ablation.py for Git LFS: bf16 Linear/Conv weights in shards below 2 GB.

GitHub stores at most 2 GB per LFS file (Free/Pro plans). Under the exports' bf16 autocast (pipeline.json
``precision: bf16``) every Linear and Conv runs on bf16 copies of its weight, so storing those weights in bf16 gives
bit-identical inference at about half the size. All other tensors (norms, biases, scale-shift tables, routers that
run in FP32) stay FP32. The output is the Hugging Face sharded layout (``model-0000i-of-0000n.safetensors`` +
``model.safetensors.index.json``) next to model.json and pipeline.json; load_export and read_weights read it.

usage: python tools/pack_export.py --export exp_vosr/head_split_f2/<run>/export --output weights/head_split_f2
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_utils import load_export, read_weights  # noqa: E402


def bf16_weight_names(model):
    """Weights of the Linear/Conv layers that autocast runs in bf16 (the FP32 routers excluded)."""
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) and '.router.' not in f'.{name}.':
            names.add(f'{name}.weight')
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export', required=True, help='Export directory (model.json, pipeline.json, weights)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-shard-gb', type=float, default=1.5, help='Shard size limit in GiB (GitHub LFS: 2 GB)')
    args = parser.parse_args()
    source, out = Path(args.export), Path(args.output)
    pipeline = json.loads((source / 'pipeline.json').read_text(encoding='utf-8'))
    if pipeline.get('precision', 'bf16') != 'bf16':
        raise ValueError('Only bf16-autocast exports run their Linear/Conv layers on bf16 weights')
    model = load_export(source)
    names = bf16_weight_names(model)
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    weights = source / 'model.safetensors'
    original = read_weights(weights if weights.is_file() else source / 'model.safetensors.index.json')
    if set(state) != set(original):
        raise ValueError('The export does not hold exactly the model weights')
    packed = {key: original[key].to(torch.bfloat16) if key in names else original[key].float() for key in state}
    from safetensors.torch import save_file
    limit = int(args.max_shard_gb * 2 ** 30)
    shards, current, size = [], {}, 0
    for key, value in packed.items():  # state-dict order keeps each block in one or two shards
        nbytes = value.numel() * value.element_size()
        if current and size + nbytes > limit:
            shards.append(current)
            current, size = {}, 0
        current[key] = value
        size += nbytes
    shards.append(current)
    out.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for i, shard in enumerate(shards, 1):
        name = f'model-{i:05d}-of-{len(shards):05d}.safetensors'
        save_file(shard, str(out / name), metadata={'format': 'pt'})
        weight_map.update({key: name for key in shard})
    total = sum(v.numel() * v.element_size() for v in packed.values())
    index = {'metadata': {'total_size': total, 'bf16_weights': sorted(names & set(packed))}, 'weight_map': weight_map}
    (out / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
    for name in ('model.json', 'pipeline.json'):
        shutil.copyfile(source / name, out / name)
    reloaded = load_export(out).state_dict()
    for key, value in original.items():
        expected = value.to(torch.bfloat16).float() if key in names else value.float()
        if not torch.equal(reloaded[key].float(), expected):
            raise RuntimeError(f'{key} did not survive the round trip')
    print(json.dumps({'shards': len(shards), 'bytes': total, 'bf16_tensors': len(names & set(packed)),
                      'fp32_tensors': len(packed) - len(names & set(packed))}))


if __name__ == '__main__':
    main()
