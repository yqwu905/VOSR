"""Self-attention locality of every head of the dense teacher, for ``student.attn_compression.head_stats``.

For each block and head: ``mass3x3``, the attention mass on the 3x3 token neighbourhood of the query; ``mean_distance``,
the mean attention distance in tokens; ``pattern``, the mean attention weight at each offset (dy, dx) of the
(2 radius + 1)^2 window around the query (row-major, over queries at least ``radius`` tokens from the border).
Measured at the one-step input (t=1, r=0, Gaussian noise) on text-centred crops of the config's paired training data
(``data.dataset_type: paired``), with the teacher of ``teacher_checkpoint`` in FP32. Crop i is drawn by
``random.Random(seed * 1000003 + i)``: an image, then a crop centred on one of its text boxes.

The shipped configs/ablations/head_split/vosr2_head_stats.json comes from the defaults:

    python tools/attn_head_stats.py --config configs/ablations/head_split_f2.yml \\
        --output configs/ablations/head_split/vosr2_head_stats.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_utils import (dino_features, load_backbone_state, load_config, load_dino,  # noqa: E402
                            read_weights, vae_tiling)


def write_stats(path, meta, mass, distance, pattern):
    """One block per line (one head per line for the patterns); FP32 values as their shortest exact decimals."""
    import numpy as np
    exact = lambda v: float(np.format_float_positional(np.float32(v), unique=True, trim='-'))
    row = lambda values: '[' + ','.join(json.dumps(v) for v in values) + ']'
    rows = lambda table, indent, convert: [f'{indent}{row([convert(v) for v in r])}' + (',' if i < len(table) - 1 else '')
                                           for i, r in enumerate(table.tolist())]
    lines = ['{'] + [f'  {json.dumps(k)}: {json.dumps(v)},' for k, v in meta.items()]
    lines += ['  "mass3x3": ['] + rows(mass, '    ', exact) + ['  ],']
    lines += ['  "mean_distance": ['] + rows(distance, '    ', lambda v: round(v, 4)) + ['  ],']
    lines.append('  "pattern": [')
    for i, block in enumerate(pattern):
        lines += ['    ['] + rows(block, '      ', exact) + ['    ]' + (',' if i < len(pattern) - 1 else '')]
    lines += ['  ]', '}']
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='Training config: teacher, model, dino, VAE and paired data')
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--radius', type=int, default=3, help='pattern window (2 radius + 1)^2 tokens')
    parser.add_argument('--device', default='cpu', help='cpu reproduces the shipped values (FP32, no TF32)')
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg['data']
    if data.get('dataset_type') != 'paired' or not data.get('text_boxes'):
        raise ValueError('The config needs data.dataset_type: paired with data.text_boxes')
    device = torch.device(args.device)
    from dataloaders.paired_crops import PairedCropDataset
    from models.lightningdit_ablation import AblationLightningDiT
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    from tiled_vae import encode_dispatch
    crop = int(data.get('crop_size') or data['resolution'])
    state = read_weights(cfg['teacher_checkpoint'])
    model_cfg = dict(cfg['model'])
    aux = model_cfg.pop('auxiliary_time_cond', 'auto')
    model_cfg['auxiliary_time_cond'] = any(k.startswith('r_embedder.') for k in state) if aux == 'auto' else bool(aux)
    model_cfg['input_size'] = int(data['resolution']) // 8
    model = AblationLightningDiT(**model_cfg, use_cross_attention=True)
    load_backbone_state(model, state)
    model = model.to(device).eval()
    del state
    vae = AutoencoderKLQwenImage2D.from_pretrained(cfg['vae_path']).to(device).eval()
    venc = load_dino(cfg['dino'], device)
    dataset = PairedCropDataset(data['lq_dir'], data['hq_dir'], crop, data['text_boxes'], 1.0)
    side, r = crop // 8 // model.patch_size, args.radius
    depth, heads, dim = len(model.blocks), model.num_heads, model.hidden_size
    yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing='ij')
    positions = torch.stack((yy.flatten(), xx.flatten()), 1).float()
    distance = torch.cdist(positions, positions).to(device)
    near = (distance <= 1.5).float()  # the 3x3 neighbourhood
    inner = ((yy >= r) & (yy < side - r) & (xx >= r) & (xx < side - r)).flatten().nonzero().flatten().to(device)
    window = torch.stack([inner + dy * side + dx for dy in range(-r, r + 1) for dx in range(-r, r + 1)], 1)
    mass = torch.zeros(depth, heads, device=device)
    mean_distance = torch.zeros(depth, heads, device=device)
    pattern = torch.zeros(depth, heads, window.shape[1], device=device)
    rope = model._get_dynamic_rope(side, device, torch.float32) if side != model.x_embedder.img_size[0] // \
        model.patch_size else model.feat_rope
    for index in range(args.samples):
        rng = random.Random(args.seed * 1_000_003 + index)
        lq = dataset.sample(rng.choice(dataset.names), rng)['lq'][None].to(device) * 2 - 1
        latent = encode_dispatch(vae, lq, vae_tiling(cfg)[1], device)[0]
        features = dino_features(venc, lq, cfg['dino'])
        torch.manual_seed(index)
        inp = torch.cat((latent, torch.randn_like(latent)), 1)
        x = model.x_embedder(inp)
        t, r0 = inp.new_ones(1), inp.new_zeros(1)
        c = model.t_embedder(t)
        if model.r_embedder is not None:
            c = c + model.r_embedder(r0) * (t - r0).unsqueeze(-1)
        c0 = model.t_block(c)
        z = model.mlp_ca(model.layer_norm(features[0]))
        for i, block in enumerate(model.blocks):
            shift, scale, *_ = (block.scale_shift_table[None] + c0.reshape(1, 6, -1)).chunk(6, 1)
            h = block.norm1(x) * (1 + scale) + shift
            q, k = F.linear(h, block.attn.qkv.weight, block.attn.qkv.bias).split(dim, -1)[:2]
            q, k = (t_.view(1, -1, heads, dim // heads).transpose(1, 2) for t_ in (q, k))
            q, k = block.attn.q_norm(q), block.attn.k_norm(k)
            if rope is not None:
                q, k = rope(q), rope(k)
            attention = (q @ k.transpose(-1, -2) * block.attn.scale).softmax(-1)[0]  # (heads, N, N)
            mass[i] += (attention * near).sum(-1).mean(-1) / args.samples
            mean_distance[i] += (attention * distance).sum(-1).mean(-1) / args.samples
            pattern[i] += torch.gather(attention[:, inner], 2, window[None].expand(heads, -1, -1)).mean(1) / args.samples
            x = block(x, c0, z, rope, 1.0, False, True)[0]
        print(f'sample {index + 1}/{args.samples}', flush=True)
    meta = {'description': 'Self-attention locality of every head of the dense teacher (tools/attn_head_stats.py): '
                           'mass3x3 = attention mass on the 3x3 token neighbourhood of the query, mean_distance = '
                           'mean attention distance in tokens, pattern = mean attention weight at each offset '
                           '(dy, dx) of the (2 radius + 1)^2 window around the query, row-major, over queries at '
                           'least radius tokens from the border; blocks x heads. Heads with mass3x3 > '
                           'attn_compression.local_threshold become local convs initialized from their pattern.',
            'teacher': str(cfg['teacher_checkpoint']), 'data': str(data['hq_dir']), 'samples': args.samples,
            'seed': args.seed, 'crop': crop, 'radius': r}
    write_stats(args.output, meta, mass.cpu(), mean_distance.cpu(), pattern.cpu())
    local = int((mass > 0.5).sum())
    print(f'wrote {args.output}: {local} of {depth * heads} heads with mass3x3 > 0.5')


if __name__ == '__main__':
    main()
