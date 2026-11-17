"""Split SwiGLU hidden units between full and spatially pooled token grids.

Both branches read the complete normalized/modulated residual width. Pooling
precedes the coarse branch's first projection; upsampling follows its last
projection. The output bias is shared and added exactly once.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def mixed_resolution_spec(config, hidden):
    allowed = {'type', 'full_hidden', 'coarse_hidden', 'factor', 'upsample', 'grouping', 'seed', 'groups_file'}
    if not isinstance(config, dict) or config.get('type') != 'mixed_resolution':
        raise ValueError('mlp_compression must have type: mixed_resolution')
    if set(config) - allowed:
        raise ValueError(f'Unknown mixed_resolution options: {sorted(set(config) - allowed)}')
    full, coarse = config.get('full_hidden'), config.get('coarse_hidden')
    if any(isinstance(w, bool) or not isinstance(w, int) or w < 0 for w in (full, coarse)) \
            or not 0 < full + coarse <= hidden:
        raise ValueError(f'full_hidden and coarse_hidden must be nonnegative integers with sum in [1, {hidden}]')
    factor, mode = config.get('factor', 2), config.get('upsample', 'bilinear')
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError('mixed_resolution.factor must be a positive integer')
    if mode not in ('bilinear', 'nearest'):
        raise ValueError('mixed_resolution.upsample must be bilinear or nearest')
    grouping, seed = config.get('grouping', 'random'), config.get('seed', 0)
    if grouping not in ('random', 'calibrated'):
        raise ValueError('mixed_resolution.grouping must be random or calibrated')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError('mixed_resolution.seed must be an integer in [0, 2**32)')
    if grouping == 'calibrated' and not config.get('groups_file'):
        raise ValueError('grouping: calibrated needs groups_file from tools/calibrate_ffn_spatial.py')
    if grouping == 'random' and config.get('groups_file') is not None:
        raise ValueError('groups_file requires grouping: calibrated')
    spec = dict(type='mixed_resolution', full_hidden=full, coarse_hidden=coarse, factor=factor, upsample=mode)
    return spec, dict(grouping=grouping, seed=seed, groups_file=config.get('groups_file'))


def pool_grid(x, factor):
    if x.ndim != 3:
        raise ValueError('Spatial FFN expects [batch, tokens, channels]')
    b, n, d = x.shape
    side = math.isqrt(n)
    if not n or side * side != n or side % factor:
        raise ValueError('Spatial FFN needs a square token grid divisible by its pooling factor')
    if factor == 1:
        return x, side
    coarse = side // factor
    return x.reshape(b, coarse, factor, coarse, factor, d).mean((2, 4)).reshape(b, coarse * coarse, d), side


def upsample_grid(x, side, mode):
    b, n, d = x.shape
    coarse = math.isqrt(n)
    if coarse == side:
        return x
    image = x.transpose(1, 2).reshape(b, d, coarse, coarse)
    options = {'align_corners': False} if mode == 'bilinear' else {}
    return F.interpolate(image, size=(side, side), mode=mode, **options).flatten(2).transpose(1, 2)


class _SwiGLUBranch(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gate, value = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(gate) * value)


class MixedResolutionSwiGLU(nn.Module):
    def __init__(self, dim, full_hidden, coarse_hidden, factor=2, upsample='bilinear'):
        super().__init__()
        mixed_resolution_spec(dict(type='mixed_resolution', full_hidden=full_hidden, coarse_hidden=coarse_hidden,
                                   factor=factor, upsample=upsample), full_hidden + coarse_hidden)
        self.full_hidden, self.coarse_hidden = full_hidden, coarse_hidden
        self.factor, self.upsample = factor, upsample
        self.full = _SwiGLUBranch(dim, full_hidden) if full_hidden else None
        self.coarse = _SwiGLUBranch(dim, coarse_hidden) if coarse_hidden else None
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        out = self.full(x) if self.full is not None else None
        if self.coarse is not None:
            pooled, side = pool_grid(x, self.factor)
            update = upsample_grid(self.coarse(pooled), side, self.upsample)
            out = update if out is None else out + update
        # Preserve linear/autocast output dtype, including after loading FP32 biases.
        return out + self.bias.to(out.dtype)


def mlp_fingerprint(state, prefix):
    """Bind a calibration to all four source FFN tensors, in canonical FP32."""
    digest = hashlib.sha256()
    for name in ('w12.weight', 'w12.bias', 'w3.weight', 'w3.bias'):
        value = state[prefix + name].detach().float().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_groups(groups, spec, depth, hidden):
    if not isinstance(groups, dict) or groups.get('version') != 1 or groups.get('source_hidden') != hidden:
        raise ValueError('groups_file must be a version 1 grouping for this source hidden width')
    for key in ('factor', 'upsample'):
        if groups.get(key) != spec[key]:
            raise ValueError(f'groups_file {key} differs from the configured pooling/upsampling')
    fingerprints = groups.get('source_mlp_sha256')
    if not isinstance(fingerprints, list) or len(fingerprints) != depth:
        raise ValueError('groups_file needs one source_mlp_sha256 per block')
    rows = []
    for name in ('full', 'coarse'):
        indices = groups.get(name + '_indices')
        if not isinstance(indices, list) or len(indices) != depth:
            raise ValueError(f'groups_file needs one {name}_indices list per block')
        for row in indices:
            if not isinstance(row, list) or len(row) != spec[name + '_hidden'] or any(
                    isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < hidden for i in row):
                raise ValueError(f'groups_file {name}_indices have invalid widths or source indices')
        rows.append(indices)
    for full, coarse in zip(*rows):
        if len(set(full + coarse)) != len(full + coarse):
            raise ValueError('groups_file full/coarse indices must be disjoint and unique')
    return rows


def split_mlp_state(state, spec, init, depth, hidden):
    """Convert dense-source weights once. Split checkpoints pass through strictly.

    The original w12 gate and value rows travel together with the w3 column.
    No selection files are read when resuming already split weights.
    """
    converted = dict(state)
    dense_blocks = [i for i in range(depth) if f'blocks.{i}.mlp.w12.weight' in state]
    if not dense_blocks:
        return state
    groups = rows = None
    if init['grouping'] == 'calibrated':
        groups = json.loads(Path(init['groups_file']).read_text(encoding='utf-8'))
        rows = validate_groups(groups, spec, depth, hidden)
    for i in dense_blocks:
        prefix = f'blocks.{i}.mlp.'
        names = ('w12.weight', 'w12.bias', 'w3.weight', 'w3.bias')
        if any(prefix + name not in state for name in names):
            raise ValueError(f'Incomplete source SwiGLU weights in block {i}')
        w12, b12, w3, bias = (state[prefix + name] for name in names)
        dim = w3.shape[0]
        if w12.shape != (2 * hidden, dim) or b12.shape != (2 * hidden,) \
                or w3.shape != (dim, hidden) or bias.shape != (dim,):
            raise ValueError(f'Block {i}: initialize spatial FFN from the full-width source ({hidden})')
        if any(key.startswith(prefix) and key not in [prefix + n for n in names] for key in state):
            raise ValueError(f'Block {i}: mixed/fading FFN state is not a full-width source')
        if groups is not None:
            if mlp_fingerprint(state, prefix) != groups['source_mlp_sha256'][i]:
                raise ValueError(f'groups_file was calibrated on different FFN weights (block {i})')
            full, coarse = rows[0][i], rows[1][i]
        else:
            # Baseline retains the exact source ordering. Other tiers share the
            # same per-block permutation, independent of global RNG and device.
            order = (torch.arange(hidden) if spec['full_hidden'] == hidden else torch.randperm(
                hidden, generator=torch.Generator().manual_seed(init['seed'] * 1_000_003 + i)))
            full = order[:spec['full_hidden']]
            coarse = order[spec['full_hidden']:spec['full_hidden'] + spec['coarse_hidden']]
        for name in names:
            del converted[prefix + name]
        converted[prefix + 'bias'] = bias
        for branch, indices in (('full', full), ('coarse', coarse)):
            if not len(indices):
                continue
            ids = torch.as_tensor(indices, dtype=torch.long, device=w12.device)
            packed = torch.cat((ids, ids + hidden))
            converted[prefix + branch + '.w12.weight'] = w12[packed].contiguous()
            converted[prefix + branch + '.w12.bias'] = b12[packed].contiguous()
            converted[prefix + branch + '.w3.weight'] = w3[:, ids].contiguous()
    return converted
