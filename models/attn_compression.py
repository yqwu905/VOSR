"""Head-split self-attention token reduction (``student.attn_compression``) for VOSR2 ablations.

Only the self-attention of the chosen blocks changes. Cross-attention (DINO) and the MLP still run on every
token, and the pretrained weights keep their names. Each block splits its heads by how local their dense
attention is (``tools/attn_head_stats.py`` measures it on the dense teacher):

- local heads, whose dense attention mass in the 3x3 token neighbourhood of the query exceeds
  ``local_threshold``, become a ``kernel`` x ``kernel`` depthwise conv over their full-grid values
  (``attn.local_conv``): no queries, keys or softmax. The conv starts from the head's mean attention weight at
  each offset around the query, renormalized to sum 1 inside the kernel;
- the other (global) heads attend among the ``factor`` x ``factor`` average-pooled tokens (N / factor^2
  queries, keys and values; RoPE at the centroid of each pooled group). Their part of the output projection
  runs on the pooled grid and is upsampled back to the full grid (``upsample``: bilinear or nearest).
- optional ``dense_heads`` retain full-resolution Q/K/V, full attention and full-grid RoPE. They are disjoint
  from local heads; sensitivity experiments can select them under an explicit MAC budget.

Both parts project their heads' channels with their columns of the pretrained ``proj`` and add up, the bias
once. With factor 1 and no local heads this is the original attention. Pooled attention cannot represent a
head that attends to the query token itself or to one fixed neighbour, which is why plain pooling loses text
strokes and such heads become convs instead. ``student.attn_type`` (block-sparse attention) applies to the
global heads on the pooled grid.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .sparse_attention import build_sparse_attn, windowed_attention

HEAD_SPLIT_DEFAULTS = dict(factor=2, kernel=7, upsample='bilinear', local_threshold=0.5)
UPSAMPLE_MODES = ('bilinear', 'nearest')
CONFIG_KEYS = ('type', 'factor', 'kernel', 'upsample', 'blocks', 'local_heads', 'dense_heads',
               'head_stats', 'local_threshold')


def load_head_stats(path):
    """``tools/attn_head_stats.py`` output: per block and head, the 3x3 attention mass and the mean attention
    weight at each offset of a (2 * radius + 1)^2 window around the query (row-major, dy then dx)."""
    stats = json.loads(Path(path).read_text(encoding='utf-8'))
    mass = torch.tensor(stats['mass3x3'], dtype=torch.float64, device='cpu')  # also inside a meta-device context
    pattern = torch.tensor(stats['pattern'], dtype=torch.float64, device='cpu')
    radius = int(stats['radius'])
    if mass.dim() != 2 or pattern.shape != (*mass.shape, (2 * radius + 1) ** 2):
        raise ValueError(f'{path}: expected mass3x3 (blocks, heads) and pattern (blocks, heads, (2r+1)^2)')
    return {'mass3x3': mass, 'pattern': pattern, 'radius': radius}


def attn_compression_spec(config, depth, num_heads):
    """Validate ``student.attn_compression`` and resolve its local heads.

    Returns ``(spec, kernels)``. ``spec`` is the structure that model.json stores: type, factor, kernel,
    upsample, blocks (sorted, the blocks whose self-attention is split; default all), optional dense_heads and local_heads (one sorted
    head list per block, empty outside ``blocks``). ``local_heads`` comes from the config, or from ``head_stats``
    and ``local_threshold``. ``kernels`` maps each block with local heads to the initial conv weights from
    ``head_stats`` ((heads, kernel, kernel)), or is None without head_stats (the convs then start as identity).
    A resolved spec passes through unchanged, so exports rebuild the same structure without the stats file.
    """
    if not isinstance(config, dict) or config.get('type') != 'head_split':
        raise ValueError('attn_compression must be a mapping with type: head_split')
    unknown = set(config) - set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f'Unknown attn_compression options: {sorted(unknown)}')
    spec = dict(HEAD_SPLIT_DEFAULTS, **{k: v for k, v in config.items() if v is not None})
    for key, minimum in (('factor', 1), ('kernel', 1)):
        if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] < minimum:
            raise ValueError(f'attn_compression.{key} must be an integer >= {minimum}')
    if spec['kernel'] % 2 == 0:
        raise ValueError('attn_compression.kernel must be odd')
    if spec['upsample'] not in UPSAMPLE_MODES:
        raise ValueError(f'attn_compression.upsample must be one of {UPSAMPLE_MODES}')
    threshold = spec['local_threshold']
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError('attn_compression.local_threshold must be in [0, 1]')
    blocks = list(range(depth)) if spec.get('blocks') is None else list(spec['blocks'])
    if not blocks or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < depth for i in blocks) \
            or len(set(blocks)) != len(blocks):
        raise ValueError(f'attn_compression.blocks must be unique block indices in [0, {depth})')
    blocks = sorted(blocks)
    stats = load_head_stats(spec['head_stats']) if spec.get('head_stats') else None
    if stats is not None and tuple(stats['mass3x3'].shape) != (depth, num_heads):
        raise ValueError(f'attn_compression.head_stats has {tuple(stats["mass3x3"].shape)} (blocks, heads), '
                         f'the model {(depth, num_heads)}')
    if spec.get('local_heads') is not None:
        local = spec['local_heads']
        if not isinstance(local, (list, tuple)) or len(local) != depth or not all(isinstance(h, (list, tuple))
                                                                                  for h in local):
            raise ValueError(f'attn_compression.local_heads must hold one head list per block ({depth})')
        local = [list(h) for h in local]
        for index, heads in enumerate(local):
            if any(isinstance(h, bool) or not isinstance(h, int) or not 0 <= h < num_heads for h in heads) \
                    or len(set(heads)) != len(heads):
                raise ValueError(f'attn_compression.local_heads[{index}] must be unique heads in [0, {num_heads})')
            if heads and index not in blocks:
                raise ValueError(f'attn_compression.local_heads[{index}]: block {index} is not in blocks')
    elif stats is not None:
        # Strictly above the threshold, compared in the precision the stats were measured in.
        mass = stats['mass3x3'].float()
        local = [[h for h in range(num_heads) if mass[i, h] > threshold] if i in blocks else []
                 for i in range(depth)]
    else:
        raise ValueError('attn_compression needs local_heads or head_stats')
    dense = spec.get('dense_heads', [[] for _ in range(depth)])
    if not isinstance(dense, (list, tuple)) or len(dense) != depth:
        raise ValueError(f'attn_compression.dense_heads must hold one head list per block ({depth})')
    for index, heads in enumerate(dense):
        if not isinstance(heads, (list, tuple)) or any(isinstance(h, bool) or not isinstance(h, int)
                or not 0 <= h < num_heads for h in heads) or len(set(heads)) != len(heads):
            raise ValueError(f'attn_compression.dense_heads[{index}] must be unique heads in [0, {num_heads})')
        if heads and index not in blocks:
            raise ValueError(f'attn_compression.dense_heads[{index}]: block {index} is not in blocks')
        if set(heads) & set(local[index]):
            if spec.get('local_heads') is not None:
                raise ValueError(f'attn_compression: local_heads and dense_heads overlap in block {index}')
            local[index] = [h for h in local[index] if h not in heads]
    resolved = dict(type='head_split', factor=spec['factor'], kernel=spec['kernel'], upsample=spec['upsample'],
                    blocks=blocks, local_heads=[sorted(h) for h in local])
    # Preserve old exports byte-for-byte when the new branch is absent.
    if any(dense):
        resolved['dense_heads'] = [sorted(h) for h in dense]
    kernels = None
    if stats is not None:
        radius, half = stats['radius'], spec['kernel'] // 2
        if half > radius:
            raise ValueError(f'attn_compression.kernel {spec["kernel"]} exceeds the head_stats window '
                             f'{2 * radius + 1}')
        size = 2 * radius + 1
        kernels = {}
        for index, heads in enumerate(resolved['local_heads']):
            if heads:
                window = stats['pattern'][index, heads].view(len(heads), size, size)
                window = window[:, radius - half:radius + half + 1, radius - half:radius + half + 1].float()
                kernels[index] = window / window.sum((1, 2), keepdim=True)
    return resolved, kernels


def pool_grid(x, side, factor):
    """(B, side^2, C) row-major grid -> (B, (side / factor)^2, C), factor x factor averages."""
    if factor == 1:
        return x
    b, _, c = x.shape
    s = side // factor
    return x.reshape(b, s, factor, s, factor, c).mean((2, 4)).reshape(b, s * s, c)


def upsample_grid(x, side, factor, mode):
    """(B, (side / factor)^2, C) -> (B, side^2, C): nearest copies, bilinear (align_corners False) in FP32."""
    if factor == 1:
        return x
    b, _, c = x.shape
    s = side // factor
    if mode == 'nearest':
        return x.reshape(b, s, 1, s, 1, c).expand(b, s, factor, s, factor, c).reshape(b, side * side, c)
    grid = F.interpolate(x.transpose(1, 2).reshape(b, c, s, s).float(), scale_factor=factor, mode=mode,
                         align_corners=False)
    return grid.to(x.dtype).flatten(2).transpose(1, 2)


def pooled_rope(positions, half_head_dim, dtype):
    """RoPE at the given grid positions with cos/sin tables in ``dtype`` (the block's residual dtype).

    Applied as ``t * cos + rotate_half(t) * sin`` without casting the tables to ``t``'s dtype, like the
    upstream RoPE modules.
    """
    from .pos_embed import rotate_half
    from .token_hourglass import rope_tables
    cos, sin = rope_tables(positions, half_head_dim)
    cos, sin = cos.to(dtype), sin.to(dtype)

    def rope(t):
        return t * cos + rotate_half(t) * sin
    return rope


class HeadSplitAttention(nn.Module):
    """Drop-in replacement of a block's ``Attention``: global heads on pooled tokens, local heads as a conv.

    Keeps the pretrained submodules qkv, q_norm, k_norm and proj (same state-dict names) and adds
    ``local_conv`` (depthwise, no bias) when the block has local heads. ``rope`` is the RoPE of the pooled grid
    (the model passes it; at factor 1 the full grid's), or a (pooled, full) tuple when dense_heads are present.
    """
    def __init__(self, attn, local_heads, factor, kernel, upsample, dense_heads=()):
        super().__init__()
        for name in ('qkv', 'q_norm', 'k_norm', 'attn_drop', 'proj', 'proj_drop'):
            setattr(self, name, getattr(attn, name))
        self.num_heads, self.head_dim, self.scale = attn.num_heads, attn.head_dim, attn.scale
        self.fused_attn = attn.fused_attn
        self.factor, self.upsample = factor, upsample
        self.local_heads = sorted(local_heads)
        self.dense_heads = sorted(dense_heads)
        for group in (self.local_heads, self.dense_heads):
            if len(set(group)) != len(group) or any(isinstance(h, bool) or not isinstance(h, int)
                                                   or not 0 <= h < self.num_heads for h in group):
                raise ValueError('Head indices must be unique and in range')
        if set(self.local_heads) & set(self.dense_heads):
            raise ValueError('local_heads and dense_heads overlap')
        self.global_heads = [h for h in range(self.num_heads) if h not in self.local_heads + self.dense_heads]
        dim = self.num_heads * self.head_dim

        def channels(heads):
            return torch.tensor([h * self.head_dim + c for h in heads for c in range(self.head_dim)],
                                dtype=torch.long)
        global_channels, local_channels = channels(self.global_heads), channels(self.local_heads)
        # Rows of the fused qkv weight and columns of proj that belong to each group of heads.
        self.register_buffer('global_qkv_rows', torch.cat((global_channels, global_channels + dim,
                                                            global_channels + 2 * dim)), persistent=False)
        self.register_buffer('global_channels', global_channels, persistent=False)
        self.register_buffer('local_value_rows', local_channels + 2 * dim, persistent=False)
        self.register_buffer('local_channels', local_channels, persistent=False)
        dense_channels = channels(self.dense_heads)
        self.register_buffer('dense_channels', dense_channels, persistent=False)
        self.register_buffer('dense_qkv_rows', torch.cat((dense_channels, dense_channels + dim,
                                                         dense_channels + 2 * dim)), persistent=False)
        width = len(local_channels)
        self.local_conv = (nn.Conv2d(width, width, kernel, padding=kernel // 2, groups=width, bias=False)
                           if width else None)
        self.set_local_kernels(None)
        self.set_attn_type(attn.attn_type)

    @torch.no_grad()
    def set_local_kernels(self, kernels):
        """Per local head a (kernel, kernel) weight shared by its channels; None = identity (the query itself)."""
        if self.local_conv is None:
            return
        weight = self.local_conv.weight
        if kernels is None:
            weight.zero_()
            weight[:, :, weight.shape[-2] // 2, weight.shape[-1] // 2] = 1
            return
        if tuple(kernels.shape) != (len(self.local_heads), *weight.shape[-2:]):
            raise ValueError(f'Expected {len(self.local_heads)} local-head kernels of {tuple(weight.shape[-2:])}')
        weight.copy_(kernels.to(weight).repeat_interleave(self.head_dim, 0).unsqueeze(1))

    def set_attn_type(self, attn_type):
        """full | sparse | local self-attention of the global heads on the pooled grid (models/sparse_attention.py)."""
        sparse = build_sparse_attn(attn_type, self.head_dim, max(1, len(self.global_heads)))
        self.sparse_attn = sparse if self.global_heads else None
        self.attn_type = attn_type

    def _attend(self, q, k, v):
        if self.sparse_attn is not None:
            return windowed_attention(self.sparse_attn, q, k, v)
        if self.fused_attn:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        return self.attn_drop(attention.softmax(dim=-1)) @ v

    def forward(self, x, rope=None):
        # A mixed-resolution block receives both tables; old exports still receive a callable.
        dense_rope = None
        if isinstance(rope, tuple):
            rope, dense_rope = rope
        elif self.dense_heads and rope is not None:
            if self.factor != 1:
                raise ValueError('Dense heads require (pooled_rope, full_rope), not a pooled-only RoPE')
            dense_rope = rope
        b, n, _ = x.shape
        side = math.isqrt(n)
        if side * side != n or side % self.factor:
            raise ValueError(f'Head-split attention needs a square token grid with sides divisible by '
                             f'{self.factor}; got {n} tokens')
        weight, bias = self.qkv.weight, self.qkv.bias
        out = 0
        if self.global_heads:
            pooled = pool_grid(x, side, self.factor)
            heads, rows = len(self.global_heads), self.global_qkv_rows
            q, k, v = F.linear(pooled, weight[rows], None if bias is None else bias[rows]).split(heads * self.head_dim, -1)
            q, k, v = (t.reshape(b, -1, heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
            q, k = self.q_norm(q), self.k_norm(k)
            if rope is not None:
                q, k = rope(q), rope(k)
            o = self._attend(q, k, v).transpose(1, 2).reshape(b, pooled.shape[1], heads * self.head_dim)
            o = F.linear(o, self.proj.weight[:, self.global_channels], self.proj.bias)
            out = upsample_grid(o, side, self.factor, self.upsample)
        if self.local_heads:
            rows = self.local_value_rows
            v = F.linear(x, weight[rows], None if bias is None else bias[rows])
            v = self.local_conv(v.transpose(1, 2).reshape(b, -1, side, side)).flatten(2).transpose(1, 2)
            out = out + F.linear(v, self.proj.weight[:, self.local_channels],
                                 None if self.global_heads else self.proj.bias)
        if self.dense_heads:
            heads, rows = len(self.dense_heads), self.dense_qkv_rows
            q, k, v = F.linear(x, weight[rows], None if bias is None else bias[rows]).split(heads * self.head_dim, -1)
            q, k, v = (t.reshape(b, n, heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
            q, k = self.q_norm(q), self.k_norm(k)
            if dense_rope is not None:
                q, k = dense_rope(q), dense_rope(k)
            # Protected heads always use full attention, independently of the pooled branch's attn_type.
            if self.fused_attn:
                v = F.scaled_dot_product_attention(q, k, v,
                        dropout_p=self.attn_drop.p if self.training else 0.)
            else:
                v = self.attn_drop(((q * self.scale) @ k.transpose(-2, -1)).softmax(-1)) @ v
            v = v.transpose(1, 2).reshape(b, n, heads * self.head_dim)
            out = out + F.linear(v, self.proj.weight[:, self.dense_channels],
                                 None if self.global_heads or self.local_heads else self.proj.bias)
        return self.proj_drop(out)
