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

Both parts project their heads' channels with their columns of the pretrained ``proj`` and add up, the bias
once. With factor 1 and no local heads this is the original attention. Pooled attention cannot represent a
head that attends to the query token itself or to one fixed neighbour, which is why plain pooling loses text
strokes and such heads become convs instead. ``student.attn_type`` (block-sparse attention) applies to the
global heads on the pooled grid.

Two more head types keep full-grid queries (``kv_pool_heads``, ``dense_heads``; empty by default): a
key/value-pooled head attends from every full-grid token to the pooled keys and values, a dense head is the original
full attention. ``search`` learns which heads to upgrade to them (``HeadSearchAttention``): every head of the
head-split base gets one logit per upgrade; a price per MAC, solved from all logits, keeps the expected
self-attention MACs within ``search.attn_budget``; training samples a hard choice per head and image
(straight-through Gumbel-softmax) and fixing the choice at the budget gives the fixed head lists of an export.
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
CONFIG_KEYS = ('type', 'factor', 'kernel', 'upsample', 'blocks', 'local_heads', 'head_stats', 'local_threshold',
               'kv_pool_heads', 'dense_heads', 'heads_from', 'search')
HEAD_LIST_KEYS = ('local_heads', 'kv_pool_heads', 'dense_heads')
# Upgrades a head search may give a head besides its head-split base (local conv or pooled attention).
SEARCH_CANDIDATES = ('kv_pool', 'dense')
SEARCH_DEFAULTS = dict(candidates=list(SEARCH_CANDIDATES), temperature=[1.0, 0.1])
SEARCH_KEYS = ('candidates', 'attn_budget', 'temperature')


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
    """Validate ``student.attn_compression`` and resolve its head types.

    Returns ``(spec, kernels)``. ``spec`` is the structure that model.json stores: type, factor, kernel,
    upsample, blocks (sorted, the blocks whose self-attention is split; default all) and local_heads (one sorted
    head list per block, empty outside ``blocks``), plus kv_pool_heads / dense_heads when any block has such heads
    and the resolved ``search``. ``local_heads`` comes from the config, from ``heads_from`` (the fixed head lists of
    another export) or from ``head_stats`` and ``local_threshold``. ``kernels`` maps each block with local heads to
    the initial conv weights from ``head_stats`` ((heads, kernel, kernel)), or is None without head_stats (the convs
    then start as identity). A resolved spec passes through unchanged, so exports rebuild the same structure
    without the stats file.
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
    if spec.get('heads_from') is not None:
        if any(spec.get(key) is not None for key in HEAD_LIST_KEYS):
            raise ValueError('attn_compression.heads_from replaces local_heads, kv_pool_heads and dense_heads; '
                             'set only one of them')
        source = load_head_selection(spec['heads_from'])
        for key, value in (('factor', spec['factor']), ('kernel', spec['kernel']), ('upsample', spec['upsample']),
                           ('blocks', blocks)):
            if source.get(key) != value:
                raise ValueError(f'attn_compression.heads_from has {key} {source.get(key)}, this config {value}')
        spec.update({key: source.get(key) or [[]] * depth for key in HEAD_LIST_KEYS})

    def head_lists(key):
        heads = spec[key]
        if not isinstance(heads, (list, tuple)) or len(heads) != depth or not all(isinstance(h, (list, tuple))
                                                                                  for h in heads):
            raise ValueError(f'attn_compression.{key} must hold one head list per block ({depth})')
        heads = [list(h) for h in heads]
        for index, row in enumerate(heads):
            if any(isinstance(h, bool) or not isinstance(h, int) or not 0 <= h < num_heads for h in row) \
                    or len(set(row)) != len(row):
                raise ValueError(f'attn_compression.{key}[{index}] must be unique heads in [0, {num_heads})')
            if row and index not in blocks:
                raise ValueError(f'attn_compression.{key}[{index}]: block {index} is not in blocks')
        return heads
    if spec.get('local_heads') is not None:
        local = head_lists('local_heads')
    elif stats is not None:
        # Strictly above the threshold, compared in the precision the stats were measured in.
        mass = stats['mass3x3'].float()
        local = [[h for h in range(num_heads) if mass[i, h] > threshold] if i in blocks else []
                 for i in range(depth)]
    else:
        raise ValueError('attn_compression needs local_heads or head_stats (or heads_from)')
    upgraded = {key: head_lists(key) if spec.get(key) is not None else [[] for _ in range(depth)]
                for key in ('kv_pool_heads', 'dense_heads')}
    for index in range(depth):
        rows = (local[index], upgraded['kv_pool_heads'][index], upgraded['dense_heads'][index])
        if len(set().union(*rows)) != sum(map(len, rows)):
            raise ValueError(f'attn_compression block {index}: a head is at most one of local, kv_pool and dense')
    resolved = dict(type='head_split', factor=spec['factor'], kernel=spec['kernel'], upsample=spec['upsample'],
                    blocks=blocks, local_heads=[sorted(h) for h in local])
    for key, heads in upgraded.items():
        if any(heads):  # absent keeps the model.json of plain head-split exports unchanged
            resolved[key] = [sorted(h) for h in heads]
    if spec.get('search') is not None:
        if any(map(any, upgraded.values())):
            raise ValueError('attn_compression.search starts from local and pooled heads; '
                             'leave kv_pool_heads and dense_heads empty')
        resolved['search'] = search_spec(spec['search'])
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


def load_head_selection(path):
    """``heads_from``: the ``attn_compression`` of a fixed head-split export, i.e. its directory or model.json."""
    path = Path(path)
    if path.is_dir():
        path = path / 'model.json'
    if not path.is_file():
        raise FileNotFoundError(f'attn_compression.heads_from: {path} not found; point it at the export/ of a '
                                'head search run (or a directory written by tools/head_search_select.py)')
    config = json.loads(path.read_text(encoding='utf-8')).get('attn_compression')
    if not isinstance(config, dict) or config.get('type') != 'head_split':
        raise ValueError(f'attn_compression.heads_from: {path} is not a head-split export')
    if config.get('search') is not None:
        raise ValueError(f'attn_compression.heads_from: {path} is a head search state (a checkpoint); use the '
                         "run's export/ or fix its heads with tools/head_search_select.py")
    return config


def search_spec(config):
    """Validate ``attn_compression.search``: ``candidates`` (upgrades, any of kv_pool and dense), ``attn_budget``
    (the self-attention MACs allowed, as a share of the dense self-attention of all blocks at the reference grid)
    and ``temperature`` ([start, end] of the Gumbel-softmax temperature, annealed linearly over training)."""
    if not isinstance(config, dict):
        raise ValueError('attn_compression.search must be a mapping')
    unknown = set(config) - set(SEARCH_KEYS)
    if unknown:
        raise ValueError(f'Unknown attn_compression.search options: {sorted(unknown)}')
    search = dict(SEARCH_DEFAULTS, **{k: v for k, v in config.items() if v is not None})
    candidates = search['candidates']
    if not isinstance(candidates, (list, tuple)) or not candidates or len(set(candidates)) != len(candidates) \
            or any(c not in SEARCH_CANDIDATES for c in candidates):
        raise ValueError(f'attn_compression.search.candidates must be unique entries of {SEARCH_CANDIDATES}')
    budget = search.get('attn_budget')
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 < budget <= 1:
        raise ValueError('attn_compression.search.attn_budget (share of the dense self-attention MACs) must be '
                         'in (0, 1]')
    temperature = search['temperature']
    if not isinstance(temperature, (list, tuple)) or len(temperature) != 2 or any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or t <= 0 for t in temperature):
        raise ValueError('attn_compression.search.temperature must be [start, end], both > 0')
    return dict(candidates=[c for c in SEARCH_CANDIDATES if c in candidates], attn_budget=float(budget),
                temperature=[float(t) for t in temperature])


def head_macs(kind, tokens, dim, head_dim, factor, kernel):
    """MACs of one self-attention head of the given kind on ``tokens`` tokens, as FlopCounterMode counts them: the
    q/k/v rows and output-projection columns of the head and its two attention matmuls (norms, RoPE, pooling,
    upsampling and softmax are not counted)."""
    pooled = tokens // factor ** 2
    rows = lambda count, width: count * dim * width * head_dim  # input projection of `width` of q, k and v
    if kind == 'dense':
        return rows(tokens, 3) + 2 * tokens * tokens * head_dim + tokens * head_dim * dim
    if kind == 'kv_pool':
        return rows(tokens, 1) + rows(pooled, 2) + 2 * tokens * pooled * head_dim + tokens * head_dim * dim
    if kind == 'local':
        return rows(tokens, 1) + tokens * head_dim * kernel ** 2 + tokens * head_dim * dim
    if kind == 'pooled':
        return rows(pooled, 3) + 2 * pooled * pooled * head_dim + pooled * head_dim * dim
    raise ValueError(f'Unknown head kind {kind}')


def search_price(logits, costs, target, hard=False):
    """Price per unit of cost that keeps a head search within budget.

    ``logits`` and ``costs`` are (heads, options); the options of a head are chosen by softmax(logits - price *
    costs). Returns the smallest price >= 0 at which the expected cost (``hard``: the cost of the argmax options)
    summed over all heads is at most ``target``, found by bisection on the CPU in float64.
    """
    logits, costs = logits.detach().double().cpu(), costs.detach().double().cpu()

    def spend(price):
        scores = logits - price * costs
        if hard:
            return costs.gather(1, scores.argmax(1, keepdim=True)).sum().item()
        return (scores.softmax(1) * costs).sum().item()
    if spend(0.) <= target:
        return 0.
    if costs.min(1).values.sum().item() >= target:
        raise ValueError('The head search budget does not exceed the cost of the cheapest option of every head')
    high = 1.
    for _ in range(64):
        if spend(high) <= target:
            break
        high *= 2
    low = 0.
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if spend(middle) > target else (low, middle)
    return high


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
    (the model passes it; at factor 1 the full grid's). With key/value-pooled or dense heads, whose queries stay on
    the full grid, the model passes ``(pooled RoPE, full-grid RoPE)``.
    """
    def __init__(self, attn, local_heads, factor, kernel, upsample, kv_pool_heads=(), dense_heads=()):
        super().__init__()
        for name in ('qkv', 'q_norm', 'k_norm', 'attn_drop', 'proj', 'proj_drop'):
            setattr(self, name, getattr(attn, name))
        self.num_heads, self.head_dim, self.scale = attn.num_heads, attn.head_dim, attn.scale
        self.fused_attn = attn.fused_attn
        self.factor, self.upsample = factor, upsample
        self.local_heads = sorted(local_heads)
        self.kv_pool_heads, self.dense_heads = sorted(kv_pool_heads), sorted(dense_heads)
        typed = self.local_heads + self.kv_pool_heads + self.dense_heads
        if len(set(typed)) != len(typed):
            raise ValueError('A head is at most one of local, kv_pool and dense')
        self.global_heads = [h for h in range(self.num_heads) if h not in typed]
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
        # Full-grid queries: the dense heads, then the key/value-pooled heads.
        dense_channels, kv_channels = channels(self.dense_heads), channels(self.kv_pool_heads)
        query_channels = torch.cat((dense_channels, kv_channels))
        self.register_buffer('query_rows', query_channels, persistent=False)
        self.register_buffer('dense_key_value_rows', torch.cat((dense_channels + dim, dense_channels + 2 * dim)),
                             persistent=False)
        self.register_buffer('pooled_key_value_rows', torch.cat((kv_channels + dim, kv_channels + 2 * dim)),
                             persistent=False)
        # proj columns of everything that is projected on the full grid: query heads, then local heads.
        self.register_buffer('full_channels', torch.cat((query_channels, local_channels)), persistent=False)
        width = len(local_channels)
        self.local_conv = (nn.Conv2d(width, width, kernel, padding=kernel // 2, groups=width, bias=False)
                           if width else None)
        self.set_local_kernels(None)
        self.set_attn_type(attn.attn_type)

    @property
    def needs_full_rope(self):
        """Whether queries stay on the full grid, so the model must also pass the full-grid RoPE."""
        return bool(self.dense_heads or self.kv_pool_heads)

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
        """full | sparse | local self-attention of the global heads on the pooled grid (models/sparse_attention.py).
        Dense and key/value-pooled heads always attend fully."""
        sparse = build_sparse_attn(attn_type, self.head_dim, max(1, len(self.global_heads)))
        self.sparse_attn = sparse if self.global_heads else None
        self.attn_type = attn_type

    def _attend(self, q, k, v):
        if self.sparse_attn is not None:
            return windowed_attention(self.sparse_attn, q, k, v)
        return self._full_attend(q, k, v)

    def _full_attend(self, q, k, v):
        if self.fused_attn:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        return self.attn_drop(attention.softmax(dim=-1)) @ v

    @staticmethod
    def _ropes(rope):
        """(pooled-grid RoPE, full-grid RoPE) from what the model passes."""
        return rope if isinstance(rope, tuple) else (rope, None)

    def _grid_side(self, n):
        side = math.isqrt(n)
        if side * side != n or side % self.factor:
            raise ValueError(f'Head-split attention needs a square token grid with sides divisible by '
                             f'{self.factor}; got {n} tokens')
        return side

    def forward(self, x, rope=None):
        b, n, _ = x.shape
        side = self._grid_side(n)
        rope, full_rope = self._ropes(rope)
        if self.needs_full_rope and rope is not None and full_rope is None:
            raise ValueError('Dense and key/value-pooled heads need (pooled RoPE, full-grid RoPE)')
        weight, bias = self.qkv.weight, self.qkv.bias
        out, pooled = 0, None
        if self.global_heads:
            pooled = pool_grid(x, side, self.factor)
            heads, rows = len(self.global_heads), self.global_qkv_rows
            q, k, v = F.linear(pooled, weight[rows], bias[rows]).split(heads * self.head_dim, -1)
            q, k, v = (t.reshape(b, -1, heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
            q, k = self.q_norm(q), self.k_norm(k)
            if rope is not None:
                q, k = rope(q), rope(k)
            o = self._attend(q, k, v).transpose(1, 2).reshape(b, pooled.shape[1], heads * self.head_dim)
            o = F.linear(o, self.proj.weight[:, self.global_channels], self.proj.bias)
            out = upsample_grid(o, side, self.factor, self.upsample)
        full = []  # outputs on the full grid before proj: query heads, then local heads
        if self.needs_full_rope:
            full.append(self._query_heads(x, side, pooled, rope, full_rope))
        if self.local_heads:
            rows = self.local_value_rows
            v = F.linear(x, weight[rows], bias[rows])
            full.append(self.local_conv(v.transpose(1, 2).reshape(b, -1, side, side)).flatten(2).transpose(1, 2))
        if full:
            out = out + F.linear(full[0] if len(full) == 1 else torch.cat(full, -1),
                                 self.proj.weight[:, self.full_channels], None if self.global_heads else self.proj.bias)
        return self.proj_drop(out)

    def _query_heads(self, x, side, pooled, rope, full_rope):
        """Dense heads (full attention) and key/value-pooled heads (full-grid queries, pooled keys and values)."""
        b, n, _ = x.shape
        weight, bias, hd = self.qkv.weight, self.qkv.bias, self.head_dim
        dense, kv = len(self.dense_heads), len(self.kv_pool_heads)
        q = F.linear(x, weight[self.query_rows], bias[self.query_rows])
        q = self.q_norm(q.reshape(b, n, dense + kv, hd).transpose(1, 2))
        if full_rope is not None:
            q = full_rope(q)
        outputs = []
        if dense:
            rows = self.dense_key_value_rows
            k, v = (t.reshape(b, n, dense, hd).transpose(1, 2)
                    for t in F.linear(x, weight[rows], bias[rows]).split(dense * hd, -1))
            k = self.k_norm(k)
            if full_rope is not None:
                k = full_rope(k)
            outputs.append(self._full_attend(q[:, :dense], k, v))
        if kv:
            pooled = pool_grid(x, side, self.factor) if pooled is None else pooled
            rows = self.pooled_key_value_rows
            k, v = (t.reshape(b, pooled.shape[1], kv, hd).transpose(1, 2)
                    for t in F.linear(pooled, weight[rows], bias[rows]).split(kv * hd, -1))
            k = self.k_norm(k)
            if rope is not None:
                k = rope(k)
            outputs.append(self._full_attend(q[:, dense:], k, v))
        o = outputs[0] if len(outputs) == 1 else torch.cat(outputs, 1)
        return o.transpose(1, 2).reshape(b, n, (dense + kv) * hd)


class HeadSearchAttention(HeadSplitAttention):
    """Head-split attention whose heads each choose, while training, between their base type and upgrades.

    The base is the head-split structure (local heads as convs, the others pooled); ``candidates`` are the
    upgrades (``kv_pool``, ``dense``). ``head_logits`` (heads, upgrades) are the only new weights; the base option
    has logit 0. The model sets ``price`` (per unit of ``costs``, the MACs of each option in dense heads) so that
    softmax(logits - price * costs) meets the budget, and ``temperature``. Training draws a hard option per head and
    image from that softmax (Gumbel-max) and passes gradients to the logits through the Gumbel-softmax at
    ``temperature`` (straight-through); eval mode takes the argmax. All options are computed from one full-grid qkv
    projection (pooling commutes with it) and mixed per head before one full-grid ``proj``, which equals the fixed
    structure up to rounding.
    """
    def __init__(self, attn, local_heads, factor, kernel, upsample, candidates, costs):
        super().__init__(attn, local_heads, factor, kernel, upsample)
        self.candidates = list(candidates)
        self.head_logits = nn.Parameter(torch.zeros(self.num_heads, len(self.candidates)))
        if tuple(costs.shape) != (self.num_heads, 1 + len(self.candidates)):
            raise ValueError('Expected the costs of the base and every candidate per head')
        self.register_buffer('costs', costs.float(), persistent=False)
        # Base heads in head order: the pooled (global) heads' outputs, then the local heads', reordered.
        self.register_buffer('base_order', torch.argsort(torch.tensor(self.global_heads + self.local_heads)),
                             persistent=False)
        self.price, self.temperature = 0., 1.

    @property
    def needs_full_rope(self):
        return True

    def option_logits(self):
        """(heads, 1 + candidates) logits of [base, *candidates], FP32."""
        return torch.cat((self.head_logits.new_zeros(self.num_heads, 1), self.head_logits), 1).float()

    def choice(self, batch):
        """(batch, heads, options) one-hot option per head and image; straight-through gradients in training."""
        scores = self.option_logits() - self.price * self.costs
        if not self.training:
            return F.one_hot(scores.argmax(-1), scores.shape[-1]).float().expand(batch, -1, -1)
        gumbel = -torch.empty(batch, *scores.shape, device=scores.device).exponential_().log()
        noisy = scores + gumbel
        soft = (noisy / self.temperature).softmax(-1)
        hard = F.one_hot(noisy.argmax(-1), scores.shape[-1]).to(soft.dtype)
        return hard + (soft - soft.detach())  # exactly the hard choice forward

    def forward(self, x, rope=None):
        b, n, _ = x.shape
        side = self._grid_side(n)
        rope, full_rope = self._ropes(rope)
        heads, hd = self.num_heads, self.head_dim
        qkv = F.linear(x, self.qkv.weight, self.qkv.bias)
        pooled = pool_grid(qkv, side, self.factor)  # the pooled tokens' q, k and v
        q, k, v = qkv.reshape(b, n, 3, heads, hd).permute(2, 0, 3, 1, 4)
        pq, pk, pv = pooled.reshape(b, pooled.shape[1], 3, heads, hd).permute(2, 0, 3, 1, 4)
        q, k, pq, pk = self.q_norm(q), self.k_norm(k), self.q_norm(pq), self.k_norm(pk)
        if full_rope is not None:
            q, k = full_rope(q), full_rope(k)
        if rope is not None:
            pq, pk = rope(pq), rope(pk)
        base = []
        if self.global_heads:
            g = self.global_heads
            o = self._attend(pq[:, g], pk[:, g], pv[:, g])  # (B, global, pooled tokens, hd)
            o = upsample_grid(o.transpose(1, 2).reshape(b, -1, len(g) * hd), side, self.factor, self.upsample)
            base.append(o.reshape(b, n, len(g), hd).transpose(1, 2))
        if self.local_heads:
            values = v[:, self.local_heads].transpose(1, 2).reshape(b, n, -1)
            o = self.local_conv(values.transpose(1, 2).reshape(b, -1, side, side)).flatten(2)
            base.append(o.reshape(b, len(self.local_heads), hd, n).transpose(2, 3))
        options = [torch.cat(base, 1)[:, self.base_order] if len(base) > 1 else base[0]]
        for candidate in self.candidates:
            options.append(self._full_attend(q, pk, pv) if candidate == 'kv_pool' else self._full_attend(q, k, v))
        options = torch.stack(options, 2)  # (B, heads, options, N, hd)
        o = torch.einsum('bhk,bhknd->bhnd', self.choice(b).to(options.dtype), options)
        return self.proj_drop(F.linear(o.transpose(1, 2).reshape(b, n, heads * hd), self.proj.weight,
                                       self.proj.bias))
