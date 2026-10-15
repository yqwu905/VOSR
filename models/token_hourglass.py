"""Static U-shaped hourglass token merging for VOSR2 ablations.

The pretrained blocks keep their weights; only the token grid they run on
changes. The first ``fine_in`` and last ``fine_out`` active blocks see the full
p2 token grid. The blocks between them run on factor x factor merged tokens
(SpaceToDepth + Linear, initialized as the group average). After the coarse
span, Linear + DepthToSpace (initialized as a copy) returns to the full grid.
With ``bypass`` the coarse span only adds its update, ``h_fine + Up(y_out - y_in)``,
so each fine token keeps its own deviation from the group mean. ``inner_depth``
blocks in the middle of the merged span are merged once more (a UNet-like second
level) with their own merge, unmerge and bypass. ``fine_blocks`` keeps chosen blocks
inside the merged span on the full grid, splitting it; every merged stretch reuses the
same merge, unmerge and bypass. All shapes are static: no top-k, gather or scatter.

Optional ``routing`` (mixed resolution): at the merge, a router scores every factor x factor
group and the top fine-group share stays on the full grid inside the merged span, next to the
merged tokens of the other groups in one sequence. Only this mode uses top-k, gather and scatter
(once at the merge, once at the unmerge); with a fixed share the shapes stay static.
"""
import math
import random
import torch
import torch.nn.functional as F
from torch import nn

HOURGLASS_DEFAULTS = dict(factor=2, fine_in=1, fine_out=1, drop_blocks=(), bypass=True,
                          rope='centroid', fine_cross_attention=True, cond_pool=1,
                          inner_depth=0, inner_cond_pool=1, fine_blocks=())
NEW_MODULE_PREFIXES = ('token_merge.', 'token_unmerge.', 'token_merge_inner.', 'token_unmerge_inner.')


def hourglass_spec(config, depth):
    """Validate a ``student.token_compression`` mapping and fill in its defaults."""
    if not isinstance(config, dict) or config.get('type') != 'hourglass':
        raise ValueError('token_compression must be a mapping with type: hourglass')
    unknown = set(config) - set(HOURGLASS_DEFAULTS) - {'type', 'routing'}
    if unknown:
        raise ValueError(f'Unknown token_compression options: {sorted(unknown)}')
    spec = dict(HOURGLASS_DEFAULTS, **{k: v for k, v in config.items() if k not in ('type', 'routing')})
    integers = {'factor': 2, 'fine_in': 0, 'fine_out': 0, 'cond_pool': 1, 'inner_depth': 0, 'inner_cond_pool': 1}
    for key, minimum in integers.items():
        if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] < minimum:
            raise ValueError(f'token_compression.{key} must be an integer >= {minimum}')
    for key in ('bypass', 'fine_cross_attention'):
        if not isinstance(spec[key], bool):
            raise ValueError(f'token_compression.{key} must be true or false')
    if spec['rope'] not in ('centroid', 'corner'):
        raise ValueError('token_compression.rope must be centroid or corner')
    drop = list(spec['drop_blocks'] or ())
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < depth for i in drop) \
            or len(set(drop)) != len(drop):
        raise ValueError(f'token_compression.drop_blocks must be unique block indices in [0, {depth})')
    if depth - len(drop) - spec['fine_in'] - spec['fine_out'] < 1:
        raise ValueError('token_compression leaves no block for the merged span')
    if spec['inner_depth'] > depth - len(drop) - spec['fine_in'] - spec['fine_out']:
        raise ValueError('token_compression.inner_depth cannot exceed the merged blocks')
    fine = check_fine_blocks(spec['fine_blocks'] or (), merged_blocks(dict(spec, drop_blocks=drop), depth),
                             'token_compression.fine_blocks')
    spec = dict(type='hourglass', **dict(spec, drop_blocks=sorted(drop), fine_blocks=fine))
    if config.get('routing') is not None:  # absent keeps the plain hourglass spec (and model.json) unchanged
        spec['routing'] = routing_spec(config['routing'], spec)
    return spec


ROUTING_DEFAULTS = dict(keep_ratios=(0.25, 0.5, 0.75, 1.0), eval_keep_ratio=0.75, warmup_steps=0)


def routing_spec(config, spec):
    """Validate ``token_compression.routing``: the share of groups kept on the full grid in the merged span.

    keep_ratios: shares the trainer samples from, one per optimizer step (elastic capacity);
    eval_keep_ratio: the share of eval mode, previews, exports and inference (``--keep-ratio`` overrides it);
    warmup_steps: optimizer steps over which the lowest share that may be sampled falls from 1 to min(keep_ratios).
    """
    if not isinstance(config, dict):
        raise ValueError('token_compression.routing must be a mapping')
    unknown = set(config) - set(ROUTING_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown token_compression.routing options: {sorted(unknown)}')
    if spec['inner_depth'] or spec['fine_blocks']:
        raise ValueError('token_compression.routing needs inner_depth 0 and no fine_blocks (one merged span)')
    routing = dict(ROUTING_DEFAULTS, **config)
    ratios = routing['keep_ratios']
    share = lambda v: not isinstance(v, bool) and isinstance(v, (int, float)) and 0 <= v <= 1
    if not isinstance(ratios, (list, tuple)) or not ratios or not all(share(r) for r in ratios) \
            or not share(routing['eval_keep_ratio']):
        raise ValueError('token_compression.routing keep_ratios (non-empty) and eval_keep_ratio must be in [0, 1]')
    warmup = routing['warmup_steps']
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('token_compression.routing.warmup_steps must be an integer >= 0')
    return dict(keep_ratios=sorted({float(r) for r in ratios}), eval_keep_ratio=float(routing['eval_keep_ratio']),
                warmup_steps=warmup)


def merged_blocks(spec, depth):
    """Pretrained block indices of the merged span: kept blocks between the fine entry and exit."""
    active = [i for i in range(depth) if i not in spec['drop_blocks']]
    return active[spec['fine_in']:len(active) - spec['fine_out']]


def check_fine_blocks(indices, merged, name='fine_blocks'):
    """Sorted block indices kept on the full grid inside the merged span; at least one block stays merged."""
    indices = list(indices)
    if any(isinstance(i, bool) or not isinstance(i, int) or i not in merged for i in indices) \
            or len(set(indices)) != len(indices) or len(indices) >= len(merged):
        raise ValueError(f'{name} must be unique blocks of the merged span {merged[0]}-{merged[-1]}, '
                         'leaving at least one block merged')
    return sorted(indices)


def _side(tokens, factor):
    side = math.isqrt(tokens)
    if side * side != tokens or side % factor:
        raise ValueError(f'Token merging needs a square grid divisible by {factor}; got {tokens} tokens')
    return side


def pool_tokens(x, factor):
    """Average factor x factor neighbours of a square row-major token grid (B, N, C)."""
    if factor == 1:
        return x
    b, n, c = x.shape
    s = _side(n, factor) // factor
    pooled = x.float().reshape(b, s, factor, s, factor, c).mean((2, 4))
    return pooled.reshape(b, s * s, c).to(x.dtype)


def space_to_depth_tokens(x, factor):
    """(B, N, C) -> (B, N / factor^2, factor^2 * C), sub-position major, channel minor."""
    b, n, c = x.shape
    s = _side(n, factor) // factor
    x = x.reshape(b, s, factor, s, factor, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, s * s, factor * factor * c)


def depth_to_space_tokens(x, factor):
    """Inverse of space_to_depth_tokens."""
    b, n, c = x.shape
    s = _side(n, 1)
    c //= factor * factor
    x = x.reshape(b, s, s, factor, factor, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, n * factor * factor, c)


class TokenMerge(nn.Module):
    """SpaceToDepth + Linear; starts as the factor x factor average."""
    def __init__(self, dim, factor):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(factor * factor * dim, dim)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim).repeat(1, factor * factor) / factor ** 2)
            self.proj.bias.zero_()

    def forward(self, x):
        return self.proj(space_to_depth_tokens(x, self.factor))


def keep_ratio_at(routing, step, seed):
    """Elastic capacity: the share of full-grid groups for one optimizer step.

    Uniform over the allowed keep_ratios, whose lowest allowed value falls linearly from 1 to
    min(keep_ratios) over warmup_steps. Seeded by (seed, step) only, so every rank and a resumed run
    draw the same share.
    """
    from .sdt_router import linear_schedule
    floor = linear_schedule(step, 1.0, min(routing['keep_ratios']), routing['warmup_steps'])
    allowed = [r for r in routing['keep_ratios'] if r >= floor - 1e-9] or [max(routing['keep_ratios'])]
    return random.Random(seed * 1_000_003 + step).choice(allowed)


class GroupRouter(nn.Module):
    """Scores each factor x factor token group from its concatenated tokens (SDTRouter layers, run in FP32).

    The final bias starts at 0 (keep probability 0.5, steepest sigmoid); with a fixed share of fine
    groups only the ranking matters.
    """
    def __init__(self, dim, factor):
        super().__init__()
        from .sdt_router import SDTRouter
        self.router = SDTRouter(factor * factor * dim, init_keep_prob=0.5)

    def forward(self, groups):
        with torch.autocast(device_type=groups.device.type, enabled=False):
            return self.router.net(groups.float()).squeeze(-1)


class TokenUnmerge(nn.Module):
    """Linear + DepthToSpace; starts by copying each coarse token to its sub-tokens."""
    def __init__(self, dim, factor):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(dim, factor * factor * dim)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim).repeat(factor * factor, 1))
            self.proj.bias.zero_()

    def forward(self, x):
        return depth_to_space_tokens(self.proj(x), self.factor)


def rope_tables(positions, half_head_dim):
    """cos/sin rows (row-major positions x positions grid, head_dim) of the 2D axial RoPE."""
    from einops import repeat
    from .pos_embed import broadcat
    freqs = 1. / (10000 ** (torch.arange(0, half_head_dim, 2, device=positions.device)[:(half_head_dim // 2)].float()
                            / half_head_dim))
    freqs = repeat(torch.einsum('..., f -> ... f', positions.float(), freqs), '... n -> ... (n r)', r=2)
    freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
    return freqs.cos().view(-1, freqs.shape[-1]), freqs.sin().view(-1, freqs.shape[-1])


def table_rope(cos, sin):
    """RoPE with explicit cos/sin rows: (tokens, head_dim), or (batch, 1, tokens, head_dim) per sample."""
    from .pos_embed import rotate_half

    def rope(t):
        return t * cos.to(t.dtype) + rotate_half(t) * sin.to(t.dtype)
    return rope


def rope_at(positions, half_head_dim):
    """Same frequencies as LightningDiT._get_dynamic_rope, at arbitrary grid positions."""
    return table_rope(*rope_tables(positions, half_head_dim))


def new_module_key(key):
    return key.startswith(NEW_MODULE_PREFIXES)


def feature_distill_loss(student, teacher):
    """Relative MSE per layer, averaged over layers.

    A coarse student state is compared with the factor x factor average of the
    teacher's full-grid state at the same pretrained block.
    """
    losses = []
    for s, t in zip(student, teacher, strict=True):
        s, t = s.float(), t.float()
        if s.shape[1] != t.shape[1]:
            factor = math.isqrt(t.shape[1] // s.shape[1])
            if factor * factor * s.shape[1] != t.shape[1]:
                raise ValueError(f'Cannot pool {t.shape[1]} teacher tokens to {s.shape[1]}')
            t = pool_tokens(t, factor)
        losses.append(F.mse_loss(s, t) / t.square().mean().clamp_min(1e-12))
    return torch.stack(losses).mean()
