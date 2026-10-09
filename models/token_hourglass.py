"""Static U-shaped hourglass token merging for VOSR2 ablations.

The pretrained blocks keep their weights; only the token grid they run on
changes. The first ``fine_in`` and last ``fine_out`` active blocks see the full
p2 token grid. The blocks between them run on factor x factor merged tokens
(SpaceToDepth + Linear, initialized as the group average). After the coarse
span, Linear + DepthToSpace (initialized as a copy) returns to the full grid.
With ``bypass`` the coarse span only adds its update, ``h_fine + Up(y_out - y_in)``,
so each fine token keeps its own deviation from the group mean. All shapes are
static: no top-k, gather or scatter.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

HOURGLASS_DEFAULTS = dict(factor=2, fine_in=1, fine_out=1, drop_blocks=(), bypass=True,
                          rope='centroid', fine_cross_attention=True, cond_pool=1)
NEW_MODULE_PREFIXES = ('token_merge.', 'token_unmerge.')


def hourglass_spec(config, depth):
    """Validate a ``student.token_compression`` mapping and fill in its defaults."""
    if not isinstance(config, dict) or config.get('type') != 'hourglass':
        raise ValueError('token_compression must be a mapping with type: hourglass')
    unknown = set(config) - set(HOURGLASS_DEFAULTS) - {'type'}
    if unknown:
        raise ValueError(f'Unknown token_compression options: {sorted(unknown)}')
    spec = dict(HOURGLASS_DEFAULTS, **{k: v for k, v in config.items() if k != 'type'})
    integers = {'factor': 2, 'fine_in': 0, 'fine_out': 0, 'cond_pool': 1}
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
    return dict(type='hourglass', **dict(spec, drop_blocks=sorted(drop)))


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


def rope_at(positions, half_head_dim):
    """Same frequencies as LightningDiT._get_dynamic_rope, at arbitrary grid positions."""
    from einops import repeat
    from .pos_embed import broadcat, rotate_half
    freqs = 1. / (10000 ** (torch.arange(0, half_head_dim, 2, device=positions.device)[:(half_head_dim // 2)].float()
                            / half_head_dim))
    freqs = repeat(torch.einsum('..., f -> ... f', positions.float(), freqs), '... n -> ... (n r)', r=2)
    freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
    cos, sin = freqs.cos().view(-1, freqs.shape[-1]), freqs.sin().view(-1, freqs.shape[-1])

    def rope(t):
        return t * cos.to(t.dtype) + rotate_half(t) * sin.to(t.dtype)
    return rope


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
