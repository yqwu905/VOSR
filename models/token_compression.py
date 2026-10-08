"""Parameter-free spatial token bottleneck, inspired by (not reproducing) GRACE.

Keep the original dense stream as a detail bypass. Process pooled tokens in a
contiguous block span, then lift ONLY the update: x + U(F(P(x)) - P(x)).
This preserves the input exactly when the compressed blocks are identities.
It does not guarantee preservation of text or reproduce GRACE's dual-latent VAE.
"""
import math

import torch
import torch.nn.functional as F


def validate_compression_config(config, depth):
    if config is None:
        return None
    defaults = dict(enabled=True, factor=2, start_block=0, end_block=depth)
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown compression options: {sorted(unknown)}')
    cfg = dict(defaults, **config)
    if not isinstance(cfg['enabled'], bool):
        raise ValueError('compression.enabled must be boolean')
    if any(type(cfg[k]) is not int for k in ('factor', 'start_block', 'end_block')):
        raise ValueError('factor and block indices must be integers')
    if cfg['factor'] not in (1, 2):
        raise ValueError('This experiment supports compression factor 1 or 2 only')
    if not 0 <= cfg['start_block'] < cfg['end_block'] <= depth:
        raise ValueError('Require 0 <= start_block < end_block <= depth (end exclusive)')
    return cfg


def pool_tokens(x, factor):
    b, n, d = x.shape
    side = math.isqrt(n)
    if side * side != n or side % factor:
        raise ValueError('Token grid must be square and divisible by compression factor')
    # Float32 reduction avoids accumulating bf16 rounding error in the bypass.
    grid = x.transpose(1, 2).reshape(b, d, side, side)
    return F.avg_pool2d(grid.float(), factor, factor).to(x.dtype).flatten(2).transpose(1, 2)


def expand_tokens(x, factor):
    b, n, d = x.shape
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError('Compressed token grid must be square')
    grid = x.reshape(b, side, side, d)
    return grid.repeat_interleave(factor, 1).repeat_interleave(factor, 2).reshape(b, n * factor**2, d)


def restore_tokens(coarse, entry, pooled_entry, factor):
    return entry + expand_tokens(coarse - pooled_entry, factor)


def pooled_rope(grid_side, factor, pretrained_side, head_dim, device):
    """RoPE at cell centers in the ORIGINAL grid's coordinate system.

    VOSR rescales dynamic grids to its training grid. Apply that same scaling
    after mapping coarse cells to their centers; never renumber them 0..N/4.
    Compute angles in fp32 and cast at application (same as upstream RoPE).
    """
    if grid_side % factor or head_dim % 4:
        raise ValueError('RoPE requires grid divisible by factor and head_dim divisible by 4')
    axis_dim = head_dim // 2
    freq = 1.0 / (10000 ** (torch.arange(0, axis_dim, 2, device=device).float() / axis_dim))
    positions = (torch.arange(grid_side // factor, device=device).float() * factor + (factor - 1) / 2)
    positions = positions * (pretrained_side / grid_side)
    angles = (positions[:, None] * freq[None, :]).repeat_interleave(2, -1)
    side = grid_side // factor
    angles = torch.cat((angles[:, None, :].expand(side, side, -1),
                        angles[None, :, :].expand(side, side, -1)), -1).reshape(-1, head_dim)
    cos, sin = angles.cos(), angles.sin()

    def apply(x):
        from .pos_embed import rotate_half
        return (x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)).contiguous()
    return apply


def feature_distillation_loss(student, teacher):
    if not student or student.keys() != teacher.keys():
        raise ValueError('Feature KD requires the same non-empty layer set')
    losses = []
    for layer in student:
        a, b = student[layer].float(), teacher[layer].detach().float()
        if a.shape != b.shape:
            raise ValueError(f'Feature shape mismatch at layer {layer}: {a.shape} vs {b.shape}')
        losses.append((1 - F.cosine_similarity(a, b, dim=-1, eps=1e-6)).mean())
    return torch.stack(losses).mean()


def detail_distillation_loss(prediction, target):
    """Latent finite-difference KD; an edge constraint, NOT an OCR loss."""
    difference = prediction.float() - target.detach().float()
    if min(difference.shape[-2:]) < 2:
        raise ValueError('Detail KD needs spatial dimensions >= 2')
    return (difference.diff(dim=-1).square().mean() +
            difference.diff(dim=-2).square().mean()) * 0.5
