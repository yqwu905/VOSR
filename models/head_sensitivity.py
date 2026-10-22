"""Evaluation-only, same-weight attention interventions and exact MAC accounting.

Locality is descriptive; output error is measured separately. These helpers never
claim that a feature-space score is an OCR result. Production inference uses
HeadSplitAttention, including its protected full-resolution heads.
"""
from contextlib import contextmanager
from copy import deepcopy
import math

import torch
import torch.nn.functional as F

from .attn_compression import (HeadSplitAttention, attn_compression_spec, pool_grid,
                               pooled_rope, upsample_grid)


def full_spec(depth, heads, factor=2, kernel=7):
    return dict(type='head_split', factor=factor, kernel=kernel, upsample='bilinear',
                blocks=list(range(depth)), local_heads=[[] for _ in range(depth)],
                dense_heads=[list(range(heads)) for _ in range(depth)])


def head_modes(spec, depth, heads):
    modes = [['dense'] * heads for _ in range(depth)]
    if spec is not None:
        for layer in spec.get('blocks', range(depth)):
            modes[layer] = ['pool'] * heads
            for h in spec['local_heads'][layer]:
                modes[layer][h] = 'conv'
            for h in spec.get('dense_heads', [[]] * depth)[layer]:
                modes[layer][h] = 'dense'
    return modes


def assign_heads(spec, assignments, depth, heads, factor=2, kernel=7):
    """Copy a layout and assign (layer, head, dense|pool|conv), retaining all other modes."""
    result = deepcopy(spec) if spec is not None else full_spec(depth, heads, factor, kernel)
    modes = head_modes(result, depth, heads)
    seen = set()
    for layer, head, mode in assignments:
        if (layer, head) in seen or not 0 <= layer < depth or not 0 <= head < heads:
            raise ValueError('Interventions need unique in-range (layer, head) pairs')
        if mode not in ('dense', 'pool', 'conv'):
            raise ValueError(f'Unknown head mode: {mode}')
        seen.add((layer, head))
        modes[layer][head] = mode
    result.update(blocks=list(range(depth)),
                  local_heads=[[h for h, m in enumerate(row) if m == 'conv'] for row in modes],
                  dense_heads=[[h for h, m in enumerate(row) if m == 'dense'] for row in modes])
    return attn_compression_spec(result, depth, heads)[0]


def reconfigure(model, spec, kernels=None):
    """Reuse backbone weights and migrate learned conv channels by head ID, never by packed position.

    New conv heads use the provided (per-layer, all-head) kernels or identity.
    This is also the explicit export migration path before short recovery tuning.
    """
    spec, _ = attn_compression_spec(spec, len(model.blocks), model.num_heads)
    if model.token_compression is not None or any(b.router is not None for b in model.blocks):
        raise ValueError('Sensitivity interventions require ordinary VOSR2 blocks, without token/MLP routing')
    for index in spec['blocks']:
        old = model.blocks[index].attn
        new = HeadSplitAttention(old, spec['local_heads'][index], spec['factor'], spec['kernel'],
                                 spec['upsample'], spec.get('dense_heads', [[]] * len(model.blocks))[index])
        new.to(device=old.qkv.weight.device)
        new.train(old.training)
        if new.local_conv is not None:
            if kernels is not None:
                new.set_local_kernels(kernels[index][new.local_heads])
            if isinstance(old, HeadSplitAttention) and old.local_conv is not None:
                if new.local_conv.kernel_size != old.local_conv.kernel_size:
                    raise ValueError('Changing kernel size needs an explicit migration; do not silently reset learned convs')
                with torch.no_grad():
                    for slot, head in enumerate(new.local_heads):
                        if head in old.local_heads:
                            source = old.local_heads.index(head)
                            d = new.head_dim
                            new.local_conv.weight[slot*d:(slot+1)*d].copy_(old.local_conv.weight[source*d:(source+1)*d])
        model.blocks[index].attn = new
    model.attn_compression = spec
    model.export_config['attn_compression'] = spec
    return model


@contextmanager
def intervention(model, assignments, factor=2, kernel=7, kernels=None):
    """Restore exact modules and metadata even if a forward fails. No checkpoint mutation."""
    original = [b.attn for b in model.blocks]
    spec, exported = model.attn_compression, deepcopy(model.export_config)
    try:
        selected = assign_heads(spec, assignments, len(original), model.num_heads, factor, kernel)
        # Only rebuild changed blocks; leave unrelated dense modules bit-for-bit identical.
        changed = sorted({a[0] for a in assignments})
        local = deepcopy(selected)
        local['blocks'] = changed
        for i in range(len(original)):
            if i not in changed:
                local['local_heads'][i] = []
                if 'dense_heads' in local:
                    local['dense_heads'][i] = []
        reconfigure(model, local, kernels)
        model.attn_compression = deepcopy(selected)
        model.attn_compression['blocks'] = sorted(set(changed) | set((spec or {}).get('blocks', [])))
        yield
    finally:
        for block, attn in zip(model.blocks, original):
            block.attn = attn
        model.attn_compression, model.export_config = spec, exported


def head_qkv(attn, x, heads, rope=None, factor=1):
    b, n, width = x.shape
    side = math.isqrt(n)
    if side * side != n or side % factor or not heads:
        raise ValueError('Expected nonempty heads and a square grid divisible by factor')
    x = pool_grid(x, side, factor)
    d = attn.head_dim
    channels = torch.tensor([h*d+c for h in heads for c in range(d)], device=x.device)
    rows = torch.cat((channels, channels + width, channels + 2*width))
    bias = attn.qkv.bias
    qkv = F.linear(x, attn.qkv.weight[rows], None if bias is None else bias[rows])
    q, k, v = qkv.reshape(b, -1, 3, len(heads), d).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    if rope is not None:
        q, k = rope(q), rope(k)
    return q, k, v


def attention_values(attn, x, heads, rope=None, factor=1, upsample='bilinear'):
    """Per-head outputs BEFORE W_o; pool-before-QK-norm and centroid RoPE match production."""
    q, k, v = head_qkv(attn, x, heads, rope, factor)
    values = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
    b, h, _, d = values.shape
    values = values.transpose(1, 2).reshape(b, -1, h*d)
    values = upsample_grid(values, math.isqrt(x.shape[1]), factor, upsample)
    return values.reshape(b, -1, h, d).transpose(1, 2)


def conv_values(attn, x, heads, kernels):
    """All-head calibrated kernels, or learned per-channel kernels from a release."""
    b, n, width = x.shape
    side, d = math.isqrt(n), attn.head_dim
    channels = torch.tensor([h*d+c for h in heads for c in range(d)], device=x.device)
    bias = attn.qkv.bias
    values = F.linear(x, attn.qkv.weight[channels + 2*width],
                      None if bias is None else bias[channels + 2*width])
    weight = kernels[heads].to(values).repeat_interleave(d, 0).unsqueeze(1)
    if isinstance(attn, HeadSplitAttention) and attn.local_conv is not None:
        for slot, head in enumerate(heads):
            if head in attn.local_heads:
                j = attn.local_heads.index(head)
                weight[slot*d:(slot+1)*d] = attn.local_conv.weight[j*d:(j+1)*d].to(weight)
    result = F.conv2d(values.transpose(1, 2).reshape(b, -1, side, side), weight,
                      padding=weight.shape[-1] // 2, groups=len(heads)*d)
    return result.flatten(2).transpose(1, 2).reshape(b, n, len(heads), d).transpose(1, 2)


def locality(attn, x, heads, rope=None, query_chunk=128):
    """Exact mean 3x3 mass, with bounded query memory; independent of output sensitivity."""
    q, k, _ = head_qkv(attn, x, heads, rope)
    n, side = x.shape[1], math.isqrt(x.shape[1])
    indices = torch.arange(n, device=x.device)
    total = torch.zeros(len(heads), device=x.device, dtype=torch.float32)
    for start in range(0, n, query_chunk):
        idx = indices[start:start+query_chunk]
        near = ((idx[:, None] // side - indices[None] // side).abs() <= 1) & \
               ((idx[:, None] % side - indices[None] % side).abs() <= 1)
        a = ((q[:, :, start:start+query_chunk].float() * attn.scale) @ k.float().transpose(-2, -1)).softmax(-1)
        total += (a * near).sum(-1).sum((0, 2)) / (x.shape[0] * n)
    return total


def weighted_mse(error, mask=None):
    """error (B,N,C), mask (B,N,1); None means full image, empty ROI -> None (not zero)."""
    error = error.float().square()
    if mask is None:
        return error.mean().item()
    weight = mask.float().sum()
    if weight <= 0:
        return None
    return ((error * mask).sum() / (weight * error.shape[-1])).item()


def measure_heads(attn, x, gate, full_rope, coarse_rope, kernels=None, masks=None,
                  factor=2, head_chunk=4, query_chunk=128, upsample='bilinear'):
    """One dense input distribution; no upstream compression. Yields independently interpretable scores."""
    if attn.training or attn.attn_type != 'full':
        raise ValueError('Profile requires eval mode and full dense attention')
    for start in range(0, attn.num_heads, head_chunk):
        heads = list(range(start, min(start+head_chunk, attn.num_heads)))
        dense = attention_values(attn, x, heads, full_rope)
        mass = locality(attn, x, heads, full_rope, query_chunk)
        candidates = {'pool': attention_values(attn, x, heads, coarse_rope, factor, upsample)}
        if kernels is not None:
            candidates['conv'] = conv_values(attn, x, heads, kernels)
        for mode, values in candidates.items():
            for j, head in enumerate(heads):
                delta = values[:, j].float() - dense[:, j].float()
                d = attn.head_dim
                # FP32 metric arithmetic, even if the actual forward uses BF16.
                with torch.autocast(x.device.type, enabled=False):
                    weight = attn.proj.weight[:, head*d:(head+1)*d].float()
                    projected = F.linear(delta, weight)
                    reference = F.linear(dense[:, j].float(), weight)
                residual = projected * gate.float()
                mse = weighted_mse(delta)
                row = dict(head=head, mode=mode, mass3x3=mass[j].item(), head_mse=mse,
                           head_nmse=mse / max(weighted_mse(dense[:, j]), 1e-20),
                           projected_mse=weighted_mse(projected), residual_mse=weighted_mse(residual),
                           residual_nmse=weighted_mse(residual) /
                               max(weighted_mse(reference * gate.float()), 1e-20))
                for name, mask in (masks or {}).items():
                    row[f'{name}_mse'] = weighted_mse(residual, mask)
                yield row


def attention_macs(spec, depth, heads, dim, grid):
    """MACs for QKV + QK/AV + W_o + local DWConv (no bias/norm/interpolation FLOPs).

    This is full-attention theoretical MACs, not latency. Multiply by two for FLOPs.
    """
    if dim % heads or grid <= 0:
        raise ValueError('Invalid head width or grid')
    if spec is not None:
        spec, _ = attn_compression_spec(spec, depth, heads)
    factor, kernel = (spec['factor'], spec['kernel']) if spec else (1, 1)
    if grid % factor:
        raise ValueError('Grid must be divisible by pooling factor')
    n, d, p = grid*grid, dim//heads, (grid//factor)**2
    cost = {'dense': 4*n*dim*d + 2*n*n*d, 'pool': 4*p*dim*d + 2*p*p*d,
            'conv': 2*n*dim*d + n*d*kernel*kernel}
    layers = [sum(cost[m] for m in row) for row in head_modes(spec, depth, heads)]
    dense = depth * heads * cost['dense']
    return dict(macs=sum(layers), dense_macs=dense, ratio=sum(layers)/dense,
                layer_macs=layers, per_head=cost)
