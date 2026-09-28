"""Independent DyDiT SDT-style spatial MLP routing (not TDW).

Attention retains every token. Training is dense with hard straight-through
masks; no-grad evaluation can gather/scatter only the selected MLP tokens.
Reference: https://github.com/alibaba-damo-academy/DyDiT
"""
import math
import torch
from torch import nn


class SDTRouter(nn.Module):
    def __init__(self, dim, init_keep_prob=0.99, temperature=1.0, threshold=0.5):
        super().__init__()
        if dim <= 0 or not 0 < init_keep_prob < 1:
            raise ValueError('dim must be positive; init_keep_prob must be in (0, 1)')
        if temperature <= 0 or not 0 < threshold < 1:
            raise ValueError('temperature must be positive; threshold must be in (0, 1)')
        hidden = max(1, dim // 16)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.constant_(self.net[-1].bias, math.log(init_keep_prob / (1 - init_keep_prob)))
        self.temperature, self.threshold = temperature, threshold

    def forward(self, x):
        logits = self.net(x).float()
        # Expected stochastic keep probability; equals sigmoid(logits) at threshold=.5.
        cutoff = self.temperature * math.log(self.threshold / (1 - self.threshold))
        probability = (logits - cutoff).sigmoid()
        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            soft = ((logits + u.log() - torch.log1p(-u)) / self.temperature).sigmoid()
        else:
            soft = logits.sigmoid()
        hard = (soft > self.threshold).to(soft.dtype)
        mask = hard + (soft - soft.detach()) if self.training else hard
        return mask.to(x.dtype), probability


def routed_mlp(mlp, x, mask, sparse=False):
    if not sparse:
        return mlp(x) * mask
    if torch.is_grad_enabled():
        raise RuntimeError('Sparse MLP is inference-only; wrap evaluation in torch.no_grad()')
    flat = x.reshape(-1, x.shape[-1])
    indices = torch.nonzero(mask.reshape(-1) > 0, as_tuple=False).flatten()
    output = torch.zeros_like(flat)
    if indices.numel():
        selected = mlp(flat.index_select(0, indices).unsqueeze(0)).squeeze(0)
        output.index_copy_(0, indices, selected)
    return output.reshape_as(x)


def linear_schedule(step, start, end, warmup_steps):
    if step < 0 or warmup_steps < 0:
        raise ValueError('step and warmup_steps must be non-negative')
    return float(end if warmup_steps == 0 else
                 start + (end - start) * min(step / warmup_steps, 1.0))


def conditioning_key(key):
    return key.startswith(('layer_norm.', 'mlp_ca.')) or '.cross_attn.' in key


class AblationBlock(nn.Module):
    """Reuse upstream modules and state-dict names without reinitializing weights."""
    def __init__(self, block, router_config=None):
        super().__init__()
        for name in ('norm1', 'norm2', 'attn', 'mlp'):
            setattr(self, name, getattr(block, name))
        self.scale_shift_table = block.scale_shift_table
        self.z_dims = block.z_dims
        if self.z_dims is not None:
            self.cross_attn = block.cross_attn
        dim = self.scale_shift_table.shape[-1]
        self.router = SDTRouter(dim, **router_config) if router_config is not None else None

    def forward(self, x, c, z, rope, ca_scale, force_dense, sparse_eval):
        b = x.shape[0]
        s1, a1, g1, s2, a2, g2 = (self.scale_shift_table[None] + c.reshape(b, 6, -1)).chunk(6, 1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + a1) + s1, rope=rope)
        if self.z_dims is not None and ca_scale != 0:
            x = x + ca_scale * self.cross_attn(x, z)
        h = self.norm2(x) * (1 + a2) + s2
        if self.router is None:
            update = self.mlp(h)
            probability = x.new_ones((), dtype=torch.float32)
            fraction = probability
        else:
            # Match SDT: select from the residual stream before MLP normalization.
            mask, probabilities = self.router(x)
            probability = probabilities.mean()
            if force_dense:
                update = self.mlp(h) + probability.to(h.dtype) * 0
                fraction = probability.detach().new_ones(())
            else:
                update = routed_mlp(self.mlp, h, mask, sparse_eval and not self.training)
                fraction = mask.detach().float().mean()
        return x + g2 * update, torch.stack((probability, fraction))
