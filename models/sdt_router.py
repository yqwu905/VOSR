"""Independent DyDiT SDT-style spatial MLP routing (not TDW).

Attention retains every token. Training is dense with hard straight-through
masks; no-grad evaluation can gather/scatter only the selected MLP tokens.
New experiments use capacity_topk in both training and evaluation. The legacy
Gumbel/threshold mode remains the default here to preserve old model.json exports.
Reference: https://github.com/alibaba-damo-academy/DyDiT
"""
import math
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_reduce


class SDTRouter(nn.Module):
    def __init__(self, dim, init_keep_prob=0.99, temperature=1.0, threshold=0.5,
                 routing_mode='gumbel'):
        super().__init__()
        if dim <= 0 or not 0 < init_keep_prob < 1:
            raise ValueError('dim must be positive; init_keep_prob must be in (0, 1)')
        if temperature <= 0 or not 0 < threshold < 1:
            raise ValueError('temperature must be positive; threshold must be in (0, 1)')
        if routing_mode not in ('gumbel', 'capacity_topk'):
            raise ValueError('routing_mode must be gumbel or capacity_topk')
        if routing_mode == 'capacity_topk' and (temperature != 1.0 or threshold != 0.5):
            raise ValueError('capacity_topk requires temperature=1 and threshold=0.5; '
                             'capacity is learned from mean sigmoid(logits)')
        hidden = max(1, dim // 16)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.constant_(self.net[-1].bias, math.log(init_keep_prob / (1 - init_keep_prob)))
        self.temperature, self.threshold = temperature, threshold
        self.routing_mode = routing_mode

    def forward(self, x):
        # Keep the small router in FP32, including the Linear outputs. Casting
        # BF16 logits afterwards cannot recover scores rounded into ties near
        # the initial all-keep bias; stable top-k would then prefer token order.
        # Router parameters stay FP32 in both the trainer and export loader.
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.net(x.float())
        if self.routing_mode == 'capacity_topk':
            probability = logits.sigmoid()
            # Per-image/per-layer capacity, NOT the same fixed k in every layer.
            # No host .item() or batch-wide competition for token slots.
            with torch.no_grad():
                scores = logits.squeeze(-1)
                tokens = scores.shape[1]
                keep = torch.floor(probability.squeeze(-1).sum(1, keepdim=True) + 0.5).long()
                order = torch.argsort(scores, dim=1, descending=True, stable=True)
                positions = torch.arange(tokens, device=x.device).expand_as(order)
                ranks = torch.empty_like(order).scatter_(1, order, positions)
                hard = (ranks < keep).unsqueeze(-1).to(probability.dtype)
            # Rank/count decisions are discrete. This explicit surrogate gives
            # task gradients to the router; top-k indices alone would not do so.
            mask = hard + (probability - probability.detach()) if self.training else hard
            return mask.to(x.dtype), probability
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


def mlp_budget_loss(layer_keep, target_keep, scope='global'):
    """Budget over equal-cost MLPs in this backbone, not whole-pipeline FLOPs.

    Global reduction happens across layers and DDP ranks BEFORE squaring, so
    images can use different capacities even at local batch size one. All ranks
    must call this once per micro-batch with equal local sample/token counts,
    as enforced by the trainer's fixed crops and drop_last loader. Accumulation
    still averages separate micro-batch losses. 'layer' keeps the legacy local
    per-layer objective.
    """
    if scope not in ('global', 'layer'):
        raise ValueError('budget_scope must be global or layer')
    if not 0 <= target_keep <= 1:
        raise ValueError('target_keep must be in [0, 1]')
    if scope == 'layer':
        return (layer_keep.float() - target_keep).square().mean()
    mean_keep = layer_keep.float().mean()
    if dist.is_initialized() and dist.get_world_size() > 1:
        # Autograd also reduces in backward. With DDP's gradient averaging this
        # matches a single loss over the concatenated global micro-batch; no
        # additional world-size factor belongs in the loss or budget weight.
        mean_keep = all_reduce(mean_keep, op=dist.ReduceOp.SUM) / dist.get_world_size()
    return (mean_keep - target_keep).square()


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
        # Autocast may run the MLP in lower precision than the residual input.
        output.index_copy_(0, indices, selected.to(output.dtype))
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
