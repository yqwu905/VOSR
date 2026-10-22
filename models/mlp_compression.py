"""SwiGLU neuron pruning of the block MLPs (``student.mlp_compression``) for VOSR2 ablations.

Every block MLP is ``w3(silu(x1) * x2)`` with ``x1, x2 = w12(h)`` and 4096 hidden neurons in VOSR2. Pruning keeps
``keep_ratio`` of a block's neurons: the kept neuron j keeps its rows j and j + 4096 of ``w12`` and its column j of
``w3``, so the block is an ordinary, narrower ``SwiGLUFFN`` (same state-dict names) and runs as dense matmuls. A block
with ``keep_ratio`` 0 loses its whole MLP: ``ConstantMLP`` adds one learned vector to every token (``mlp.bias``).

Which neurons stay (``score``):

- ``fluctuation`` (default): the largest Var(a_j) * |g2 * w3[:, j]|^2, the token-varying energy that neuron j
  writes into the residual stream (FLAP's fluctuation metric with VOSR2's MLP gate g2, constant at t=1, r=0);
- ``random``: a random subset per block (``seed``), the control for the ranking.

``compensation: mean`` (default) moves the pruned neurons' mean output ``w3[:, drop] @ mean(a_drop)`` into the w3
bias, so a pruned neuron that never varied changes nothing; ``none`` drops them outright. Both need the statistics
of ``tools/mlp_neuron_stats.py`` (``neuron_stats``), measured on the full-width network the student is initialized
from; ``random`` with ``none`` needs none.

The pruning happens when full-width weights are loaded into the student (``load_backbone_state`` calls
``prepare_pretrained_state``): a dense VOSR2 checkpoint or a head-split export, both 4096 wide. Checkpoints and
exports of the pruned student are already narrow and load unchanged. model.json stores the resolved structure, the
hidden width of every block (``hidden``), so exports rebuild without the statistics.
"""
from pathlib import Path

import torch
from torch import nn

from .swiglu_ffn import SwiGLUFFN

SCORES = ('fluctuation', 'random')
COMPENSATIONS = ('mean', 'none')
MLP_PRUNE_DEFAULTS = dict(score='fluctuation', compensation='mean', seed=0)
CONFIG_KEYS = ('type', 'keep_ratio', 'score', 'compensation', 'neuron_stats', 'seed', 'hidden')
STATS_KEYS = ('mean', 'variance', 'output_energy', 'w3_sq_norm')


class ConstantMLP(nn.Module):
    """A block MLP with every hidden neuron pruned: one learned vector (initially the w3 bias plus the neurons'
    mean output) added to every token."""
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # copy: under no_grad a view of the parameter itself would still report requires_grad (breaks autograd hooks)
        return self.bias.to(x.dtype, copy=True).expand(x.shape)


def pruned_mlp(dim, width):
    return SwiGLUFFN(dim, width) if width else ConstantMLP(dim)


def _ratio(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0 <= value <= 1


def mlp_compression_spec(config, depth, hidden):
    """Validate ``student.mlp_compression``.

    Returns ``(spec, init)``. ``spec`` is what model.json stores: ``{'type': 'neuron_prune', 'hidden': [...]}``, the
    hidden width of each block, ``round(keep_ratio * hidden)``; ``keep_ratio`` is one ratio for every block or one
    per block. ``init`` holds how full-width weights are pruned (score, compensation, neuron_stats, seed), or is None
    for a resolved spec (an export), which only rebuilds the structure.
    """
    if not isinstance(config, dict) or config.get('type') != 'neuron_prune':
        raise ValueError('mlp_compression must be a mapping with type: neuron_prune')
    unknown = set(config) - set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f'Unknown mlp_compression options: {sorted(unknown)}')
    if config.get('hidden') is not None:
        widths = config['hidden']
        if set(config) - {'type', 'hidden'}:
            raise ValueError('A resolved mlp_compression (hidden) takes no other options')
        if not isinstance(widths, (list, tuple)) or len(widths) != depth or any(
                isinstance(w, bool) or not isinstance(w, int) or not 0 <= w <= hidden for w in widths):
            raise ValueError(f'mlp_compression.hidden must hold one width in [0, {hidden}] per block ({depth})')
        return dict(type='neuron_prune', hidden=list(widths)), None
    ratios = config.get('keep_ratio')
    if _ratio(ratios):
        ratios = [ratios] * depth
    if not isinstance(ratios, (list, tuple)) or len(ratios) != depth or not all(map(_ratio, ratios)):
        raise ValueError(f'mlp_compression.keep_ratio must be in [0, 1], or one such ratio per block ({depth})')
    init = dict(MLP_PRUNE_DEFAULTS, **{k: config[k] for k in ('score', 'compensation', 'seed', 'neuron_stats')
                                        if config.get(k) is not None})
    if init['score'] not in SCORES:
        raise ValueError(f'mlp_compression.score must be one of {SCORES}')
    if init['compensation'] not in COMPENSATIONS:
        raise ValueError(f'mlp_compression.compensation must be one of {COMPENSATIONS}')
    if isinstance(init['seed'], bool) or not isinstance(init['seed'], int):
        raise ValueError('mlp_compression.seed must be an integer')
    widths = [int(round(r * hidden)) for r in ratios]
    if needs_stats(widths, hidden, init) and not init.get('neuron_stats'):
        raise ValueError('mlp_compression needs neuron_stats (tools/mlp_neuron_stats.py) for score: fluctuation '
                         'or compensation: mean')
    return dict(type='neuron_prune', hidden=widths), init


def needs_stats(widths, hidden, init):
    ranked = init['score'] == 'fluctuation' and any(0 < w < hidden for w in widths)
    return ranked or (init['compensation'] == 'mean' and any(w < hidden for w in widths))


def load_neuron_stats(path, depth, hidden):
    """``tools/mlp_neuron_stats.py`` output: per block and neuron mean, variance and output_energy (FP32), and
    w3_sq_norm per block (the source's |w3|_F^2)."""
    from safetensors.torch import load_file
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'mlp_compression.neuron_stats {path} not found; write it with '
                                f'tools/mlp_neuron_stats.py (docs/mlp_neuron_prune.md)')
    stats = load_file(str(path), device='cpu')
    if set(stats) != set(STATS_KEYS):
        raise ValueError(f'{path}: expected the tensors {STATS_KEYS}')
    for key in STATS_KEYS[:3]:
        if tuple(stats[key].shape) != (depth, hidden):
            raise ValueError(f'{path}: {key} has {tuple(stats[key].shape)} (blocks, neurons), the model '
                             f'{(depth, hidden)}')
    if tuple(stats['w3_sq_norm'].shape) != (depth,):
        raise ValueError(f'{path}: w3_sq_norm must have one value per block ({depth})')
    return {key: value.double() for key, value in stats.items()}


def neuron_order(index, hidden, init, stats):
    """Neurons of block ``index`` from the most to the least worth keeping."""
    if init['score'] == 'random':
        return torch.randperm(hidden, generator=torch.Generator().manual_seed(init['seed'] * 1_000_003 + index))
    score = stats['variance'][index].clamp_min(0) * stats['output_energy'][index]
    return torch.argsort(score, descending=True, stable=True)


def prune_mlp_state(state, spec, init, hidden):
    """Replace the full-width MLP weights of the pruned blocks in ``state`` by the student's narrow ones.

    Blocks whose weights are already narrow (checkpoints of the pruned student) pass through unchanged.
    """
    widths = spec['hidden']
    pending = [i for i, width in enumerate(widths)
               if width < hidden and f'blocks.{i}.mlp.w3.weight' in state
               and state[f'blocks.{i}.mlp.w3.weight'].shape[-1] == hidden]
    if not pending:
        return state
    if init is None:
        raise ValueError('This student was built from a resolved mlp_compression (hidden widths); prune '
                         'full-width weights with a keep_ratio config')
    stats = load_neuron_stats(init['neuron_stats'], len(widths), hidden) \
        if needs_stats([widths[i] for i in pending], hidden, init) else None
    state = dict(state)
    for index in pending:
        prefix = f'blocks.{index}.mlp.'
        w12, b12 = state.pop(prefix + 'w12.weight'), state.pop(prefix + 'w12.bias')
        w3, b3 = state.pop(prefix + 'w3.weight'), state.pop(prefix + 'w3.bias')
        if stats is not None:
            norm = w3.double().square().sum()
            if not torch.isclose(norm, stats['w3_sq_norm'][index], rtol=1e-4, atol=0):
                raise ValueError(f'mlp_compression.neuron_stats {init["neuron_stats"]} was measured on another '
                                 f'checkpoint (block {index}: |w3|^2 {norm.item():.6g} != '
                                 f'{stats["w3_sq_norm"][index].item():.6g}); rerun tools/mlp_neuron_stats.py')
        width = widths[index]
        keep = neuron_order(index, hidden, init, stats)[:width].sort().values
        drop = torch.ones(hidden, dtype=torch.bool)
        drop[keep] = False
        bias = b3.double()
        if init['compensation'] == 'mean':
            bias = bias + w3[:, drop].double() @ stats['mean'][index][drop]
        if not width:
            state[prefix + 'bias'] = bias.to(b3.dtype)
            continue
        rows = torch.cat((keep, keep + hidden))
        state[prefix + 'w12.weight'], state[prefix + 'w12.bias'] = w12[rows].clone(), b12[rows].clone()
        state[prefix + 'w3.weight'], state[prefix + 'w3.bias'] = w3[:, keep].clone(), bias.to(b3.dtype)
    return state

