"""SwiGLU neuron pruning of the block MLPs (``student.mlp_compression``) for VOSR2 ablations.

Every block MLP is ``w3(silu(x1) * x2)`` with ``x1, x2 = w12(h)`` and 4096 hidden neurons in VOSR2. Pruning keeps
``keep_ratio`` of a block's neurons: the kept neuron j keeps its rows j and j + 4096 of ``w12`` and its column j of
``w3``, so the block is an ordinary, narrower ``SwiGLUFFN`` (same state-dict names) and runs as dense matmuls. A block
with ``keep_ratio`` 0 loses its whole MLP: ``ConstantMLP`` adds one learned vector to every token (``mlp.bias``).

Which neurons stay (``score``):

- ``fluctuation`` (default): the largest Var(a_j) * |g2 * w3[:, j]|^2, the token-varying energy that neuron j
  writes into the residual stream (FLAP's fluctuation metric with VOSR2's MLP gate g2, constant at t=1, r=0);
- ``random``: a random subset per block (``seed``), the control for the ranking;
- ``obs``: Optimal Brain Surgeon with the activation covariance, greedy (rounds of ``OBS_ROUND``): the neurons whose
  removal adds the least output energy once the remaining ones are refit, |g2 * w3_j|^2 / [C^-1]_jj, then the
  covariance inverse and the refit weights are downdated. ``fluctuation`` is this score with a diagonal covariance.
  It goes with ``compensation: lstsq``, the refit it assumes.

``compensation: mean`` (default) moves the pruned neurons' mean output ``w3[:, drop] @ mean(a_drop)`` into the w3
bias, so a pruned neuron that never varied changes nothing; ``none`` drops them outright. ``lstsq`` regresses the
pruned activations on the kept ones over the statistics' tokens, a_drop ~ B a_keep + c (ridge ``LSTSQ_RIDGE``), and
folds the fit into the kept neurons: w3_keep += w3_drop B, bias += w3_drop c. ``mean`` is the special case B = 0.
These need the statistics of ``tools/mlp_neuron_stats.py`` (``neuron_stats``), measured on the full-width network
the student is initialized from; ``lstsq`` also their activation covariance (``--covariance``); ``random`` with
``none`` needs none.

``fade_steps`` > 0 removes the pruned neurons gradually instead of at step 0: the student keeps full-width
``FadingSwiGLU`` MLPs whose pruned neurons are frozen and scaled from 1 (the source MLP) to 0 (the pruned MLP with
its compensation) by the trainer, as ``ca_fade_steps`` fades cross-attention. The final export is the narrow MLP.

The pruning happens when full-width weights are loaded into the student (``load_backbone_state`` calls
``prepare_pretrained_state``): a dense VOSR2 checkpoint or a head-split export, both 4096 wide. Checkpoints and
exports of the pruned student are already narrow (or, during a fade, in the fade layout) and load unchanged.
model.json stores the resolved structure, the hidden width of every block (``hidden``) and during a fade its scale
(``fade_scale``), so exports rebuild without the statistics.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .swiglu_ffn import SwiGLUFFN

SCORES = ('fluctuation', 'random', 'obs')
COMPENSATIONS = ('mean', 'none', 'lstsq')
MLP_PRUNE_DEFAULTS = dict(score='fluctuation', compensation='mean', seed=0)
CONFIG_KEYS = ('type', 'keep_ratio', 'score', 'compensation', 'neuron_stats', 'seed', 'fade_steps', 'hidden',
               'fade_scale')
STATS_KEYS = ('mean', 'variance', 'output_energy', 'w3_sq_norm')
COVARIANCE_KEYS = ('covariance', 'gate')  # optional (tools/mlp_neuron_stats.py --covariance): lstsq, obs
LSTSQ_RIDGE = 1e-3  # ridge of the lstsq fit and of score: obs, relative to the mean activation variance
OBS_ROUND = 64  # neurons removed per round of score: obs


class ConstantMLP(nn.Module):
    """A block MLP with every hidden neuron pruned: one learned vector (initially the w3 bias plus the neurons'
    mean output) added to every token."""
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # copy: under no_grad a view of the parameter itself would still report requires_grad (breaks autograd hooks)
        return self.bias.to(x.dtype, copy=True).expand(x.shape)


class FadingSwiGLU(SwiGLUFFN):
    """A pruned block MLP during ``fade_steps``: full width, the ``width`` kept neurons first, the compensation
    already folded into the kept neurons' w3 columns and bias (as in the narrow MLP).

    out = w3_keep a_keep + bias + scale * (w3_drop a_drop - fade_weight a_keep - fade_bias), with fade_weight and
    fade_bias the folded compensation (w3_drop B and w3_drop c): scale 1 is the source MLP exactly, scale 0 the
    pruned MLP. The pruned neurons' weights are frozen (no gradient; their input still gets one), so they do not
    grow to undo the fade; only a nonzero ``weight_decay`` still shrinks them (0 in the head_split_f2 recipe).
    ``narrow_state`` gives the pruned ``SwiGLUFFN`` weights.
    """
    def __init__(self, dim, hidden, width, scale=1.0):
        super().__init__(dim, hidden)
        if not 0 < width < hidden:
            raise ValueError('A fading MLP keeps between 1 and hidden - 1 neurons')
        self.width, self.scale = width, float(scale)
        self.register_buffer('fade_weight', torch.zeros(dim, width))
        self.register_buffer('fade_bias', torch.zeros(dim))

    def forward(self, x):
        hidden, width = self.w3.in_features, self.width
        rows = lambda t, start, end: torch.cat((t[start:end], t[hidden + start:hidden + end]))
        x1, x2 = F.linear(x, rows(self.w12.weight, 0, width), rows(self.w12.bias, 0, width)).chunk(2, -1)
        kept = F.silu(x1) * x2
        out = F.linear(kept, self.w3.weight[:, :width], self.w3.bias)
        if self.scale:
            d1, d2 = F.linear(x, rows(self.w12.weight, width, hidden).detach(),
                              rows(self.w12.bias, width, hidden).detach()).chunk(2, -1)
            pruned = F.linear(F.silu(d1) * d2, self.w3.weight[:, width:].detach())
            out = out + self.scale * (pruned - F.linear(kept, self.fade_weight, self.fade_bias))
        return out

    def narrow_state(self, prefix):
        hidden, width = self.w3.in_features, self.width
        rows = torch.cat((torch.arange(width), torch.arange(hidden, hidden + width)))
        return {prefix + 'w12.weight': self.w12.weight[rows], prefix + 'w12.bias': self.w12.bias[rows],
                prefix + 'w3.weight': self.w3.weight[:, :width], prefix + 'w3.bias': self.w3.bias}


def pruned_mlp(dim, width, hidden=None, fade_scale=None):
    """The student's MLP of one pruned block; ``fade_scale`` (not None) builds the full-width fading MLP."""
    if fade_scale is not None:
        return FadingSwiGLU(dim, hidden, width, fade_scale)
    return SwiGLUFFN(dim, width) if width else ConstantMLP(dim)


def _ratio(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0 <= value <= 1


def mlp_compression_spec(config, depth, hidden):
    """Validate ``student.mlp_compression``.

    Returns ``(spec, init)``. ``spec`` is what model.json stores: ``{'type': 'neuron_prune', 'hidden': [...]}``, the
    hidden width of each block, ``round(keep_ratio * hidden)``; ``keep_ratio`` is one ratio for every block or one
    per block. With ``fade_steps`` the spec also holds ``fade_scale`` (1 at the start of the fade): the pruned blocks
    are full-width ``FadingSwiGLU``. ``init`` holds how full-width weights are pruned (score, compensation,
    neuron_stats, seed, fade_steps), or is None for a resolved spec (an export), which only rebuilds the structure.
    """
    if not isinstance(config, dict) or config.get('type') != 'neuron_prune':
        raise ValueError('mlp_compression must be a mapping with type: neuron_prune')
    unknown = set(config) - set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f'Unknown mlp_compression options: {sorted(unknown)}')
    if config.get('hidden') is not None:
        widths = config['hidden']
        if set(config) - {'type', 'hidden', 'fade_scale'}:
            raise ValueError('A resolved mlp_compression (hidden, fade_scale) takes no other options')
        if not isinstance(widths, (list, tuple)) or len(widths) != depth or any(
                isinstance(w, bool) or not isinstance(w, int) or not 0 <= w <= hidden for w in widths):
            raise ValueError(f'mlp_compression.hidden must hold one width in [0, {hidden}] per block ({depth})')
        spec = dict(type='neuron_prune', hidden=list(widths))
        if config.get('fade_scale') is not None:
            if not _ratio(config['fade_scale']) or 0 in widths:
                raise ValueError('mlp_compression.fade_scale must be in [0, 1], with no block pruned whole')
            spec['fade_scale'] = float(config['fade_scale'])
        return spec, None
    if 'fade_scale' in config:
        raise ValueError('mlp_compression.fade_scale belongs to a resolved spec (with hidden); set fade_steps')
    ratios = config.get('keep_ratio')
    if _ratio(ratios):
        ratios = [ratios] * depth
    if not isinstance(ratios, (list, tuple)) or len(ratios) != depth or not all(map(_ratio, ratios)):
        raise ValueError(f'mlp_compression.keep_ratio must be in [0, 1], or one such ratio per block ({depth})')
    init = dict(MLP_PRUNE_DEFAULTS, **{k: config[k] for k in ('score', 'compensation', 'seed', 'neuron_stats',
                                                              'fade_steps') if config.get(k) is not None})
    fade_steps = init.get('fade_steps', 0)
    if init['score'] not in SCORES:
        raise ValueError(f'mlp_compression.score must be one of {SCORES}')
    if init['compensation'] not in COMPENSATIONS:
        raise ValueError(f'mlp_compression.compensation must be one of {COMPENSATIONS}')
    if init['score'] == 'obs' and init['compensation'] != 'lstsq':
        raise ValueError('mlp_compression.score: obs refits the kept neurons; use it with compensation: lstsq')
    if isinstance(init['seed'], bool) or not isinstance(init['seed'], int):
        raise ValueError('mlp_compression.seed must be an integer')
    if isinstance(fade_steps, bool) or not isinstance(fade_steps, int) or fade_steps < 0:
        raise ValueError('mlp_compression.fade_steps must be a non-negative integer')
    widths = [int(round(r * hidden)) for r in ratios]
    if fade_steps and 0 in widths:
        raise ValueError('mlp_compression.fade_steps fades neurons out of a narrower MLP; keep_ratio 0 (whole MLPs) '
                         'is not supported with it')
    if needs_stats(widths, hidden, init) and not init.get('neuron_stats'):
        raise ValueError('mlp_compression needs neuron_stats (tools/mlp_neuron_stats.py) for score: fluctuation '
                         '/ obs or compensation: mean / lstsq')
    spec = dict(type='neuron_prune', hidden=widths)
    if fade_steps:
        spec['fade_scale'] = 1.0
    return spec, init


def needs_stats(widths, hidden, init):
    ranked = init['score'] != 'random' and any(0 < w < hidden for w in widths)
    return ranked or (init['compensation'] != 'none' and any(w < hidden for w in widths))


def load_neuron_stats(path, depth, hidden, covariance=False):
    """``tools/mlp_neuron_stats.py`` output: per block and neuron mean, variance and output_energy (FP32), and
    w3_sq_norm per block (the source's |w3|_F^2); with ``covariance`` also the per-block activation covariance
    (blocks, neurons, neurons; kept FP32 here, converted block by block) and MLP gate g2 (blocks, dim)."""
    from safetensors.torch import load_file
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'mlp_compression.neuron_stats {path} not found; write it with '
                                f'tools/mlp_neuron_stats.py (docs/mlp_neuron_prune.md)')
    stats = load_file(str(path), device='cpu')
    if not set(STATS_KEYS) <= set(stats) <= set(STATS_KEYS + COVARIANCE_KEYS):
        raise ValueError(f'{path}: expected the tensors {STATS_KEYS} (and optionally {COVARIANCE_KEYS})')
    for key in STATS_KEYS[:3]:
        if tuple(stats[key].shape) != (depth, hidden):
            raise ValueError(f'{path}: {key} has {tuple(stats[key].shape)} (blocks, neurons), the model '
                             f'{(depth, hidden)}')
    if tuple(stats['w3_sq_norm'].shape) != (depth,):
        raise ValueError(f'{path}: w3_sq_norm must have one value per block ({depth})')
    if covariance and not set(COVARIANCE_KEYS) <= set(stats):
        raise ValueError(f'{path} has no activation covariance, which compensation: lstsq and score: obs need; '
                         'rerun tools/mlp_neuron_stats.py with --covariance')
    if 'covariance' in stats and (tuple(stats['covariance'].shape) != (depth, hidden, hidden)
                                  or stats.get('gate', stats['covariance']).shape[0] != depth):
        raise ValueError(f'{path}: covariance must have the shape {(depth, hidden, hidden)}, gate one row per block')
    return {key: value if key == 'covariance' else value.double() for key, value in stats.items()}


def ridged(cov):
    return cov + LSTSQ_RIDGE * cov.diagonal().mean().clamp_min(1e-12) * torch.eye(len(cov), dtype=cov.dtype)


def neuron_order(index, hidden, init, stats):
    """Neurons of block ``index`` from the most to the least worth keeping."""
    if init['score'] == 'random':
        return torch.randperm(hidden, generator=torch.Generator().manual_seed(init['seed'] * 1_000_003 + index))
    score = stats['variance'][index].clamp_min(0) * stats['output_energy'][index]
    return torch.argsort(score, descending=True, stable=True)


def obs_keep(index, width, w3, stats):
    """``score: obs``: the ``width`` neurons of block ``index`` left by greedy OBS elimination (sorted)."""
    inverse = torch.linalg.inv(ridged(stats['covariance'][index].double()))
    weight = stats['gate'][index][:, None] * w3.double()  # what each neuron writes into the residual stream
    alive = torch.arange(w3.shape[1])
    while len(alive) > width:
        drop = torch.argsort(weight.square().sum(0) / inverse.diagonal(), stable=True)[:min(OBS_ROUND,
                                                                                            len(alive) - width)]
        keep = torch.ones(len(alive), dtype=torch.bool)
        keep[drop] = False
        pivot = torch.linalg.inv(inverse[drop][:, drop])
        cross = inverse[drop][:, keep]
        weight = weight[:, keep] - weight[:, drop] @ pivot @ cross  # the remaining neurons refit
        inverse = inverse[keep][:, keep] - cross.T @ pivot @ cross  # = the inverse of their covariance
        alive = alive[keep]
    return alive


def compensation(w3, keep, drop, index, init, stats):
    """The pruned neurons' compensation folded into the kept ones (FP64): w3_keep += weight (None: unchanged),
    bias += shift. ``lstsq`` regresses a_drop on a_keep (a_drop ~ B a_keep + c): weight = w3_drop B and
    shift = w3_drop c; ``mean`` is B = 0, c = mean(a_drop)."""
    if init['compensation'] == 'none':
        return None, w3.new_zeros(w3.shape[0], dtype=torch.float64)
    w3_drop, mean = w3[:, drop].double(), stats['mean'][index]
    if init['compensation'] == 'mean' or not len(keep):
        return None, w3_drop @ mean[drop]
    cov = stats['covariance'][index].double()
    fit = torch.linalg.solve(ridged(cov[keep][:, keep]), cov[keep][:, drop]).T
    return w3_drop @ fit, w3_drop @ (mean[drop] - fit @ mean[keep])


def prune_mlp_state(state, spec, init, hidden):
    """Replace the full-width MLP weights of the pruned blocks in ``state`` by the student's narrow ones, or with
    ``fade_scale`` in the spec by the fade layout of ``FadingSwiGLU`` (kept neurons first, compensation folded in
    and stored as fade_weight / fade_bias).

    Blocks whose weights are already narrow or in the fade layout (checkpoints of the pruned student) pass through
    unchanged.
    """
    widths, fade = spec['hidden'], spec.get('fade_scale') is not None
    pending = [i for i, width in enumerate(widths)
               if width < hidden and f'blocks.{i}.mlp.w3.weight' in state
               and state[f'blocks.{i}.mlp.w3.weight'].shape[-1] == hidden
               and f'blocks.{i}.mlp.fade_weight' not in state]
    if not pending:
        return state
    if init is None:
        raise ValueError('This student was built from a resolved mlp_compression (hidden widths); prune '
                         'full-width weights with a keep_ratio config')
    stats = load_neuron_stats(init['neuron_stats'], len(widths), hidden, init['compensation'] == 'lstsq') \
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
        keep = obs_keep(index, width, w3, stats) if init['score'] == 'obs' and width else \
            neuron_order(index, hidden, init, stats)[:width].sort().values
        drop = torch.ones(hidden, dtype=torch.bool)
        drop[keep] = False
        weight, shift = compensation(w3, keep, drop, index, init, stats)
        bias = b3.double() + shift
        if not width:
            state[prefix + 'bias'] = bias.to(b3.dtype)
            continue
        dtype = w3.dtype if weight is None else torch.promote_types(w3.dtype, torch.float32)
        kept_w3 = w3[:, keep].to(dtype) if weight is None else (w3[:, keep].double() + weight).to(dtype)
        if fade:
            order = torch.cat((keep, drop.nonzero().flatten()))
            kept_w3 = torch.cat((kept_w3, w3[:, drop].to(dtype)), 1)
            state[prefix + 'fade_weight'] = (torch.zeros(w3.shape[0], width) if weight is None
                                             else weight.float())
            state[prefix + 'fade_bias'] = shift.float()
        else:
            order = keep
        rows = torch.cat((order, order + hidden))
        state[prefix + 'w12.weight'], state[prefix + 'w12.bias'] = w12[rows].clone(), b12[rows].clone()
        state[prefix + 'w3.weight'], state[prefix + 'w3.bias'] = kept_w3.clone(), bias.to(b3.dtype)
    return state
