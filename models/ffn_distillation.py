"""Opt-in text-aware distillation and optimizer groups for physical FFN pruning.

These helpers have no VAE/OCR dependencies and work on CPU for diagnostics.
All spatial weights are area pooled from native image masks. Internal targets
are residual-width tensors, never a forced match of 4096/2048 hidden channels.
"""
import math
import re

import torch
import torch.nn.functional as F


def text_distillation_spec(config):
    tc = config['training']
    defaults = dict(kd_small_text_weight=0.0, internal_text_weight=0.0,
                    internal_small_text_weight=0.0, mlp_update_weight=0.0,
                    distill_energy_floor=1e-6, ffn_lr_multiplier=1.0,
                    roi_charbonnier_weight=0.0, roi_edge_weight=0.0,
                    roi_ocr_weight=0.0, pixel_loss_warmup_steps=0,
                    internal_kd_end_ratio=1.0)
    spec = {key: tc.get(key, default) for key, default in defaults.items()}
    for key, value in spec.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'training.{key} must be finite and numeric')
        if value < 0:
            raise ValueError(f'training.{key} must be nonnegative')
    if spec['ffn_lr_multiplier'] <= 0 or spec['distill_energy_floor'] <= 0:
        raise ValueError('ffn_lr_multiplier and distill_energy_floor must be positive')
    if not isinstance(spec['pixel_loss_warmup_steps'], int):
        raise ValueError('pixel_loss_warmup_steps must be an integer')
    if not 0 <= spec['internal_kd_end_ratio'] <= 1:
        raise ValueError('internal_kd_end_ratio must be in [0, 1]')
    layers = tc.get('mlp_update_layers') or []
    if layers == 'all':
        layers = list(range(int(config['model']['depth'])))
    if not isinstance(layers, list) or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in layers):
        raise ValueError('mlp_update_layers must be a list of nonnegative integers or all')
    if len(layers) != len(set(layers)) or (spec['mlp_update_weight'] and not layers):
        raise ValueError('mlp_update_weight needs unique mlp_update_layers')
    spec['mlp_update_layers'] = layers if spec['mlp_update_weight'] else []
    spec['roi_enabled'] = any(spec[key] for key in
                              ('roi_charbonnier_weight', 'roi_edge_weight', 'roi_ocr_weight'))
    spec['masks_enabled'] = bool(spec['kd_small_text_weight'] or spec['internal_text_weight']
                                 or spec['internal_small_text_weight'] or spec['roi_enabled'])
    if spec['kd_small_text_weight'] and float(tc.get('kd_weight', 1)) <= 0:
        raise ValueError('kd_small_text_weight requires kd_weight > 0')
    if (spec['internal_text_weight'] or spec['internal_small_text_weight']) and not (
            spec['mlp_update_weight'] or float(tc.get('feature_distill_weight', 0))):
        raise ValueError('Internal text weights require feature or MLP update distillation')
    if spec['mlp_update_weight'] and config['student'].get('token_compression') is not None:
        raise ValueError('MLP update KD requires an unchanged residual token grid')
    if config['training'].get('objective', 'one_step') != 'one_step' and (
            spec['mlp_update_weight'] or spec['kd_small_text_weight']):
        raise ValueError('Latent FFN distillation is only supported for objective: one_step')
    return spec


def validate_text_data(data):
    """At least one annotated source is needed; unannotated natural images are valid.

    Per-source text_boxes overrides a global default, including explicit null.
    Dataset construction validates actual paths and annotation geometry.
    """
    sources = data.get('sources')
    annotated = (any(source.get('text_boxes', data.get('text_boxes')) for source in sources)
                 if isinstance(sources, list) else bool(data.get('text_boxes')))
    if data.get('dataset_type') != 'paired' or not annotated:
        raise ValueError('Text losses need paired data with text_boxes at the top level or in data.sources')


def spatial_weights(batch, size, device, text_weight=0.0, small_weight=0.0, trust=False):
    weights = None
    for key, scale in (('text_mask', text_weight), ('small_text_mask', small_weight)):
        if scale:
            if key not in batch:
                raise ValueError(f'Text distillation needs batch[{key!r}]')
            mask = batch[key].to(device=device, dtype=torch.float32)
            pooled = F.adaptive_avg_pool2d(mask, size).clamp(0, 1)
            weights = 1 + scale * pooled if weights is None else weights + scale * pooled
    if trust and 'kd_trust_mask' in batch:
        mask = F.adaptive_avg_pool2d(batch['kd_trust_mask'].to(device=device, dtype=torch.float32), size)
        weights = mask if weights is None else weights * mask
    return weights


def weighted_mse(prediction, target, weights=None):
    error = (prediction.float() - target.detach().float()).square()
    if weights is None:
        return error.mean()
    if weights.shape[0] != error.shape[0] or weights.shape[-2:] != error.shape[-2:]:
        raise ValueError('Spatial loss weight shape does not match prediction')
    return (error * weights).mean() / weights.mean().clamp_min(1e-12)


def weighted_feature_loss(student, teacher, batch, text_weight=0.0, small_weight=0.0,
                          energy_floor=1e-6, trust=True):
    """Relative weighted MSE per layer, averaged equally across selected layers."""
    from models.token_hourglass import pool_tokens
    if not student or len(student) != len(teacher):
        raise ValueError('Internal KD needs equal nonempty lists of student and teacher tensors')
    losses = []
    for value, target in zip(student, teacher, strict=True):
        value, target = value.float(), target.detach().float()
        if value.shape[1] != target.shape[1]:
            factor = math.isqrt(target.shape[1] // value.shape[1])
            if factor * factor * value.shape[1] != target.shape[1]:
                raise ValueError('Teacher/student token grids cannot be aligned by integer pooling')
            target = pool_tokens(target, factor)
        if value.shape != target.shape:
            raise ValueError('Internal KD requires equal residual dimensions and batch sizes')
        side = math.isqrt(value.shape[1])
        if side * side != value.shape[1]:
            raise ValueError('Text-aware feature KD requires a square residual grid')
        weight = spatial_weights(batch, (side, side), value.device, text_weight, small_weight, trust)
        error, energy = (value - target).square().mean(-1), target.square().mean(-1)
        if weight is not None:
            weight = weight.flatten(1)
            mass = weight.sum().clamp_min(1e-12)
            error, energy = (error * weight).sum() / mass, (energy * weight).sum() / mass
        else:
            error, energy = error.mean(), energy.mean()
        losses.append(error / energy.clamp_min(energy_floor))
    return torch.stack(losses).mean()


def ffn_optimizer_groups(model, training):
    """Disjoint named groups; FFN multiplier is relative to the base learning_rate."""
    from models.token_hourglass import new_module_key
    lr = float(training['learning_rate'])
    scales = dict(base=1.0, router=float(training.get('router_lr_multiplier', 10)),
                  token_merge=float(training.get('token_merge_lr_multiplier', 1)),
                  ffn=float(training.get('ffn_lr_multiplier', 1)))
    if not math.isfinite(lr) or lr <= 0 or any(not math.isfinite(v) or v <= 0 for v in scales.values()):
        raise ValueError('Learning rate and group multipliers must be finite and positive')
    grouped = {name: [] for name in scales}
    fading = any(getattr(block, 'mlp', None).__class__.__name__ == 'FadingSwiGLU'
                 for block in getattr(model, 'blocks', ()))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = ('router' if '.router.' in name else
                 'token_merge' if scales['token_merge'] != 1 and new_module_key(name) else
                 'ffn' if (scales['ffn'] != 1 or fading) and re.match(r'^blocks\.\d+\.mlp\.', name) else 'base')
        grouped[group].append(parameter)
    if not any(grouped.values()):
        raise ValueError('No trainable parameters')
    groups = [dict(params=parameters, lr=lr * scales[name], lr_scale=scales[name], name=name)
              for name, parameters in grouped.items() if parameters]
    # AdamW decay applies to the entire packed tensor, including detached dropped
    # rows. Isolate fading FFNs and disable their decay even for legacy configs.
    for group in groups:
        if fading and group['name'] == 'ffn':
            group['weight_decay'] = 0.0
    return groups


def loss_schedule(step, total_steps, spec):
    progress = min(max(step, 0) / max(total_steps - 1, 1), 1.0)
    internal = 1 + progress * (spec['internal_kd_end_ratio'] - 1)
    warmup = spec['pixel_loss_warmup_steps']
    pixel = min((step + 1) / warmup, 1.0) if warmup else 1.0
    return internal, pixel


def optimizer_state_for_resume(state, optimizer, model, training):
    """Migrate the historical unnamed SFT groups without losing Adam moments.

    Old fade runs packed frozen dropped rows in the decayed base group. The new
    FFN group disables decay for those rows. Recover parameter identity using
    the exact legacy grouping order; never match tensors by shape alone.
    Named states (including all new runs) are returned unchanged.
    """
    from models.token_hourglass import new_module_key
    old_groups = state['param_groups']
    if not old_groups or any('name' in group for group in old_groups):
        return state
    legacy = {'base': [], 'router': [], 'token_merge': []}
    merge_scale = float(training.get('token_merge_lr_multiplier', 1))
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            label = ('router' if '.router.' in name else
                     'token_merge' if merge_scale != 1 and new_module_key(name) else 'base')
            legacy[label].append(parameter)
    # The historical trainer always created base, including an empty base group.
    buckets = [legacy['base']] + [legacy[key] for key in ('router', 'token_merge') if legacy[key]]
    if len(buckets) != len(old_groups) or any(len(params) != len(group['params'])
                                             for params, group in zip(buckets, old_groups)):
        raise ValueError('Legacy optimizer groups do not match this model; use --init-from for a fresh run')
    by_parameter = {}
    for params, group in zip(buckets, old_groups):
        for parameter, old_id in zip(params, group['params']):
            moment = state['state'].get(old_id)
            if moment is not None:
                for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    if key in moment and moment[key].shape != parameter.shape:
                        raise ValueError('Legacy optimizer tensor shape differs; use --init-from for a fresh run')
            by_parameter[id(parameter)] = (group, moment)
    migrated, next_id = dict(state, state={}, param_groups=[]), 0
    for current in optimizer.param_groups:
        params = current['params']
        if not params or any(id(parameter) not in by_parameter for parameter in params):
            raise ValueError('Cannot map legacy optimizer parameters; use --init-from for a fresh run')
        origin = by_parameter[id(params[0])][0]
        if any(by_parameter[id(parameter)][0] is not origin for parameter in params):
            raise ValueError('A new optimizer group combines legacy groups with different settings')
        group = dict(origin)
        # Current groups carry the intentional fade no-decay correction and names.
        group.update({key: current[key] for key in ('name', 'lr_scale', 'weight_decay') if key in current})
        group['params'] = list(range(next_id, next_id + len(params)))
        for new_id, parameter in zip(group['params'], params):
            moment = by_parameter[id(parameter)][1]
            if moment is not None:
                migrated['state'][new_id] = moment
        migrated['param_groups'].append(group)
        next_id += len(params)
    if next_id != len(by_parameter):
        raise ValueError('Legacy optimizer contains unmatched parameters; use --init-from for a fresh run')
    return migrated
