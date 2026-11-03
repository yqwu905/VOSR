"""Route B: noise-time pixel flow, clean-image prediction and decreasing-time Euler sampling.

t=0 is clean and t=1 is noise. Training's denominator floor is a loss weight only;
the ODE sampler uses the actual positive t, with a direct x-pred at its last step.
No VAE, latent/RGB velocity KD or endpoint distillation is involved.
"""
import math
import re

import torch


def pixel_flow_spec(config):
    if not isinstance(config, dict):
        raise ValueError('pdm must be a mapping')
    defaults = dict(noise_scale=1.0, timestep_mean=0.0, timestep_std=1.0,
                    loss_t_floor=0.05, sampling_steps=[4, 8, 16], inference_steps=8)
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown pdm keys: {sorted(unknown)}')
    spec = dict(defaults, **config)
    for key in ('noise_scale', 'timestep_mean', 'timestep_std', 'loss_t_floor'):
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f'pdm.{key} must be finite')
        spec[key] = float(value)
    if spec['noise_scale'] <= 0 or spec['timestep_std'] <= 0 or not 0 < spec['loss_t_floor'] <= 1:
        raise ValueError('pdm needs noise_scale > 0, timestep_std > 0 and loss_t_floor in (0, 1]')
    counts = spec['sampling_steps']
    if not isinstance(counts, list) or not counts:
        raise ValueError('pdm.sampling_steps must be a nonempty list of unique positive integers')
    for count in [*counts, spec['inference_steps']]:
        validate_steps(count)
    if len(set(counts)) != len(counts):
        raise ValueError('pdm.sampling_steps must be a nonempty list of unique positive integers')
    return spec


def validate_steps(steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError('Pixel flow sampling steps must be a positive integer')


def training_flow_spec(config):
    """Fail before loading data/weights if a one-step configuration is mixed into route B."""
    objective = config['training'].get('objective', 'one_step')
    if objective not in ('one_step', 'pixel_flow'):
        raise ValueError('training.objective must be one_step or pixel_flow')
    pixel = config['student'].get('pixel') or {}
    if objective == 'one_step':
        if 'pdm' in config or pixel.get('prediction_type') == 'x':
            raise ValueError('pdm / pixel.prediction_type: x require training.objective: pixel_flow')
        return None
    if pixel.get('prediction_type') != 'x':
        raise ValueError('pixel_flow requires student.pixel.prediction_type: x')
    tc, student = config['training'], config['student']
    if student.get('noise_input', 'gaussian') != 'gaussian':
        raise ValueError('pixel_flow requires Gaussian pixel noise')
    if any(float(tc.get(key, default)) != 0 for key, default in
           (('kd_weight', 1), ('feature_distill_weight', 0), ('kd_text_weight', 0))):
        raise ValueError('pixel_flow requires kd_weight, feature_distill_weight and kd_text_weight = 0; '
                         'latent KD and route A endpoint KD are not pixel FM')
    weight = tc.get('flow_weight', 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError('training.flow_weight must be finite and positive')
    return pixel_flow_spec(config.get('pdm', {}))


def make_flow_batch(clean, spec):
    """Each example gets an independent logit-normal t and fresh pixel noise, in FP32."""
    with torch.autocast(clean.device.type, enabled=False):
        t = (torch.randn(clean.shape[0], device=clean.device, dtype=torch.float32)
             * spec['timestep_std'] + spec['timestep_mean']).sigmoid().clamp(1e-5, 1 - 1e-5)
        noise = torch.randn_like(clean, dtype=torch.float32) * spec['noise_scale']
        time = t[:, None, None, None]
        noisy = (1 - time) * clean.float() + time * noise
    return noisy, t


def pixel_flow_loss(prediction, clean, t, floor):
    """MSE(v_pred, v_target) = mean((x_pred - clean)^2 / max(t, floor)^2).

    Both velocities have the SAME denominator. Computing their difference directly
    in x-space avoids subtracting two nearly equal velocities. No hard RGB clipping.
    """
    with torch.autocast(prediction.device.type, enabled=False):
        denominator = t.float().clamp_min(floor)[:, None, None, None]
        return ((prediction.float() - clean.float()) / denominator).square().mean()


@torch.no_grad()
def sample_pixel_flow(predict_x, initial_noise, steps):
    """Uniform decreasing grid 1 -> 0, with exactly ``steps`` positive-time model calls.

    ``initial_noise`` already includes gamma. ``predict_x(x_t, t_batch)`` returns clean
    RGB; the caller owns autocast, conditioning and (optionally) tiled prediction.
    The final Euler update to 0 equals x-pred, so return it directly in FP32.
    """
    validate_steps(steps)
    state = initial_noise.float()
    for index in range(steps):
        time = 1 - index / steps
        t = torch.full((state.shape[0],), time, device=state.device, dtype=torch.float32)
        clean = predict_x(state, t).float()
        if index == steps - 1:
            return clean
        with torch.autocast(state.device.type, enabled=False):
            state = state - (state - clean) / (steps * time)


def flow_optimizer_groups(model, config):
    """Disjoint regex LR groups, split into matrix decay and bias/norm no-decay.

    trainable_parameters owns freezing; this helper groups only selected weights.
    Reject unmatched/overlapping groups so inherited blocks cannot silently get the
    interface LR. Calling a frozen backbone is still differentiable w.r.t. its input.
    """
    definitions = config.get('parameter_groups')
    if definitions is None:
        definitions = [dict(name='trainable', patterns=['.*'], lr_multiplier=1.0)]
    if not isinstance(definitions, list) or not definitions:
        raise ValueError('training.parameter_groups must be a nonempty list')
    groups, names = [], set()
    for definition in definitions:
        if not isinstance(definition, dict) or set(definition) != {'name', 'patterns', 'lr_multiplier'}:
            raise ValueError('Each parameter group needs name, patterns and lr_multiplier')
        name, patterns, scale = (definition[key] for key in ('name', 'patterns', 'lr_multiplier'))
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Parameter group names must be nonempty and unique')
        if not isinstance(patterns, list) or not patterns or not all(isinstance(p, str) for p in patterns):
            raise ValueError('Parameter group patterns must be a nonempty list of regex strings')
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('Parameter group lr_multiplier must be finite and positive')
        names.add(name)
        groups.append((name, [re.compile(p) for p in patterns], float(scale)))
    buckets = {(name, decay): [] for name, _, _ in groups for decay in (False, True)}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [label for label, patterns, _ in groups if any(p.search(name) for p in patterns)]
        if len(matches) != 1:
            raise ValueError(f'Trainable parameter {name} must match exactly one LR group; got {matches}')
        buckets[matches[0], parameter.ndim > 1].append(parameter)
    result = []
    for name, _, scale in groups:
        for decay in (False, True):
            parameters = buckets[name, decay]
            if parameters:
                result.append(dict(params=parameters, name=name + ('/decay' if decay else '/no_decay'),
                                   lr=float(config['learning_rate']) * scale, lr_scale=scale,
                                   weight_decay=float(config.get('weight_decay', .01)) if decay else 0.0))
    if not result:
        raise ValueError('No trainable pixel flow parameters')
    return result
