"""Small-text weighted FFN statistics and teacher-input reconstruction primitives.

These utilities do not import the full DiT, a VAE, OCR, or an accelerator backend.
The CLI tools provide those dependencies. Local reconstruction runs a frozen
teacher once and backpropagates each selected narrow MLP immediately in a forward
hook: no student transformer graph or cache of all teacher layers is retained.
"""
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import re

import torch
import torch.nn.functional as F


WEIGHTING_MODES = ('uniform', 'text', 'balanced')


def token_weights(text_mask, small_text_mask=None, *, tokens, mode='balanced',
                  strata_weights=(0.5, 0.3, 0.2), text_weight=4.0, valid_mask=None):
    """Return [B,N] nonnegative weights, each image normalized to mean one.

    ``balanced`` allocates probability mass to small text / other text /
    background. Empty strata relinquish their mass, distributed over the
    remaining nonempty strata. Area pooling retains sub-token small characters.
    Masks are on the original output-pixel grid, before any OCR resizing.
    """
    if mode not in WEIGHTING_MODES:
        raise ValueError(f'weighting must be one of {WEIGHTING_MODES}')
    side = math.isqrt(tokens)
    if tokens <= 0 or side * side != tokens:
        raise ValueError('FFN calibration needs a positive square token grid')
    if text_mask.ndim != 4 or text_mask.shape[1] != 1:
        raise ValueError('text_mask must have shape [B,1,H,W]')
    if not torch.isfinite(text_mask).all():
        raise ValueError('text_mask contains non-finite values')
    text = F.adaptive_avg_pool2d(text_mask.float().clamp(0, 1), (side, side)).flatten(1)
    valid = torch.ones_like(text)
    if valid_mask is not None:
        if valid_mask.shape != text_mask.shape or not torch.isfinite(valid_mask).all():
            raise ValueError('valid_mask must be finite with the same shape as text_mask')
        valid = F.adaptive_avg_pool2d(valid_mask.float().clamp(0, 1), (side, side)).flatten(1)
        if (valid.sum(1) <= 0).any():
            raise ValueError('Each calibration image must contain valid, unpadded pixels')
    text = torch.minimum(text, valid)
    if mode == 'uniform':
        return valid / valid.mean(1, keepdim=True).clamp_min(1e-12)
    if mode == 'text':
        if not math.isfinite(text_weight) or text_weight < 0:
            raise ValueError('text_weight must be finite and nonnegative')
        weights = valid + float(text_weight) * text
        return weights / weights.mean(1, keepdim=True)
    if small_text_mask is None or small_text_mask.shape != text_mask.shape:
        raise ValueError('balanced weighting needs a same-shape small_text_mask')
    if not torch.isfinite(small_text_mask).all():
        raise ValueError('small_text_mask contains non-finite values')
    masses = torch.as_tensor(strata_weights, dtype=torch.float32, device=text.device)
    if masses.shape != (3,) or not torch.isfinite(masses).all() or (masses < 0).any() or masses.sum() <= 0:
        raise ValueError('strata_weights must be three finite nonnegative values with a positive sum')
    small = F.adaptive_avg_pool2d(small_text_mask.float().clamp(0, 1), (side, side)).flatten(1)
    small = torch.minimum(small, text)
    strata = torch.stack((small, (text - small).clamp_min(0), (valid - text).clamp_min(0)), 1)
    totals = strata.sum(-1, keepdim=True)
    shares = masses[None, :, None] * (totals > 1e-12)
    denom = shares.sum(1, keepdim=True)
    shares = shares / denom.clamp_min(1e-12)
    weights = (shares * strata / totals.clamp_min(1e-12)).sum(1) * tokens
    # If the user assigned zero mass to every stratum present in an image,
    # retain that image with uniform weights rather than creating NaN/zero loss.
    uniform = valid / valid.mean(1, keepdim=True).clamp_min(1e-12)
    return torch.where(denom[:, 0] > 0, weights, uniform)


@torch.no_grad()
def subsample_token_weights(weights, max_tokens_per_sample=0, *, seed=0):
    """Uniformly sample positive-weight tokens per image without replacement.

    Returns ``(weights, flat_indices)``. ``flat_indices=None`` preserves the
    exact all-token path when the limit is zero or no image exceeds the limit.
    Otherwise both outputs are flat; indices address a flattened [B,N,D]
    activation. Each selected weight is multiplied by positive_count / K, the
    inverse inclusion probability. Weighted sums are unbiased estimators of
    the full sums, but their ratio and the resulting centered covariance are
    approximations, not exact full-token statistics or extra image samples.

    Selection uses a dedicated CPU generator so it is independent of global
    RNG state, accelerator type, and layer grouping. Padding/zero-mass tokens
    are never sampled. The teacher still runs on its complete token grid.
    """
    if isinstance(max_tokens_per_sample, bool) or not isinstance(max_tokens_per_sample, int) \
            or max_tokens_per_sample < 0:
        raise ValueError('max_tokens_per_sample must be a nonnegative integer')
    if weights.ndim != 2 or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Token weights must be finite nonnegative [B,N]')
    counts = (weights > 0).sum(1).cpu()
    if (counts == 0).any():
        raise ValueError('Each sampled image must contain positive-weight tokens')
    if max_tokens_per_sample == 0 or counts.max() <= max_tokens_per_sample:
        return weights, None
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    positive = (weights > 0).cpu()
    indices, scales = [], []
    for batch_index, row in enumerate(positive):
        available = row.nonzero(as_tuple=False).flatten()
        count = available.numel()
        chosen = min(count, max_tokens_per_sample)
        if chosen < count:
            available = available[torch.randperm(count, generator=generator)[:chosen]].sort().values
        indices.append(available + batch_index * weights.shape[1])
        scales.append(torch.full((chosen,), count / chosen, dtype=torch.float32))
    indices = torch.cat(indices)
    selected = weights.flatten().index_select(0, indices.to(weights.device))
    return selected * torch.cat(scales).to(selected), indices


class WeightedMoments:
    """Stable, mergeable population moments using weighted parallel Welford.

    Only one layer/group's M2 matrices need live on the chosen moment device.
    CPU float64 is the default; accelerator float32 is useful on NPU, where
    float64 is not supported. Centering each chunk avoids E[x*x]-E[x]^2 loss
    of precision. ``chunk_tokens`` bounds the activation temporary.
    """
    def __init__(self, width, covariance=False, *, device='cpu', dtype=torch.float64, chunk_tokens=1024):
        if width <= 0 or chunk_tokens <= 0:
            raise ValueError('width and chunk_tokens must be positive')
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('moment dtype must be float32 or float64')
        if torch.device(device).type == 'npu' and dtype == torch.float64:
            raise ValueError('NPU moments require float32, or use moment_device: cpu with float64')
        self.width, self.covariance, self.chunk_tokens = width, bool(covariance), int(chunk_tokens)
        self.mean = torch.zeros(width, dtype=dtype, device=device)
        self.m2 = torch.zeros((width, width) if covariance else (width,), dtype=dtype, device=device)
        self.mass = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, activation, weights=None):
        activation = activation.detach().reshape(-1, self.width)
        if weights is None:
            weights = torch.ones(activation.shape[0], dtype=torch.float32, device=activation.device)
        weights = weights.detach().reshape(-1)
        if weights.numel() != activation.shape[0]:
            raise ValueError('One calibration weight is required per activation token')
        for start in range(0, activation.shape[0], self.chunk_tokens):
            a = activation[start:start + self.chunk_tokens].to(self.mean)
            w = weights[start:start + self.chunk_tokens].to(self.mean)
            if not torch.isfinite(a).all() or not torch.isfinite(w).all() or (w < 0).any():
                raise ValueError('Calibration activations/weights must be finite; weights must be nonnegative')
            mass = float(w.sum())
            if mass == 0:
                continue
            mean = (a * w[:, None]).sum(0) / mass
            centered = a - mean
            m2 = centered.T @ (centered * w[:, None]) if self.covariance else (centered.square() * w[:, None]).sum(0)
            total = self.mass + mass
            delta = mean - self.mean
            correction = torch.outer(delta, delta) if self.covariance else delta.square()
            self.m2.add_(m2).add_(correction, alpha=self.mass * mass / total)
            self.mean.add_(delta, alpha=mass / total)
            self.mass = total
            self.count += int((w > 0).sum())

    def result(self):
        if self.mass <= 0:
            raise ValueError('No positive-weight calibration tokens reached the MLP')
        moment = self.m2 / self.mass
        if self.covariance:
            moment = (moment + moment.T) * 0.5
            variance = moment.diagonal().clamp_min(0)
        else:
            variance = moment.clamp_min(0)
        result = dict(mean=self.mean.float().cpu(), variance=variance.float().cpu())
        if self.covariance:
            result['covariance'] = moment.float().cpu()
        return result


@torch.no_grad()
def mlp_residual_gates(model, t, r):
    """Actual adaLN MLP gates for each sample/layer, including auxiliary time."""
    c = model.t_embedder(t)
    if model.r_embedder is not None:
        c = c + model.r_embedder(r) * (t - r).unsqueeze(-1)
    c0 = model.t_block(c).reshape(t.shape[0], 6, -1)
    return [(block.scale_shift_table[None] + c0)[:, 5].detach() for block in model.blocks]


def local_reconstruction_loss(prediction, target, gate, weights, *, energy_floor=1e-4, gated=True):
    """Weighted relative MSE of the MLP's contribution to the residual stream."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('Local MLP prediction and target must have matching [B,N,D] shapes')
    if weights.shape != prediction.shape[:2] or (weights < 0).any() or not torch.isfinite(weights).all():
        raise ValueError('Local reconstruction weights must be finite nonnegative [B,N]')
    if not math.isfinite(energy_floor) or energy_floor <= 0:
        raise ValueError('energy_floor must be finite and positive')
    s, y = prediction.float(), target.detach().float()
    if gated:
        if gate.shape != (s.shape[0], s.shape[-1]):
            raise ValueError('MLP gate must have shape [B,D]')
        s, y = s * gate.float()[:, None], y * gate.float()[:, None]
    weights = weights.to(s)
    if weights.sum() <= 0:
        raise ValueError('Local reconstruction requires positive token weight')
    error = ((s - y).square().mean(-1) * weights).sum() / weights.sum()
    energy = (y.square().mean(-1) * weights).sum() / weights.sum()
    return error / energy.clamp_min(energy_floor)


def select_reconstruction_parameters(student, layers=None, patterns=None):
    """Freeze the full student, enabling selected physical MLPs (w12 AND w3 by default)."""
    layers = list(range(len(student.blocks))) if layers is None else list(layers)
    if not layers or len(set(layers)) != len(layers) or any(isinstance(i, bool) or not isinstance(i, int)
                                                          or not 0 <= i < len(student.blocks) for i in layers):
        raise ValueError('reconstruction.layers must be distinct, existing block indices')
    if patterns is not None and (not isinstance(patterns, list) or not patterns
                                 or not all(isinstance(p, str) for p in patterns)):
        raise ValueError('reconstruction.trainable_parameters must be a nonempty list of regex patterns')
    compiled = [re.compile(p) for p in patterns] if patterns is not None else None
    student.requires_grad_(False)
    parameters = []
    for index in layers:
        mlp = student.blocks[index].mlp
        if not hasattr(mlp, 'w12') or not hasattr(mlp, 'w3') or hasattr(mlp, 'fade_weight'):
            raise ValueError('Local reconstruction needs physical SwiGLU MLPs; disable fade and whole-MLP removal')
        chosen = []
        for name, parameter in mlp.named_parameters():
            full_name = f'blocks.{index}.mlp.{name}'
            trainable = compiled is None or any(pattern.search(full_name) for pattern in compiled)
            parameter.requires_grad_(trainable)
            if trainable:
                chosen.append(parameter)
        if not chosen:
            raise ValueError(f'No reconstruction parameter matched block {index}')
        parameters.extend(chosen)
    return layers, parameters


def reconstruction_microstep(teacher, student, inp, features, weights, *, layers, loss_scale=1.0,
                             autocast=None, energy_floor=1e-4, gated=True):
    """Every selected layer gets one local backward, with no full student forward.

    The teacher's hooks provide normalized+modulated pre-MLP inputs and outputs.
    The teacher is evaluated under no_grad; only the narrow MLP call re-enables
    gradients. Each local graph is freed before proceeding to the next block.
    """
    autocast = autocast or nullcontext
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError('loss_scale must be positive')
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError('Reconstruction teacher must be frozen')
    t, r = inp.new_ones(inp.shape[0]), inp.new_zeros(inp.shape[0])
    with torch.no_grad(), autocast():
        gates = mlp_residual_gates(teacher, t, r)
    losses, calls = {}, {index: 0 for index in layers}

    def hook(index):
        def reconstruct(module, args, output):
            calls[index] += 1
            with torch.enable_grad(), autocast():
                prediction = student.blocks[index].mlp(args[0].detach())
                loss = local_reconstruction_loss(prediction, output, gates[index], weights,
                                                 energy_floor=energy_floor, gated=gated)
                scaled_loss = loss * loss_scale / len(layers)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite local reconstruction loss in block {index}')
            scaled_loss.backward()
            losses[index] = float(loss.detach())
        return reconstruct

    handles = [teacher.blocks[index].mlp.register_forward_hook(hook(index)) for index in layers]
    try:
        with torch.no_grad(), autocast():
            teacher(inp, t, r, features)
    finally:
        for handle in handles:
            handle.remove()
    if any(count != 1 for count in calls.values()):
        raise RuntimeError(f'Every selected MLP must run once per microstep; got {calls}')
    return losses


def seeded_noise_like(tensor, seed):
    """CPU generator gives explicit, backend-independent Gaussian noise streams."""
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    return torch.randn(tensor.shape, generator=generator, dtype=torch.float32).to(tensor)


def sample_paired_batch(dataset, size, seed):
    """Deterministic samples without altering Python's global/worker RNG state."""
    from dataloaders.paired_crops import collate_paired_samples
    rng, samples = random.Random(int(seed)), []
    for _ in range(size):
        index = rng.randrange(len(dataset))
        if hasattr(dataset, 'sample_index'):
            sample = dataset.sample_index(index, rng)
        elif hasattr(dataset, 'sample') and hasattr(dataset, 'names'):
            sample = dataset.sample(dataset.names[index], rng)
        else:
            raise ValueError('Calibration/reconstruction requires a deterministic paired dataset')
        samples.append(sample)
    return collate_paired_samples(samples)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_provenance(path):
    """Strong, streamed hashes of the actual checkpoint/shard bytes, not only a norm."""
    path = Path(path).resolve()
    if path.is_dir():
        candidates = [path / sub / name for sub in ('clean_weights', 'checkpoints', '')
                      for name in ('ema_model.safetensors', 'model.safetensors', 'model.safetensors.index.json')]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError('No clean checkpoint found for provenance')
    files = [path]
    if path.name.endswith('.index.json'):
        index = json.loads(path.read_text(encoding='utf-8'))
        files += [path.parent / name for name in sorted(set(index['weight_map'].values()))]
    records = [dict(path=str(p), bytes=p.stat().st_size, sha256=file_sha256(p)) for p in files]
    fingerprint = hashlib.sha256(json.dumps([(p['bytes'], p['sha256']) for p in records]).encode()).hexdigest()
    return dict(fingerprint=fingerprint, files=records)


def paired_data_provenance(dataset, data_config):
    """Hash ordered paths/sizes/mtimes and annotation bytes, recording the policy.

    Images are represented by filesystem manifests (NOT content hashes); this
    is stated explicitly so users can archive immutable datasets when needed.
    Annotation JSON is content-hashed since it changes token weighting.
    """
    datasets = getattr(dataset, 'datasets', [dataset])
    sources = data_config.get('sources') or [data_config]
    records = []
    for source, child in zip(sources, datasets, strict=True):
        digest = hashlib.sha256()
        for name in sorted(child.names):
            for kind in ('lq', 'hq'):
                path = Path(getattr(child, kind)[name]).resolve()
                st = path.stat()
                digest.update(json.dumps((name, kind, str(path), st.st_size, st.st_mtime_ns)).encode())
        annotation = source.get('text_boxes', data_config.get('text_boxes'))
        records.append(dict(name=source.get('name', 'paired'), images=len(child.names),
                            image_manifest_sha256=digest.hexdigest(),
                            annotation_sha256=file_sha256(annotation) if annotation else None,
                            lq_dir=str(Path(source['lq_dir']).resolve()),
                            hq_dir=str(Path(source['hq_dir']).resolve()), weight=source.get('weight', 1.0)))
    payload = dict(policy='images: path/size/mtime manifest; annotations: sha256 bytes',
                   config=data_config, sources=records)
    payload['fingerprint'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload
