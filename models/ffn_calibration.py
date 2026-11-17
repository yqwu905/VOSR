"""Deterministic sampling, spatial weights and provenance for FFN spatial calibration."""
import hashlib
import json
import math
from pathlib import Path
import random

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
def mlp_residual_gates(model, t, r):
    """Actual adaLN MLP gates for each sample/layer, including auxiliary time."""
    c = model.t_embedder(t)
    if model.r_embedder is not None:
        c = c + model.r_embedder(r) * (t - r).unsqueeze(-1)
    c0 = model.t_block(c).reshape(t.shape[0], 6, -1)
    return [(block.scale_shift_table[None] + c0)[:, 5].detach() for block in model.blocks]


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
            raise ValueError('Spatial calibration requires a deterministic paired dataset')
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
