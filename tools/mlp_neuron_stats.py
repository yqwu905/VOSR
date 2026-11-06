"""Deterministic, small-text weighted SwiGLU neuron calibration.

Writes the original mean/variance/output_energy/w3_sq_norm tensor contract and,
with --covariance, covariance/gate. Statistics describe the *full-width source*
from which the student is pruned. Full covariance is accumulated in bounded
layer groups, with stable weighted Welford updates, never all FP64 covariance
matrices on the model device. The final FP32 file remains about 2.4 GB for VOSR2.

Both single paired data and data.sources mixtures are supported. Their source
probabilities and data.text_sampling policy are honored. Balanced weighting
gives small text / other text / background configurable per-image mass. A
sidecar .provenance.json records checkpoint hashes, data manifest policy,
sample/noise seeds, and the exact settings. Runtime defaults can be supplied in
the optional calibration config block; explicit CLI options override them.

Cost controls are independent: layer_group_size limits covariance memory but
repeats the complete VAE/DINO/teacher pass for each group; chunk_tokens bounds
temporary matrix sizes without subsampling. max_tokens_per_sample=0 uses all
tokens (exact default). A positive limit uniformly samples valid, positive-
weight tokens per image with inverse-inclusion-probability weights, reducing
covariance work while estimating, rather than exactly computing, full-token
moments. It does not reduce the teacher's full-grid forward pass.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_utils import (build_train_dataset, dino_features, load_backbone_state, load_config, load_dino,
                            read_weights, student_dino_config, vae_tiling)  # noqa: E402
from models.ffn_reconstruction import (WEIGHTING_MODES, WeightedMoments, checkpoint_provenance,
                                      mlp_residual_gates, paired_data_provenance, sample_paired_batch,
                                      seeded_noise_like, subsample_token_weights, token_weights)  # noqa: E402
from training_runtime import select_device  # noqa: E402


@torch.no_grad()
def neuron_statistics(model, samples, covariance=False, *, layer_group_size=None,
                      moment_device='cpu', moment_dtype=torch.float64, chunk_tokens=1024):
    """Weighted statistics, preserving the existing programmatic API.

    Samples yield (input, DINO features), (input, features, [B,N] weights), or
    (input, features, [K] sampled weights, [K] flat token indices). Indices
    address the flattened B*N axis and must be unique; sampling is performed
    before this API so callers can explicitly reproduce the same selection.
    For grouped passes, supply a callable returning an identical fresh iterator,
    or a re-iterable sequence. A one-shot generator cannot be reused and is
    rejected instead of silently measuring different examples in different layers.
    """
    from models.swiglu_ffn import SwiGLUFFN
    if not model.blocks or not all(isinstance(block.mlp, SwiGLUFFN) and not hasattr(block.mlp, 'fade_weight')
                                    for block in model.blocks):
        raise ValueError('Neuron statistics need the full-width SwiGLU MLPs without a fade')
    depth, hidden = len(model.blocks), model.blocks[0].mlp.w3.in_features
    if any(block.mlp.w3.in_features != hidden for block in model.blocks):
        raise ValueError('Statistics require the same full source width in each block')
    group_size = depth if layer_group_size is None else layer_group_size
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError('layer_group_size must be a positive integer')
    if not callable(samples) and iter(samples) is samples and group_size < depth:
        raise ValueError('Grouped calibration needs a re-iterable sequence or a sample factory')
    factory = samples if callable(samples) else lambda: iter(samples)
    device = next(model.parameters()).device
    stats_device = device if str(moment_device) == 'model' else torch.device(moment_device)
    stats = {key: torch.empty(depth, hidden) for key in ('mean', 'variance')}
    if covariance:
        stats['covariance'] = torch.empty(depth, hidden, hidden, device='cpu')
    last = None
    for start in range(0, depth, group_size):
        indices = list(range(start, min(start + group_size, depth)))
        moments = {index: WeightedMoments(hidden, covariance, device=stats_device, dtype=moment_dtype,
                                         chunk_tokens=chunk_tokens) for index in indices}
        current_weights, current_indices = None, None

        def hook(index):
            def gather(module, args):
                activation = args[0]
                if current_indices is not None:
                    activation = activation.reshape(-1, hidden).index_select(0, current_indices.to(activation.device))
                moments[index].update(activation, current_weights)
            return gather

        handles = [model.blocks[index].mlp.w3.register_forward_pre_hook(hook(index)) for index in indices]
        try:
            for sample in factory():
                if len(sample) not in (2, 3, 4):
                    raise ValueError('Samples need input, DINO features, optional token weights and flat indices')
                inp, features = sample[:2]
                current_weights = sample[2] if len(sample) >= 3 else None
                current_indices = sample[3] if len(sample) == 4 else None
                if current_indices is not None:
                    if current_indices.ndim != 1 or current_indices.dtype != torch.long \
                            or current_indices.numel() == 0 or current_indices.min() < 0 \
                            or current_indices.unique().numel() != current_indices.numel():
                        raise ValueError('Sampled token indices must be distinct, nonnegative int64 values')
                    if current_weights is None or current_weights.numel() != current_indices.numel():
                        raise ValueError('One weight is required for every sampled token index')
                t, r = inp.new_ones(inp.shape[0]), inp.new_zeros(inp.shape[0])
                model(inp, t, r, features)
                last = (t[:1], r[:1])
        finally:
            for handle in handles:
                handle.remove()
        for index, moment in moments.items():
            for key, value in moment.result().items():
                stats[key][index] = value
        del moments
    if last is None:
        raise ValueError('No samples reached the MLPs')
    gates = mlp_residual_gates(model, *last)
    energy, norms = [], []
    for index, block in enumerate(model.blocks):
        # CPU float64 also works on NPU hosts (which do not support float64).
        w3 = block.mlp.w3.weight.detach().cpu().double()
        gate = gates[index][0].cpu().double()
        energy.append((gate[:, None] * w3).square().sum(0))
        norms.append(w3.square().sum())
    stats['output_energy'], stats['w3_sq_norm'] = torch.stack(energy).float(), torch.stack(norms).float()
    if covariance:
        stats['gate'] = torch.stack([g[0].cpu().float() for g in gates])
    return {key: value.float().cpu().contiguous() for key, value in stats.items()}


def source_model(cfg):
    """Student architecture before pruning, loaded from its actual source checkpoint."""
    from models.lightningdit_ablation import AblationLightningDiT
    state = read_weights(cfg['teacher_checkpoint'])
    model_cfg = dict(cfg['model'])
    aux = model_cfg.pop('auxiliary_time_cond', 'auto')
    model_cfg['auxiliary_time_cond'] = any(k.startswith('r_embedder.') for k in state) if aux == 'auto' else bool(aux)
    model_cfg['input_size'] = int(cfg['data']['resolution']) // 8
    student = cfg['student']
    if student.get('token_compression') is not None or student.get('router_config') is not None or student.get('pixel'):
        raise ValueError('Neuron calibration requires the latent SwiGLU model without token compression/routing')
    model = AblationLightningDiT(**model_cfg, use_cross_attention=student['use_cross_attention'],
                                 attn_compression=student.get('attn_compression'),
                                 attn_type=student.get('attn_type', 'full'))
    initial = state if not cfg.get('student_checkpoint') else read_weights(cfg['student_checkpoint'])
    if any(k.endswith('.mlp.w3.weight') and value.shape[-1] != model.blocks[0].mlp.w3.in_features
           for k, value in initial.items()):
        raise ValueError('Calibrate a full-width source checkpoint, not an already pruned student')
    load_backbone_state(model, initial, allow_new_modules=True)
    return model


CALIBRATION_DEFAULTS = dict(samples=64, seed=11, noise_seed=0, weighting='uniform', text_weight=4.0,
                            small_text_height=64.0, strata_weights=[0.5, 0.3, 0.2], layer_group_size=2,
                            moment_device='cpu', moment_dtype='float64', chunk_tokens=1024, covariance=False,
                            max_tokens_per_sample=0)


def calibration_options(cfg, args):
    supplied = cfg.get('calibration') or {}
    if not isinstance(supplied, dict) or set(supplied) - set(CALIBRATION_DEFAULTS):
        raise ValueError(f'calibration accepts only {sorted(CALIBRATION_DEFAULTS)}')
    options = dict(CALIBRATION_DEFAULTS, **supplied)
    for key in options:
        override = getattr(args, key, None)
        if override is not None:
            options[key] = override
    for key in ('samples', 'layer_group_size', 'chunk_tokens'):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] <= 0:
            raise ValueError(f'calibration.{key} must be a positive integer')
    for key in ('seed', 'noise_seed', 'max_tokens_per_sample'):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 0:
            raise ValueError(f'calibration.{key} must be a nonnegative integer')
    if options['moment_device'] not in ('cpu', 'model') or options['moment_dtype'] not in ('float32', 'float64'):
        raise ValueError('Use moment_device cpu|model and moment_dtype float32|float64')
    if options['weighting'] not in WEIGHTING_MODES:
        raise ValueError(f'weighting must be one of {WEIGHTING_MODES}')
    if not isinstance(options['covariance'], bool):
        raise ValueError('calibration.covariance must be a boolean')
    if not isinstance(options['small_text_height'], (int, float)) or not 0 < options['small_text_height'] < float('inf'):
        raise ValueError('small_text_height must be finite and positive')
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True, help='.safetensors file for mlp_compression.neuron_stats')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda[:index], or npu[:index]')
    for key in ('samples', 'seed', 'noise_seed', 'layer_group_size', 'chunk_tokens', 'max_tokens_per_sample'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    for key in ('small_text_height', 'text_weight'):
        parser.add_argument('--' + key.replace('_', '-'), type=float)
    parser.add_argument('--strata-weights', nargs=3, type=float)
    parser.add_argument('--weighting', choices=WEIGHTING_MODES)
    parser.add_argument('--moment-device', choices=('cpu', 'model'))
    parser.add_argument('--moment-dtype', choices=('float32', 'float64'))
    parser.add_argument('--covariance', action='store_true', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    options = calibration_options(cfg, args)
    data = copy.deepcopy(cfg['data'])
    if data.get('dataset_type') != 'paired':
        raise ValueError('Neuron calibration needs data.dataset_type: paired (single source or sources mixture)')
    data['resolution'] = int(data.get('crop_size') or data['resolution'])
    data.update(text_metadata=True, small_text_height=options['small_text_height'])
    dataset = build_train_dataset(data, text_mask=True)
    device = select_device(args.device)
    if device.type == 'npu' and options['moment_device'] == 'model' and options['moment_dtype'] == 'float64':
        raise ValueError('On NPU choose moment_dtype: float32, or moment_device: cpu')
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    from tiled_vae import encode_dispatch
    model = source_model(cfg).to(device).eval().requires_grad_(False)
    vae = AutoencoderKLQwenImage2D.from_pretrained(cfg['vae_path']).to(device).eval().requires_grad_(False)
    venc = load_dino(cfg['dino'], device)
    dino = student_dino_config(cfg)
    sampled, passes = [], 0

    @torch.no_grad()
    def samples():
        nonlocal passes
        passes += 1
        for index in range(options['samples']):
            crop_seed = options['seed'] * 1_000_003 + index
            noise_seed = options['noise_seed'] + index
            token_seed = options['seed'] * 19_000_033 + index
            batch = sample_paired_batch(dataset, 1, crop_seed)
            lq = batch['lq'].to(device) * 2 - 1
            latent = encode_dispatch(vae, lq, vae_tiling(cfg)[1], device)[0]
            tokens = (latent.shape[-2] // model.patch_size) * (latent.shape[-1] // model.patch_size)
            weights = token_weights(batch['text_mask'], batch['small_text_mask'], tokens=tokens,
                                    mode=options['weighting'], strata_weights=options['strata_weights'],
                                    text_weight=options['text_weight'], valid_mask=batch.get('valid_mask'))
            positive_tokens = int((weights > 0).sum())
            weights, selected_indices = subsample_token_weights(weights, options['max_tokens_per_sample'],
                                                               seed=token_seed)
            if passes == 1:
                sampled.append(dict(index=index, crop_seed=crop_seed, noise_seed=noise_seed,
                                    token_seed=token_seed, grid_tokens=tokens, positive_tokens=positive_tokens,
                                    selected_tokens=(positive_tokens if selected_indices is None
                                                     else selected_indices.numel()),
                                    sample_name=batch.get('sample_name'), source_id=batch.get('source_id')))
            yield (torch.cat((latent, seeded_noise_like(latent, noise_seed)), 1),
                   dino_features(venc, lq, dino), weights, selected_indices)
            if index == 0 or (index + 1) % 64 == 0 or index + 1 == options['samples']:
                print(f'pass {passes}: sample {index + 1}/{options["samples"]}', flush=True)

    stats = neuron_statistics(model, samples, options['covariance'], layer_group_size=options['layer_group_size'],
                              moment_device=options['moment_device'], moment_dtype=getattr(torch, options['moment_dtype']),
                              chunk_tokens=options['chunk_tokens'])
    source = str(cfg.get('student_checkpoint') or cfg['teacher_checkpoint'])
    provenance = dict(version=2, source=checkpoint_provenance(source), data=paired_data_provenance(dataset, data),
                      calibration=options, samples=sampled, torch_version=str(torch.__version__),
                      model_device=str(device), model=cfg['model'], student=cfg['student'],
                      source_model=model.export_config, dino=dino,
                      vae_path=str(cfg['vae_path']), endpoint=dict(t=1, r=0), layer_group_passes=passes,
                      token_sampling=dict(policy='uniform_without_replacement_among_positive_weight_tokens',
                                          inverse_inclusion_weighting=True,
                                          approximate=any(row['selected_tokens'] < row['positive_tokens']
                                                          for row in sampled),
                                          sampled_tokens=sum(row['selected_tokens'] for row in sampled),
                                          crop_draws=len(sampled)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix('.provenance.json')
    sidecar.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    metadata = {key: str(value) for key, value in dict(description='Weighted SwiGLU neuron statistics, format v2',
                source=source, samples=options['samples'], seed=options['seed'], crop=data['resolution'],
                weighting=options['weighting'], source_sha256=provenance['source']['fingerprint'],
                max_tokens_per_sample=options['max_tokens_per_sample'],
                data_sha256=provenance['data']['fingerprint'], provenance_file=sidecar.name,
                attn_compression=cfg['student'].get('attn_compression') is not None).items()}
    from safetensors.torch import save_file
    temporary = output.with_name(output.name + '.tmp')
    try:
        save_file(stats, str(temporary), metadata=metadata)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    score = stats['variance'] * stats['output_energy']
    top = score.sort(1, descending=True).values
    share = (top[:, :max(1, top.shape[1] // 4)].sum(1) / top.sum(1).clamp_min(1e-12)).mean().item()
    print(f'wrote {output}: {score.shape[0]} blocks x {score.shape[1]} neurons; top 25% fluctuation {share:.1%}')


if __name__ == '__main__':
    main()
