"""Calibrate spatial SwiGLU groups on the actual full-width student source.

For each neuron, measure the squared difference between its full-grid hidden
activation and its pooled-input / upsampled activation, weighted by the squared
norm of its gated output column. This is a diagonal approximation: correlations
between neurons are not modeled. Also measure the cost of dropping each neuron
outright (no mean compensation). See docs/ffn_mixed_resolution.md.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from ablation_utils import (build_train_dataset, dino_features, load_backbone_state, load_config, load_dino, read_weights,
                            student_dino_config, vae_tiling)
from models.ffn_calibration import (WEIGHTING_MODES, checkpoint_provenance, mlp_residual_gates,
                                    paired_data_provenance, sample_paired_batch, seeded_noise_like, token_weights)
from models.mixed_resolution_ffn import mlp_fingerprint, pool_grid, upsample_grid, validate_groups
from models.swiglu_ffn import SwiGLUFFN
from training_runtime import select_device


TIERS = ((4096, 0), (0, 4096), (1024, 3072), (512, 3584), (512, 2048))


@torch.no_grad()
def source_model(cfg):
    """Student architecture before spatial splitting, loaded from its actual source checkpoint."""
    from models.lightningdit_ablation import AblationLightningDiT
    state = read_weights(cfg['teacher_checkpoint'])
    model_cfg = dict(cfg['model'])
    aux = model_cfg.pop('auxiliary_time_cond', 'auto')
    model_cfg['auxiliary_time_cond'] = any(k.startswith('r_embedder.') for k in state) if aux == 'auto' else bool(aux)
    model_cfg['input_size'] = int(cfg['data']['resolution']) // 8
    student = cfg['student']
    if student.get('router_config') is not None or student.get('pixel'):
        raise ValueError('Neuron calibration requires the latent SwiGLU model without routing')
    model = AblationLightningDiT(**model_cfg, use_cross_attention=student['use_cross_attention'],
                                 attn_compression=student.get('attn_compression'),
                                 attn_type=student.get('attn_type', 'full'))
    initial = state if not cfg.get('student_checkpoint') else read_weights(cfg['student_checkpoint'])
    if any(k.endswith('.mlp.w3.weight') and value.shape[-1] != model.blocks[0].mlp.w3.in_features
           for k, value in initial.items()):
        raise ValueError('Calibrate a full-width source checkpoint, not an already pruned student')
    load_backbone_state(model, initial, allow_new_modules=True)
    return model


@torch.no_grad()
def spatial_statistics(model, samples, factor=2, upsample='bilinear', chunk_hidden=256):
    """Samples are (model_input, DINO_features[, BxN token weights]).

    Full spatial grids are retained; chunks bound temporary hidden activations.
    The pre-hook sees the exact normalized and modulated input to the source FFN.
    Accumulators live on CPU in FP64, so this also works on NPU hosts.
    """
    if model.training or not model.blocks or not all(type(b.mlp) is SwiGLUFFN for b in model.blocks):
        raise ValueError('Spatial calibration needs an eval model with ordinary full-width SwiGLU FFNs')
    if isinstance(chunk_hidden, bool) or not isinstance(chunk_hidden, int) or chunk_hidden < 1:
        raise ValueError('chunk_hidden must be a positive integer')
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1 \
            or upsample not in ('bilinear', 'nearest'):
        raise ValueError('Use a positive pooling factor and bilinear or nearest upsampling')
    depth, hidden = len(model.blocks), model.blocks[0].mlp.w3.in_features
    if any(b.mlp.w3.in_features != hidden for b in model.blocks):
        raise ValueError('Spatial calibration needs the same source width in all blocks')
    pool_sum = torch.zeros(depth, hidden, dtype=torch.float64)
    drop_sum = torch.zeros_like(pool_sum)
    mass = torch.zeros(depth, dtype=torch.float64)
    weights = last = None

    def hook(index):
        def gather(mlp, args):
            x = args[0]
            coarse, side = pool_grid(x, factor)
            weight = torch.ones(x.shape[:2], device=x.device) if weights is None else weights.to(x.device)
            if weight.shape != x.shape[:2] or not torch.isfinite(weight).all() or (weight < 0).any() \
                    or weight.sum() <= 0:
                raise ValueError('Calibration weights must be finite, nonnegative [B,N] with positive mass')
            weight = weight.float().unsqueeze(-1)
            mass[index] += weight.sum().cpu().double()
            for start in range(0, hidden, chunk_hidden):
                end = min(start + chunk_hidden, hidden)
                rows = torch.cat((torch.arange(start, end, device=x.device),
                                  torch.arange(start + hidden, end + hidden, device=x.device)))
                w12, b12 = mlp.w12.weight[rows], mlp.w12.bias[rows]
                gate, value = F.linear(x, w12, b12).chunk(2, -1)
                activation = F.silu(gate) * value
                gate, value = F.linear(coarse, w12, b12).chunk(2, -1)
                approximate = upsample_grid(F.silu(gate) * value, side, upsample)
                error = activation.float() - approximate.float()
                pool_sum[index, start:end] += (error.square() * weight).sum((0, 1)).cpu().double()
                drop_sum[index, start:end] += (activation.float().square() * weight).sum((0, 1)).cpu().double()
        return gather

    handles = [block.mlp.register_forward_pre_hook(hook(i)) for i, block in enumerate(model.blocks)]
    try:
        for sample in samples:
            if len(sample) not in (2, 3):
                raise ValueError('Samples need input, DINO features and optional full-grid token weights')
            inp, features = sample[:2]
            weights = sample[2] if len(sample) == 3 else None
            t, r = inp.new_ones(inp.shape[0]), inp.new_zeros(inp.shape[0])
            model(inp, t, r, features)
            last = (t[:1], r[:1])
    finally:
        for handle in handles:
            handle.remove()
    if last is None or (mass <= 0).any():
        raise ValueError('No samples reached every FFN')
    gates = mlp_residual_gates(model, *last)
    energy = torch.stack([(g[0].detach().cpu().double()[:, None] * b.mlp.w3.weight.detach().cpu().double())
                          .square().sum(0) for b, g in zip(model.blocks, gates)])
    result = dict(pool_error=pool_sum / mass[:, None] * energy,
                  drop_error=drop_sum / mass[:, None] * energy)
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise ValueError('Non-finite calibration statistics')
    return {k: v.float().contiguous() for k, v in result.items()}


def select_groups(stats, full_hidden, coarse_hidden, *, factor, upsample, fingerprints):
    """Retain neurons by drop cost, then protect retained ones by pooling error.

    Stable sorting resolves ties by source index. Lists use source neuron IDs and
    may be edited before training; uniqueness, size and source hashes are checked.
    """
    from models.mixed_resolution_ffn import mixed_resolution_spec
    pool, drop = stats['pool_error'], stats['drop_error']
    if pool.ndim != 2 or pool.shape != drop.shape or any(
            not torch.isfinite(v).all() or (v < 0).any() for v in (pool, drop)):
        raise ValueError('Calibration scores must be finite, nonnegative [depth, hidden] tensors')
    depth, hidden = pool.shape
    spec, _ = mixed_resolution_spec(dict(type='mixed_resolution', full_hidden=full_hidden,
                                         coarse_hidden=coarse_hidden, factor=factor, upsample=upsample), hidden)
    result = dict(version=1, source_hidden=hidden, factor=factor, upsample=upsample,
                  source_mlp_sha256=fingerprints, full_indices=[], coarse_indices=[])
    for pooling, dropping in zip(pool, drop):
        kept = torch.argsort(dropping, descending=True, stable=True)[:full_hidden + coarse_hidden].sort().values
        protected = kept[torch.argsort(pooling[kept], descending=True, stable=True)[:full_hidden]]
        full = set(protected.tolist())
        result['full_indices'].append(sorted(full))
        result['coarse_indices'].append([i for i in kept.tolist() if i not in full])
    validate_groups(result, spec, depth, hidden)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Any spatial-FFN training config; calibrates its source')
    parser.add_argument('--output-dir', required=True, help='New directory for scores, provenance and five groups')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--noise-seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--factor', type=int, default=2)
    parser.add_argument('--upsample', choices=('bilinear', 'nearest'), default='bilinear')
    parser.add_argument('--chunk-hidden', type=int, default=256)
    parser.add_argument('--weighting', choices=WEIGHTING_MODES, default='balanced')
    parser.add_argument('--text-weight', type=float, default=4.)
    parser.add_argument('--small-text-height', type=float, default=64.)
    parser.add_argument('--strata-weights', nargs=3, type=float, default=[.5, .3, .2])
    args = parser.parse_args()
    if args.samples < 1 or args.seed < 0 or args.noise_seed < 0 or args.factor < 1 or args.chunk_hidden < 1:
        parser.error('samples, factor and chunk-hidden must be positive; seeds must be nonnegative')
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a new calibration directory')
    cfg = load_config(args.config)
    data = copy.deepcopy(cfg['data'])
    if data.get('dataset_type') != 'paired':
        raise ValueError('Spatial calibration requires paired training data')
    data['resolution'] = int(data.get('crop_size') or data['resolution'])
    data.update(text_metadata=True, small_text_height=args.small_text_height)
    dataset = build_train_dataset(data, text_mask=True)
    device = select_device(args.device)
    model = source_model(cfg).to(device).eval().requires_grad_(False)
    if model.blocks[0].mlp.w3.in_features != 4096:
        raise ValueError('The CLI five-tier suite requires a 4096-wide source FFN')
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    from tiled_vae import encode_dispatch
    vae = AutoencoderKLQwenImage2D.from_pretrained(cfg['vae_path']).to(device).eval().requires_grad_(False)
    venc = load_dino(cfg['dino'], device)
    dino = student_dino_config(cfg)
    sampled = []

    @torch.no_grad()
    def samples():
        for i in range(args.samples):
            crop_seed, noise_seed = args.seed * 1_000_003 + i, args.noise_seed + i
            batch = sample_paired_batch(dataset, 1, crop_seed)
            lq = batch['lq'].to(device) * 2 - 1
            latent = encode_dispatch(vae, lq, vae_tiling(cfg)[1], device)[0]
            weights = token_weights(batch['text_mask'], batch['small_text_mask'],
                                    tokens=(latent.shape[-1] // model.patch_size) ** 2,
                                    mode=args.weighting, strata_weights=args.strata_weights,
                                    text_weight=args.text_weight, valid_mask=batch.get('valid_mask'))
            sampled.append(dict(crop_seed=crop_seed, noise_seed=noise_seed,
                                sample_name=batch.get('sample_name'), source_id=batch.get('source_id')))
            yield torch.cat((latent, seeded_noise_like(latent, noise_seed)), 1), dino_features(venc, lq, dino), weights
            print(f'calibrated {i + 1}/{args.samples}', flush=True)

    stats = spatial_statistics(model, samples(), args.factor, args.upsample, args.chunk_hidden)
    state = model.state_dict()
    fingerprints = [mlp_fingerprint(state, f'blocks.{i}.mlp.') for i in range(len(model.blocks))]
    provenance = dict(version=1, source=checkpoint_provenance(cfg.get('student_checkpoint') or cfg['teacher_checkpoint']),
                      source_model=model.export_config, source_mlp_sha256=fingerprints,
                      data=paired_data_provenance(dataset, data), options=vars(args), samples=sampled,
                      dino=dino, vae_path=str(cfg['vae_path']), endpoint=dict(t=1, r=0),
                      torch_version=str(torch.__version__), device=str(device))
    output.mkdir(parents=True, exist_ok=False)
    save_file(stats, str(output / 'scores.safetensors'))
    (output / 'calibration.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    for full, coarse in TIERS:
        groups = select_groups(stats, full, coarse, factor=args.factor, upsample=args.upsample,
                               fingerprints=fingerprints)
        path = output / f'groups_f{full}_c{coarse}.json'
        path.write_text(json.dumps(groups, indent=2), encoding='utf-8')
        print(f'wrote {path}', flush=True)


if __name__ == '__main__':
    main()
