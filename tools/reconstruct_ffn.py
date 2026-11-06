"""Teacher-input nonlinear reconstruction of physically narrower SwiGLU FFNs.

Example (one device):
    python tools/reconstruct_ffn.py --config configs/experiment.yml --output runs/local --device auto
Example (effective batch 16 with 8 ranks, microbatch 1, accumulation 2):
    torchrun --standalone --nproc_per_node=8 tools/reconstruct_ffn.py --config configs/experiment.yml --output runs/local

Every selected layer receives every optimizer step, training BOTH w12 and w3 by
default. The local teacher is the full-width source used for pruning (Dense by
default, or the configured full-width head-split source). End-to-end training
can subsequently use the separate dense teacher_checkpoint as usual.

Output/export and checkpoint-*/ are normal save_export layouts, accepted by
train_vosr_ablation.py --init-from. Reconstruction checkpoints also contain an
optimizer/RNG/step state and checkpoint.complete for --resume. A new run never
overwrites an existing output directory; resume verifies source/data/config.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_utils import (build_train_dataset, dino_features, load_backbone_state, load_config, load_dino,
                            load_export, save_export, student_dino_config, vae_tiling)  # noqa: E402
from models.ffn_reconstruction import (WEIGHTING_MODES, checkpoint_provenance, paired_data_provenance,
                                      reconstruction_microstep, sample_paired_batch, seeded_noise_like,
                                      select_reconstruction_parameters, token_weights)  # noqa: E402
from tools.mlp_neuron_stats import source_model  # noqa: E402
from training_runtime import (atomic_torch_save, capture_rng, distributed_backend, restore_rng,
                              select_device)  # noqa: E402


RECONSTRUCTION_DEFAULTS = dict(max_steps=2000, batch_size_per_gpu=1, gradient_accumulation_steps=2,
    learning_rate=1e-4, weight_decay=0.0, lr_warmup_steps=100, lr_schedule='cosine', min_lr_ratio=0.1,
    precision='bf16', max_grad_norm=1.0, save_every=500, log_every=10, layers=None, seed=42,
    weighting='balanced', text_weight=4.0, small_text_height=64.0, strata_weights=[0.5, 0.3, 0.2],
    energy_floor=1e-4, trainable_parameters=None, gated=True)


def reconstruction_options(cfg):
    supplied = cfg.get('reconstruction') or {}
    if not isinstance(supplied, dict) or set(supplied) - set(RECONSTRUCTION_DEFAULTS):
        raise ValueError(f'reconstruction accepts only {sorted(RECONSTRUCTION_DEFAULTS)}')
    options = dict(RECONSTRUCTION_DEFAULTS, **supplied)
    for key in ('max_steps', 'batch_size_per_gpu', 'gradient_accumulation_steps', 'save_every', 'log_every'):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] <= 0:
            raise ValueError(f'reconstruction.{key} must be a positive integer')
    for key in ('seed', 'lr_warmup_steps'):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 0:
            raise ValueError(f'reconstruction.{key} must be a nonnegative integer')
    for key in ('learning_rate', 'energy_floor', 'small_text_height', 'max_grad_norm'):
        if isinstance(options[key], bool) or not isinstance(options[key], (float, int)) \
                or not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f'reconstruction.{key} must be finite and positive')
    if not isinstance(options['weight_decay'], (float, int)) or not math.isfinite(options['weight_decay']) \
            or options['weight_decay'] < 0:
        raise ValueError('reconstruction.weight_decay must be finite and nonnegative')
    if options['precision'] not in ('bf16', 'fp32'):
        raise ValueError('reconstruction.precision must be bf16 or fp32')
    if options['weighting'] not in WEIGHTING_MODES:
        raise ValueError(f'reconstruction.weighting must be one of {WEIGHTING_MODES}')
    if options['lr_schedule'] not in ('constant', 'cosine') or not 0 <= options['min_lr_ratio'] <= 1:
        raise ValueError('Use lr_schedule constant|cosine and min_lr_ratio in [0,1]')
    if not isinstance(options['gated'], bool):
        raise ValueError('reconstruction.gated must be a boolean')
    return options


def synchronize_gradients(parameters, world, bucket_bytes=32 * 1024 * 1024):
    """Average independent local-MLP gradients, in bounded communication buckets.

    Hooks run inside a no-grad teacher, so wrapping the full student in DDP
    would not provide a matching DDP forward. Explicit reduction is used here.
    All ranks execute the same ordered parameter list and collectives.
    """
    if not parameters:
        raise ValueError('No reconstruction parameters')
    if all(p.grad is not None for p in parameters):
        flag = torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all().to(torch.int32)
    else:
        flag = torch.tensor(0, device=parameters[0].device, dtype=torch.int32)
    if world > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not int(flag):
        raise FloatingPointError('Missing or non-finite reconstruction gradient on at least one rank')
    if world == 1:
        return
    # Split large linear matrices too, keeping the temporary below the bucket
    # bound. Gradients are contiguous for dense linear weights.
    bucket, elements = [], 0
    limit = max(1, bucket_bytes // parameters[0].element_size())

    def reduce_bucket():
        nonlocal bucket, elements
        flat = torch.cat(bucket)
        dist.all_reduce(flat)
        flat.div_(world)
        offset = 0
        for view in bucket:
            view.copy_(flat[offset:offset + view.numel()])
            offset += view.numel()
        bucket, elements = [], 0

    for parameter in parameters:
        gradient = parameter.grad.view(-1)
        for start in range(0, gradient.numel(), limit):
            view = gradient[start:start + limit]
            if elements and elements + view.numel() > limit:
                reduce_bucket()
            bucket.append(view)
            elements += view.numel()
    if bucket:
        reduce_bucket()


def learning_rate_at(options, step):
    warmup = options['lr_warmup_steps']
    fraction = min((step + 1) / max(1, warmup), 1.0)
    if options['lr_schedule'] == 'cosine' and step + 1 > warmup:
        progress = (step + 1 - warmup) / max(1, options['max_steps'] - warmup)
        fraction = options['min_lr_ratio'] + (1 - options['min_lr_ratio']) * 0.5 * (1 + math.cos(math.pi * progress))
    return options['learning_rate'] * fraction


def prepare_output(output, resume, rank, world):
    error = [None]
    if rank == 0:
        try:
            if resume is None and output.exists() and any(output.iterdir()):
                raise FileExistsError(f'{output} is not empty; choose a new --output or use --resume')
            if resume is not None and (output / 'export').exists():
                raise FileExistsError(f'{output} already has an export; resume into a fresh --output to avoid stale artifacts')
            if resume is not None and output.exists() and any(output.iterdir()) \
                    and output.resolve() != resume.parent.resolve():
                raise FileExistsError('A resumed output must be empty or the directory containing its checkpoint')
            output.mkdir(parents=True, exist_ok=True)
            if resume is not None and not (resume / 'checkpoint.complete').is_file():
                raise ValueError(f'{resume} is not a complete reconstruction checkpoint')
        except (OSError, ValueError) as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', help='A completed reconstruction checkpoint directory')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    cfg = load_config(args.config)
    options = reconstruction_options(cfg)
    world, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    device = select_device(args.device, int(os.environ.get('LOCAL_RANK', 0)))
    if world > 1:
        dist.init_process_group(backend=distributed_backend(device))
    try:
        run(cfg, options, Path(args.output), Path(args.resume) if args.resume else None, device, rank, world)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def run(cfg, options, output, resume, device, rank=0, world=1):
    prepare_output(output, resume, rank, world)
    seed = options['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    data = copy.deepcopy(cfg['data'])
    if data.get('dataset_type') != 'paired':
        raise ValueError('Local FFN reconstruction requires aligned paired data (data.sources is supported)')
    data.update(resolution=int(data.get('crop_size') or data['resolution']), text_metadata=True,
                small_text_height=options['small_text_height'])
    if data['resolution'] % (8 * int(cfg['model']['patch_size'])):
        raise ValueError('Reconstruction crop must be divisible by 8 * patch_size')
    if cfg['student'].get('noise_input', 'gaussian') != 'gaussian':
        raise ValueError('Local reconstruction uses the source model at Gaussian noise, t=1,r=0')
    compression = cfg['student'].get('mlp_compression')
    if not compression or compression.get('fade_steps', 0) or compression.get('fade_scale') is not None:
        raise ValueError('Reconstruction requires physical mlp_compression with fade_steps: 0')
    dataset = build_train_dataset(data, text_mask=True)
    teacher = source_model(cfg).to(device).eval().requires_grad_(False)
    if resume is not None:
        student = load_export(resume, device=device)
    else:
        from models.lightningdit_ablation import AblationLightningDiT
        model_cfg = dict(teacher.export_config, mlp_compression=compression, use_checkpoint=False)
        student = AblationLightningDiT(**model_cfg)
        # Pruning stats operate on CPU source weights. Device-local source state
        # would otherwise conflict with CPU covariance/index calculations.
        source = {name: value.detach().cpu() for name, value in teacher.state_dict().items()}
        load_backbone_state(student, source, allow_new_modules=True)
        student = student.to(device)
        del source
    layers, parameters = select_reconstruction_parameters(student, options['layers'], options['trainable_parameters'])
    if any(student.blocks[index].mlp.w3.in_features > teacher.blocks[index].mlp.w3.in_features for index in layers):
        raise ValueError('Reconstruction student cannot be wider than its source teacher')
    student.train()
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    from tiled_vae import encode_dispatch
    vae = AutoencoderKLQwenImage2D.from_pretrained(cfg['vae_path']).to(device).eval().requires_grad_(False)
    venc = load_dino(cfg['dino'], device)
    dino = student_dino_config(cfg)
    tile, ae_args = vae_tiling(cfg)
    pipeline = dict(vae_path=cfg['vae_path'], dino=dino, resolution=data['resolution'], vae_tile_size=tile,
                    upscale=int(data['upscale']), precision=options['precision'])
    amp = options['precision'] == 'bf16'
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp)
    optimizer = torch.optim.AdamW(parameters, lr=options['learning_rate'], betas=(0.9, 0.95),
                                  weight_decay=options['weight_decay'])
    manifest = [None]
    if rank == 0:
        manifest[0] = dict(source=checkpoint_provenance(cfg.get('student_checkpoint') or cfg['teacher_checkpoint']),
                           data=paired_data_provenance(dataset, data), torch_version=str(torch.__version__))
    if world > 1:
        dist.broadcast_object_list(manifest, src=0)
    provenance = manifest[0]
    step, updates = 0, {index: 0 for index in layers}
    if resume is not None:
        state = torch.load(resume / 'reconstruction_state.pt', map_location='cpu', weights_only=True)
        if state['world_size'] != world or state['options'] != options or state['config'] != cfg:
            raise ValueError('Resume needs the same config, reconstruction options and world size')
        if state['provenance']['source']['fingerprint'] != provenance['source']['fingerprint'] \
                or state['provenance']['data']['fingerprint'] != provenance['data']['fingerprint']:
            raise ValueError('Resume source checkpoint or data manifest has changed')
        if state['layers'] != layers:
            raise ValueError('Resume selected layers do not match')
        optimizer.load_state_dict(state['optimizer'])
        step, updates = int(state['step']), state['updates']
        restore_rng(state['rng'][rank], device)
        if any(updates[index] != step for index in layers):
            raise ValueError('Checkpoint does not show one update per selected layer per optimizer step')
        later = [p for p in output.glob('checkpoint-*') if p.name.removeprefix('checkpoint-').isdigit()
                 and int(p.name.removeprefix('checkpoint-')) > step and (p / 'checkpoint.complete').is_file()]
        if later:
            raise ValueError('Output already has a later completed checkpoint; resume into a new --output')
    if rank == 0:
        (output / 'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(dict(device=str(device), world_size=world, selected_layers=layers,
                              effective_batch=world * options['batch_size_per_gpu'] * options['gradient_accumulation_steps'],
                              trainable_parameters=sum(p.numel() for p in parameters), source='full-width pruning source'),
                         ensure_ascii=False), flush=True)

    def checkpoint():
        states = [None] * world
        rng = capture_rng(device)
        if world > 1:
            dist.all_gather_object(states, rng)
        else:
            states[0] = rng
        error = [None]
        if rank == 0:
            try:
                directory = output / f'checkpoint-{step:08d}'
                if directory.exists():
                    raise FileExistsError(f'Refusing to overwrite checkpoint {directory}')
                temporary = Path(tempfile.mkdtemp(prefix=f'.checkpoint-{step:08d}.tmp-', dir=output))
                save_export(student, temporary, pipeline)
                atomic_torch_save(dict(step=step, optimizer=optimizer.state_dict(), options=options, config=cfg,
                                       world_size=world, layers=layers, updates=updates, rng=states,
                                       provenance=provenance), temporary / 'reconstruction_state.pt')
                (temporary / 'checkpoint.complete').write_text(json.dumps(dict(step=step, updates=updates)), encoding='utf-8')
                os.replace(temporary, directory)
            except Exception as exc:
                error[0] = f'{type(exc).__name__}: {exc}'
        if world > 1:
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f'Reconstruction checkpoint write failed: {error[0]}')

    while step < options['max_steps']:
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate_at(options, step)
        for group in optimizer.param_groups:
            group['lr'] = lr
        aggregate = torch.zeros(len(layers), dtype=torch.float32, device=device)
        accumulation = options['gradient_accumulation_steps']
        for micro in range(accumulation):
            stream = ((step * accumulation + micro) * world + rank)
            crop_seed = seed * 1_000_003 + stream
            noise_seed = seed * 10_000_019 + stream
            batch = sample_paired_batch(dataset, options['batch_size_per_gpu'], crop_seed)
            lq = batch['lq'].to(device) * 2 - 1
            with torch.no_grad():
                latent = encode_dispatch(vae, lq, ae_args, device)[0]
                inp = torch.cat((latent, seeded_noise_like(latent, noise_seed)), 1)
                with autocast():
                    features = dino_features(venc, lq, dino)
            tokens = (latent.shape[-2] // student.patch_size) * (latent.shape[-1] // student.patch_size)
            weights = token_weights(batch['text_mask'], batch['small_text_mask'], tokens=tokens,
                                     mode=options['weighting'], strata_weights=options['strata_weights'],
                                     text_weight=options['text_weight'], valid_mask=batch.get('valid_mask')).to(device)
            losses = reconstruction_microstep(teacher, student, inp, features, weights, layers=layers,
                        loss_scale=1 / accumulation, autocast=autocast, energy_floor=options['energy_floor'],
                        gated=options['gated'])
            aggregate += aggregate.new_tensor([losses[index] for index in layers]) / accumulation
        synchronize_gradients(parameters, world)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, options['max_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        updates = {index: value + 1 for index, value in updates.items()}
        if world > 1:
            dist.all_reduce(aggregate)
            aggregate /= world
        if rank == 0 and (step == 1 or step % options['log_every'] == 0 or step == options['max_steps']):
            record = dict(step=step, learning_rate=lr, loss=float(aggregate.mean()), grad_norm=float(grad_norm),
                          layer_loss={str(index): float(value) for index, value in zip(layers, aggregate)},
                          updates_per_layer={str(index): updates[index] for index in layers})
            with (output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        if step % options['save_every'] == 0 or step == options['max_steps']:
            checkpoint()
    error = [None]
    if rank == 0:
        try:
            export = output / 'export'
            if export.exists():
                raise FileExistsError(f'Refusing to reuse or overwrite existing export {export}')
            else:
                temporary = Path(tempfile.mkdtemp(prefix='.export.tmp-', dir=output))
                save_export(student, temporary, pipeline)
                (temporary / 'export.complete').write_text(json.dumps(dict(step=step, updates=updates)), encoding='utf-8')
                os.replace(temporary, export)
            print(f'Exported reconstructed student: {export}', flush=True)
        except Exception as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(f'Reconstruction export write failed: {error[0]}')


if __name__ == '__main__':
    main()
