"""One-step VOSR2 teacher-preserving SFT, not the original FM/RCGM recipe.

Run with torchrun (DDP), or python for one device. Only bf16/fp32 are supported.
The teacher and student see identical LQ latents/noise at t=1, r=0.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ablation_utils import (build_txt_dataset, prepare_training_batch, load_config, read_weights,
                            load_backbone_state, save_export, load_dino, dino_features)
from models.sdt_router import linear_schedule
from ablation_logging import TrainingLogger, build_preview_samples, render_previews, resume_configs_match, logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', help='A checkpoint directory written by this trainer')
    args = parser.parse_args()
    cfg = load_config(args.config)
    tc = cfg['training']
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    precision = tc.get('precision', 'bf16')
    if precision not in ('bf16', 'fp32'):
        raise ValueError('Only bf16/fp32 supported (fp16 needs gradient scaling)')
    amp = precision == 'bf16'
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp)
    seed = int(tc.get('seed', 42))
    torch.manual_seed(seed)
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2 ** 32))
    resolution = int(cfg['data']['resolution'])
    patch = int(cfg['model']['patch_size'])
    if resolution % (8 * patch):
        raise ValueError('resolution must be divisible by 8 * patch_size')
    dataset = build_txt_dataset(cfg['data'])
    batch_size, accumulation = int(tc['batch_size_per_gpu']), int(tc['gradient_accumulation_steps'])
    steps = int(tc['max_steps'])
    preview_every = int(tc.get('preview_every', 500))
    preview_count = int(tc.get('preview_num_images', 2))
    if preview_every < 0 or (preview_every and preview_count <= 0):
        raise ValueError('preview_every must be nonnegative and enabled previews require preview_num_images > 0')
    if min(batch_size, accumulation, steps, int(tc['save_every']), int(tc.get('log_every', 10))) <= 0:
        raise ValueError('batch size, accumulation, max_steps and save_every must be positive')
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True,
                        num_workers=int(tc.get('num_workers', 4)), pin_memory=device.type == 'cuda')
    if not len(loader):
        raise ValueError('Dataset is too small for world_size * batch_size_per_gpu')
    out = Path(tc['output_dir'])
    occupied = torch.tensor(int(rank == 0 and not args.resume and out.exists() and any(out.iterdir())), device=device)
    if world > 1:
        dist.broadcast(occupied, src=0)
    if occupied.item():
        raise ValueError('Output directory is not empty; use a fresh directory or --resume')
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    if world > 1:
        dist.barrier()

    # Import heavy upstream dependencies only after CLI/config/data validation.
    from models.lightningdit_ablation import AblationLightningDiT
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    from tiled_vae import encode_latent
    from types import SimpleNamespace
    from dataloaders.realesrgan_gpu import RealESRGAN_degradation
    degradation = RealESRGAN_degradation('params_realsr.yml', device=device)
    state = read_weights(cfg['teacher_checkpoint'])
    model_cfg = dict(cfg['model'])
    aux = model_cfg.pop('auxiliary_time_cond', 'auto')
    model_cfg['auxiliary_time_cond'] = any(k.startswith('r_embedder.') for k in state) if aux == 'auto' else bool(aux)
    model_cfg['input_size'] = resolution // 8
    teacher = AblationLightningDiT(**model_cfg, use_cross_attention=True)
    report = load_backbone_state(teacher, state)
    teacher = teacher.to(device=device, dtype=torch.bfloat16 if amp else torch.float32).eval().requires_grad_(False)
    student_cfg = cfg['student']
    student = AblationLightningDiT(**model_cfg,
                                  use_cross_attention=student_cfg['use_cross_attention'],
                                  router_config=student_cfg.get('router_config'),
                                  use_checkpoint=tc.get('gradient_checkpointing', True))
    initial = state if not cfg.get('student_checkpoint') else read_weights(cfg['student_checkpoint'])
    if args.resume:
        initial = read_weights(Path(args.resume) / 'model.safetensors')
    report_student = load_backbone_state(student, initial, allow_new_router=not bool(args.resume))
    del state, initial
    fade = student_cfg.get('ca_fade_steps', 0)
    if fade and not student.use_cross_attention:
        raise ValueError('CA fading requires a CA branch during training')
    if fade and steps < fade:
        raise ValueError('max_steps must finish the CA fade')
    if fade:
        # These frozen weights may become unused after scale reaches zero.
        student.freeze_conditioning()
    student.to(device).train()
    vae = AutoencoderKLQwenImage2D.from_pretrained(cfg['vae_path']).to(device).eval().requires_grad_(False)
    venc = load_dino(cfg['dino'], device)  # Frozen, full teacher always uses DINO.
    ae_args = SimpleNamespace(ae_type='qwen')
    pipeline = dict(vae_path=cfg['vae_path'], dino=cfg['dino'], resolution=resolution,
                    upscale=int(degradation.opt['scale']), precision=precision)
    base, routers = [], []
    for name, parameter in student.named_parameters():
        if parameter.requires_grad:
            (routers if '.router.' in name else base).append(parameter)
    lr = float(tc['learning_rate'])
    groups = [{'params': base, 'lr': lr, 'lr_scale': 1.0}]
    if routers:
        scale = float(tc.get('router_lr_multiplier', 10.0))
        groups.append({'params': routers, 'lr': lr * scale, 'lr_scale': scale})
    opt_kwargs = dict(lr=lr, betas=(0.9, 0.95), weight_decay=float(tc.get('weight_decay', 0.01)))
    zero = world > 1 and tc.get('zero_optimizer', True)
    if zero:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW, **opt_kwargs)
    else:
        optimizer = torch.optim.AdamW(groups, **opt_kwargs)
    step = 0
    if args.resume:
        saved = torch.load(Path(args.resume) / 'training_state.pt', map_location='cpu', weights_only=True)
        if not resume_configs_match(saved['config'], cfg):
            raise ValueError('Resume requires the same training configuration (logging/preview options may change)')
        optimizer.load_state_dict(saved['optimizer'])
        step = int(saved['step'])
    tracking = TrainingLogger(cfg, device, start_step=step)
    model = DDP(student, device_ids=[local_rank] if device.type == 'cuda' else None,
                broadcast_buffers=False) if world > 1 else student
    torch.manual_seed(seed + rank + step)
    epoch = 0
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    if rank == 0:
        logger.info(json.dumps({'teacher': report, 'student': report_student,
                          'effective_batch': world * batch_size * accumulation, 'zero_optimizer': zero}))
    routing = student_cfg.get('router_config') is not None
    dense_steps = int(tc.get('dense_warmup_steps', 0)) if routing else 0
    dense_weight = float(tc.get('dense_distill_weight', 0)) if routing else 0
    gt_weight = float(tc.get('gt_weight', 0))

    preview_samples = None

    try:
        while step < steps:
            student.set_ca_scale(linear_schedule(step, 1, 0, fade) if fade else
                                 (1.0 if student.use_cross_attention else 0.0))
            target_keep = linear_schedule(max(0, step - dense_steps), 0.99,
                                          tc.get('target_keep_ratio', 0.75), tc.get('budget_warmup_steps', 4500))
            force_dense = step < dense_steps
            lr_factor = min((step + 1) / max(1, int(tc.get('lr_warmup_steps', 100))), 1.0)
            for group in optimizer.param_groups:
                group['lr'] = lr * group['lr_scale'] * lr_factor
            optimizer.zero_grad(set_to_none=True)
            metrics = torch.zeros(6, device=device)
            for micro in range(accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                hq, lq = prepare_training_batch(batch, degradation, device)
                with torch.no_grad():
                    lq_latent, _, _ = encode_latent(vae, lq, ae_args, device, posterior_mode=True)
                    gt = encode_latent(vae, hq, ae_args, device, posterior_mode=True)[0] if gt_weight else None
                    with autocast():
                        features = dino_features(venc, lq, cfg['dino'])
                    noise = torch.randn_like(lq_latent)
                    inp = torch.cat((lq_latent, noise), 1)
                    t, r = lq_latent.new_ones(lq.shape[0]), lq_latent.new_zeros(lq.shape[0])
                    with autocast():
                        target = teacher(inp, t, r, features).float()
                sync = model.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
                with sync:
                    with autocast():
                        result = model(inp, t, r, features, return_stats=True, force_dense=force_dense,
                                       include_dense=dense_weight > 0 and not force_dense)
                        prediction, stats = result[:2]
                    kd = F.mse_loss(prediction.float(), target)
                    dense_kd = F.mse_loss(result[2].float(), target) if len(result) == 3 else kd.new_zeros(())
                    gt_loss = F.mse_loss(noise - prediction.float(), gt.float()) if gt_weight else kd.new_zeros(())
                    loss = kd + dense_weight * dense_kd + gt_weight * gt_loss
                    budget = ((stats['keep_probabilities'] - target_keep) ** 2).mean() if routing else kd.new_zeros(())
                    loss = loss + float(tc.get('budget_weight', 0.1)) * budget
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite loss; stop instead of saving corrupt weights')
                    (loss / accumulation).backward()
                metrics += torch.stack((loss.detach(), kd.detach(), budget.detach(), stats['keep_fraction'].detach(),
                                        dense_kd.detach(), gt_loss.detach())) / accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float(tc.get('max_grad_norm', 1.0)), error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if world > 1:
                dist.all_reduce(metrics)
                metrics /= world
            preview_due = preview_every > 0 and (step == 1 or step % preview_every == 0 or step == steps)
            log_due = step == 1 or step % int(tc.get('log_every', 10)) == 0 or step == steps
            # All ranks participate in the status broadcast, but only rank 0 runs the
            # unwrapped student. No DDP forward/collectives occur inside previews.
            logging_error = [None]
            if rank == 0 and (log_due or preview_due):
                try:
                    record = dict(step=step, loss=metrics[0].item(), kd=metrics[1].item(), budget=metrics[2].item(),
                                  stochastic_mlp_keep=metrics[3].item(), dense_kd=metrics[4].item(), gt=metrics[5].item(),
                                  target_keep=target_keep, ca_scale=student.ca_scale,
                                  learning_rate=optimizer.param_groups[0]['lr'], grad_norm=grad_norm.item())
                    if routers:
                        record['router_learning_rate'] = optimizer.param_groups[1]['lr']
                    images = None
                    if preview_due:
                        if preview_samples is None:
                            preview_samples = build_preview_samples(dataset, degradation, device, preview_count,
                                                                    int(tc.get('preview_seed', 1234)))
                        images, preview_metrics = render_previews(student, teacher, vae, venc, preview_samples, cfg, device)
                        record.update(preview_metrics)
                    tracking.log(record, images)
                except Exception as error:
                    logger.exception('Training logging/preview failed')
                    logging_error[0] = f'{type(error).__name__}: {error}'
            if world > 1 and (log_due or preview_due):
                dist.broadcast_object_list(logging_error, src=0)
            if logging_error[0] is not None:
                raise RuntimeError(f'Training logging/preview failed: {logging_error[0]}')
            if step % int(tc['save_every']) == 0 or step == steps:
                if fade:
                    student.set_ca_scale(linear_schedule(step, 1, 0, fade))
                if zero:
                    optimizer.consolidate_state_dict(to=0)
                if rank == 0:
                    checkpoint_dir = out / f'checkpoint-{step:08d}'
                    save_export(student, checkpoint_dir, pipeline)
                    torch.save({'step': step, 'optimizer': optimizer.state_dict(), 'config': cfg},
                               checkpoint_dir / 'training_state.pt')
                if world > 1:
                    dist.barrier()
        if rank == 0:
            save_export(student, out / 'export', pipeline, strip_conditioning=student.ca_scale == 0)
            logger.info(f'Exported inference model: {out / "export"}')
    finally:
        tracking.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
