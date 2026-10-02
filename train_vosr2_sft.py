"""Continue the original one-step VOSR2 1.4B on controlled synthetic data.

All original cross-attention modules remain trainable. There is no router or
architecture ablation. Frozen Qwen VAE/DINO and endpoint t=1,r=0 are retained.
Supervision is clean-target latent MSE with optional frozen-initial-model MSE.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ablation_utils import read_weights, dino_features
from data_pipeline.pipeline import atomic_json, file_digest
from experiments.training_data import ResearchDataset


def make_model(args, gradient_checkpointing=False):
    from models.lightningdit import LightningDiT
    expected = dict(dim=1536, depth=36, num_heads=24, ae_type='qwen', patch_size=2,
                    auxiliary_time_cond=False, distill_type='onestep')
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f'Expected original VOSR2 1.4B {key}={value}')
    return LightningDiT(input_size=args['resolution']//8, patch_size=args['patch_size'],
                        in_channels=32, out_channels=16, hidden_size=args['dim'], depth=args['depth'],
                        num_heads=args['num_heads'], mlp_ratio=args['mlp_ratio'],
                        z_dims=args['enc_dim'], encdim_ratio=args['encdim_ratio'],
                        num_fused_layers=len(args['layer_dinov2b_list']), auxiliary_time_cond=False,
                        use_qknorm=args['use_qknorm'], use_swiglu=args['use_swiglu'],
                        use_rope=args['use_rope'], use_rmsnorm=args['use_rmsnorm'],
                        wo_shift=args['wo_shift'], use_checkpoint=gradient_checkpointing)


def save_model(student, directory, model_args):
    from safetensors.torch import save_file
    directory = Path(directory)
    (directory / 'checkpoints').mkdir(parents=True, exist_ok=True)
    temp = directory / 'checkpoints/model.safetensors.tmp'
    save_file({k: v.detach().cpu().contiguous() for k, v in student.state_dict().items()}, str(temp))
    temp.replace(directory / 'checkpoints/model.safetensors')
    atomic_json(directory / 'args.json', model_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Save a resumable checkpoint after this step for a smoke run')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    tc, dc = cfg['training'], cfg['data']
    if not torch.cuda.is_available():
        raise RuntimeError('Real 1.4B training requires CUDA')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This experiment runner targets the available single GPU')
    torch.set_num_threads(int(tc.get('cpu_threads', 4)))
    torch.backends.cuda.matmul.allow_tf32 = True
    seed = tc['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = Path(tc['output_dir'])
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Use a fresh output directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    # An OS lock prevents two writers even if a session was reattached.
    import fcntl
    lock = (out / '.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial_checkpoint = Path(cfg['checkpoint'])
    model_args = json.loads((initial_checkpoint / 'args.json').read_text())
    if dc['resolution'] != model_args['resolution']:
        raise ValueError('This study uses the released model configuration resolution of 512 pixels')
    model_args['ae_path'] = str(Path(cfg['vae_path']).resolve())
    provenance = dict(config=cfg, manifest_sha256=file_digest(dc['manifest']),
                      initial_weight_sha256=file_digest(initial_checkpoint / 'checkpoints/ema_model.safetensors'),
                      torch_version=str(torch.__version__), parameter_update='all_original_DiT_parameters',
                      code_sha256={p: file_digest(p) for p in [__file__, 'experiments/training_data.py',
                                   'models/lightningdit.py', 'data_pipeline/augmentation.py']})
    state_path = out / 'latest' / 'training_state.pt'
    start = 0
    saved = None
    actual_sources = Counter()
    if args.resume:
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        if saved['provenance'] != provenance:
            raise ValueError('Resume requires unchanged data, configuration and training code')
        start = saved['step']
        actual_sources.update(saved.get('source_counts', {}))
    else:
        atomic_json(out / 'provenance.json', provenance)
    batch_size, accumulation, steps = tc['batch_size'], tc['accumulation'], tc['max_steps']
    dataset = ResearchDataset(dc, steps*batch_size*accumulation, seed, start*batch_size*accumulation)
    atomic_json(out / 'data_summary.json', dataset.summary)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=tc.get('workers', 2),
                        pin_memory=True, persistent_workers=tc.get('workers', 2) > 0)
    weights = read_weights(out / 'latest' if args.resume else initial_checkpoint)
    student = make_model(model_args, tc['gradient_checkpointing'])
    student.load_state_dict(weights, strict=True)
    del weights
    student.cuda().train()
    parameter_count = sum(p.numel() for p in student.parameters())
    if parameter_count != 1393943616 or any('router' in n for n, _ in student.named_parameters()):
        raise ValueError('Model is not the original dense VOSR2 1.4B')
    if not all(hasattr(b, 'cross_attn') for b in student.blocks):
        raise ValueError('Cross-attention must be retained in all 36 blocks')
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    kd_weight = tc.get('teacher_weight', 0.1)
    teacher = None
    if kd_weight:
        teacher = make_model(model_args)
        teacher.load_state_dict(read_weights(initial_checkpoint), strict=True)
        teacher.to(device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    from tiled_vae import encode_latent, decode_latent
    vae = AutoencoderKLQwenImage2D.from_pretrained(cfg['vae_path']).cuda().eval().requires_grad_(False)
    torch.hub.set_dir('preset/ckpts/torch_cache')
    dino = torch.hub.load('preset/ckpts/torch_cache/facebookresearch_dinov2_main',
                          'dinov2_vitl14', source='local').cuda().eval().requires_grad_(False)
    dino_config = dict(size=model_args['dinov2_size'], layer=model_args['layer_dinov2b_list'][0])
    ae_args = SimpleNamespace(ae_type='qwen')
    optimizer = torch.optim.AdamW(student.parameters(), lr=tc['learning_rate'],
                                 weight_decay=tc['weight_decay'], betas=(0.9, 0.999), fused=True)
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        del saved
    iterator = iter(loader)
    watch_names = ['x_embedder.proj.weight', 'blocks.0.cross_attn.q_linear.weight']
    parameters = dict(student.named_parameters())
    initial_watches = {name: parameters[name].detach().cpu().clone() for name in watch_names}
    print(json.dumps(dict(event='training_started', start_step=start, parameters=parameter_count,
                          cross_attention_blocks=36, router_count=0, source_sampling=dataset.summary)), flush=True)
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    end = min(steps, args.stop_after) if args.stop_after else steps
    if end <= start:
        raise ValueError('Training end must be after resume step')
    for step in range(start, end):
        step_start = time.monotonic()
        # Noise and data order are reproducible across checkpoint/resume.
        torch.manual_seed(seed + step)
        lr = tc['learning_rate'] * min((step+1) / max(1, tc['warmup_steps']), 1)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        total_gt, total_kd = 0.0, 0.0
        for micro in range(accumulation):
            batch = next(iterator)
            actual_sources.update(batch['source'])
            hq, lq = batch['hq'].cuda(non_blocking=True), batch['lq'].cuda(non_blocking=True)
            with torch.no_grad(), amp():
                clean, mean, std = encode_latent(vae, hq, ae_args, 'cuda', posterior_mode=True)
                low = encode_latent(vae, lq, ae_args, 'cuda', posterior_mode=True)[0]
                features = dino_features(dino, lq, dino_config)
                noise = torch.randn_like(low)
                inp = torch.cat((low, noise), 1)
                t, r = low.new_ones(len(low)), low.new_zeros(len(low))
                target = teacher(inp, t, r, features).float() if teacher is not None else None
            with amp():
                prediction = student(inp, t, r, features)
            gt_loss = F.mse_loss(noise.float() - prediction.float(), clean.float())
            kd = F.mse_loss(prediction.float(), target) if target is not None else gt_loss.new_zeros(())
            loss = gt_loss + kd_weight*kd
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            (loss/accumulation).backward()
            total_gt += gt_loss.item()/accumulation
            total_kd += kd.item()/accumulation
        if step == start:
            missing = [n for n, p in student.named_parameters() if p.grad is None]
            if missing:
                raise RuntimeError(f'Original trainable weights have no gradient: {missing[:10]}')
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), tc['max_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        record = dict(step=step+1, loss=total_gt + kd_weight*total_kd, gt=total_gt, kd=total_kd,
                      grad_norm=grad_norm.item(), learning_rate=lr, seconds=time.monotonic()-step_start,
                      peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                      sampled_sources=dict(actual_sources))
        with (out / 'metrics.jsonl').open('a') as log:
            log.write(json.dumps(record) + '\n')
        if step == start or (step+1) % tc['log_every'] == 0:
            print(json.dumps(record), flush=True)
        if (step+1) % tc['preview_every'] == 0 or step == start or step+1 == end:
            with torch.no_grad(), amp():
                sr = decode_latent(vae, noise[:1]-prediction[:1].float(), ae_args, mean, std)
            from torchvision.utils import save_image
            preview = torch.cat((lq[:1], sr, hq[:1]), dim=3)
            (out / 'previews').mkdir(exist_ok=True)
            save_image(preview.float().cpu().add(1).div(2).clamp(0, 1), out / 'previews' / f'{step+1:06}.png')
        if (step+1) % tc['save_every'] == 0 or step+1 == end:
            # Stage model and optimizer together, then atomically change the
            # symlink. A crash cannot pair new weights with old optimizer state.
            checkpoint = out / f'checkpoint-{step+1:08}'
            save_model(student, checkpoint, model_args)
            torch.save(dict(step=step+1, optimizer=optimizer.state_dict(), provenance=provenance,
                            source_counts=dict(actual_sources)),
                       checkpoint / 'training_state.pt')
            latest = out / 'latest'
            previous = latest.resolve() if latest.is_symlink() else None
            temporary = out / 'latest.tmp'
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(checkpoint.name)
            temporary.replace(latest)
            if previous and previous.parent == out.resolve() and previous != checkpoint.resolve():
                shutil.rmtree(previous)
            print(json.dumps(dict(event='checkpoint', step=step+1, path=str(checkpoint))), flush=True)
    delta = {name: (parameters[name].detach().cpu()-value).norm().item() for name, value in initial_watches.items()}
    atomic_json(out / 'weight_update.json', dict(start_step=start, end_step=end, l2_change=delta))
    if end == steps:
        # Inference export shares the completed checkpoint weights, without a
        # second 5.6GB copy. Optimizer retention is configurable for long studies.
        export = out / 'export'
        if not export.exists():
            export.symlink_to((out / 'latest').resolve().name, target_is_directory=True)
        if not tc.get('retain_final_optimizer', True):
            (out / 'latest' / 'training_state.pt').unlink()
        atomic_json(out / 'complete.json', dict(steps=end, parameters=parameter_count,
                    cross_attention_blocks=36, router_count=0, sampled_sources=dict(actual_sources), weight_change=delta))


if __name__ == '__main__':
    main()
