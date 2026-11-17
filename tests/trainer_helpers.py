"""Shared tiny real-backbone trainer setup."""
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from models.lightningdit_ablation import AblationLightningDiT

ROOT = Path(__file__).resolve().parents[1]

MODEL = dict(patch_size=1, in_channels=6, out_channels=3, hidden_size=32, depth=4, num_heads=2, mlp_ratio=2,
             z_dims=6, encdim_ratio=2, auxiliary_time_cond='auto', use_qknorm=True, use_swiglu=True,
             use_rope=True, use_rmsnorm=True)


def trainer_config(tmp_path, student, **training):
    import yaml
    from PIL import Image
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (80, 80, 3), dtype=np.uint8)).save(tmp_path / 'hq.png')
    (tmp_path / 'images.txt').write_text(str(tmp_path / 'hq.png') + '\n')
    (tmp_path / 'datasets.txt').write_text(str(tmp_path / 'images.txt') + ', 2\n')
    torch.manual_seed(0)
    teacher = AblationLightningDiT(**dict(MODEL, auxiliary_time_cond=True), input_size=8)
    torch.nn.init.normal_(teacher.final_layer.linear.weight, std=.02)
    torch.save(teacher.state_dict(), tmp_path / 'teacher.pt')
    cfg = {'teacher_checkpoint': str(tmp_path / 'teacher.pt'), 'vae_path': 'fixture', 'model': MODEL,
           'dino': {'size': 56, 'layer': 0},  # fixture DINO: (size // 14)^2 tokens
           'data': {'resolution': 64, 'dataset_type': 'txt', 'train_dataset_config': str(tmp_path / 'datasets.txt')},
           'student': student,
           'training': dict(output_dir=str(tmp_path / 'run'), precision='fp32', report_to='none',
                            batch_size_per_gpu=1, gradient_accumulation_steps=2, max_steps=3, learning_rate=1e-3,
                            save_every=2, log_every=1, preview_every=2, preview_num_images=1, num_workers=0,
                            zero_optimizer=False, **training)}
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump(cfg))
    return path, teacher.state_dict()


def run_trainer(config_path, world=1, resume=None):
    command = [sys.executable]
    if world > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}']
    command += ['tests/ablation_model_smoke.py', '--config', str(config_path)]
    if resume:
        command += ['--resume', str(resume)]
    env = dict(os.environ, OMP_NUM_THREADS='1', WANDB_MODE='offline', TORCHDYNAMO_DISABLE='1')
    run = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
