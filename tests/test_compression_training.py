import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from ablation_utils import load_export
from models.lightningdit_ablation import AblationLightningDiT


@pytest.mark.parametrize('world', [1, 2])
def test_real_trainer_checkpoint_feature_kd_and_resume(tmp_path, world):
    if world == 2 and os.environ.get('VOSR_TEST_DDP') != '1':
        pytest.skip('Enable VOSR_TEST_DDP=1 for local Gloo smoke')
    kwargs = dict(patch_size=2, in_channels=4, out_channels=2, hidden_size=64, depth=4,
                  num_heads=4, mlp_ratio=2, z_dims=16, use_rope=True, auxiliary_time_cond=True)
    teacher = AblationLightningDiT(input_size=8, **kwargs)
    torch.nn.init.normal_(teacher.final_layer.linear.weight, std=.02)
    torch.save(teacher.state_dict(), tmp_path/'teacher.pt')
    output = tmp_path/'run'
    cfg = dict(teacher_checkpoint=str(tmp_path/'teacher.pt'), vae_path='synthetic', model=kwargs, dino={},
               data=dict(resolution=64), student=dict(use_cross_attention=True, compression_config=dict(
                   enabled=True, factor=2, start_block=1, end_block=3)), training=dict(
                   output_dir=str(output), precision='fp32', seed=42, batch_size_per_gpu=1,
                   gradient_accumulation_steps=2, max_steps=3, learning_rate=1e-4,
                   save_every=2, log_every=1, preview_every=0, num_workers=0,
                   gradient_checkpointing=True, zero_optimizer=False, report_to='none',
                   dense_warmup_steps=1, dense_distill_weight=.1, feature_distill_weight=.1,
                   feature_distill_layers=[1, 3], detail_distill_weight=.1, gt_weight=.1))
    config_path = tmp_path/'config.yml'
    config_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    command = [sys.executable]
    if world > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}']
    command += ['tests/compression_trainer_smoke.py', '--config', str(config_path)]
    env = dict(os.environ, OMP_NUM_THREADS='1', TORCHDYNAMO_DISABLE='1', WANDB_MODE='offline')
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    metrics = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in metrics] == [1, 2, 3]
    assert metrics[0]['attention_pair_ratio'] == 1
    assert all(r['attention_pair_ratio'] == .53125 and r['feature_kd'] > 0 and r['detail_kd'] > 0 for r in metrics[1:])
    assert all(r['grad_norm'] > 0 for r in metrics)
    net = load_export(output/'export')
    assert net.compression_config['enabled']
    run = subprocess.run(command + ['--resume', str(output/'checkpoint-00000002')], cwd=root,
                         env=env, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
