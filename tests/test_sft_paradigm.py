"""SFT training-paradigm options: EMA, adversarial loss, LPIPS text mask, and their configs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml
from PIL import Image

from ablation_utils import load_config, read_weights
from models.discriminator import UNetDiscriminatorSN, adaptive_scale, discriminator_loss, gan_spec, generator_loss

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs/ablations'


def test_unet_discriminator_matches_real_esrgan_layout():
    net = UNetDiscriminatorSN(3, 64)
    state = net.state_dict()
    # conv0 and conv9 are plain convolutions with bias; conv1-8 carry spectral norm, no bias.
    assert state['conv0.weight'].shape == (64, 3, 3, 3) and 'conv0.bias' in state
    assert state['conv3.weight_orig'].shape == (512, 256, 4, 4) and 'conv3.bias' not in state
    assert state['conv4.weight_orig'].shape == (256, 512, 3, 3)
    assert state['conv9.weight'].shape == (1, 64, 3, 3)
    assert sum('weight_u' in k for k in state) == 8
    assert sum(p.numel() for p in net.parameters()) == 4_376_897
    x = torch.randn(2, 3, 32, 48, requires_grad=True)
    logits = net(x)
    assert logits.shape == (2, 1, 32, 48)
    generator_loss(logits).backward()
    assert x.grad.abs().sum() > 0


def test_gan_losses_and_adaptive_scale():
    real, fake = torch.full((1, 1, 4, 4), 8.), torch.full((1, 1, 4, 4), -8.)
    assert discriminator_loss(real, fake) < 1e-3  # a confident, correct discriminator
    assert generator_loss(fake) == pytest.approx(8., abs=1e-3)
    image = torch.randn(1, 3, 8, 8, requires_grad=True)
    reference = (image * 3).square().mean()
    adversarial = (image * 0.01).sum()
    scale = adaptive_scale(reference, adversarial, image, 0.5, 1e4)
    grad_ref, = torch.autograd.grad(reference, image, retain_graph=True)
    grad_adv, = torch.autograd.grad(scale * adversarial, image, retain_graph=True)
    assert grad_adv.norm() == pytest.approx(0.5 * grad_ref.norm().item(), rel=1e-5)
    assert not scale.requires_grad
    assert adaptive_scale(reference, image.sum() * 0, image, 0.5, 7.) == 7.  # vanishing gradient: capped


def test_gan_spec_validation():
    assert gan_spec(None) is None
    spec = gan_spec({'weight': 0.25})
    assert spec['weight'] == 0.25 and spec['num_feat'] == 64 and spec['betas'] == [0.9, 0.99]
    for bad in ({'weight': 0}, {'start_step': -1}, {'num_feat': 0}, {'betas': [0.9]}, {'unknown': 1}, 'yes'):
        with pytest.raises(ValueError):
            gan_spec(bad)


def changed_keys(base, cfg, prefix=''):
    keys = set()
    for key in set(base) | set(cfg):
        a, b = base.get(key), cfg.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            keys |= changed_keys(a, b, f'{prefix}{key}.')
        elif a != b:
            keys.add(f'{prefix}{key}')
    return keys


@pytest.mark.parametrize('name, base, changes', [
    ('sft_ema.yml', 'loss_all.yml', {'training.ema_decay'}),
    ('sft_gan.yml', 'sft_ema.yml', {'training.gan'}),
    ('sft_lpips_textmask.yml', 'loss_all.yml', {'training.lpips_text_mask'}),
])
def test_paradigm_configs_change_one_point(name, base, changes):
    cfg = load_config(CONFIGS / name)
    assert yaml.safe_load((CONFIGS / name).read_text())['_base_'] == base
    assert changed_keys(load_config(CONFIGS / base), cfg) == changes | {'training.output_dir'}
    assert cfg['training']['output_dir'] == f'exp_vosr/{name[:-4]}'
    assert cfg['training']['kd_weight'] == 0 and cfg['training']['gt_weight'] == 1  # no teacher
    gan_spec(cfg['training'].get('gan'))


def test_sft_hourglass_is_hourglass_student_with_loss_all_objective():
    cfg = load_config(CONFIGS / 'sft_hourglass.yml')
    loss_all, hourglass = load_config(CONFIGS / 'loss_all.yml'), load_config(CONFIGS / 'hourglass.yml')
    assert cfg['student'] == hourglass['student']
    assert changed_keys(loss_all, cfg) - {'training.output_dir'} == {
        f'student.{k}' for k in ('dino', 'token_compression', 'merge_curriculum_start', 'merge_curriculum_steps')}
    tc = cfg['training']
    assert tc['kd_weight'] == 0 and not tc.get('feature_distill_weight')  # the teacher is never loaded


def run_smoke(tmp_path, training, extra=None, resume=None, world=1):
    Image.new('RGB', (16, 16), 'white').save(tmp_path / 'hq.png')
    (tmp_path / 'images.txt').write_text(str(tmp_path / 'hq.png') + '\n')
    (tmp_path / 'datasets.txt').write_text(str(tmp_path / 'images.txt') + ', 2\n')
    torch.save({'weight': torch.tensor(.1)}, tmp_path / 'teacher.pt')
    cfg = dict(teacher_checkpoint=str(tmp_path / 'teacher.pt'), vae_path='fixture', model={'patch_size': 1},
               dino={'size': 14}, data={'resolution': 16, 'dataset_type': 'txt',
                                        'train_dataset_config': str(tmp_path / 'datasets.txt')},
               student={'use_cross_attention': False}, ocr={'strip_height': 8, 'strip_stride': 4},
               training=dict(output_dir=str(tmp_path / 'run'), report_to='none', batch_size_per_gpu=1,
                             gradient_accumulation_steps=2, max_steps=3, learning_rate=.001, save_every=2,
                             log_every=1, preview_every=0, num_workers=0, zero_optimizer=False,
                             kd_weight=0., gt_weight=1., **training), **(extra or {}))
    config_path = tmp_path / 'config.yml'
    config_path.write_text(yaml.safe_dump(cfg))
    command = [sys.executable]
    if world > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}']
    command += ['tests/ablation_trainer_smoke.py', '--config', str(config_path)]
    if resume:
        command += ['--resume', str(resume)]
    return subprocess.run(command, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS='1'), text=True,
                          capture_output=True, timeout=120)


def test_training_with_ema_saves_ema_and_raw_and_resumes(tmp_path):
    run = run_smoke(tmp_path, {'ema_decay': 0.5})
    assert run.returncode == 0, run.stdout + run.stderr
    output, = (tmp_path / 'run').iterdir()
    checkpoint = output / 'checkpoint-00000002'
    ema, raw = read_weights(checkpoint)['weight'], read_weights(checkpoint / 'raw')['weight']
    assert (checkpoint / 'raw/model.json').is_file()  # raw/ is itself a loadable export
    assert raw != 0.1 and ema != raw
    assert abs(ema - 0.1) < abs(raw - 0.1) and (ema - 0.1) * (raw - 0.1) > 0  # EMA lags behind training
    run = run_smoke(tmp_path, {'ema_decay': 0.5}, resume=checkpoint)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (output / 'export/model.safetensors').is_file()


@pytest.mark.parametrize('world', [1, 2])
def test_training_with_gan_and_ema_trains_discriminator_and_resumes(tmp_path, world):
    if world == 2 and os.environ.get('VOSR_TEST_DDP') != '1':
        pytest.skip('Set VOSR_TEST_DDP=1 on a host that permits Gloo sockets')
    gan = {'gan': {'weight': 0.5, 'start_step': 1, 'num_feat': 4}, 'lpips_weight': 1., 'ema_decay': 0.5}
    run = run_smoke(tmp_path, gan, world=world)
    assert run.returncode == 0, run.stdout + run.stderr
    output, = (tmp_path / 'run').iterdir()
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in records] == [1, 2, 3]
    assert records[0]['gan_scale'] == 0 and records[0]['gan_g'] == 0  # discriminator-only first step
    for row in records:
        assert row['gan_d'] > 0
    for row in records[1:]:
        assert row['gan_scale'] > 0 and row['gan_g'] > 0
        assert row['loss'] == pytest.approx(row['gt'] + row['lpips'] + row['gan_scale'] * row['gan_g'], rel=1e-3)
    state = torch.load(output / 'checkpoint-00000002/training_state.pt', weights_only=True)
    assert 'conv1.weight_orig' in state['discriminator'] and state['discriminator_optimizer']['state']
    assert (output / 'checkpoint-00000002/raw/model.safetensors').is_file()
    run = run_smoke(tmp_path, gan, resume=output / 'checkpoint-00000002', world=world)
    assert run.returncode == 0, run.stdout + run.stderr
    run = run_smoke(tmp_path, {'gan': {'weight': 0.5}})
    assert run.returncode != 0 and 'enable lpips_weight' in run.stderr


def test_training_with_lpips_text_mask(tmp_path):
    run = run_smoke(tmp_path, {'lpips_weight': 1., 'lpips_text_mask': {'threshold': 0.5, 'smooth_frames': 1}})
    assert run.returncode == 0, run.stdout + run.stderr
    output, = (tmp_path / 'run').iterdir()
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    for row in records:
        assert 0 <= row['text_fraction'] <= 1 and row['ocr_kl'] == 0  # recognizer used for the mask only
    run = run_smoke(tmp_path, {'lpips_text_mask': {'threshold': 0.5}})
    assert run.returncode != 0 and 'lpips_weight > 0' in run.stderr
    run = run_smoke(tmp_path, {'lpips_weight': 1., 'lpips_text_mask': {'smooth_frames': 2}})
    assert run.returncode != 0 and 'odd positive smooth_frames' in run.stderr


def test_hourglass_sft_with_ema_on_the_real_student(tmp_path):
    """Real small AblationLightningDiT: SFT without teacher, merge curriculum, EMA export and resume."""
    from ablation_utils import load_export
    from test_ablation_hourglass import HOURGLASS, run_trainer, trainer_config
    student = dict(HOURGLASS, merge_curriculum_start=1, merge_curriculum_steps=2)
    config_path, _ = trainer_config(tmp_path, student, kd_weight=0., gt_weight=1., ema_decay=0.5)
    run_trainer(config_path)
    output, = (tmp_path / 'run').iterdir()
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [row['coarse_depth'] for row in records] == [1, 2, 2]
    assert all(row['kd'] == 0 and row['gt'] > 0 for row in records)  # no teacher loaded
    checkpoint = output / 'checkpoint-00000002'
    ema, raw = load_export(checkpoint), load_export(checkpoint / 'raw')
    assert ema.coarse_depth == raw.coarse_depth == 2  # the EMA copy follows the curriculum
    differs = [not torch.equal(a, b) for a, b in zip(ema.state_dict().values(), raw.state_dict().values())]
    assert any(differs)
    run_trainer(config_path, resume=checkpoint)
    assert load_export(output / 'export').coarse_depth == 2
