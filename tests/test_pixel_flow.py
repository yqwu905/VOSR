"""Route B mathematics, frozen-backbone gradients and real trainer/export integration (CPU)."""
import copy
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import numpy as np
import pytest
import torch
from torch import nn
import yaml

from ablation_utils import build_train_dataset, load_config, load_export, read_weights
from models.pixel_flow import (flow_optimizer_groups, make_flow_batch, pixel_flow_loss, pixel_flow_spec,
                               sample_pixel_flow, training_flow_spec)

ROOT = Path(__file__).resolve().parents[1]


def test_clamped_velocity_loss_uses_same_target_denominator_and_fp32():
    torch.manual_seed(4)
    clean, predicted, eps = (torch.randn(3, 3, 8, 8) for _ in range(3))
    t = torch.tensor([.001, .2, .8])  # includes t below the loss floor
    noisy = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * 2 * eps
    d = t.clamp_min(.05)[:, None, None, None]
    predicted = predicted.bfloat16().requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = pixel_flow_loss(predicted, clean, t, .05)
    expected = ((noisy - predicted.float()) / d - (noisy - clean) / d).square().mean()
    torch.testing.assert_close(loss, expected)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() > 0
    incorrect = ((noisy - predicted.float()) / d - (2 * eps - clean)).square().mean()
    assert not torch.isclose(loss, incorrect)


def test_noising_has_independent_per_example_times_and_correct_gamma():
    clean = torch.rand(8, 3, 8, 8) * 2 - 1
    spec = pixel_flow_spec(dict(noise_scale=2, timestep_mean=.8, timestep_std=.8))
    torch.manual_seed(3)
    noisy, t = make_flow_batch(clean, spec)
    torch.manual_seed(3)
    expected_t = (torch.randn(8) * .8 + .8).sigmoid()
    expected_eps = torch.randn_like(clean) * 2
    torch.testing.assert_close(t, expected_t)
    torch.testing.assert_close(noisy, (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * expected_eps)
    assert torch.unique(t).numel() == 8 and ((t > 0) & (t < 1)).all()


@pytest.mark.parametrize('steps', [1, 4, 8, 16, 32])
@pytest.mark.parametrize('gamma', [1, 2])
def test_oracle_ode_sign_grid_and_endpoint(steps, gamma):
    clean, noise = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4) * gamma
    seen = []
    def predict(state, t):
        seen.append(t[0].item())
        torch.testing.assert_close(state, (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise)
        return clean
    output = sample_pixel_flow(predict, noise, steps)
    torch.testing.assert_close(output, clean, rtol=0, atol=0)
    assert len(seen) == steps and seen[0] == 1 and min(seen) > 0
    assert all(a > b for a, b in zip(seen, seen[1:]))


@pytest.mark.parametrize('config', [dict(noise_scale=0), dict(noise_scale=float('nan')),
    dict(timestep_std=0), dict(loss_t_floor=0), dict(sampling_steps=[0]), dict(sampling_steps=[4, 4]),
    dict(inference_steps=True), dict(unknown=True)])
def test_flow_spec_rejects_invalid_values(config):
    with pytest.raises(ValueError):
        pixel_flow_spec(config)


def test_route_b_rejects_endpoint_or_mismatched_objectives():
    cfg = load_config(ROOT / 'configs/pixel_pdm/warmup.yml')
    assert training_flow_spec(cfg)['noise_scale'] == 1
    for key, value in [('kd_weight', 1), ('feature_distill_weight', 1), ('flow_weight', 0)]:
        changed = copy.deepcopy(cfg)
        changed['training'][key] = value
        with pytest.raises(ValueError):
            training_flow_spec(changed)
    cfg['training']['objective'] = 'one_step'
    with pytest.raises(ValueError, match='require training.objective'):
        training_flow_spec(cfg)


def test_direct_x_prediction_and_gradient_through_frozen_middle():
    from test_pixel_dit import MODEL
    from models.pixel_dit import PixelLightningDiT, pixel_model_config
    model = PixelLightningDiT(**pixel_model_config(MODEL, 64, {'prediction_type': 'x'}), use_checkpoint=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('x_embedder.', 'decoder.')))
    inputs, t = torch.randn(2, 6, 64, 64), torch.tensor([.001, .9])
    decoded = []
    handle = model.decoder.out_conv.register_forward_hook(lambda m, a, out: decoded.append(out))
    output = model(inputs, t, torch.zeros_like(t), [torch.randn(2, 16, 16)])
    torch.testing.assert_close(output, decoded[0].float(), rtol=0, atol=0)
    handle.remove()
    pixel_flow_loss(output, torch.randn_like(output), t, .05).backward()
    assert model.x_embedder.proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.blocks.parameters())


@pytest.mark.parametrize('filename', ['warmup.yml', 'adapt.yml', 'warmup_gamma2.yml', 'adapt_gamma2.yml', 'refine.yml'])
def test_shipped_stage_configs_have_disjoint_lrs_and_freezing(filename):
    from models.pixel_dit import PixelLightningDiT, pixel_model_config
    cfg = load_config(ROOT / 'configs/pixel_pdm' / filename)
    spec = training_flow_spec(cfg)
    assert spec['noise_scale'] == (2 if 'gamma2' in filename else 1)
    assert spec['sampling_steps'] == [4, 8, 16] and cfg['data']['resolution'] == 512
    assert cfg['training']['batch_size_per_gpu'] * cfg['training']['gradient_accumulation_steps'] * 8 == 32
    with torch.device('meta'):
        model = PixelLightningDiT(**pixel_model_config(dict(cfg['model'], auxiliary_time_cond=True), 512,
                                                       cfg['student']['pixel']))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(re.search(p, name) for p in cfg['training']['trainable_parameters']))
        if name.startswith(('blocks.12.', 'mlp_ca.', 'layer_norm.')):
            assert not parameter.requires_grad
    groups = flow_optimizer_groups(model, cfg['training'])
    assert {id(p) for g in groups for p in g['params']} == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(p.ndim > 1 or g['weight_decay'] == 0 for g in groups for p in g['params'])
    if 'warmup' in filename:
        assert all(g['lr'] == 1e-4 for g in groups)
    else:
        assert sorted({g['lr'] for g in groups}) == pytest.approx([1e-6, 2e-6, 5e-5])


def test_optimizer_rejects_ambiguous_groups():
    model = nn.Linear(4, 4)
    group = dict(name='a', patterns=['.*'], lr_multiplier=1.)
    with pytest.raises(ValueError, match='exactly one'):
        flow_optimizer_groups(model, dict(learning_rate=.001, parameter_groups=[group, dict(group, name='b')]))
    with pytest.raises(ValueError, match='exactly one'):
        flow_optimizer_groups(model, dict(learning_rate=.001,
                                         parameter_groups=[dict(group, patterns=['^weight$'])]))


def make_pair_folders(tmp_path):
    from PIL import Image
    for folder, value in [('lq', 96), ('teacher', 32), ('gt', 192)]:
        (tmp_path / folder).mkdir(exist_ok=True)
        Image.new('RGB', (80, 80), (value,) * 3).save(tmp_path / folder / 'sample.png')
    return dict(dataset_type='paired', resolution=64, upscale=4,
                sources=[dict(name='teacher', weight=4, lq_dir=str(tmp_path / 'lq'), hq_dir=str(tmp_path / 'teacher')),
                         dict(name='gt', weight=1, lq_dir=str(tmp_path / 'lq'), hq_dir=str(tmp_path / 'gt'))])


def test_paired_teacher_gt_sampling_probabilities_and_seed(tmp_path):
    dataset = build_train_dataset(make_pair_folders(tmp_path))
    random.seed(14)
    first = [dataset[i]['hq'].mean().item() for i in range(300)]
    random.seed(14)
    second = [dataset[i]['hq'].mean().item() for i in range(300)]
    assert first == second
    assert .7 < sum(value < .5 for value in first) / 300 < .9
    assert torch.allclose(dataset[0]['lq'], torch.full((3, 64, 64), 96 / 255))


def test_tiled_flow_uses_global_state_and_caches_dino_once(monkeypatch):
    from PIL import Image
    import inference_vosr_ablation as inference
    from tiled_vae import _make_tile_grid
    class Pointwise(nn.Module):
        pixel_space, latent_multiple = True, 16
        pixel = {'prediction_type': 'x'}
        def __init__(self):
            super().__init__()
            self.times = []
        def forward(self, inp, t, r, z, return_stats=False):
            assert (t > 0).all() and (r == 0).all()
            self.times.append(t[0].item())
            prediction = inp[:, :3] * .25 + inp[:, 3:] * .1 + t[:, None, None, None] * .02
            return prediction, {'keep_fraction': t.new_tensor(1.)}
    seen = []
    def dino(*args):
        seen.append(1)
        return [torch.zeros(1, 4, 2)]
    monkeypatch.setattr(inference, 'dino_features', dino)
    image = Image.fromarray(np.random.default_rng(5).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    pipeline = dict(precision='fp32', dino={}, pixel_flow=pixel_flow_spec(dict(noise_scale=2)))
    model = Pointwise()
    torch.manual_seed(13)
    tiled, keep = inference.restore(model, None, nn.Identity(), image, pipeline, torch.device('cpu'),
                                    tile_size=32, tile_overlap=16, upscale=1, sampling_steps=4)
    count = len(_make_tile_grid(64, 32, 16)) ** 2
    assert len(seen) == count and len(model.times) == 4 * count and keep == 1
    torch.manual_seed(13)
    full, _ = inference.restore(model, None, nn.Identity(), image, pipeline, torch.device('cpu'),
                                tile_size=64, tile_overlap=0, upscale=1, sampling_steps=4)
    assert np.abs(np.asarray(tiled).astype(int) - np.asarray(full).astype(int)).max() <= 1


def test_trainer_flow_stages_ema_resume_validation_and_export(tmp_path):
    from test_pixel_dit import trainer_config, run_trainer
    student = dict(use_cross_attention=True, router_config=None, pixel=dict(patch_size=16, prediction_type='x'))
    path, teacher = trainer_config(tmp_path, student, ema_decay=.9,
        trainable_parameters=['^(x_embedder|decoder)\\.'])
    cfg = yaml.safe_load(path.read_text())
    cfg['data'] = make_pair_folders(tmp_path)
    cfg['training'].update(objective='pixel_flow', gt_weight=0, flow_weight=1)
    cfg['pdm'] = dict(sampling_steps=[2, 4])
    cfg['validation'] = dict(data=dict(dataset_type='paired', upscale=4, lq_dir=str(tmp_path / 'lq'),
                                     hq_dir=str(tmp_path / 'gt')), every=2, num_images=1, seed=77)
    path.write_text(yaml.safe_dump(cfg))
    result = run_trainer(path)
    assert result.returncode == 0, result.stdout + result.stderr
    run, = (tmp_path / 'run').iterdir()
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert all(row['flow'] > 0 and 0 < row['timestep_mean'] < 1 and row['kd'] == 0 for row in records)
    assert all('preview/4_step_psnr_rgb' in row and 'validation/4_step_psnr_rgb' in row for row in records)
    assert (run / 'validation/step-00000003/sample-00.png').is_file()
    state = read_weights(run / 'checkpoint-00000003')
    assert all(torch.equal(value, teacher[name]) for name, value in state.items() if name.startswith('blocks.'))
    restored = load_export(run / 'export')
    assert restored.pixel['prediction_type'] == 'x'
    assert json.loads((run / 'export/pipeline.json').read_text())['pixel_flow']['noise_scale'] == 1
    result = run_trainer(path, resume=run / 'checkpoint-00000002')
    assert result.returncode == 0, result.stdout + result.stderr

    # A new stage starts with strict pixel weights, a fresh optimizer and low-LR inherited block 0.
    cfg['training'].update(output_dir=str(tmp_path / 'adapt'), require_pixel_init=True,
                          trainable_parameters=['^(x_embedder|decoder)\\.', '^blocks\\.0\\.'])
    cfg['training']['parameter_groups'] = [dict(name='new', patterns=['^(x_embedder|decoder)\\.'], lr_multiplier=1.),
                                          dict(name='inherited', patterns=['^blocks\\.0\\.'], lr_multiplier=.1)]
    path.write_text(yaml.safe_dump(cfg))
    result = run_trainer(path)
    assert result.returncode != 0 and 'requires --init-from' in result.stderr
    result = run_trainer(path, init_from=run / 'export')
    assert result.returncode == 0, result.stdout + result.stderr
    adapted, = (tmp_path / 'adapt').iterdir()
    final = read_weights(adapted / 'checkpoint-00000003')
    assert any(not torch.equal(value, state[name]) for name, value in final.items() if name.startswith('blocks.0.'))
    assert all(torch.equal(value, state[name]) for name, value in final.items() if name.startswith('blocks.2.'))
    result = run_trainer(path, resume=adapted / 'checkpoint-00000002')
    assert result.returncode == 0, result.stdout + result.stderr
    cfg['pdm']['noise_scale'] = 2
    path.write_text(yaml.safe_dump(cfg))
    result = run_trainer(path, init_from=run / 'export')
    assert result.returncode != 0 and 'same pdm.noise_scale' in result.stderr


@pytest.mark.skipif(os.environ.get('VOSR_TEST_DDP') != '1',
                    reason='Set VOSR_TEST_DDP=1 on a host that permits Gloo sockets')
def test_two_rank_flow_with_zero_groups_and_rank_zero_validation(tmp_path):
    from test_pixel_dit import trainer_config
    student = dict(use_cross_attention=True, router_config=None, pixel=dict(prediction_type='x'))
    path, _ = trainer_config(tmp_path, student, ema_decay=.9,
                             trainable_parameters=['^(x_embedder|decoder)\\.'])
    cfg = yaml.safe_load(path.read_text())
    cfg['data'] = dict(make_pair_folders(tmp_path), samples_per_epoch=4)
    cfg['training'].update(objective='pixel_flow', gt_weight=0, max_steps=2, zero_optimizer=True)
    cfg['pdm'] = dict(sampling_steps=[2])
    cfg['validation'] = dict(data=dict(dataset_type='paired', upscale=4, lq_dir=str(tmp_path / 'lq'),
                                     hq_dir=str(tmp_path / 'gt')), every=2, num_images=1)
    path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, OMP_NUM_THREADS='1', TORCHDYNAMO_DISABLE='1', WANDB_MODE='offline')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
               'tests/ablation_hourglass_smoke.py', '--config', str(path)]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    run, = (tmp_path / 'run').iterdir()
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in records] == [1, 2]
    assert all(row['flow'] > 0 and 'validation/2_step_psnr_rgb' in row for row in records)
    saved = torch.load(run / 'checkpoint-00000002/training_state.pt', weights_only=True)
    assert saved['optimizer']['state'] and len(saved['optimizer']['param_groups']) == 2
    assert (run / 'checkpoint-00000002/ema/model.safetensors').is_file()
