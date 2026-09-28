import json
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import ablation_logging as logmod
from ablation_logging import TrainingLogger, build_preview_samples, render_previews


class VAE(nn.Module):
    config = SimpleNamespace(latents_mean=[0, 0, 0], latents_std=[1, 1, 1])

    def encode(self, x):
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: x))

    def decode(self, x, return_dict=False):
        return (x,)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.sparse_eval = True
        self.inputs = []

    def forward(self, x, t, r, features, return_stats=False):
        assert not self.training
        self.inputs.append(x.clone())
        y = self.weight * x[:, 3:]
        return (y, {'keep_fraction': x.new_tensor(.75), 'keep_probabilities': x.new_tensor([.7, .8]),
                    'keep_fractions': x.new_tensor([.5, 1.])}) if return_stats else y


def config(tmp_path):
    return {'training': {'output_dir': str(tmp_path), 'precision': 'fp32', 'preview_seed': 1234,
                         'report_to': ['tensorboard', 'wandb'], 'wandb_mode': 'offline',
                         'tracker_project_name': 'logging-test'}, 'dino': {}}


def test_trackers_write_scalars_and_images_same_step(tmp_path, monkeypatch):
    monkeypatch.setenv('WANDB_MODE', 'offline')
    tracking = TrainingLogger(config(tmp_path), torch.device('cpu'))
    try:
        tracking.log({'step': 1, 'loss': 1.25, 'kd': .5}, [Image.new('RGB', (32, 16), 'red')])
        tracking.log({'step': 2, 'loss': .75, 'kd': .25})
    finally:
        tracking.close()
    events = EventAccumulator(str(tmp_path / 'logs/logging-test')).Reload()
    assert [e.step for e in events.Scalars('loss')] == [1, 2]
    assert events.Scalars('loss')[0].value == 1.25
    assert events.Images('preview/LQ_student_teacher_HQ')[0].step == 1
    assert (tmp_path / 'previews/step-00000001/sample-00.png').is_file()
    assert len((tmp_path / 'metrics.jsonl').read_text().splitlines()) == 2
    # Read the actual offline W&B history, checking images weren't discarded after
    # a scalar call committed the same step.
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto.wandb_internal_pb2 import Record
    run_files = list((tmp_path / 'logs/wandb').glob('offline-run-*/run-*.wandb'))
    assert len(run_files) == 1
    store = DataStore()
    store.open_for_scan(str(run_files[0]))
    histories = []
    try:
        while (data := store.scan_data()) is not None:
            record = Record()
            record.ParseFromString(data)
            if record.HasField('history'):
                histories.append({item.key or '.'.join(item.nested_key): json.loads(item.value_json)
                                  for item in record.history.item})
    finally:
        store.close()
    assert any(row.get('loss') == 1.25 and row.get('preview/LQ_student_teacher_HQ.count') == 1 and row.get('_step') == 1 for row in histories)


def test_preview_is_repeatable_and_preserves_rng_and_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(logmod, 'dino_features', lambda *args: [torch.zeros(1, 1, 1)])
    class Dataset:
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return {'hq': torch.rand(3, 16, 16) * random.random() * np.random.random()}
    class Degradation:
        def degrade_process(self, hq, resize_bak):
            return hq, hq * torch.rand(())
    device = torch.device('cpu')
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    samples = build_preview_samples(Dataset(), Degradation(), device, 2, 55)
    student, teacher = Model().train(), Model().eval()
    cfg = config(tmp_path)
    images, stats = render_previews(student, teacher, VAE(), None, samples, cfg, device)
    images2, _ = render_previews(student, teacher, VAE(), None, samples, cfg, device)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(images, images2))
    assert images[0].size == (64, 40)
    assert stats['preview/deterministic_mlp_keep'] == .75
    assert stats['preview/expected_mlp_keep'] == .75
    assert stats['preview/keep_gap'] == 0.
    assert stats['preview/layer_00/actual_keep'] == .5
    assert stats['preview/layer_01/actual_keep'] == 1.
    assert student.training and student.sparse_eval
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    for a, b in zip(student.inputs, teacher.inputs):
        torch.testing.assert_close(a, b)
    class BrokenVAE(VAE):
        def encode(self, x):
            raise RuntimeError('preview failure')
    with pytest.raises(RuntimeError, match='preview failure'):
        render_previews(student, teacher, BrokenVAE(), None, samples, cfg, device)
    assert student.training and student.sparse_eval
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_tracking_disabled_still_saves_local_previews(tmp_path):
    cfg = config(tmp_path)
    cfg['training']['report_to'] = 'none'
    tracking = TrainingLogger(cfg, torch.device('cpu'))
    try:
        assert not tracking.accelerator.trackers
        tracking.log({'step': 1, 'loss': 0.0}, [Image.new('RGB', (16, 16))])
    finally:
        tracking.close()
    assert (tmp_path / 'previews/step-00000001/sample-00.png').exists()


@pytest.mark.parametrize('world', [1, 2])
@pytest.mark.parametrize('routing_mode', [None, 'capacity_topk'])
def test_training_and_resume(tmp_path, world, routing_mode):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import yaml
    if world == 2 and os.environ.get('VOSR_TEST_DDP') != '1':
        pytest.skip('Set VOSR_TEST_DDP=1 on a host that permits Gloo sockets')
    root = Path(__file__).resolve().parents[1]
    Image.new('RGB', (16, 16), 'white').save(tmp_path / 'hq.png')
    (tmp_path / 'images.txt').write_text(str(tmp_path / 'hq.png') + '\n')
    (tmp_path / 'datasets.txt').write_text(str(tmp_path / 'images.txt') + ', 2\n')
    torch.save({'weight': torch.tensor(.1)}, tmp_path / 'teacher.pt')
    cfg = config(tmp_path / 'run')
    cfg.update(teacher_checkpoint=str(tmp_path / 'teacher.pt'), vae_path='fixture',
               model={'patch_size': 1}, data={'resolution': 16, 'dataset_type': 'txt',
                'train_dataset_config': str(tmp_path / 'datasets.txt')},
               student={'use_cross_attention': False})
    cfg['training'].update(report_to=['tensorboard'], batch_size_per_gpu=1,
                           gradient_accumulation_steps=2, max_steps=3, learning_rate=.001,
                           save_every=2, log_every=2, preview_every=2, preview_num_images=1,
                           num_workers=0, zero_optimizer=False, gt_weight=1.)
    if routing_mode:
        cfg['student']['router_config'] = {'init_keep_prob': .75, 'routing_mode': routing_mode}
        cfg['training'].update(budget_scope='global', dense_warmup_steps=1, dense_distill_weight=.1,
                               budget_warmup_steps=2, target_keep_ratio=.5)
    config_path = tmp_path / 'config.yml'
    config_path.write_text(yaml.safe_dump(cfg))
    command = [sys.executable]
    if world > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}']
    command += ['tests/ablation_trainer_smoke.py', '--config', str(config_path)]
    env = dict(os.environ, OMP_NUM_THREADS='1', WANDB_MODE='offline')
    run = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    output = tmp_path / 'run'
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in records] == [1, 2, 3]
    assert all(row['gt'] > 0 and row['grad_norm'] > 0 for row in records)
    if routing_mode:
        assert records[0]['mlp_keep'] == 1.
        for row in records[1:]:
            assert 'stochastic_mlp_keep' not in row
            assert abs(row['mlp_keep_gap']) <= .5 / (16 * 16) + 1e-6
            assert abs(row['preview/keep_gap']) <= .5 / (16 * 16) + 1e-6
            assert 'router/layer_01/expected_keep' in row
            assert 'preview/layer_01/actual_keep' in row
        assert 'mlp_keep' in EventAccumulator(str(output / 'logs/logging-test')).Reload().Tags()['scalars']
    assert len(list((output / 'previews').glob('*/sample-00.png'))) == 3
    assert (output / 'checkpoint-00000002/training_state.pt').is_file()
    events = EventAccumulator(str(output / 'logs/logging-test')).Reload()
    assert [e.step for e in events.Scalars('loss')] == [1, 2, 3]
    cfg['training']['run_name'] = 'resumed-with-new-logging-options'
    config_path.write_text(yaml.safe_dump(cfg))
    run = subprocess.run(command + ['--resume', str(output / 'checkpoint-00000002')],
                         cwd=root, env=env, text=True, capture_output=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    events = EventAccumulator(str(output / 'logs/logging-test')).Reload()
    assert [e.step for e in events.Scalars('loss')] == [1, 2, 3]
    assert (output / 'export/model.safetensors').is_file()


def test_resume_accepts_logging_changes_but_rejects_training_changes():
    from ablation_logging import resume_configs_match
    saved = {'model': {'depth': 36}, 'training': {'learning_rate': .001, 'log_every': 10}}
    changed = {'model': {'depth': 36}, 'training': {'learning_rate': .001, 'log_every': 20,
               'report_to': ['wandb', 'tensorboard'], 'preview_every': 500}}
    assert resume_configs_match(saved, changed)
    changed['training']['learning_rate'] = .01
    assert not resume_configs_match(saved, changed)


def test_resume_rejects_implicit_routing_or_budget_migration():
    import copy
    from ablation_logging import resume_configs_match
    saved = {'student': {'router_config': {'init_keep_prob': .99}}, 'training': {'learning_rate': .001}}
    changed = copy.deepcopy(saved)
    changed['training']['budget_scope'] = 'global'
    assert not resume_configs_match(saved, changed)
    changed = copy.deepcopy(saved)
    changed['student']['router_config']['routing_mode'] = 'capacity_topk'
    assert not resume_configs_match(saved, changed)
