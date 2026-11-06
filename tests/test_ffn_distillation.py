"""CPU checks for weighted losses, true gated updates, optimizer partitioning and resume."""
import json
import os
from pathlib import Path
import random

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import numpy as np
import pytest
import torch
from torch import nn

from models.ffn_distillation import (ffn_optimizer_groups, loss_schedule, spatial_weights,
                                    text_distillation_spec, validate_text_data, weighted_feature_loss, weighted_mse,
                                    optimizer_state_for_resume)
from training_runtime import (atomic_torch_save, capture_rng, restore_rng, select_device,
                              validate_checkpoint_complete)


def test_spatial_weights_small_text_and_normalization():
    batch = {'text_mask': torch.tensor([[[[1., 1.], [0., 0.]]]]),
             'small_text_mask': torch.tensor([[[[1., 0.], [0., 0.]]]])}
    weights = spatial_weights(batch, (2, 2), 'cpu', 2, 4)
    torch.testing.assert_close(weights, torch.tensor([[[[7., 3.], [1., 1.]]]]))
    prediction = torch.tensor([[[[2., 0.], [0., 0.]]]], requires_grad=True)
    target = torch.zeros_like(prediction, requires_grad=True)
    loss = weighted_mse(prediction, target, weights)
    assert loss.item() == pytest.approx(28 / 12)
    loss.backward()
    assert prediction.grad[0, 0, 0, 0] > 0
    assert target.grad is None
    torch.testing.assert_close(spatial_weights(batch, (1, 1), 'cpu', 2, 4), torch.tensor([[[[3.]]]]))


def test_teacher_error_mask_and_energy_floor_are_finite():
    batch = {'kd_trust_mask': torch.zeros(1, 1, 2, 2)}
    student = torch.ones(1, 4, 3, requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    loss = weighted_feature_loss([student], [teacher], batch, energy_floor=1e-4)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(student.grad) == 0 and teacher.grad is None
    loss = weighted_feature_loss([student], [teacher], {}, energy_floor=1e-4)
    assert loss.item() == pytest.approx(1e4)


def test_config_guards_and_mixed_natural_source():
    cfg = {'model': {'depth': 3}, 'student': {}, 'training': {
        'kd_small_text_weight': 4, 'mlp_update_weight': .1, 'mlp_update_layers': 'all'}}
    spec = text_distillation_spec(cfg)
    assert spec['mlp_update_layers'] == [0, 1, 2] and spec['masks_enabled']
    validate_text_data({'dataset_type': 'paired', 'sources': [
        {'text_boxes': '/annotated.json'}, {'text_boxes': None}]})
    with pytest.raises(ValueError, match='text_boxes'):
        validate_text_data({'dataset_type': 'paired', 'sources': [{'text_boxes': None}]})
    for key, value in [('ffn_lr_multiplier', 0), ('roi_edge_weight', float('nan')),
                       ('pixel_loss_warmup_steps', 1.5), ('internal_kd_end_ratio', 2)]:
        bad = dict(cfg, training=dict(cfg['training'], **{key: value}))
        with pytest.raises(ValueError):
            text_distillation_spec(bad)
    spec.update(pixel_loss_warmup_steps=100, internal_kd_end_ratio=.5)
    assert loss_schedule(0, 1000, spec) == (1., .01)
    assert loss_schedule(999, 1000, spec) == (.5, 1.)


def test_disjoint_ffn_learning_rate_groups():
    model = nn.Module()
    model.embedding = nn.Linear(3, 3)
    block = nn.Module()
    block.mlp, block.attn = nn.Linear(3, 6), nn.Linear(3, 3)
    model.blocks = nn.ModuleList([block])
    groups = ffn_optimizer_groups(model, {'learning_rate': 2e-6, 'ffn_lr_multiplier': 10})
    assert [g['name'] for g in groups] == ['base', 'ffn']
    assert [g['lr'] for g in groups] == pytest.approx([2e-6, 2e-5])
    identities = [id(p) for g in groups for p in g['params']]
    assert len(identities) == len(set(identities)) == len(list(model.parameters()))
    assert {id(p) for p in groups[1]['params']} == {id(p) for p in block.mlp.parameters()}


def test_legacy_fade_optimizer_migration_preserves_moments_and_disables_decay():
    model = nn.Module()
    model.embedding = nn.Linear(3, 3)
    block = nn.Module()
    # This test exercises optimizer identity mapping, independent of fade forward.
    FadingSwiGLU = type('FadingSwiGLU', (nn.Linear,), {})
    block.mlp, block.attn = FadingSwiGLU(3, 6), nn.Linear(3, 3)
    model.blocks = nn.ModuleList([block])
    training = dict(learning_rate=1e-3, ffn_lr_multiplier=1)
    old = torch.optim.AdamW([dict(params=list(model.parameters()), lr_scale=1.)], lr=1e-3, weight_decay=.01)
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, index + 1.)
    old.step()
    state = old.state_dict()
    new = torch.optim.AdamW(ffn_optimizer_groups(model, training), lr=1e-3, weight_decay=.01)
    migrated = optimizer_state_for_resume(state, new, model, training)
    new.load_state_dict(migrated)
    assert [group['name'] for group in new.param_groups] == ['base', 'ffn']
    assert [group['weight_decay'] for group in new.param_groups] == [.01, 0.]
    for parameter in model.parameters():
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(new.state[parameter][key], old.state[parameter][key], rtol=0, atol=0)
    assert optimizer_state_for_resume(migrated, new, model, training) is migrated
    bad = dict(state, param_groups=[])
    bad['param_groups'] = [dict(state['param_groups'][0], params=state['param_groups'][0]['params'][:-1])]
    with pytest.raises(ValueError, match='Legacy optimizer'):
        optimizer_state_for_resume(bad, new, model, training)


@pytest.mark.parametrize('checkpointing', [False, True])
def test_actual_gated_updates_have_gradients_with_checkpointing(checkpointing):
    pytest.importorskip('timm')
    from models.lightningdit_ablation import AblationLightningDiT
    from test_ablation_head_split import inputs, kwargs
    model = AblationLightningDiT(**kwargs(depth=2, use_checkpoint=checkpointing)).train()
    x, t, r, features = inputs()
    captured = {}
    handle = model.blocks[0].mlp.register_forward_hook(lambda module, args, output: captured.update(value=output))
    output, stats = model(x, t, r, features, return_stats=True, mlp_update_layers=[0, 1])
    handle.remove()
    cond = model.t_embedder(t) + model.r_embedder(r) * (t - r)[:, None]
    gate = (model.blocks[0].scale_shift_table[None] + model.t_block(cond).reshape(2, 6, -1))[:, 5:6]
    torch.testing.assert_close(stats['mlp_updates'][0], gate * captured['value'])
    loss = sum(update.square().mean() for update in stats['mlp_updates'])
    loss.backward()
    assert torch.isfinite(output).all()
    for block in model.blocks:
        for weight in (block.mlp.w12.weight, block.mlp.w3.weight):
            assert weight.grad is not None and torch.isfinite(weight.grad).all()
            assert weight.grad.abs().sum() > 0


def test_rng_checkpoint_weights_only_roundtrip(tmp_path):
    device = select_device('cpu')
    state = capture_rng(device)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    path = tmp_path / 'training.pt'
    atomic_torch_save(state, path)
    restore_rng(torch.load(path, weights_only=True), device)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2])


def test_interrupted_checkpoint_marker_is_rejected(tmp_path):
    state = {'step': 2, 'rng_by_rank': []}
    with pytest.raises(ValueError, match='Incomplete checkpoint'):
        validate_checkpoint_complete(state, tmp_path)
    marker = tmp_path / 'checkpoint.complete'
    marker.write_text('1')
    with pytest.raises(ValueError, match='Incomplete checkpoint'):
        validate_checkpoint_complete(state, tmp_path)
    marker.write_text('2')
    validate_checkpoint_complete(state, tmp_path)
    validate_checkpoint_complete({'step': 2}, tmp_path)  # legacy format


def test_distributed_device_index_is_rejected_before_backend_initialization(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    with pytest.raises(ValueError, match='LOCAL_RANK'):
        select_device('cuda:0', local_rank=1)


def test_halfwidth_trainer_text_losses_and_exact_resume(tmp_path):
    pytest.importorskip('timm')
    import yaml
    from safetensors.torch import load_file
    from test_ablation_head_split import paired_trainer_config, run_trainer
    path = paired_trainer_config(tmp_path, kd_weight=1., kd_text_weight=2., kd_small_text_weight=4.,
                                 internal_text_weight=2., internal_small_text_weight=4.,
                                 feature_distill_weight=.05, feature_distill_layers=[1, 3],
                                 mlp_update_weight=.1, mlp_update_layers=[1, 3],
                                 roi_charbonnier_weight=.1, roi_edge_weight=.05,
                                 ffn_lr_multiplier=10., pixel_loss_warmup_steps=2,
                                 ema_decay=.9, weight_decay=0., gradient_checkpointing=True)
    cfg = yaml.safe_load(path.read_text())
    cfg['student']['attn_compression'] = None
    cfg['student']['mlp_compression'] = dict(type='neuron_prune', keep_ratio=.5, score='random', compensation='none')
    cfg['data'].update(text_metadata=True, deterministic_crops=True, seed=42)
    cfg['training'].update(num_workers=0)
    path.write_text(yaml.safe_dump(cfg))
    run_trainer(path)
    out, = (tmp_path / 'run').iterdir()
    original = load_file(str(out / 'export/model.safetensors'))
    assert (out / 'export/export.complete').is_file()
    assert (out / 'checkpoint-00000002/checkpoint.complete').is_file()
    logs = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert all(np.isfinite(row['loss']) for row in logs)
    assert any(row['mlp_update_kd'] > 0 and row['roi_charbonnier'] > 0 for row in logs)
    assert logs[-1]['lr/ffn'] == pytest.approx(10 * logs[-1]['lr/base'])
    run_trainer(path, resume=out / 'checkpoint-00000002')
    resumed = load_file(str(out / 'export/model.safetensors'))
    for key in original:
        torch.testing.assert_close(original[key], resumed[key], rtol=0, atol=0)
