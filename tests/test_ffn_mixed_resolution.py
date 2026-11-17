"""Spatial FFN algebra, physical compute, calibration and real trainer round trips."""
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ablation_utils import load_backbone_state, load_config, load_export, read_weights, save_export
from models.lightningdit_ablation import AblationLightningDiT
from models.mixed_resolution_ffn import (MixedResolutionSwiGLU, mixed_resolution_spec, mlp_fingerprint,
                                        split_mlp_state, validate_groups)
from models.swiglu_ffn import SwiGLUFFN
from tools.benchmark_ffn_export import check_widths
from tools.calibrate_ffn_spatial import TIERS, select_groups, spatial_statistics
from test_ablation_head_split import inputs, kwargs, paired_trainer_config, run_trainer

ROOT = Path(__file__).resolve().parents[1]


def config(full=21, coarse=64, **extra):
    return dict(type='mixed_resolution', full_hidden=full, coarse_hidden=coarse, **extra)


def source():
    torch.manual_seed(3)
    model = AblationLightningDiT(**kwargs()).eval()
    nn.init.normal_(model.final_layer.linear.weight, std=.02)
    return model


def student_of(model, spec, **extra):
    student = AblationLightningDiT(**kwargs(**extra), mlp_compression=spec).eval()
    load_backbone_state(student, model.state_dict())
    return student


@pytest.mark.parametrize('full,coarse', [(16, 0), (0, 16), (4, 12), (2, 14), (2, 8)])
@pytest.mark.parametrize('mode', ['nearest', 'bilinear'])
def test_forward_and_gradients_against_independent_neuron_reference(full, coarse, mode):
    torch.manual_seed(7)
    dense = SwiGLUFFN(5, 16).double()
    spec, init = mixed_resolution_spec(config(full, coarse, seed=19, upsample=mode), 16)
    state = split_mlp_state({'blocks.0.mlp.' + k: v for k, v in dense.state_dict().items()}, spec, init, 1, 16)
    split = MixedResolutionSwiGLU(5, full, coarse, upsample=mode).double()
    split.load_state_dict({k.removeprefix('blocks.0.mlp.'): v for k, v in state.items()})
    x = torch.randn(2, 16, 5, dtype=torch.float64, requires_grad=True)
    # Explicit per-neuron reference: each unit reads all 5 channels, and its
    # scalar activation is upsampled before contributing its output vector.
    expected = dense.w3.bias.expand(2, 16, 5)
    order = (torch.arange(16) if full == 16 else torch.randperm(
        16, generator=torch.Generator().manual_seed(19 * 1_000_003)))
    pooled = F.avg_pool2d(x.transpose(1, 2).reshape(2, 5, 4, 4), 2).flatten(2).transpose(1, 2)
    for position, j in enumerate(order[:full + coarse]):
        inp = x if position < full else pooled
        gate = inp @ dense.w12.weight[j] + dense.w12.bias[j]
        value = inp @ dense.w12.weight[16 + j] + dense.w12.bias[16 + j]
        activation = F.silu(gate) * value
        if position >= full:
            options = {'align_corners': False} if mode == 'bilinear' else {}
            activation = F.interpolate(activation.reshape(2, 1, 2, 2), size=(4, 4), mode=mode,
                                       **options).reshape(2, 16)
        expected = expected + activation[..., None] * dense.w3.weight[:, j]
    actual = split(x)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    grad, = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)
    reference_grad, = torch.autograd.grad(expected.square().sum(), x)
    torch.testing.assert_close(grad, reference_grad, rtol=1e-10, atol=1e-10)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in split.parameters())
    assert sum(p.numel() for p in split.parameters()) == (3 * 5 + 2) * (full + coarse) + 5


def test_no_pooling_full_width_and_factor_one_recover_source():
    model = source()
    full = student_of(model, config(85, 0))
    split = student_of(model, config(factor=1))
    with torch.no_grad():
        torch.testing.assert_close(full(*inputs()), model(*inputs()), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(split(*inputs()), model(*inputs()), rtol=1e-5, atol=1e-6)
    assert full.latent_multiple == 2


def test_random_grouping_is_deterministic_and_tiers_share_indices():
    model = source()
    first = student_of(model, config(seed=12))
    torch.manual_seed(12345)
    second = student_of(model, config(seed=12))
    narrow = student_of(model, config(21, 32, seed=12))
    for i in range(6):
        a, b, n = (m.blocks[i].mlp for m in (first, second, narrow))
        assert torch.equal(a.full.w12.weight, b.full.w12.weight)
        assert torch.equal(a.full.w12.weight, n.full.w12.weight)
        assert torch.equal(a.coarse.w3.weight[:, :32], n.coarse.w3.weight)
    different = student_of(model, config(seed=13))
    assert not torch.equal(first.blocks[0].mlp.full.w12.weight, different.blocks[0].mlp.full.w12.weight)


@pytest.mark.parametrize('extra', [dict(full_hidden=-1), dict(coarse_hidden=True), dict(full_hidden=86),
                                  dict(full_hidden=0, coarse_hidden=0), dict(factor=0), dict(factor=True),
                                  dict(upsample='bicubic'), dict(grouping='learned'), dict(seed=-1),
                                  dict(grouping='calibrated'), dict(groups_file='unused'), dict(typo=True)])
def test_invalid_spec(extra):
    with pytest.raises(ValueError):
        mixed_resolution_spec(dict(config(), **extra), 85)


def test_invalid_combinations_and_grid_alignment():
    for extra in (dict(router_config={}), dict(use_swiglu=False)):
        with pytest.raises(ValueError):
            AblationLightningDiT(**dict(kwargs(), **extra), mlp_compression=config())
    with pytest.raises(ValueError, match='divisible by mixed_resolution.factor'):
        AblationLightningDiT(**kwargs(input_size=14), mlp_compression=config())
    model = AblationLightningDiT(**kwargs(input_size=24), mlp_compression=config(factor=3),
                                 attn_compression={'type': 'head_split', 'factor': 2, 'local_heads': [[]] * 6})
    assert model.latent_multiple == 12  # lcm of both pooling factors, times patch size
    with pytest.raises(ValueError, match='pooling factors'):
        model(*inputs(side=16))
    with pytest.raises(ValueError, match='square token grid'):
        MixedResolutionSwiGLU(4, 2, 6)(torch.randn(1, 15, 4))


@pytest.mark.parametrize('full,coarse', [(85, 0), (0, 85), (21, 64), (10, 75), (10, 42)])
def test_checkpointed_backward_and_export(full, coarse, tmp_path):
    model = student_of(source(), config(full, coarse), use_checkpoint=True).train()
    output, stats = model(*inputs(), return_stats=True, mlp_update_layers=[0, 3])
    (output.square().mean() + sum(v.square().mean() for v in stats['mlp_updates'])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        expected = model(*inputs())
    save_export(model, tmp_path / 'export', {})
    exported = load_export(tmp_path / 'export')
    assert 'grouping' not in exported.mlp_compression
    assert 'groups_file' not in exported.mlp_compression
    with torch.no_grad():
        torch.testing.assert_close(exported(*inputs()), expected, rtol=0, atol=0)
    assert check_widths(exported, full + coarse)[0]['coarse_hidden'] == coarse


@pytest.mark.parametrize('full,coarse', TIERS)
def test_five_shipped_tiers_and_physical_matmul_counts(full, coarse):
    from torch.utils.flop_counter import FlopCounterMode
    name = f'ffn_spatial_f{full}_c{coarse}'
    cfg = load_config(ROOT / f'configs/ablations/ffn_spatial/{name}.yml')
    spec, init = mixed_resolution_spec(cfg['student']['mlp_compression'], 4096)
    assert (spec['full_hidden'], spec['coarse_hidden']) == (full, coarse)
    assert init['grouping'] == 'random' and init['seed'] == 0
    assert cfg['student_checkpoint'] == 'weights/head_split_f2'
    assert cfg['student']['attn_compression']['factor'] == 2
    assert cfg['training']['output_dir'] == f'exp_vosr/{name}'
    calibrated = load_config(ROOT / f'configs/ablations/ffn_spatial/{name}_calibrated.yml')
    assert calibrated['student']['mlp_compression']['grouping'] == 'calibrated'
    with torch.device('meta'):
        module = MixedResolutionSwiGLU(1536, full, coarse)
        x = torch.randn(1, 1024, 1536)
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        module(x)
    assert counter.get_total_flops() / 2 == 3 * 1536 * (1024 * full + 256 * coarse)


def test_spatial_calibration_scores_and_manual_groups(tmp_path):
    model = source()
    inp, _, _, features = inputs()
    weights = torch.linspace(.1, 2, 64).expand(2, -1)
    captured = []
    handle = model.blocks[0].mlp.register_forward_pre_hook(lambda m, args: captured.append(args[0].detach()))
    stats = spatial_statistics(model, [(inp, features, weights)], chunk_hidden=7)
    handle.remove()
    x, mlp = captured[0], model.blocks[0].mlp
    # Independently measure one neuron's gated vector contribution per token.
    from models.ffn_calibration import mlp_residual_gates
    gate = mlp_residual_gates(model, inp.new_ones(1), inp.new_zeros(1))[0][0]
    a, b = mlp.w12(x).chunk(2, -1)
    activation = F.silu(a) * b
    pooled = F.avg_pool2d(x.transpose(1, 2).reshape(2, 64, 8, 8), 2).flatten(2).transpose(1, 2)
    a, b = mlp.w12(pooled).chunk(2, -1)
    approx = F.interpolate((F.silu(a) * b).transpose(1, 2).reshape(2, 85, 4, 4), (8, 8),
                            mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
    j = 17
    contribution = (activation[..., j] - approx[..., j])[..., None] * (mlp.w3.weight[:, j] * gate)
    expected = (contribution.square().sum(-1) * weights).sum() / weights.sum()
    torch.testing.assert_close(stats['pool_error'][0, j], expected)
    torch.testing.assert_close(stats['drop_error'][0, j],
                               (activation[..., j].square() * weights).sum() / weights.sum()
                               * (mlp.w3.weight[:, j] * gate).square().sum())
    no_pool = spatial_statistics(model, [(inp, features)], factor=1)
    assert torch.count_nonzero(no_pool['pool_error']) == 0
    state = model.state_dict()
    fingerprints = [mlp_fingerprint(state, f'blocks.{i}.mlp.') for i in range(6)]
    groups = select_groups(stats, 21, 32, factor=2, upsample='bilinear', fingerprints=fingerprints)
    # User edits: swap two valid source neurons between the groups, retaining widths.
    groups['full_indices'][0][0], groups['coarse_indices'][0][0] = \
        groups['coarse_indices'][0][0], groups['full_indices'][0][0]
    path = tmp_path / 'groups.json'
    path.write_text(json.dumps(groups))
    spec = config(21, 32, grouping='calibrated', groups_file=str(path))
    student = student_of(model, spec)
    ids = groups['full_indices'][0]
    torch.testing.assert_close(student.blocks[0].mlp.full.w12.weight, mlp.w12.weight[ids + [i + 85 for i in ids]])
    torch.testing.assert_close(student.blocks[0].mlp.coarse.w3.weight, mlp.w3.weight[:, groups['coarse_indices'][0]])
    changed = dict(state)
    changed['blocks.0.mlp.w12.bias'] = changed['blocks.0.mlp.w12.bias'] + .001
    with pytest.raises(ValueError, match='different FFN weights'):
        load_backbone_state(student, changed)
    groups['coarse_indices'][0][0] = groups['full_indices'][0][0]
    with pytest.raises(ValueError, match='disjoint and unique'):
        validate_groups(groups, student.mlp_compression, 6, 85)
    path.unlink()
    # Resume uses already split tensors and never opens the missing calibration file.
    resumed = AblationLightningDiT(**kwargs(), mlp_compression=spec)
    load_backbone_state(resumed, student.state_dict())
    save_export(student, tmp_path / 'export', {})
    exported = load_export(tmp_path / 'export')
    with torch.no_grad():
        torch.testing.assert_close(exported(*inputs()), student(*inputs()), rtol=0, atol=0)


def test_group_selection_keeps_output_energy_then_protects_pool_sensitive_units():
    stats = dict(drop_error=torch.tensor([[8., 7., 6., 5., 4.]]),
                 pool_error=torch.tensor([[1., 2., 9., 20., 30.]]))
    groups = select_groups(stats, 1, 2, factor=2, upsample='bilinear', fingerprints=['fixture'])
    assert groups['full_indices'] == [[2]] and groups['coarse_indices'] == [[0, 1]]


def test_calibration_cleans_hooks_after_invalid_input():
    model = source()
    inp, _, _, features = inputs()
    with pytest.raises(ValueError, match='Calibration weights'):
        spatial_statistics(model, [(inp, features, torch.zeros(2, 64))])
    assert all(not block.mlp._forward_pre_hooks for block in model.blocks)
    with pytest.raises(ValueError, match='No samples'):
        spatial_statistics(model, [])


def test_calibration_cli_writes_all_five_loadable_group_files(tmp_path, monkeypatch):
    import types
    import yaml
    from ablation_model_smoke import TinyVAE, fake_dino_features
    from tools import calibrate_ffn_spatial as calibration
    path = paired_trainer_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    # Full 4096-hidden source with a small residual width and depth; real data,
    # model loading, hooks, scores, provenance and all five manifests run here.
    cfg['model'].update(mlp_ratio=192, depth=1)
    cfg['student']['attn_compression'] = None
    initial = AblationLightningDiT(**dict(cfg['model'], auxiliary_time_cond=True), input_size=8).eval()
    torch.save(initial.state_dict(), tmp_path / 'teacher.pt')
    path.write_text(yaml.safe_dump(cfg))
    fake_vae_module = types.ModuleType('models.qwenimage_vae2d')
    fake_vae_module.AutoencoderKLQwenImage2D = TinyVAE
    monkeypatch.setitem(sys.modules, 'models.qwenimage_vae2d', fake_vae_module)
    monkeypatch.setattr(calibration, 'load_dino', lambda *args: nn.Identity())
    monkeypatch.setattr(calibration, 'dino_features', fake_dino_features)
    output = tmp_path / 'calibration'
    monkeypatch.setattr(sys, 'argv', ['calibrate_ffn_spatial.py', '--config', str(path), '--output-dir', str(output),
                                     '--samples', '1', '--device', 'cpu', '--chunk-hidden', '512'])
    calibration.main()
    assert (output / 'scores.safetensors').is_file()
    provenance = json.loads((output / 'calibration.json').read_text())
    assert len(provenance['samples']) == 1 and provenance['source']['fingerprint']
    for full, coarse in TIERS:
        group_file = output / f'groups_f{full}_c{coarse}.json'
        student = AblationLightningDiT(**dict(cfg['model'], auxiliary_time_cond=True), input_size=8,
                                       mlp_compression=config(full, coarse, grouping='calibrated',
                                                              groups_file=str(group_file)))
        load_backbone_state(student, initial.state_dict())
        assert check_widths(student, full + coarse)[0]['coarse_hidden'] == coarse


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('VOSR_TEST_DDP') != '1', reason='Set VOSR_TEST_DDP=1 for Gloo socket tests')
@pytest.mark.parametrize('full,coarse', [(0, 42), (10, 32)])
def test_two_rank_trainer_uses_every_branch(tmp_path, full, coarse):
    import yaml
    path = paired_trainer_config(tmp_path, gradient_checkpointing=True)
    cfg = yaml.safe_load(path.read_text())
    cfg['student']['mlp_compression'] = config(full, coarse)
    cfg['training'].update(max_steps=2, preview_every=0, zero_optimizer=True)
    path.write_text(yaml.safe_dump(cfg))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               'tests/ablation_model_smoke.py', '--config', str(path)]
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120,
                         env=dict(os.environ, OMP_NUM_THREADS='1', TORCHDYNAMO_DISABLE='1', WANDB_MODE='offline'))
    assert run.returncode == 0, run.stdout + run.stderr
    output, = (tmp_path / 'run').iterdir()
    model = load_export(output / 'export')
    assert check_widths(model, full + coarse)[0]['full_hidden'] == full


@pytest.mark.integration
def test_trainer_calibrated_split_ema_resume_and_pack(tmp_path):
    import yaml
    path = paired_trainer_config(tmp_path, lr_schedule='cosine', min_lr_ratio=.1, ema_decay=.5,
                                 gradient_checkpointing=True, mlp_update_weight=.1, mlp_update_layers=[0, 2],
                                 feature_distill_weight=.05, feature_distill_layers=[0, 2])
    cfg = yaml.safe_load(path.read_text())
    from tools.calibrate_ffn_spatial import source_model
    initial = source_model(cfg).eval()
    # The actual source is a head-split export, as in the release recipe.
    save_export(initial, tmp_path / 'head_split_source', {})
    cfg['student_checkpoint'] = str(tmp_path / 'head_split_source')
    state = initial.state_dict()
    stats = spatial_statistics(initial, [(torch.randn(1, 6, 8, 8), [torch.randn(1, 16, 6)])])
    groups = select_groups(stats, 10, 21, factor=2, upsample='bilinear',
                           fingerprints=[mlp_fingerprint(state, f'blocks.{i}.mlp.') for i in range(4)])
    group_path = tmp_path / 'groups.json'
    group_path.write_text(json.dumps(groups))
    cfg['student']['mlp_compression'] = config(10, 21, grouping='calibrated', groups_file=str(group_path))
    path.write_text(yaml.safe_dump(cfg))
    run_trainer(path)
    output, = (tmp_path / 'run').iterdir()
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert all(row['kd'] > 0 and row['feature_kd'] > 0 for row in records)
    export = read_weights(output / 'export')
    before_resume = {k: v.clone() for k, v in export.items()}
    for name in ('full', 'coarse'):
        assert export[f'blocks.0.mlp.{name}.w12.weight'].shape[0] == (20 if name == 'full' else 42)
    group_path.unlink()
    run_trainer(path, resume=output / 'checkpoint-00000002')
    after_resume = read_weights(output / 'export')
    for key in before_resume:
        torch.testing.assert_close(after_resume[key], before_resume[key], rtol=0, atol=0)
    # Pack utility requires a bf16-autocast pipeline; exercise split weight names.
    pipeline_path = output / 'export/pipeline.json'
    pipeline = json.loads(pipeline_path.read_text())
    pipeline['precision'] = 'bf16'
    pipeline_path.write_text(json.dumps(pipeline))
    run = subprocess.run([sys.executable, 'tools/pack_export.py', '--export', str(output / 'export'),
                          '--output', str(tmp_path / 'packed'), '--max-shard-gb', '0.0001'],
                         cwd=ROOT, text=True, capture_output=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    packed = load_export(tmp_path / 'packed')
    assert check_widths(packed, 31)[0]['full_hidden'] == 10


def test_balanced_weights_allocate_mass_and_redistribute_empty_strata():
    from models.ffn_calibration import token_weights
    text = torch.zeros(2, 1, 4, 4)
    small = torch.zeros_like(text)
    text[0, :, :2] = 1
    small[0, :, :2, :2] = 1
    weights = token_weights(text, small, tokens=4)
    torch.testing.assert_close(weights[0], torch.tensor([2.0, 1.2, .4, .4]))
    torch.testing.assert_close(weights[1], torch.ones(4))  # natural image, no annotation
    torch.testing.assert_close(weights.mean(1), torch.ones(2))
    # One sub-token character is retained by area pooling, not thresholded away.
    text.zero_()
    small.zero_()
    text[0, 0, 0, 0] = small[0, 0, 0, 0] = 1
    weights = token_weights(text, small, tokens=4)
    assert weights[0, 0] > weights[0, 1]
    assert torch.isfinite(weights).all()
