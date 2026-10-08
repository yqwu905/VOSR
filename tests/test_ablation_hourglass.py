"""U-shaped hourglass token merging; requires the upstream timm environment."""
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import numpy as np
import pytest
import torch
from torch import nn
pytest.importorskip('timm', reason='Full backbone requires the upstream timm environment')
from models.lightningdit_ablation import AblationLightningDiT
from models.token_hourglass import (TokenMerge, TokenUnmerge, depth_to_space_tokens, feature_distill_loss,
                                    hourglass_spec, pool_tokens, rope_at, space_to_depth_tokens)
from ablation_utils import load_backbone_state, load_config, load_export, read_weights, save_export, student_dino_config

ROOT = Path(__file__).resolve().parents[1]
MERGE_KEYS = ['token_merge.proj.bias', 'token_merge.proj.weight', 'token_unmerge.proj.bias', 'token_unmerge.proj.weight']


def kwargs(**extra):
    base = dict(input_size=16, patch_size=2, in_channels=4, out_channels=2, hidden_size=64, depth=6,
                num_heads=4, mlp_ratio=2, z_dims=16, auxiliary_time_cond=True, use_rope=True,
                use_qknorm=True, use_swiglu=True, use_rmsnorm=True)
    return dict(base, **extra)


def inputs(dino_tokens=16):
    torch.manual_seed(5)
    return torch.randn(2, 4, 16, 16), torch.ones(2), torch.zeros(2), [torch.randn(2, dino_tokens, 16)]


def teacher_and_student(token_compression, **extra):
    torch.manual_seed(0)
    teacher = AblationLightningDiT(**kwargs(**extra)).eval()
    torch.nn.init.normal_(teacher.final_layer.linear.weight, std=.02)
    student = AblationLightningDiT(**kwargs(**extra), token_compression=token_compression).eval()
    load_backbone_state(student, teacher.state_dict(), allow_new_modules=True)
    return teacher, student


def test_merge_starts_as_2x2_average_and_unmerge_as_copy():
    grid = torch.arange(16.).view(1, 16, 1)  # 4x4 row-major token grid
    torch.testing.assert_close(pool_tokens(grid, 2).flatten(), torch.tensor([2.5, 4.5, 10.5, 12.5]))
    x = torch.randn(2, 64, 8)
    torch.testing.assert_close(depth_to_space_tokens(space_to_depth_tokens(x, 2), 2), x, rtol=0, atol=0)
    with torch.no_grad():
        merged = TokenMerge(8, 2)(x)
        copied = TokenUnmerge(8, 2)(merged).reshape(2, 4, 2, 4, 2, 8)
    torch.testing.assert_close(merged, pool_tokens(x, 2))
    torch.testing.assert_close(copied, copied[:, :, :1, :, :1].expand_as(copied), rtol=0, atol=0)
    torch.testing.assert_close(copied[:, :, 0, :, 0].reshape(2, 16, 8), merged, rtol=0, atol=0)


def test_spec_defaults_and_validation():
    assert hourglass_spec({'type': 'hourglass'}, 36) == {
        'type': 'hourglass', 'factor': 2, 'fine_in': 1, 'fine_out': 1, 'drop_blocks': [], 'bypass': True,
        'rope': 'centroid', 'fine_cross_attention': False, 'cond_pool': 1}
    for bad in ({'type': 'p4'}, {'type': 'hourglass', 'facter': 2}, {'type': 'hourglass', 'factor': 1},
                {'type': 'hourglass', 'bypass': 'yes'}, {'type': 'hourglass', 'rope': 'center'},
                {'type': 'hourglass', 'drop_blocks': [36]}, {'type': 'hourglass', 'drop_blocks': [3, 3]},
                {'type': 'hourglass', 'fine_in': 18, 'fine_out': 18}):
        with pytest.raises(ValueError):
            hourglass_spec(bad, 36)
    with pytest.raises(ValueError, match='joint compute budget'):
        AblationLightningDiT(**kwargs(), token_compression={'type': 'hourglass'}, router_config={'init_keep_prob': .9})
    with pytest.raises(ValueError, match='divisible'):
        AblationLightningDiT(**kwargs(input_size=14), token_compression={'type': 'hourglass'})
    with pytest.raises(ValueError, match='coarse_depth'):
        AblationLightningDiT(**kwargs(), coarse_depth=2)
    _, student = teacher_and_student({'type': 'hourglass'})
    with pytest.raises(ValueError, match='coarse_depth'):
        student.set_coarse_depth(5)
    x, t, r, z = inputs()
    with pytest.raises(ValueError, match='merge'):
        student(x[:, :, :14, :14], t, r, z)  # 7x7 tokens cannot be merged 2x2


def test_zero_merged_blocks_is_the_dense_student():
    teacher, student = teacher_and_student({'type': 'hourglass', 'fine_cross_attention': True})
    student.set_coarse_depth(0)
    x, t, r, z = inputs()
    with torch.no_grad():
        torch.testing.assert_close(student(x, t, r, z), teacher(x, t, r, z), rtol=0, atol=0)


@pytest.mark.parametrize('bypass', [True, False])
def test_bypass_keeps_full_grid_detail_exactly(bypass):
    # With identity merged blocks the bypass must reproduce the dense model bit for bit;
    # without it, every 2x2 group collapses to its mean (the pure large-patch control).
    teacher, student = teacher_and_student({'type': 'hourglass', 'bypass': bypass}, use_cross_attention=False)
    for model in (teacher, student):
        with torch.no_grad():
            model.t_block[1].weight.zero_()
            model.t_block[1].bias.zero_()
            for index in range(1, 5):
                model.blocks[index].scale_shift_table[[2, 5]] = 0  # attention and MLP gates
    x, t, r, _ = inputs()
    with torch.no_grad():
        same = torch.equal(student(x, t, r), teacher(x, t, r))
    assert same == bypass


def test_blocks_see_full_and_merged_token_counts():
    _, student = teacher_and_student({'type': 'hourglass', 'drop_blocks': [2]})
    seen = {}
    for index in student.active_blocks:
        student.blocks[index].attn.register_forward_hook(
            lambda module, args, output, i=index: seen.__setitem__(i, args[0].shape[1]))
    assert isinstance(student.blocks[2], nn.Identity) and student.max_coarse_depth == 3
    x, t, r, z = inputs()
    with torch.no_grad():
        student(x, t, r, z)
        assert seen == {0: 64, 1: 16, 3: 16, 4: 16, 5: 64}
        student.set_coarse_depth(1)  # the curriculum grows the merged span from the middle
        student(x, t, r, z)
        assert seen == {0: 64, 1: 64, 3: 16, 4: 64, 5: 64}


def test_coarse_rope_corner_matches_dynamic_rope_and_centroid_is_offset():
    _, corner = teacher_and_student({'type': 'hourglass', 'rope': 'corner'})
    _, centroid = teacher_and_student({'type': 'hourglass'})
    q = torch.randn(1, 4, 16, 16)  # (batch, heads, 4x4 merged tokens, head_dim)
    torch.testing.assert_close(corner._coarse_rope(8, q.device)(q), corner._get_dynamic_rope(4, q.device, q.dtype)(q))
    torch.testing.assert_close(centroid._coarse_rope(8, q.device)(q), rope_at(torch.arange(4.) * 2 + .5, 8)(q))
    full = torch.randn(1, 4, 64, 16)
    torch.testing.assert_close(rope_at(torch.arange(8.), 8)(full), centroid.feat_rope(full))


def test_features_follow_block_boundaries_and_distillation_targets():
    teacher, student = teacher_and_student({'type': 'hourglass', 'fine_cross_attention': True})
    x, t, r, z = inputs()
    layers = [0, 2, 4, 5]
    with torch.no_grad():
        _, teacher_stats = teacher(x, t, r, z, return_stats=True, feature_layers=layers)
        _, student_stats = student(x, t, r, z, return_stats=True, feature_layers=layers)
    assert [f.shape[1] for f in teacher_stats['features']] == [64] * 4
    # After block 0: merged trunk input; after block 4: unmerged state for block 5.
    assert [f.shape[1] for f in student_stats['features']] == [16, 16, 64, 64]
    torch.testing.assert_close(student_stats['features'][0], pool_tokens(teacher_stats['features'][0], 2))
    assert feature_distill_loss(student_stats['features'][:1], teacher_stats['features'][:1]) < 1e-10
    assert feature_distill_loss(student_stats['features'], teacher_stats['features']) > 0
    with pytest.raises(ValueError, match='feature_layers'):
        student(x, t, r, z, feature_layers=[0, 0])


def test_dense_teacher_initializes_student_and_new_layers_are_explicit():
    teacher = AblationLightningDiT(**kwargs())
    student = AblationLightningDiT(**kwargs(), token_compression={'type': 'hourglass', 'drop_blocks': [2]})
    with pytest.raises(ValueError, match='Missing'):
        load_backbone_state(student, teacher.state_dict())
    report = load_backbone_state(student, teacher.state_dict(), allow_new_modules=True)
    assert sorted(report['new_token_merge_parameters']) == MERGE_KEYS
    assert any(k.startswith('blocks.2.') for k in report['discarded'])
    assert any(k.startswith('blocks.5.cross_attn.') for k in report['discarded'])
    assert not any(k.startswith('blocks.1.cross_attn.') for k in report['discarded'])
    reference = teacher.state_dict()
    for key, value in student.state_dict().items():
        if key not in MERGE_KEYS:
            torch.testing.assert_close(value, reference[key], rtol=0, atol=0)


def test_in_model_dino_pooling_matches_pooled_input():
    _, pooled = teacher_and_student({'type': 'hourglass', 'cond_pool': 2})
    _, plain = teacher_and_student({'type': 'hourglass'})
    x, t, r, z = inputs()
    with torch.no_grad():
        torch.testing.assert_close(pooled(x, t, r, z), plain(x, t, r, [pool_tokens(z[0], 2)]))


def test_checkpointed_backward_reaches_every_parameter():
    # DDP without find_unused_parameters requires every trainable weight to get a gradient.
    _, student = teacher_and_student({'type': 'hourglass', 'cond_pool': 2})
    student.use_checkpoint = True
    student.train()
    x, t, r, z = inputs()
    output, stats = student(x, t, r, z, return_stats=True, feature_layers=[0, 4])
    (output.square().mean() + sum(f.square().mean() for f in stats['features'])).backward()
    assert [name for name, p in student.named_parameters() if p.grad is None] == []


def test_export_roundtrip_keeps_structure_and_curriculum_depth(tmp_path):
    _, student = teacher_and_student({'type': 'hourglass', 'drop_blocks': [3], 'cond_pool': 2})
    student.set_coarse_depth(2)
    save_export(student, tmp_path, {})
    config = json.loads((tmp_path / 'model.json').read_text())
    assert config['token_compression']['drop_blocks'] == [3] and config['coarse_depth'] == 2
    restored = load_export(tmp_path)
    assert not any(k.startswith(('blocks.3.', 'blocks.0.cross_attn.')) for k in restored.state_dict())
    assert restored.latent_multiple == 4 and restored.coarse_depth == 2
    x, t, r, z = inputs()
    with torch.no_grad():
        torch.testing.assert_close(restored(x, t, r, z), student(x, t, r, z))


@pytest.mark.parametrize('spec, expected', [
    ({'type': 'hourglass'}, .254),  # blocks 0/35 full grid, 1-34 merged
    ({'type': 'hourglass', 'drop_blocks': [17, 18]}, .241),
    ({'type': 'hourglass', 'fine_cross_attention': True}, .262),
    ({'type': 'hourglass', 'fine_in': 0, 'fine_out': 0, 'bypass': False}, .227),
])
def test_vosr2_size_compute_ratio(spec, expected):
    from torch.utils.flop_counter import FlopCounterMode
    model = dict(load_config(ROOT / 'configs/ablations/base_vosr2.yml')['model'], auxiliary_time_cond=True, input_size=64)

    def macs(token_compression, dino_tokens):
        with torch.device('meta'):
            network = AblationLightningDiT(**model, token_compression=token_compression).eval()
            x, t, r = torch.randn(1, 32, 64, 64), torch.ones(1), torch.zeros(1)
            z = [torch.randn(1, dino_tokens, 1024)]
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            network(x, t, r, z)
        return counter.get_total_flops() / 2
    # 512^2 crop: teacher DINO@448 gives 1024 tokens, the student's DINO@224 gives 256.
    assert abs(macs(spec, 256) / macs(None, 1024) - expected) < 1e-3


MODEL = dict(patch_size=1, in_channels=6, out_channels=3, hidden_size=32, depth=4, num_heads=2, mlp_ratio=2,
             z_dims=6, encdim_ratio=2, auxiliary_time_cond='auto', use_qknorm=True, use_swiglu=True,
             use_rope=True, use_rmsnorm=True)
HOURGLASS = {'use_cross_attention': True, 'router_config': None, 'dino': {'size': 28},
             'token_compression': {'type': 'hourglass'}}


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
    command += ['tests/ablation_hourglass_smoke.py', '--config', str(config_path)]
    if resume:
        command += ['--resume', str(resume)]
    env = dict(os.environ, OMP_NUM_THREADS='1', WANDB_MODE='offline', TORCHDYNAMO_DISABLE='1')
    run = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize('world', [1, 2])
def test_trainer_curriculum_feature_distill_resume_and_inference(tmp_path, world, monkeypatch):
    if world == 2 and os.environ.get('VOSR_TEST_DDP') != '1':
        pytest.skip('Set VOSR_TEST_DDP=1 on a host that permits Gloo sockets')
    student = dict(HOURGLASS, merge_curriculum_start=1, merge_curriculum_steps=2)
    config_path, _ = trainer_config(tmp_path, student, gt_weight=1., feature_distill_weight=1.,
                                    feature_distill_layers=[0, 1, 2])
    run_trainer(config_path, world)
    output = tmp_path / 'run'
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in records] == [1, 2, 3]
    assert [row['coarse_depth'] for row in records] == [1, 2, 2]
    assert all(row['feature_kd'] > 0 and row['kd'] > 0 and row['grad_norm'] > 0 for row in records)
    assert len(list((output / 'previews').glob('*/sample-00.png'))) == 3
    checkpoint = output / 'checkpoint-00000002'
    assert json.loads((checkpoint / 'model.json').read_text())['coarse_depth'] == 2
    assert json.loads((checkpoint / 'pipeline.json').read_text())['dino'] == {'size': 28, 'layer': 0}
    run_trainer(config_path, world, resume=checkpoint)
    export = output / 'export'
    model = load_export(export)
    assert model.token_compression['fine_in'] == 1 and model.coarse_depth == model.max_coarse_depth == 2

    import inference_vosr_ablation as inference
    from PIL import Image
    from ablation_hourglass_smoke import TinyVAE, fake_dino_features
    monkeypatch.setattr(inference, 'dino_features', fake_dino_features)
    pipeline = json.loads((export / 'pipeline.json').read_text())
    image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (40, 48, 3), dtype=np.uint8))
    restored, keep = inference.restore(model, TinyVAE(), nn.Identity(), image, pipeline, torch.device('cpu'),
                                       tile_size=64, tile_overlap=32, vae_tile_size=0, upscale=1)
    assert restored.size == (48, 40) and keep == 1.
    with pytest.raises(ValueError, match='align to 16 pixels'):
        inference.restore(model, TinyVAE(), nn.Identity(), image, pipeline, torch.device('cpu'),
                          tile_size=56, tile_overlap=0, vae_tile_size=0, upscale=1)


def test_alignment_stage_updates_only_merge_layers(tmp_path):
    config_path, teacher_state = trainer_config(
        tmp_path, HOURGLASS, trainable_parameters=['^token_(merge|unmerge)\\.'],
        feature_distill_weight=1., feature_distill_layers=[0, 2])
    run_trainer(config_path)
    exported = read_weights(tmp_path / 'run/export/model.safetensors')
    for key, value in exported.items():
        if key not in MERGE_KEYS:
            torch.testing.assert_close(value, teacher_state[key], rtol=0, atol=0)
    assert not torch.equal(exported['token_merge.proj.weight'], TokenMerge(32, 2).proj.weight)
    assert not torch.equal(exported['token_unmerge.proj.weight'], TokenUnmerge(32, 2).proj.weight)


@pytest.mark.parametrize('name', sorted(p.name for p in (ROOT / 'configs/ablations').glob('hourglass*.yml')))
def test_shipped_hourglass_configs_resolve(name):
    cfg = load_config(ROOT / 'configs/ablations' / name)
    spec = hourglass_spec(cfg['student']['token_compression'], cfg['model']['depth'])
    assert cfg['student']['router_config'] is None
    assert cfg['dino']['size'] == 448  # the frozen teacher keeps its DINO input
    assert student_dino_config(cfg)['size'] // 14 // spec['cond_pool'] == 16  # 256 DINO tokens
    assert cfg['training']['output_dir'] == f'exp_vosr/{name[:-4]}'
