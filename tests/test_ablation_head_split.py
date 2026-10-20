"""Head-split self-attention token reduction (student.attn_compression); requires the upstream timm environment."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
pytest.importorskip('timm', reason='Full backbone requires the upstream timm environment')
from models.attn_compression import HeadSplitAttention, attn_compression_spec, pooled_rope
from models.lightningdit import Attention
from models.lightningdit_ablation import AblationLightningDiT
from models.token_hourglass import new_module_key
from ablation_utils import load_backbone_state, load_config, load_export, read_weights, save_export

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_STATS = ROOT / 'configs/ablations/head_split/vosr2_head_stats.json'


def kwargs(**extra):
    base = dict(input_size=16, patch_size=2, in_channels=4, out_channels=2, hidden_size=64, depth=6,
                num_heads=4, mlp_ratio=2, z_dims=16, auxiliary_time_cond=True, use_rope=True,
                use_qknorm=True, use_swiglu=True, use_rmsnorm=True)
    return dict(base, **extra)


def inputs(side=16):
    torch.manual_seed(5)
    return torch.randn(2, 4, side, side), torch.ones(2), torch.zeros(2), [torch.randn(2, 16, 16)]


def write_stats(path, depth=6, heads=4, radius=3, seed=0):
    """Synthetic tools/attn_head_stats.py output: random 3x3 masses and positive attention patterns."""
    generator = torch.Generator().manual_seed(seed)
    mass = torch.rand(depth, heads, generator=generator)
    pattern = torch.rand(depth, heads, (2 * radius + 1) ** 2, generator=generator)
    path.write_text(json.dumps({'radius': radius, 'mass3x3': mass.tolist(), 'pattern': pattern.tolist()}))
    return mass, pattern


def split(path, **extra):
    return dict({'type': 'head_split', 'head_stats': str(path)}, **extra)


def teacher_and_student(attn_compression, **extra):
    torch.manual_seed(0)
    teacher = AblationLightningDiT(**kwargs(**extra)).eval()
    torch.nn.init.normal_(teacher.final_layer.linear.weight, std=.02)
    student = AblationLightningDiT(**kwargs(**extra), attn_compression=attn_compression).eval()
    load_backbone_state(student, teacher.state_dict(), allow_new_modules=True)
    return teacher, student


def test_spec_resolves_local_heads_and_kernels_from_head_stats(tmp_path):
    mass, pattern = write_stats(tmp_path / 'stats.json')
    spec, kernels = attn_compression_spec(split(tmp_path / 'stats.json'), 6, 4)
    local = [[h for h in range(4) if mass[i, h] > .5] for i in range(6)]
    assert spec == {'type': 'head_split', 'factor': 2, 'kernel': 7, 'upsample': 'bilinear',
                    'blocks': list(range(6)), 'local_heads': local}
    assert sorted(kernels) == [i for i in range(6) if local[i]]
    for index, heads in enumerate(local):
        if heads:
            window = pattern[index, heads].view(len(heads), 7, 7)
            torch.testing.assert_close(kernels[index], window / window.sum((1, 2), keepdim=True))
    # A 3x3 kernel crops the centre of the 7x7 pattern before renormalizing.
    _, small = attn_compression_spec(split(tmp_path / 'stats.json', kernel=3), 6, 4)
    index = sorted(small)[0]
    window = pattern[index, local[index]].view(-1, 7, 7)[:, 2:5, 2:5]
    torch.testing.assert_close(small[index], window / window.sum((1, 2), keepdim=True))
    # Threshold and block subset; blocks outside the subset keep dense attention and no local heads.
    spec, _ = attn_compression_spec(split(tmp_path / 'stats.json', local_threshold=0, blocks=[4, 1]), 6, 4)
    assert spec['blocks'] == [1, 4] and spec['local_heads'] == [[], [0, 1, 2, 3], [], [], [0, 1, 2, 3], []]
    spec, kernels = attn_compression_spec(split(tmp_path / 'stats.json', local_threshold=1), 6, 4)
    assert spec['local_heads'] == [[]] * 6 and kernels == {}
    # The resolved structure (model.json) needs no stats file and passes through unchanged.
    resolved, _ = attn_compression_spec(split(tmp_path / 'stats.json', factor=4, upsample='nearest'), 6, 4)
    assert attn_compression_spec(resolved, 6, 4) == (resolved, None)


@pytest.mark.parametrize('extra, message', [
    ({'type': 'pool'}, 'type: head_split'),
    ({'factr': 2}, 'Unknown attn_compression options'),
    ({'factor': 0}, 'factor must be an integer >= 1'),
    ({'factor': True}, 'factor must be an integer >= 1'),
    ({'kernel': 4}, 'kernel must be odd'),
    ({'kernel': 9}, 'exceeds the head_stats window 7'),
    ({'upsample': 'bicubic'}, 'upsample must be one of'),
    ({'local_threshold': 1.5}, 'local_threshold must be in'),
    ({'blocks': []}, 'blocks must be unique'),
    ({'blocks': [6]}, 'blocks must be unique'),
    ({'blocks': [1, 1]}, 'blocks must be unique'),
    ({'local_heads': [[0]] * 5}, 'one head list per block'),
    ({'local_heads': [[4]] + [[]] * 5}, r'local_heads\[0\] must be unique heads'),
    ({'local_heads': [[0, 0]] + [[]] * 5}, r'local_heads\[0\] must be unique heads'),
    ({'blocks': [1], 'local_heads': [[0]] + [[]] * 5}, 'block 0 is not in blocks'),
    ({'head_stats': None}, 'needs local_heads or head_stats'),
])
def test_spec_validation(tmp_path, extra, message):
    write_stats(tmp_path / 'stats.json')
    with pytest.raises(ValueError, match=message):
        attn_compression_spec(split(tmp_path / 'stats.json', **extra), 6, 4)


def test_spec_rejects_stats_of_another_model_and_bad_combinations(tmp_path):
    write_stats(tmp_path / 'stats.json', depth=5)
    with pytest.raises(ValueError, match=r'has \(5, 4\) \(blocks, heads\), the model \(6, 4\)'):
        attn_compression_spec(split(tmp_path / 'stats.json'), 6, 4)
    write_stats(tmp_path / 'stats.json')
    for extra in ({'token_compression': {'type': 'hourglass'}}, {'router_config': {'init_keep_prob': .9}}):
        with pytest.raises(ValueError, match='cannot be combined'):
            AblationLightningDiT(**kwargs(), attn_compression=split(tmp_path / 'stats.json'), **extra)
    with pytest.raises(ValueError, match='divisible by attn_compression.factor'):  # 6x6 tokens, 4x4 pooling
        AblationLightningDiT(**kwargs(input_size=12), attn_compression=split(tmp_path / 'stats.json', factor=4))


def test_factor_1_without_local_heads_is_the_dense_student():
    teacher, student = teacher_and_student({'type': 'head_split', 'factor': 1, 'local_heads': [[]] * 6})
    assert all(isinstance(block.attn, HeadSplitAttention) for block in student.blocks)
    with torch.no_grad():
        torch.testing.assert_close(student(*inputs()), teacher(*inputs()))


def attention_module(seed=0):
    torch.manual_seed(seed)
    attn = Attention(64, num_heads=4, qkv_bias=True, qk_norm=True, use_rmsnorm=True)
    with torch.no_grad():
        attn.q_norm.weight.uniform_(.5, 1.5)
        attn.k_norm.weight.uniform_(.5, 1.5)
    return attn


def centroid_rope(side, factor, head_dim=16):
    return pooled_rope(torch.arange(side // factor).float() * factor + (factor - 1) / 2, head_dim // 2, torch.float32)


def reference_head_split(attn, x, local_heads, kernels, factor, rope):
    """Per-head reference: global heads attend among factor x factor averages, each local head is attention with
    the fixed relative-offset weights of its kernel (zero outside the grid); every head projects with its columns
    of proj, the bias is added once."""
    b, n, c = x.shape
    side, hd = math.isqrt(n), c // attn.num_heads
    s = side // factor
    pooled = x.reshape(b, s, factor, s, factor, c).mean((2, 4)).reshape(b, s * s, c)
    q, k, v = F.linear(pooled, attn.qkv.weight, attn.qkv.bias).chunk(3, -1)
    q, k, v = (t.reshape(b, s * s, attn.num_heads, hd).transpose(1, 2) for t in (q, k, v))
    q, k = rope(attn.q_norm(q)), rope(attn.k_norm(k))
    coarse = torch.zeros(b, s * s, c)
    for h in range(attn.num_heads):
        if h not in local_heads:
            weights = (q[:, h] @ k[:, h].transpose(-1, -2) / math.sqrt(hd)).softmax(-1)
            coarse += (weights @ v[:, h]) @ attn.proj.weight[:, h * hd:(h + 1) * hd].T
    coarse = F.interpolate(coarse.transpose(1, 2).reshape(b, c, s, s), scale_factor=factor, mode='bilinear',
                           align_corners=False).flatten(2).transpose(1, 2)
    values = F.linear(x, attn.qkv.weight[2 * c:], attn.qkv.bias[2 * c:]).reshape(b, n, attn.num_heads, hd)
    fine = torch.zeros(b, n, c)
    radius = kernels.shape[-1] // 2 if len(local_heads) else 0
    for i, h in enumerate(local_heads):
        weights = torch.zeros(n, n)
        for y in range(side):
            for x_ in range(side):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if 0 <= y + dy < side and 0 <= x_ + dx < side:
                            weights[y * side + x_, (y + dy) * side + x_ + dx] = kernels[i, dy + radius, dx + radius]
        fine += (weights @ values[:, :, h]) @ attn.proj.weight[:, h * hd:(h + 1) * hd].T
    return coarse + fine + attn.proj.bias


@pytest.mark.parametrize('local_heads', [[], [1, 3], [0, 1, 2, 3]])
def test_head_split_matches_per_head_reference(local_heads):
    attn = attention_module()
    module = HeadSplitAttention(attn, local_heads, factor=2, kernel=3, upsample='bilinear')
    kernels = torch.rand(len(local_heads), 3, 3)
    module.set_local_kernels(kernels)
    x = torch.randn(2, 64, 64)  # an 8x8 token grid
    with torch.no_grad():
        out = module(x, rope=centroid_rope(8, 2))
        expected = reference_head_split(attn, x, local_heads, kernels, 2, centroid_rope(8, 2))
    torch.testing.assert_close(out, expected)
    assert (module.local_conv is None) == (not local_heads)
    assert module.global_heads == [h for h in range(4) if h not in local_heads]


def test_local_convs_start_as_identity_and_nearest_upsampling_copies_groups():
    attn = attention_module()
    module = HeadSplitAttention(attn, [2], factor=2, kernel=5, upsample='nearest')
    weight = module.local_conv.weight
    assert weight.shape == (16, 1, 5, 5) and torch.equal(weight.sum((1, 2, 3)), torch.ones(16))
    assert torch.equal(weight[:, 0, 2, 2], torch.ones(16))
    x = torch.randn(1, 64, 64)
    with torch.no_grad():
        local = F.linear(F.linear(x, attn.qkv.weight[160:176], attn.qkv.bias[160:176]), attn.proj.weight[:, 32:48])
        coarse = module(x, centroid_rope(8, 2)) - local
    grid = coarse.reshape(1, 4, 2, 4, 2, 64)
    torch.testing.assert_close(grid, grid[:, :, :1, :, :1].expand_as(grid), rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match='Expected 1 local-head kernels'):
        module.set_local_kernels(torch.ones(2, 5, 5))
    with pytest.raises(ValueError, match='sides divisible by 2'):
        module(torch.randn(1, 49, 64))


def test_head_stats_initialize_the_local_convs_per_head(tmp_path):
    write_stats(tmp_path / 'stats.json')
    spec, kernels = attn_compression_spec(split(tmp_path / 'stats.json', kernel=5), 6, 4)
    model = AblationLightningDiT(**kwargs(), attn_compression=split(tmp_path / 'stats.json', kernel=5))
    for index, heads in enumerate(spec['local_heads']):
        attn = model.blocks[index].attn
        assert attn.local_heads == heads
        if heads:
            expected = kernels[index].repeat_interleave(16, 0).unsqueeze(1)  # each head's 16 channels
            torch.testing.assert_close(attn.local_conv.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize('rope_mode', ['interpolate', 'extrapolate'])
@pytest.mark.parametrize('side', [8, 16])
def test_pooled_rope_sits_at_group_centroids(side, rope_mode):
    model = AblationLightningDiT(**kwargs(), attn_compression={'type': 'head_split', 'local_heads': [[]] * 6},
                                 rope_mode=rope_mode)
    scale = 8 / side if rope_mode == 'interpolate' else 1.  # reference grid: input_size 16 / patch 2
    expected = pooled_rope((torch.arange(side // 2).float() * 2 + .5) * scale, 8, torch.float32)
    t = torch.randn(2, 4, (side // 2) ** 2, 16)
    torch.testing.assert_close(model._pooled_rope(side, t.device, torch.float32)(t), expected(t))
    with torch.no_grad():  # a non-reference grid runs end to end
        assert model(*inputs(side * 2))[0].shape == (2, side * 2, side * 2)


def test_sparse_attn_type_applies_to_the_global_heads_on_the_pooled_grid(tmp_path):
    common = dict(input_size=32, depth=2)  # 16x16 tokens -> 8x8 pooled: one 8x8 window
    write_stats(tmp_path / 'stats.json', depth=2)
    torch.manual_seed(0)
    full = AblationLightningDiT(**kwargs(**common), attn_compression=split(tmp_path / 'stats.json', kernel=3)).eval()
    sparse = AblationLightningDiT(**kwargs(**common, attn_type='sparse'),
                                  attn_compression=split(tmp_path / 'stats.json', kernel=3)).eval()
    sparse.load_state_dict(full.state_dict())
    for block in sparse.blocks:
        heads = len(block.attn.global_heads)
        assert block.attn.attn_type == 'sparse' and (block.attn.sparse_attn is None) == (heads == 0)
        if heads:
            assert block.attn.sparse_attn.heads == heads
    with torch.no_grad():  # 8 or fewer windows: block-sparse attention keeps them all
        torch.testing.assert_close(sparse(*inputs(32)), full(*inputs(32)))
    sparse.set_attn_type('local')
    assert sparse.export_config['attn_type'] == 'local'


def test_dense_teacher_initializes_the_student_and_local_convs_are_new(tmp_path):
    write_stats(tmp_path / 'stats.json')
    teacher, student = teacher_and_student(split(tmp_path / 'stats.json', kernel=3))
    teacher_state, student_state = teacher.state_dict(), student.state_dict()
    new = sorted(k for k in student_state if k not in teacher_state)
    assert new == sorted(f'blocks.{i}.attn.local_conv.weight' for i, heads in
                         enumerate(student.attn_compression['local_heads']) if heads)
    assert all(new_module_key(k) for k in new) and not any(new_module_key(k) for k in teacher_state)
    for key, value in teacher_state.items():
        torch.testing.assert_close(student_state[key], value, rtol=0, atol=0)
    report = load_backbone_state(student, teacher_state, allow_new_modules=True)
    assert report['new_token_merge_parameters'] == new
    with pytest.raises(ValueError, match='Missing backbone weights'):
        load_backbone_state(student, teacher_state)


def test_checkpointed_backward_reaches_every_parameter(tmp_path):
    write_stats(tmp_path / 'stats.json')
    torch.manual_seed(0)
    model = AblationLightningDiT(**kwargs(), attn_compression=split(tmp_path / 'stats.json', kernel=3),
                                 use_checkpoint=True).train()
    torch.nn.init.normal_(model.final_layer.linear.weight, std=.02)  # zero-initialized in DiT
    model(*inputs()).square().mean().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    conv = next(block.attn.local_conv for block in model.blocks if block.attn.local_conv is not None)
    assert conv.weight.grad.abs().sum() > 0


def test_export_roundtrip_keeps_the_resolved_structure(tmp_path):
    write_stats(tmp_path / 'stats.json')
    _, student = teacher_and_student(split(tmp_path / 'stats.json', kernel=3, factor=4, upsample='nearest'))
    with torch.no_grad():
        for block in student.blocks:  # trained-looking convs, not the stats init
            if block.attn.local_conv is not None:
                block.attn.local_conv.weight.normal_()
    save_export(student, tmp_path / 'export', {'dino': {'size': 28, 'layer': 0}})
    config = json.loads((tmp_path / 'export/model.json').read_text())
    assert config['attn_compression'] == student.attn_compression and 'head_stats' not in config['attn_compression']
    (tmp_path / 'stats.json').unlink()  # the export no longer needs the statistics
    model = load_export(tmp_path / 'export')
    assert model.latent_multiple == 8 and model.attn_compression['upsample'] == 'nearest'
    with torch.no_grad():
        torch.testing.assert_close(model(*inputs()), student(*inputs()), rtol=0, atol=0)
    plain = AblationLightningDiT(**kwargs())
    assert 'attn_compression' not in plain.export_config  # other exports keep their model.json


def test_sharded_bf16_pack_reads_back_and_matches_under_autocast(tmp_path):
    write_stats(tmp_path / 'stats.json')
    _, student = teacher_and_student(split(tmp_path / 'stats.json', kernel=3))
    save_export(student, tmp_path / 'export', {'dino': {'size': 28, 'layer': 0}, 'precision': 'bf16'})
    run = subprocess.run([sys.executable, 'tools/pack_export.py', '--export', str(tmp_path / 'export'),
                          '--output', str(tmp_path / 'packed'), '--max-shard-gb', str(2e5 / 2 ** 30)],
                         cwd=ROOT, text=True, capture_output=True, timeout=300,
                         env=dict(os.environ, TORCHDYNAMO_DISABLE='1'))
    assert run.returncode == 0, run.stdout + run.stderr
    summary = json.loads(run.stdout.strip().splitlines()[-1])
    assert summary['shards'] > 1
    index = json.loads((tmp_path / 'packed/model.safetensors.index.json').read_text())
    state = read_weights(tmp_path / 'packed')  # directory -> index -> shards
    assert set(state) == set(index['weight_map']) == set(student.state_dict())
    assert state['blocks.0.attn.qkv.weight'].dtype == torch.bfloat16
    assert state['blocks.0.scale_shift_table'].dtype == torch.float32
    assert state['blocks.0.attn.qkv.bias'].dtype == torch.float32
    packed = load_export(tmp_path / 'packed')
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        torch.testing.assert_close(packed(*inputs()), student(*inputs()), rtol=0, atol=0)


@pytest.mark.parametrize('factor, total, attention', [(2, .782, .233), (4, .746, .103)])
def test_vosr2_size_compute_ratio(factor, total, attention):
    from torch.utils.flop_counter import FlopCounterMode
    cfg = load_config(ROOT / 'configs/ablations/head_split_f2.yml')
    model = dict(cfg['model'], auxiliary_time_cond=False, input_size=64)
    spec = dict(cfg['student']['attn_compression'], head_stats=str(SHIPPED_STATS), factor=factor)

    def macs(attn_compression):
        with torch.device('meta'):
            network = AblationLightningDiT(**model, attn_compression=attn_compression).eval()
            x, t, r, z = torch.randn(1, 32, 64, 64), torch.ones(1), torch.zeros(1), [torch.randn(1, 1024, 1024)]
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            network(x, t, r, z)
        self_attention = sum(sum(ops.values()) for name, ops in counter.get_flop_counts().items()
                             if name.count('.') == 3 and name.endswith('.attn'))
        return counter.get_total_flops() / 2, self_attention / 2
    (dense, dense_attention), (split_total, split_attention) = macs(None), macs(spec)
    assert abs(split_total / dense - total) < 1e-3
    assert abs(split_attention / dense_attention - attention) < 1e-3


@pytest.mark.parametrize('name', ['head_split_f2.yml', 'head_split_f4.yml'])
def test_shipped_head_split_configs_resolve(name):
    cfg = load_config(ROOT / 'configs/ablations' / name)
    spec, kernels = attn_compression_spec(cfg['student']['attn_compression'], cfg['model']['depth'],
                                          cfg['model']['num_heads'])
    assert spec['factor'] == int(name[-5]) and spec['blocks'] == list(range(36)) and spec['kernel'] == 7
    assert sum(map(len, spec['local_heads'])) == 143 and len(kernels) == 34
    assert cfg['student']['router_config'] is None and cfg['student'].get('token_compression') is None
    assert cfg['data']['dataset_type'] == 'paired' and cfg['data']['upscale'] == 4
    assert cfg['dino'] == load_config(ROOT / 'configs/ablations/base_vosr2.yml')['dino']  # the teacher's DINO input
    tc = cfg['training']
    assert tc['output_dir'] == f'exp_vosr/{name[:-4]}' and tc['kd_weight'] == 1 and tc['gt_weight'] == 0
    assert (tc['lr_schedule'], tc['min_lr_ratio'], tc['ema_decay']) == ('cosine', .1, .999)


def test_shipped_stats_select_the_release_heads():
    stats = json.loads(SHIPPED_STATS.read_text())
    release = json.loads((ROOT / 'weights/head_split_f2/model.json').read_text())['attn_compression']
    spec, _ = attn_compression_spec({'type': 'head_split', 'head_stats': str(SHIPPED_STATS)}, 36, 24)
    assert spec == release
    assert stats['radius'] == 3 and len(stats['pattern']) == 36 and len(stats['pattern'][0][0]) == 49


MODEL = dict(patch_size=1, in_channels=6, out_channels=3, hidden_size=32, depth=4, num_heads=2, mlp_ratio=2,
             z_dims=6, encdim_ratio=2, auxiliary_time_cond='auto', use_qknorm=True, use_swiglu=True,
             use_rope=True, use_rmsnorm=True)


def paired_trainer_config(tmp_path, **training):
    import yaml
    from PIL import Image
    rng = np.random.default_rng(0)
    for folder in ('LQ', 'HQ'):
        (tmp_path / folder).mkdir()
    for name in ('a', 'b'):
        hq = rng.integers(0, 255, (80, 96, 3), dtype=np.uint8)
        Image.fromarray(hq).save(tmp_path / 'HQ' / f'{name}.png')
        Image.fromarray((hq * .8).astype(np.uint8)).save(tmp_path / 'LQ' / f'{name}.png')
    (tmp_path / 'boxes.json').write_text(json.dumps({'a': [[10, 10, 40, 10, 40, 30, 10, 30]], 'b': []}))
    write_stats(tmp_path / 'stats.json', depth=4, heads=2)
    torch.manual_seed(0)
    teacher = AblationLightningDiT(**dict(MODEL, auxiliary_time_cond=True), input_size=8)
    torch.nn.init.normal_(teacher.final_layer.linear.weight, std=.02)
    torch.save(teacher.state_dict(), tmp_path / 'teacher.pt')
    student = {'use_cross_attention': True, 'router_config': None,
               'attn_compression': split(tmp_path / 'stats.json', kernel=3)}
    cfg = {'teacher_checkpoint': str(tmp_path / 'teacher.pt'), 'vae_path': 'fixture', 'model': MODEL,
           'dino': {'size': 56, 'layer': 0},
           'data': {'resolution': 64, 'dataset_type': 'paired', 'lq_dir': str(tmp_path / 'LQ'),
                    'hq_dir': str(tmp_path / 'HQ'), 'text_boxes': str(tmp_path / 'boxes.json'),
                    'text_crop_prob': .5, 'upscale': 4},
           'student': student,
           'training': dict(output_dir=str(tmp_path / 'run'), precision='fp32', report_to='none',
                            batch_size_per_gpu=1, gradient_accumulation_steps=1, max_steps=3, learning_rate=1e-3,
                            lr_warmup_steps=1, save_every=2, log_every=1, preview_every=2, preview_num_images=1,
                            num_workers=0, zero_optimizer=False, **training)}
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump(cfg))
    return path


def run_trainer(config_path, resume=None):
    command = [sys.executable, 'tests/ablation_hourglass_smoke.py', '--config', str(config_path)]
    if resume:
        command += ['--resume', str(resume)]
    env = dict(os.environ, OMP_NUM_THREADS='1', WANDB_MODE='offline', TORCHDYNAMO_DISABLE='1')
    run = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr


def test_trainer_paired_data_cosine_ema_resume_and_inference(tmp_path, monkeypatch):
    config_path = paired_trainer_config(tmp_path, lr_schedule='cosine', min_lr_ratio=.1, ema_decay=.5,
                                        token_merge_lr_multiplier=10.)
    run_trainer(config_path)
    output, = (tmp_path / 'run').iterdir()
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    # Warmup 1 step, then cosine from 1e-3 to 1e-4 at step 3.
    assert [row['learning_rate'] for row in records] == pytest.approx([1e-3, 5.5e-4, 1e-4])
    assert all(row['token_merge_learning_rate'] == pytest.approx(10 * row['learning_rate']) for row in records)
    assert all(row['kd'] > 0 for row in records)
    checkpoint = output / 'checkpoint-00000002'
    assert json.loads((checkpoint / 'pipeline.json').read_text())['upscale'] == 4
    raw, ema = read_weights(checkpoint / 'model.safetensors'), read_weights(checkpoint / 'ema/model.safetensors')
    teacher = torch.load(tmp_path / 'teacher.pt')
    key = 'blocks.1.attn.qkv.weight'
    # decay .5: the EMA lies between the initial (teacher) and the trained weights.
    assert not torch.equal(raw[key], ema[key]) and not torch.equal(ema[key], teacher[key])
    export = read_weights(output / 'export/model.safetensors')
    final_ema = read_weights(output / 'checkpoint-00000003/ema/model.safetensors')
    assert all(torch.equal(export[k], final_ema[k]) for k in export)  # export/ = the EMA weights
    assert not torch.equal(export[key], read_weights(output / 'checkpoint-00000003/model.safetensors')[key])
    run_trainer(config_path, resume=checkpoint)  # resumes the EMA from checkpoint/ema
    model = load_export(output / 'export')
    assert model.attn_compression['factor'] == 2 and model.latent_multiple == 2

    import inference_vosr_ablation as inference
    from PIL import Image
    from ablation_hourglass_smoke import TinyVAE, fake_dino_features
    monkeypatch.setattr(inference, 'dino_features', fake_dino_features)
    pipeline = json.loads((output / 'export/pipeline.json').read_text())
    image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (40, 48, 3), dtype=np.uint8))
    restored, _ = inference.restore(model, TinyVAE(), nn.Identity(), image, pipeline, torch.device('cpu'),
                                    tile_size=64, tile_overlap=32, vae_tile_size=0, upscale=1)
    assert restored.size == (48, 40)
    with pytest.raises(ValueError, match='align to 16 pixels'):
        inference.restore(model, TinyVAE(), nn.Identity(), image, pipeline, torch.device('cpu'),
                          tile_size=56, tile_overlap=0, vae_tile_size=0, upscale=1)


@pytest.mark.parametrize('training, message', [
    ({'lr_schedule': 'linear'}, 'lr_schedule must be constant or cosine'),
    ({'min_lr_ratio': 2}, 'min_lr_ratio in'),
    ({'ema_decay': 1}, 'ema_decay must be in'),
])
def test_trainer_rejects_bad_schedule_and_ema(tmp_path, training, message):
    with pytest.raises(AssertionError, match=message):
        run_trainer(paired_trainer_config(tmp_path, **training))
