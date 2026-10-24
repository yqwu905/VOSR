"""SwiGLU neuron pruning of the block MLPs (student.mlp_compression); requires the upstream timm environment."""
import json
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
from safetensors.torch import save_file
from models.lightningdit_ablation import AblationLightningDiT
from models.mlp_compression import ConstantMLP, mlp_compression_spec
from models.swiglu_ffn import SwiGLUFFN
from ablation_utils import load_backbone_state, load_config, load_export, read_weights, save_export
from tools.mlp_neuron_stats import neuron_statistics
from test_ablation_head_split import (MODEL, SHIPPED_STATS, inputs, kwargs, paired_trainer_config, run_trainer,
                                      split, write_stats as write_head_stats)

ROOT = Path(__file__).resolve().parents[1]
HIDDEN = 85  # SwiGLU width of kwargs(): int(2/3 * hidden_size * mlp_ratio)


def teacher():
    torch.manual_seed(0)
    model = AblationLightningDiT(**kwargs()).eval()
    torch.nn.init.normal_(model.final_layer.linear.weight, std=.02)  # zero-initialized in DiT
    return model


def stats_file(path, model):
    """tools/mlp_neuron_stats.py statistics of ``model`` on the test inputs."""
    x, t, r, z = inputs()
    save_file(neuron_statistics(model, [(x, z)]), str(path))
    return path


def prune(path, **extra):
    return dict({'type': 'neuron_prune', 'neuron_stats': str(path)}, **extra)


def student_of(source, mlp_compression, **extra):
    model = AblationLightningDiT(**kwargs(**extra), mlp_compression=mlp_compression).eval()
    load_backbone_state(model, source.state_dict(), allow_new_modules=True)
    return model


def mlp_input(model, index):
    """The normalized, modulated input h of block ``index``'s MLP on the test inputs."""
    captured = {}
    handle = model.blocks[index].mlp.register_forward_pre_hook(lambda module, args: captured.setdefault('h', args[0]))
    with torch.no_grad():
        model(*inputs())
    handle.remove()
    return captured['h']


@pytest.mark.parametrize('config, message', [
    ({'type': 'prune', 'keep_ratio': .5}, 'type: neuron_prune'),
    ({'type': 'neuron_prune', 'keep_ratio': .5, 'ratio': 1}, 'Unknown mlp_compression options'),
    ({'type': 'neuron_prune'}, 'keep_ratio must be in'),
    ({'type': 'neuron_prune', 'keep_ratio': 1.5}, 'keep_ratio must be in'),
    ({'type': 'neuron_prune', 'keep_ratio': True}, 'keep_ratio must be in'),
    ({'type': 'neuron_prune', 'keep_ratio': [.5] * 5}, r'one such ratio per block \(6\)'),
    ({'type': 'neuron_prune', 'keep_ratio': .5, 'score': 'magnitude', 'neuron_stats': 's'}, 'score must be one of'),
    ({'type': 'neuron_prune', 'keep_ratio': .5, 'compensation': 'zero', 'neuron_stats': 's'}, 'compensation must be'),
    ({'type': 'neuron_prune', 'keep_ratio': .5, 'seed': 1.5, 'neuron_stats': 's'}, 'seed must be an integer'),
    ({'type': 'neuron_prune', 'keep_ratio': .5}, 'needs neuron_stats'),
    ({'type': 'neuron_prune', 'keep_ratio': .5, 'score': 'random'}, 'needs neuron_stats'),  # mean compensation
    ({'type': 'neuron_prune', 'hidden': [10] * 6, 'keep_ratio': .5}, 'takes no other options'),
    ({'type': 'neuron_prune', 'hidden': [10] * 5}, r'one width in \[0, 85\] per block'),
    ({'type': 'neuron_prune', 'hidden': [86] * 6}, r'one width in \[0, 85\] per block'),
])
def test_spec_validation(config, message):
    with pytest.raises(ValueError, match=message):
        mlp_compression_spec(config, 6, HIDDEN)


def test_spec_resolves_widths_and_a_resolved_spec_passes_through():
    spec, init = mlp_compression_spec(prune('s.safetensors', keep_ratio=.75), 6, HIDDEN)
    assert spec == {'type': 'neuron_prune', 'hidden': [64] * 6}  # round(63.75)
    assert init == {'score': 'fluctuation', 'compensation': 'mean', 'seed': 0, 'neuron_stats': 's.safetensors'}
    spec, _ = mlp_compression_spec(prune('s', keep_ratio=[1, 0, .5, 1, 1, 0]), 6, HIDDEN)
    assert spec['hidden'] == [85, 0, 42, 85, 85, 0]
    # Random neurons without compensation need no statistics.
    _, init = mlp_compression_spec({'type': 'neuron_prune', 'keep_ratio': .5, 'score': 'random',
                                    'compensation': 'none'}, 6, HIDDEN)
    assert 'neuron_stats' not in init
    assert mlp_compression_spec(spec, 6, HIDDEN) == (spec, None)


def test_keep_ratio_one_is_the_unpruned_model(tmp_path):
    source = teacher()
    model = student_of(source, prune(tmp_path / 'absent.safetensors', keep_ratio=1))
    assert all(type(block.mlp) is SwiGLUFFN and block.mlp.w3.in_features == HIDDEN for block in model.blocks)
    with torch.no_grad():
        torch.testing.assert_close(model(*inputs()), source(*inputs()), rtol=0, atol=0)


def test_fluctuation_keeps_the_top_neurons_and_moves_the_others_mean_into_the_bias(tmp_path):
    source = teacher()
    stats = neuron_statistics(source, [inputs()[::3]])
    model = student_of(source, prune(stats_file(tmp_path / 'stats.safetensors', source), keep_ratio=.5))
    full, pruned = source.state_dict(), model.state_dict()
    for i, block in enumerate(model.blocks):
        assert block.mlp.w3.in_features == 42
        score = stats['variance'][i].double() * stats['output_energy'][i].double()
        keep = score.argsort(descending=True, stable=True)[:42].sort().values
        drop = torch.ones(HIDDEN, dtype=torch.bool)
        drop[keep] = False
        p = f'blocks.{i}.mlp.'
        torch.testing.assert_close(pruned[p + 'w12.weight'], full[p + 'w12.weight'][torch.cat((keep, keep + HIDDEN))])
        torch.testing.assert_close(pruned[p + 'w3.weight'], full[p + 'w3.weight'][:, keep])
        bias = full[p + 'w3.bias'].double() + full[p + 'w3.weight'][:, drop].double() @ stats['mean'][i][drop].double()
        torch.testing.assert_close(pruned[p + 'w3.bias'], bias.float())
    # The pruned MLP = the full MLP with every pruned neuron's activation replaced by its mean.
    h, full_mlp = mlp_input(source, 2), source.blocks[2].mlp
    x1, x2 = full_mlp.w12(h).chunk(2, -1)
    a = F.silu(x1) * x2
    keep = (stats['variance'][2].double() * stats['output_energy'][2].double()).argsort(descending=True,
                                                                                        stable=True)[:42]
    mask = torch.zeros(HIDDEN, dtype=torch.bool)
    mask[keep] = True
    reference = full_mlp.w3(torch.where(mask, a, stats['mean'][2]))
    with torch.no_grad():
        torch.testing.assert_close(model.blocks[2].mlp(h), reference, rtol=1e-5, atol=1e-6)


def test_random_score_is_seeded_and_compensation_none_keeps_the_bias(tmp_path):
    source = teacher()
    path = stats_file(tmp_path / 'stats.safetensors', source)
    ranked = student_of(source, prune(path, keep_ratio=.5)).state_dict()
    first = student_of(source, prune(path, keep_ratio=.5, score='random')).state_dict()
    again = student_of(source, prune(path, keep_ratio=.5, score='random')).state_dict()
    other = student_of(source, prune(path, keep_ratio=.5, score='random', seed=1)).state_dict()
    key = 'blocks.0.mlp.w3.weight'
    assert torch.equal(first[key], again[key]) and not torch.equal(first[key], other[key])
    assert not torch.equal(first[key], ranked[key])
    plain = student_of(source, {'type': 'neuron_prune', 'keep_ratio': .5, 'score': 'random',
                                'compensation': 'none'}).state_dict()  # no statistics needed
    assert torch.equal(plain[key], first[key])  # the same random neurons
    assert torch.equal(plain['blocks.0.mlp.w3.bias'], source.state_dict()['blocks.0.mlp.w3.bias'])
    assert not torch.equal(first['blocks.0.mlp.w3.bias'], plain['blocks.0.mlp.w3.bias'])


def test_keep_ratio_zero_replaces_the_mlp_by_its_mean_output(tmp_path):
    source = teacher()
    stats = neuron_statistics(source, [inputs()[::3]])
    model = student_of(source, prune(stats_file(tmp_path / 'stats.safetensors', source),
                                     keep_ratio=[1, 0, 1, 1, .5, 0]))
    assert [type(block.mlp) for block in model.blocks].count(ConstantMLP) == 2
    assert model.blocks[4].mlp.w3.in_features == 42 and model.blocks[0].mlp.w3.in_features == HIDDEN
    full = source.state_dict()
    for i in (1, 5):
        assert {k for k in model.state_dict() if k.startswith(f'blocks.{i}.mlp.')} == {f'blocks.{i}.mlp.bias'}
        expected = full[f'blocks.{i}.mlp.w3.bias'].double() + \
            full[f'blocks.{i}.mlp.w3.weight'].double() @ stats['mean'][i].double()
        torch.testing.assert_close(model.blocks[i].mlp.bias.data, expected.float())
        h = mlp_input(model, i)
        torch.testing.assert_close(model.blocks[i].mlp(h), expected.float().expand(h.shape))
    # The kept blocks are the source's.
    for key in ('blocks.0.mlp.w12.weight', 'blocks.3.mlp.w3.bias'):
        assert torch.equal(model.state_dict()[key], full[key])


def test_statistics_of_another_checkpoint_or_a_missing_file_are_rejected(tmp_path):
    source = teacher()
    path = stats_file(tmp_path / 'stats.safetensors', source)
    with torch.no_grad():
        source.blocks[3].mlp.w3.weight.mul_(1.01)
    with pytest.raises(ValueError, match=r'measured on another checkpoint \(block 3'):
        student_of(source, prune(path, keep_ratio=.5))
    with pytest.raises(FileNotFoundError, match='tools/mlp_neuron_stats.py'):
        student_of(source, prune(tmp_path / 'absent.safetensors', keep_ratio=.5))
    bad = dict(neuron_statistics(source, [inputs()[::3]]), extra=torch.zeros(1))
    save_file(bad, str(tmp_path / 'bad.safetensors'))
    with pytest.raises(ValueError, match='expected the tensors'):
        student_of(source, prune(tmp_path / 'bad.safetensors', keep_ratio=.5))


def test_export_roundtrip_and_resume_need_no_statistics(tmp_path):
    source = teacher()
    path = stats_file(tmp_path / 'stats.safetensors', source)
    model = student_of(source, prune(path, keep_ratio=[.5, 0, 1, .75, .5, .5]))
    with torch.no_grad():
        for parameter in model.parameters():  # trained-looking weights, not the pruned init
            parameter.add_(torch.randn_like(parameter) * .01)
    save_export(model, tmp_path / 'export', {'dino': {'size': 28, 'layer': 0}})
    config = json.loads((tmp_path / 'export/model.json').read_text())
    assert config['mlp_compression'] == {'type': 'neuron_prune', 'hidden': [42, 0, 85, 64, 42, 42]}
    path.unlink()  # neither the export nor a resume reads the statistics
    exported = load_export(tmp_path / 'export')
    resumed = AblationLightningDiT(**kwargs(), mlp_compression=prune(path, keep_ratio=[.5, 0, 1, .75, .5, .5]))
    load_backbone_state(resumed, read_weights(tmp_path / 'export/model.safetensors'))
    with torch.no_grad():
        for network in (exported, resumed.eval()):
            torch.testing.assert_close(network(*inputs()), model(*inputs()), rtol=0, atol=0)
    with pytest.raises(ValueError, match='resolved mlp_compression'):
        load_backbone_state(exported, source.state_dict())  # an export cannot prune full-width weights
    assert 'mlp_compression' not in AblationLightningDiT(**kwargs()).export_config  # other exports unchanged


def test_combines_with_head_split_and_backward_reaches_every_parameter(tmp_path):
    write_head_stats(tmp_path / 'heads.json')
    source = AblationLightningDiT(**kwargs(), attn_compression=split(tmp_path / 'heads.json', kernel=3)).eval()
    load_backbone_state(source, teacher().state_dict(), allow_new_modules=True)
    path = stats_file(tmp_path / 'stats.safetensors', source)
    torch.manual_seed(1)
    model = AblationLightningDiT(**kwargs(), attn_compression=split(tmp_path / 'heads.json', kernel=3),
                                 mlp_compression=prune(path, keep_ratio=[.5, 0, .5, .5, .5, .5]),
                                 use_checkpoint=True)
    load_backbone_state(model, source.state_dict())  # a head-split export: its local convs load as they are
    model.train()(*inputs()).square().mean().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    assert model.blocks[1].mlp.bias.grad.abs().sum() > 0


def test_rejects_token_compression_router_and_plain_mlps():
    spec = {'type': 'neuron_prune', 'hidden': [42] * 6}
    with pytest.raises(ValueError, match='token_compression or router_config'):
        AblationLightningDiT(**kwargs(), mlp_compression=spec, router_config={})
    with pytest.raises(ValueError, match='use_swiglu: true'):
        AblationLightningDiT(**kwargs(use_swiglu=False), mlp_compression=spec)


@pytest.mark.parametrize('hidden, total, of_head_split', [
    ([3072] * 36, .676, .864),  # ffn_prune_k75
    ([2048] * 36, .570, .728),  # ffn_prune_k50
    ([4096, 4096] + [0] * 7 + [4096] * 20 + [0, 0] + [4096] * 5, .676, .864),  # ffn_prune_k75_depth
])
def test_vosr2_size_compute_ratio(hidden, total, of_head_split):
    from torch.utils.flop_counter import FlopCounterMode
    cfg = load_config(ROOT / 'configs/ablations/head_split_f2.yml')
    model = dict(cfg['model'], auxiliary_time_cond=False, input_size=64)
    heads = dict(cfg['student']['attn_compression'], head_stats=str(SHIPPED_STATS))

    def macs(**compression):
        with torch.device('meta'):
            network = AblationLightningDiT(**model, **compression).eval()
            x, t, r, z = torch.randn(1, 32, 64, 64), torch.ones(1), torch.zeros(1), [torch.randn(1, 1024, 1024)]
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            network(x, t, r, z)
        return counter.get_total_flops() / 2
    dense, head_split = macs(), macs(attn_compression=heads)
    pruned = macs(attn_compression=heads, mlp_compression={'type': 'neuron_prune', 'hidden': hidden})
    assert abs(pruned / dense - total) < 1e-3 and abs(pruned / head_split - of_head_split) < 1e-3


def flat(config, prefix=''):
    items = {}
    for key, value in config.items():
        if isinstance(value, dict):
            items.update(flat(value, f'{prefix}{key}.'))
        else:
            items[prefix + key] = value
    return items


@pytest.mark.parametrize('name, base, changed', [
    ('ffn_prune_ctrl.yml', 'head_split_f2.yml', {'student_checkpoint'}),
    ('ffn_prune_k75.yml', 'ffn_prune_ctrl.yml', {'student.mlp_compression.type', 'student.mlp_compression.keep_ratio',
                                                 'student.mlp_compression.score',
                                                 'student.mlp_compression.compensation',
                                                 'student.mlp_compression.neuron_stats'}),
    ('ffn_prune_k50.yml', 'ffn_prune_k75.yml', {'student.mlp_compression.keep_ratio'}),
    ('ffn_prune_k75_random.yml', 'ffn_prune_k75.yml', {'student.mlp_compression.score',
                                                       'student.mlp_compression.seed'}),
    ('ffn_prune_k75_nocomp.yml', 'ffn_prune_k75.yml', {'student.mlp_compression.compensation'}),
    ('ffn_prune_k75_depth.yml', 'ffn_prune_k75.yml', {'student.mlp_compression.keep_ratio'}),
    ('ffn_prune_k75_joint.yml', 'ffn_prune_k75.yml', {'student_checkpoint', 'student.mlp_compression.neuron_stats',
                                                      'training.max_steps'}),
])
def test_shipped_ffn_prune_configs_change_only_their_ablation_point(name, base, changed):
    root = ROOT / 'configs/ablations'
    config, reference = flat(load_config(root / name)), flat(load_config(root / base))
    diff = {k for k in set(config) | set(reference) if config.get(k, '<absent>') != reference.get(k, '<absent>')}
    assert diff == changed | {'training.output_dir'}
    assert config['training.output_dir'] == f'exp_vosr/{name[:-4]}'
    cfg = load_config(root / name)
    spec = cfg['student'].get('mlp_compression')
    if spec is not None:
        resolved, init = mlp_compression_spec(spec, cfg['model']['depth'], 4096)
        assert sum(resolved['hidden']) == {'ffn_prune_k50.yml': 36 * 2048}.get(name, 27 * 4096)
        assert init['neuron_stats'].startswith('configs/ablations/ffn_prune/')
    assert cfg['training']['kd_weight'] == 1 and cfg['data']['dataset_type'] == 'paired'


def test_stats_tool_feeds_the_trainer_which_prunes_resumes_and_exports_for_inference(tmp_path, monkeypatch):
    import yaml
    config_path = paired_trainer_config(tmp_path, token_merge_lr_multiplier=10.)
    stats = tmp_path / 'neuron_stats.safetensors'
    cfg = yaml.safe_load(config_path.read_text())
    cfg['student']['mlp_compression'] = {'type': 'neuron_prune', 'keep_ratio': [.5, 0, 1, .5],
                                         'neuron_stats': str(stats)}
    config_path.write_text(yaml.safe_dump(cfg))
    # The tool with the smoke fixture's VAE and DINO stand-ins (tests/ablation_hourglass_smoke.py).
    script = ('import sys; sys.path.insert(0, "tests"); import torch\n'
              'from ablation_hourglass_smoke import TinyVAE, fake_dino_features, inject\n'
              'inject("models.qwenimage_vae2d", "AutoencoderKLQwenImage2D", TinyVAE)\n'
              'import tools.mlp_neuron_stats as tool\n'
              'tool.load_dino = lambda *args: torch.nn.Identity(); tool.dino_features = fake_dino_features\n'
              f'sys.argv = ["tool", "--config", {str(config_path)!r}, "--output", {str(stats)!r}, "--samples", "3"]\n'
              'tool.main()\n')
    run = subprocess.run([sys.executable, '-c', script], cwd=ROOT, text=True, capture_output=True, timeout=300,
                         env=dict(os.environ, OMP_NUM_THREADS='1', TORCHDYNAMO_DISABLE='1'))
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'wrote' in run.stdout and 'sample 3/3' in run.stdout
    from safetensors import safe_open
    with safe_open(str(stats), 'pt') as handle:
        assert handle.metadata()['source'] == str(tmp_path / 'teacher.pt') and handle.metadata()['samples'] == '3'
    teacher_state = torch.load(tmp_path / 'teacher.pt')
    written = read_weights(stats)
    assert written['mean'].shape == (4, 42) and (written['variance'] >= 0).all()
    torch.testing.assert_close(written['w3_sq_norm'], torch.stack(
        [teacher_state[f'blocks.{i}.mlp.w3.weight'].double().square().sum() for i in range(4)]).float())

    run_trainer(config_path)
    output, = (tmp_path / 'run').iterdir()
    model_json = json.loads((output / 'export/model.json').read_text())
    assert model_json['mlp_compression'] == {'type': 'neuron_prune', 'hidden': [21, 0, 42, 21]}
    assert model_json['attn_compression']['factor'] == 2
    first = read_weights(output / 'checkpoint-00000002/model.safetensors')
    assert first['blocks.0.mlp.w3.weight'].shape == (32, 21) and first['blocks.1.mlp.bias'].shape == (32,)
    stats.unlink()  # resume and inference read the pruned weights, not the statistics
    run_trainer(config_path, resume=output / 'checkpoint-00000002')
    model = load_export(output / 'export')
    assert model.blocks[2].mlp.w3.in_features == 42 and isinstance(model.blocks[1].mlp, ConstantMLP)

    import inference_vosr_ablation as inference
    from PIL import Image
    from ablation_hourglass_smoke import TinyVAE, fake_dino_features
    monkeypatch.setattr(inference, 'dino_features', fake_dino_features)
    pipeline = json.loads((output / 'export/pipeline.json').read_text())
    image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (40, 48, 3), dtype=np.uint8))
    restored, _ = inference.restore(model, TinyVAE(), nn.Identity(), image, pipeline, torch.device('cpu'),
                                    tile_size=64, tile_overlap=32, vae_tile_size=0, upscale=1)
    assert restored.size == (48, 40)
