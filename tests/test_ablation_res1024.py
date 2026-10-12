"""1024 tiles: RoPE extrapolation, DINO tiling, tiled teacher, res1024 configs and 1024 crops."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from ablation_utils import dino_features, load_config, student_dino_config, teacher_velocity, vae_tiling

ROOT = Path(__file__).resolve().parents[1]


def kwargs(**extra):
    base = dict(input_size=16, patch_size=2, in_channels=4, out_channels=2, hidden_size=64, depth=6,
                num_heads=4, mlp_ratio=2, z_dims=16, auxiliary_time_cond=True, use_rope=True,
                use_qknorm=True, use_swiglu=True, use_rmsnorm=True)
    return dict(base, **extra)


def same_weights(source, *targets):
    from ablation_utils import load_backbone_state
    for target in targets:
        load_backbone_state(target, source.state_dict(), allow_new_modules=True)
        if getattr(source, 'token_compression', None):
            target.load_state_dict({k: v for k, v in source.state_dict().items() if 'token_' in k}, strict=False)


@pytest.mark.parametrize('compression', [None, {'type': 'hourglass'}])
def test_extrapolate_matches_a_model_built_for_the_larger_grid(compression):
    pytest.importorskip('timm')
    from models.lightningdit_ablation import AblationLightningDiT
    torch.manual_seed(0)
    small = AblationLightningDiT(**kwargs(), rope_mode='extrapolate', token_compression=compression).eval()
    torch.nn.init.normal_(small.final_layer.linear.weight, std=.02)
    native = AblationLightningDiT(**kwargs(input_size=32), token_compression=compression).eval()
    interpolated = AblationLightningDiT(**kwargs(), token_compression=compression).eval()
    same_weights(small, native, interpolated)
    torch.manual_seed(5)
    x, t, r, z = torch.randn(2, 4, 32, 32), torch.ones(2), torch.zeros(2), [torch.randn(2, 16, 16)]
    with torch.no_grad():
        # Integer positions 0..15 on a reference grid of 8: the RoPE a 32-latent model is built with.
        torch.testing.assert_close(small(x, t, r, z), native(x, t, r, z))
        assert not torch.allclose(interpolated(x, t, r, z), native(x, t, r, z), atol=1e-4)
        # On the reference grid both modes use the trained RoPE buffer: bit-identical.
        torch.testing.assert_close(small(x[..., :16, :16], t, r, z), interpolated(x[..., :16, :16], t, r, z),
                                   rtol=0, atol=0)


def test_rope_mode_default_export_and_override(tmp_path):
    pytest.importorskip('timm')
    from models.lightningdit_ablation import AblationLightningDiT
    from ablation_utils import load_export, save_export
    with pytest.raises(ValueError, match='rope_mode'):
        AblationLightningDiT(**kwargs(), rope_mode='ntk')
    model = AblationLightningDiT(**kwargs())
    assert model.rope_mode == 'interpolate' and model.export_config['rope_mode'] == 'interpolate'
    model = AblationLightningDiT(**kwargs(), rope_mode='extrapolate')
    save_export(model, tmp_path, {})
    assert json.loads((tmp_path / 'model.json').read_text())['rope_mode'] == 'extrapolate'
    restored = load_export(tmp_path)
    assert restored.rope_mode == 'extrapolate'
    restored.set_rope_mode('interpolate')
    assert restored.export_config['rope_mode'] == 'interpolate'
    config = json.loads((tmp_path / 'model.json').read_text())
    del config['rope_mode']  # exports written before rope_mode existed
    assert AblationLightningDiT(**config).rope_mode == 'interpolate'


class FakeDino(nn.Module):
    """One token per 14x14 patch (mean colour) after a CLS token, like DINOv2's token layout."""
    def __init__(self):
        super().__init__()
        self.blocks, self.norm = nn.ModuleList([nn.Identity()]), nn.Identity()
        self.inputs = 0  # images DINO has encoded

    def prepare_tokens_with_masks(self, x):
        self.inputs += x.shape[0]
        tokens = F.avg_pool2d(x, 14).flatten(2).transpose(1, 2)
        return torch.cat((tokens.new_zeros(tokens.shape[0], 1, tokens.shape[2]), tokens), 1)


def test_dino_tiles_are_laid_out_as_one_spatial_grid():
    torch.manual_seed(0)
    lq, dino = torch.rand(2, 3, 64, 96) * 2 - 1, FakeDino()
    config = {'size': 28, 'layer': 0}
    tiled = dino_features(dino, lq, dict(config, tile=32))[0]
    assert tiled.shape == (2, 4 * 6, 3)
    grid = tiled.reshape(2, 4, 6, 3)
    for i in range(2):
        for j in range(3):
            expected = dino_features(dino, lq[:, :, 32 * i:32 * i + 32, 32 * j:32 * j + 32], config)[0]
            torch.testing.assert_close(grid[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2].reshape(2, 4, 3), expected)
    # A crop no larger than the tile is the whole-image path.
    torch.testing.assert_close(dino_features(dino, lq[..., :32, :32], dict(config, tile=32))[0],
                               dino_features(dino, lq[..., :32, :32], config)[0], rtol=0, atol=0)
    with pytest.raises(ValueError, match='tile'):
        dino_features(dino, lq[..., :48], dict(config, tile=32))
    assert student_dino_config({'dino': config, 'student': {'dino': {'tile': 32}}})['tile'] == 32


def test_teacher_velocity_tiles_like_inference():
    calls = []

    def teacher(inp, t, r, features):
        calls.append((inp.shape[-1], features[0].shape))
        return inp[:, :2] * 3 + features[0].mean()  # pointwise in the latent, plus the tile's DINO mean

    lq, inp = torch.zeros(1, 3, 1024, 1024), torch.randn(1, 4, 128, 128)
    t, r = torch.ones(1), torch.zeros(1)
    config = {'training': {'teacher_tile': 512, 'teacher_tile_overlap': 64}, 'dino': {'size': 28, 'layer': 0}}
    tiled = teacher_velocity(teacher, inp, t, r, lq, None, FakeDino(), config)
    assert len(calls) == 9 and all(size == 64 and shape == (1, 4, 3) for size, shape in calls)
    constant = dino_features(FakeDino(), lq[..., :512, :512], config['dino'])[0].mean()  # same LQ in every tile
    torch.testing.assert_close(tiled, inp[:, :2] * 3 + constant)  # blending a pointwise map reproduces it
    calls.clear()
    whole = [torch.ones(1, 4, 3)]
    for training in ({}, {'teacher_tile': None}, {'teacher_tile': 1024}):  # one forward on the given features
        out = teacher_velocity(teacher, inp, t, r, lq, whole, FakeDino(), dict(config, training=training))
        torch.testing.assert_close(out, inp[:, :2] * 3 + 1, rtol=0, atol=0)
    assert [size for size, _ in calls] == [128, 128, 128]
    # Without features the single forward computes the top-level DINO of the whole crop itself.
    out = teacher_velocity(teacher, inp, t, r, lq, None, FakeDino(), dict(config, training={}))
    torch.testing.assert_close(out, inp[:, :2] * 3 + dino_features(FakeDino(), lq, config['dino'])[0].mean())


class PoolVAE(nn.Module):
    """8x average-pool encoder and nearest decoder: the Qwen VAE's shapes, purely local."""
    config = SimpleNamespace(latents_mean=[0, 0, 0], latents_std=[1, 1, 1])

    def encode(self, x):
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: F.avg_pool2d(x, 8)))

    def decode(self, z, return_dict=False):
        return (F.interpolate(z, scale_factor=8, mode='nearest'),)


class PreviewStudent(nn.Module):
    sparse_eval = True

    def forward(self, inp, t, r, features, return_stats=False):
        return inp[:, 3:] * .5, {'keep_fraction': inp.new_ones(())}


@pytest.mark.parametrize('student_tile, teacher_tile, inputs', [
    (None, 16, 1 + 9),   # res1024_dino_whole: whole-crop student, 3x3 tiled teacher; no unused 4-tile DINO
    (16, 16, 4 + 9),     # res1024: the student's 4 tiles; the teacher DINOs its own tiles
    (None, None, 1 + 4),  # one teacher forward reads the top-level 4-tile DINO
])
def test_previews_compute_only_the_dino_inputs_they_use(student_tile, teacher_tile, inputs):
    from ablation_logging import render_previews
    config = {'data': {'resolution': 16, 'crop_size': 32}, 'dino': {'size': 28, 'layer': 0, 'tile': 16},
              'student': {'dino': {'tile': student_tile}},
              'training': {'teacher_tile': teacher_tile, 'teacher_tile_overlap': 8, 'precision': 'fp32'}}
    dino, sample = FakeDino(), torch.rand(1, 3, 32, 32) * 2 - 1
    images, _ = render_previews(PreviewStudent(), lambda inp, t, r, features: inp[:, 3:], PoolVAE(), dino,
                                [(sample, sample)], config, torch.device('cpu'))
    assert len(images) == 1 and dino.inputs == inputs


def test_vae_tiling_defaults_to_one_pass_per_training_crop(capsys):
    from tiled_vae import decode_dispatch, decode_latent, encode_dispatch, encode_latent

    def config(**training):
        return {'data': {'resolution': 16, 'crop_size': 32}, 'training': training}
    assert vae_tiling({'data': {'resolution': 512}, 'training': {}})[0] == 512  # 512 exports keep inference's 512
    tile, args = vae_tiling(config())
    assert tile == 32 and args.vae_tile_size == 0  # one pass over the crop, as before the key existed
    assert vae_tiling(config(vae_tile_size=0))[0] == 0 and vae_tiling(config(vae_tile_size=64))[1].vae_tile_size == 0
    assert vae_tiling({'data': {'resolution': 512, 'crop_size': 1024},
                       'training': {'vae_tile_size': 512}})[1].vae_tile_overlap == 64  # restore's overlap
    vae, x = PoolVAE(), torch.rand(1, 3, 32, 32) * 2 - 1
    plain = SimpleNamespace(ae_type='qwen')
    whole, mean, std = encode_latent(vae, x, plain, 'cpu')
    torch.testing.assert_close(encode_dispatch(vae, x, args, 'cpu')[0], whole, rtol=0, atol=0)
    torch.testing.assert_close(decode_dispatch(vae, whole, args, mean, std),
                               decode_latent(vae, whole, plain, mean, std), rtol=0, atol=0)
    tile, args = vae_tiling(config(vae_tile_size=16))
    assert tile == 16 and args.vae_tile_size == 16 and args.vae_tile_overlap == 8
    # 2x2-latent tiles of a purely local VAE blend back to the whole-image result.
    torch.testing.assert_close(encode_dispatch(vae, x, args, 'cpu')[0], whole)
    torch.testing.assert_close(decode_dispatch(vae, whole, args, mean, std), decode_latent(vae, whole, plain, mean, std))
    assert capsys.readouterr().out == ''  # training does not print the tile grid every step


@pytest.mark.parametrize('recorded, flag, expected', [
    (None, [], 512),  # exports written before pipeline.json had vae_tile_size
    (1024, [], 1024),
    (1024, ['--vae-tile-size', '0'], 0),
])
def test_inference_defaults_to_the_exported_vae_tile(tmp_path, monkeypatch, recorded, flag, expected):
    import inference_vosr_ablation as inference
    pipeline = {'vae_path': 'fixture', 'dino': {}, 'resolution': 32, 'upscale': 1, 'precision': 'fp32'}
    if recorded is not None:
        pipeline['vae_tile_size'] = recorded
    (tmp_path / 'pipeline.json').write_text(json.dumps(pipeline))
    Image.new('RGB', (8, 8)).save(tmp_path / 'in.png')
    model = SimpleNamespace(use_cross_attention=False, ca_scale=0., export_config={}, rope_mode='interpolate')
    monkeypatch.setattr(inference, 'load_export', lambda path, device: model)
    module = ModuleType('models.qwenimage_vae2d')
    module.AutoencoderKLQwenImage2D = SimpleNamespace(from_pretrained=lambda path: nn.Identity())
    monkeypatch.setitem(sys.modules, 'models.qwenimage_vae2d', module)
    seen = []
    monkeypatch.setattr(inference, 'restore', lambda *a: (seen.append(a[8]), (Image.new('RGB', (8, 8)), 1.))[1])
    monkeypatch.setattr(sys, 'argv', ['inference_vosr_ablation.py', '--export', str(tmp_path),
                                      '--input', str(tmp_path / 'in.png'), '--output', str(tmp_path / 'out'), *flag])
    inference.main()
    assert seen == [expected]


def flat(config, prefix=''):
    out = {}
    for key, value in config.items():
        out.update(flat(value, f'{prefix}{key}.') if isinstance(value, dict) else {f'{prefix}{key}': value})
    return out


RES1024 = {
    'res1024.yml': ('base_vosr2.yml', {'model.rope_mode', 'dino.tile', 'data.crop_size', 'training.teacher_tile',
                                       'training.teacher_tile_overlap'}),
    'res1024_rope_interp.yml': ('res1024.yml', {'model.rope_mode'}),
    'res1024_dino_whole.yml': ('res1024.yml', {'student.dino.tile'}),
    'res1024_loss_all.yml': ('res1024.yml', {'training.kd_weight', 'training.gt_weight', 'training.lpips_weight',
                                             'training.lpips_net', 'training.ocr_kl_weight',
                                             'ocr.type', 'ocr.checkpoint',
                                             'ocr.strip_height', 'ocr.strip_stride', 'ocr.temperature'}),
}


@pytest.mark.parametrize('name', sorted(RES1024))
def test_res1024_configs_change_one_ablation_point(name):
    root = ROOT / 'configs/ablations'
    base_name, keys = RES1024[name]
    assert (root / name).read_text().startswith(f'_base_: {base_name}\n')
    config, base = flat(load_config(root / name)), flat(load_config(root / base_name))
    changed = {k for k in set(config) | set(base) if config.get(k, '<absent>') != base.get(k, '<absent>')}
    assert changed == keys | {'training.output_dir'}
    data = load_config(root / name)['data']
    assert data['resolution'] == 512 and data['crop_size'] == 1024  # reference grid 512 (input_size 64), 1024 crops


def test_existing_configs_keep_their_semantics():
    for path in (ROOT / 'configs/ablations').glob('*.yml'):
        if path.name.startswith('res1024'):
            continue
        config = load_config(path)
        assert 'rope_mode' not in config['model'] and 'tile' not in config['dino']
        assert 'crop_size' not in config['data'] and 'teacher_tile' not in config['training']
        assert 'vae_tile_size' not in config['training']


def test_trainer_runs_larger_crops_with_tiled_teacher_and_pixel_loss(tmp_path):
    import yaml
    Image.new('RGB', (40, 40), 'white').save(tmp_path / 'hq.png')
    (tmp_path / 'images.txt').write_text(str(tmp_path / 'hq.png') + '\n')
    (tmp_path / 'datasets.txt').write_text(str(tmp_path / 'images.txt') + ', 4\n')
    torch.save({'weight': torch.tensor(.1)}, tmp_path / 'teacher.pt')
    config = {
        'teacher_checkpoint': str(tmp_path / 'teacher.pt'), 'vae_path': 'fixture', 'model': {'patch_size': 1},
        'dino': {'size': 14, 'layer': 0},
        'data': {'resolution': 16, 'crop_size': 32,
                 'dataset_type': 'txt', 'train_dataset_config': str(tmp_path / 'datasets.txt')},
        'student': {'use_cross_attention': True},
        'training': {'output_dir': str(tmp_path / 'run'), 'precision': 'fp32', 'report_to': 'none',
                     'batch_size_per_gpu': 1, 'gradient_accumulation_steps': 2, 'max_steps': 3,
                     'learning_rate': .001, 'save_every': 3, 'log_every': 1, 'preview_every': 3,
                     'preview_num_images': 1, 'num_workers': 0, 'zero_optimizer': False,
                     'teacher_tile': 16, 'teacher_tile_overlap': 8, 'lpips_weight': .1},
    }
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump(config))
    env = dict(os.environ, OMP_NUM_THREADS='1', WANDB_MODE='offline')
    run = subprocess.run([sys.executable, 'tests/ablation_trainer_smoke.py', '--config', str(path)],
                         cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    output = next((tmp_path / 'run').rglob('metrics.jsonl')).parent  # output_dir or a run directory in it
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in records] == [1, 2, 3]
    assert all(row['kd'] >= 0 and row['lpips'] >= 0 and row['grad_norm'] > 0 for row in records)
    pipeline = json.loads((output / 'export/pipeline.json').read_text())
    assert pipeline['resolution'] == 32 and pipeline['vae_tile_size'] == 32  # inference VAE = one pass per crop
    config['training']['feature_distill_weight'], config['training']['feature_distill_layers'] = .1, [0]
    config['training']['output_dir'] = str(tmp_path / 'feature')
    path.write_text(yaml.safe_dump(config))
    run = subprocess.run([sys.executable, 'tests/ablation_trainer_smoke.py', '--config', str(path)],
                         cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
    assert run.returncode != 0 and 'teacher_tile' in run.stderr
    config['training'].update(feature_distill_weight=0, vae_tile_size=12)
    path.write_text(yaml.safe_dump(config))
    run = subprocess.run([sys.executable, 'tests/ablation_trainer_smoke.py', '--config', str(path)],
                         cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
    assert run.returncode != 0 and 'vae_tile_size' in run.stderr
