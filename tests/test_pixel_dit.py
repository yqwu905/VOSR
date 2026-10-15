"""Pixel-space (L2P) student; requires the upstream timm environment."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
pytest.importorskip('timm', reason='Full backbone requires the upstream timm environment')
from models.lightningdit_ablation import AblationLightningDiT
from models.pixel_dit import MicroDiffusionModelTop3SCWide, PixelLightningDiT, pixel_model_config, pixel_spec
from ablation_utils import load_backbone_state, load_config, load_export, read_weights, save_export

ROOT = Path(__file__).resolve().parents[1]
# Latent stand-in of VOSR2: 3 latent channels, patch 2 on an 8x8 latent = 4x4 tokens of 16 pixels at 64^2.
MODEL = dict(patch_size=2, in_channels=6, out_channels=3, hidden_size=64, depth=4, num_heads=4, mlp_ratio=2,
             z_dims=16, auxiliary_time_cond=True, use_rope=True, use_qknorm=True, use_swiglu=True, use_rmsnorm=True)


def latent_and_pixel(encoder='patch', patch_size=16, **extra):
    torch.manual_seed(0)
    latent = AblationLightningDiT(**MODEL, input_size=8, **extra)
    torch.nn.init.normal_(latent.final_layer.linear.weight, std=.02)
    pixel = PixelLightningDiT(**pixel_model_config(MODEL, 64, {'patch_size': patch_size, 'encoder': encoder}), **extra)
    return latent, pixel, load_backbone_state(pixel, latent.state_dict(), allow_new_modules=True)


def pixel_inputs(size=64):
    torch.manual_seed(3)
    lq, noise = torch.rand(2, 3, size, size) * 2 - 1, torch.randn(2, 3, size, size)
    return lq, noise, torch.cat((lq, noise), 1), torch.ones(2), torch.zeros(2)


def supplied_forward(d, x, c):
    """The decoder's forward as supplied (stage closures inlined)."""
    p1 = d.ds1(x)
    enc2 = d.enc2(p1)
    enc3 = d.enc3(d.ds2(enc2))
    enc4 = d.enc4(d.ds3(enc3))
    b = d.bottleneck(torch.cat([d.ds4(enc4), d.c_upsample(c)], dim=1))
    d4 = d.dec4(d.up4(b) + enc4)
    d3 = d.dec3(d.up3(d4) + enc3)
    return d.out_conv(d.up1(d.dec2(d.up2(d3) + enc2)))


@pytest.mark.parametrize('checkpointing', [(False, False), (True, False), (True, True)])
def test_decoder_keeps_the_supplied_forward(checkpointing):
    torch.manual_seed(0)
    decoder = MicroDiffusionModelTop3SCWide(6, 8)  # c_scale 2: the supplied 2x c_upsample
    c = torch.randn(1, 8, 2, 2)
    assert torch.equal(decoder.c_upsample(c), F.interpolate(c, scale_factor=2, mode='nearest'))
    x = torch.randn(1, 6, 64, 64, requires_grad=True)
    expected = supplied_forward(decoder, x, c)
    (grad,) = torch.autograd.grad(expected.square().sum(), x)
    out = decoder(x, c, *checkpointing)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(out.square().sum(), x)[0], grad)
    # 16-pixel tokens: c already has the bottleneck's 1/16 grid, no upsampling layer
    same_grid = MicroDiffusionModelTop3SCWide(6, 8, c_scale=1)
    assert isinstance(same_grid.c_upsample, nn.Identity)
    assert same_grid(x, torch.randn(1, 8, 4, 4)).shape == (1, 3, 64, 64)


def test_latent_checkpoint_loads_like_l2p():
    latent, pixel, report = latent_and_pixel()
    # Shape-mismatched patch embedding: fresh init; same-shaped bias: loaded; final layer: removed.
    assert {'x_embedder.proj.weight', 'final_layer.linear.weight', 'final_layer.adaLN_modulation.1.weight'} \
        <= set(report['discarded'])
    assert not hasattr(pixel, 'final_layer')
    assert 'x_embedder.proj.weight' in report['new_pixel_parameters']
    assert all(k.startswith(('x_embedder.', 'decoder.')) for k in report['new_pixel_parameters'])
    reference = latent.state_dict()
    for key, value in pixel.state_dict().items():
        if key.startswith(('blocks.', 't_embedder.', 'r_embedder.', 't_block.', 'mlp_ca.')) \
                or key == 'x_embedder.proj.bias':
            torch.testing.assert_close(value, reference[key])
    with pytest.raises(ValueError, match='Shape mismatch for x_embedder.proj.weight'):
        load_backbone_state(PixelLightningDiT(**pixel_model_config(MODEL, 64, {})), latent.state_dict())
    # Resume: a pixel checkpoint loads strictly.
    resumed = PixelLightningDiT(**pixel_model_config(MODEL, 64, {}))
    assert all(k.startswith('feat_rope.') for k in load_backbone_state(resumed, pixel.state_dict())['discarded'])


@pytest.mark.parametrize('encoder', ['patch', 'unet'])
def test_x_prediction_returns_velocity_in_fp32(encoder):
    _, pixel, _ = latent_and_pixel(encoder, use_cross_attention=False)
    pixel.eval()
    lq, noise, x, t, r = pixel_inputs()
    images = []
    pixel.decoder.out_conv.register_forward_hook(lambda module, args, output: images.append(output))
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        velocity = pixel(x, t, r)
    assert velocity.dtype == torch.float32 and velocity.shape == (2, 3, 64, 64)
    # one step at t = 1: x_t is the noise, so noise - velocity is exactly the decoder's image
    torch.testing.assert_close(noise - velocity, images[-1].float(), rtol=0, atol=1e-6)


@pytest.mark.parametrize('encoder', ['patch', 'unet'])
def test_export_roundtrip_and_checkpointed_backward(tmp_path, encoder):
    _, pixel, _ = latent_and_pixel(encoder, use_checkpoint=True)
    _, _, x, t, r = pixel_inputs()
    z = [torch.randn(2, 5, 16)]
    pixel.train()
    pixel(x, t, r, z).square().mean().backward()
    assert pixel.x_embedder.proj.weight.grad is not None and pixel.decoder.ds1[0].weight.grad is not None
    pixel.eval()
    save_export(pixel, tmp_path, {'vae_path': None})
    restored = load_export(tmp_path)
    assert isinstance(restored, PixelLightningDiT) and restored.pixel == {'patch_size': 16, 'encoder': encoder}
    with torch.no_grad():
        torch.testing.assert_close(restored(x, t, r, z), pixel(x, t, r, z))


def test_patch_32_quarters_the_tokens_and_upsamples_c():
    _, pixel, _ = latent_and_pixel(patch_size=32, use_cross_attention=False)
    assert pixel.latent_multiple == 32 and isinstance(pixel.decoder.c_upsample, nn.ConvTranspose2d)
    tokens = []
    pixel.blocks[0].register_forward_hook(lambda module, args, output: tokens.append(output[0].shape[1]))
    _, _, x, t, r = pixel_inputs()
    assert pixel(x, t, r).shape == (2, 3, 64, 64) and tokens == [4]  # 2x2 tokens instead of 4x4


@pytest.mark.parametrize('config, message', [
    ({'patch_size': 24}, 'multiple of 16'),
    ({'patch_size': 32, 'encoder': 'unet'}, 'needs patch_size 16'),
    ({'encoder': 'conv'}, 'encoder must be one of'),
    ({'decoder': 'x'}, 'Unknown student.pixel keys'),
])
def test_pixel_spec_rejects(config, message):
    with pytest.raises(ValueError, match=message):
        pixel_spec(config)


@pytest.mark.parametrize('extra', [{'router_config': {'init_keep_prob': .5}}, {'token_compression': {'type': 'hourglass'}}])
def test_pixel_student_rejects_token_reduction(extra):
    with pytest.raises(ValueError, match='supports neither'):
        PixelLightningDiT(**pixel_model_config(MODEL, 64, {}), **extra)


def flat(config, prefix=''):
    items = {}
    for key, value in config.items():
        items.update(flat(value, f'{prefix}{key}.') if isinstance(value, dict) else {prefix + key: value})
    return items


def test_configs_only_add_the_pixel_student():
    root = ROOT / 'configs/ablations'
    assert (root / 'pixel_l2p.yml').read_text().startswith('_base_: no_dino_sft.yml\n')
    assert (root / 'pixel_l2p_unet.yml').read_text().startswith('_base_: pixel_l2p.yml\n')
    sft, cfg, unet = (flat(load_config(root / name)) for name in ('no_dino_sft.yml', 'pixel_l2p.yml', 'pixel_l2p_unet.yml'))
    changed = lambda a, b: {k for k in set(a) | set(b) if a.get(k, '<absent>') != b.get(k, '<absent>')}
    assert changed(cfg, sft) == {'student.pixel.patch_size', 'student.pixel.encoder', 'training.output_dir',
                                 'training.learning_rate', 'training.trainable_parameters',
                                 'training.lpips_weight', 'training.ocr_kl_weight'}
    assert changed(unet, cfg) == {'student.pixel.encoder', 'training.output_dir'}
    assert cfg['training.kd_weight'] == 0 and cfg['training.gt_weight'] == 1
    assert pixel_spec({'patch_size': cfg['student.pixel.patch_size'], 'encoder': unet['student.pixel.encoder']})
    # L2P partial fine-tuning: blocks 5-30 of 36 frozen, every other module trains.
    with torch.device('meta'):
        model = PixelLightningDiT(**pixel_model_config(dict(load_config(root / 'pixel_l2p.yml')['model'],
                                                            auxiliary_time_cond=False), 512,
                                                       {'patch_size': 16}), use_cross_attention=False)
    trained = {name for name, _ in model.named_parameters()
               if any(re.search(p, name) for p in cfg['training.trainable_parameters'])}
    blocks = {int(name.split('.')[1]) for name in trained if name.startswith('blocks.')}
    assert blocks == set(range(5)) | set(range(31, 36))
    assert {name for name, _ in model.named_parameters() if not name.startswith('blocks.')} <= trained


def trainer_config(tmp_path, student, **training):
    import yaml
    from PIL import Image
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (80, 80, 3), dtype=np.uint8)).save(tmp_path / 'hq.png')
    (tmp_path / 'images.txt').write_text(str(tmp_path / 'hq.png') + '\n')
    (tmp_path / 'datasets.txt').write_text(str(tmp_path / 'images.txt') + ', 2\n')
    model = dict(MODEL, z_dims=6, auxiliary_time_cond='auto')  # fixture DINO: 6 channels
    torch.manual_seed(0)
    teacher = AblationLightningDiT(**dict(model, auxiliary_time_cond=True), input_size=8)
    torch.nn.init.normal_(teacher.final_layer.linear.weight, std=.02)
    torch.save(teacher.state_dict(), tmp_path / 'teacher.pt')
    cfg = {'teacher_checkpoint': str(tmp_path / 'teacher.pt'), 'vae_path': 'fixture', 'model': model,
           'dino': {'size': 56, 'layer': 0},
           'data': {'resolution': 64, 'dataset_type': 'txt', 'train_dataset_config': str(tmp_path / 'datasets.txt')},
           'student': student,
           'training': dict(output_dir=str(tmp_path / 'run'), precision='fp32', report_to='none',
                            batch_size_per_gpu=1, gradient_accumulation_steps=2, max_steps=3, learning_rate=1e-3,
                            save_every=2, log_every=1, preview_every=2, preview_num_images=1, num_workers=0,
                            zero_optimizer=False, kd_weight=0, gt_weight=1., **training)}
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump(cfg))
    return path, teacher.state_dict()


def run_trainer(config_path, resume=None):
    command = [sys.executable, 'tests/ablation_hourglass_smoke.py', '--config', str(config_path)]
    if resume:
        command += ['--resume', str(resume)]
    env = dict(os.environ, OMP_NUM_THREADS='1', WANDB_MODE='offline', TORCHDYNAMO_DISABLE='1')
    return subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=300)


def test_trainer_freezes_blocks_resumes_and_restores_without_vae(tmp_path, monkeypatch):
    student = {'use_cross_attention': False, 'router_config': None, 'pixel': {'patch_size': 16}}
    config_path, teacher = trainer_config(tmp_path, student, trainable_parameters=['^(?!blocks\\.)', '^blocks\\.0\\.'])
    run = run_trainer(config_path)
    assert run.returncode == 0, run.stdout + run.stderr
    output, = (tmp_path / 'run').iterdir()
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in records] == [1, 2, 3] and all(row['gt'] > 0 for row in records)
    assert len(list((output / 'previews').glob('*/sample-00.png'))) == 3
    checkpoint = output / 'checkpoint-00000002'
    assert json.loads((checkpoint / 'pipeline.json').read_text())['vae_path'] is None
    run = run_trainer(config_path, resume=checkpoint)
    assert run.returncode == 0, run.stdout + run.stderr
    export = output / 'export'
    state = read_weights(export / 'model.safetensors')
    assert not any(k.startswith('final_layer.') for k in state)
    for key in state:  # frozen blocks keep the latent weights; block 0 trained
        if key.startswith(('blocks.1.', 'blocks.2.', 'blocks.3.')):
            torch.testing.assert_close(state[key], teacher[key])
    assert not torch.equal(state['blocks.0.attn.qkv.weight'], teacher['blocks.0.attn.qkv.weight'])

    import inference_vosr_ablation as inference
    from PIL import Image
    model = load_export(export)
    pipeline = json.loads((export / 'pipeline.json').read_text())
    image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (40, 48, 3), dtype=np.uint8))
    restored, keep = inference.restore(model, None, None, image, pipeline, torch.device('cpu'),
                                       tile_size=32, tile_overlap=16, vae_tile_size=0, upscale=1)
    assert restored.size == (48, 40) and keep == 1.
    with pytest.raises(ValueError, match='align to 16 pixels'):
        inference.restore(model, None, None, image, pipeline, torch.device('cpu'), tile_size=40, tile_overlap=0,
                          vae_tile_size=0, upscale=1)


def test_trainer_rejects_a_teacher_for_the_pixel_student(tmp_path):
    student = {'use_cross_attention': False, 'router_config': None, 'pixel': {'patch_size': 16}}
    config_path, _ = trainer_config(tmp_path, student)
    import yaml
    cfg = yaml.safe_load(config_path.read_text())
    cfg['training']['kd_weight'] = 1.
    config_path.write_text(yaml.safe_dump(cfg))
    run = run_trainer(config_path)
    assert run.returncode != 0 and 'student.pixel trains without a teacher' in run.stderr
