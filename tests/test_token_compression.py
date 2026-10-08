"""Real tiny LightningDiT CPU tests; no pretrained quality/performance claim."""
import copy
import os
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

from ablation_utils import load_backbone_state, load_export, save_export, load_config
from models.lightningdit import LightningDiT
from models.lightningdit_ablation import AblationLightningDiT
from models.token_compression import (pool_tokens, restore_tokens, pooled_rope,
    validate_compression_config, feature_distillation_loss, detail_distillation_loss)
from models.pos_embed import VisionRotaryEmbeddingFast
from inference_vosr_ablation import restore


def options():
    return dict(input_size=8, patch_size=2, in_channels=4, out_channels=2, hidden_size=64,
                depth=4, num_heads=4, mlp_ratio=2, z_dims=16, use_rope=True, auxiliary_time_cond=True)


def compression(**kwargs):
    return dict(enabled=True, factor=2, start_block=1, end_block=3, **kwargs)


def inputs(size=8):
    return torch.randn(2, 4, size, size), torch.ones(2), torch.zeros(2), [torch.randn(2, 3, 16)]


def model(**kwargs):
    out = AblationLightningDiT(**options(), **kwargs)
    # Upstream zero-initialized output head would make parity tests vacuous.
    nn.init.normal_(out.final_layer.linear.weight, std=.02)
    return out


@pytest.mark.parametrize('factor', [1, 2])
def test_identity_update_keeps_details_and_gradients(factor):
    entry = torch.randn(2, 64, 12, requires_grad=True)
    pooled = pool_tokens(entry, factor)
    restored = restore_tokens(pooled, entry, pooled, factor)
    assert torch.equal(restored, entry)
    restored.sum().backward()
    assert torch.equal(entry.grad, torch.ones_like(entry))


@pytest.mark.parametrize('size', [8, 12])
def test_actual_attention_shapes_and_feature_grids(size):
    net = model(compression_config=compression()).eval()
    lengths, ca_lengths = [], []
    handles = [block.attn.register_forward_pre_hook(lambda m, args: lengths.append(args[0].shape[1])) for block in net.blocks]
    handles += [block.cross_attn.register_forward_pre_hook(lambda m, args: ca_lengths.append(args[0].shape[1])) for block in net.blocks]
    with torch.no_grad():
        out, stats = net(*inputs(size), return_stats=True, feature_layers=[0, 1, 2, 3])
    for handle in handles:
        handle.remove()
    n = (size//2)**2
    assert lengths == ca_lengths == [n, n//4, n//4, n]
    assert stats['attention_tokens'].tolist() == lengths
    assert stats['attention_pair_ratio'].item() == pytest.approx(.53125)
    assert stats['attention_token_ratio'].item() == pytest.approx(.625)
    assert out.shape == (2, 2, size, size)
    assert all(feature.shape == (2, n, 64) for feature in stats['features'].values())


def test_weight_compatibility_and_all_off_paths_are_original():
    base = LightningDiT(**options()).eval()
    nn.init.normal_(base.final_layer.linear.weight, std=.02)
    net = model(compression_config=compression()).eval()
    assert set(base.state_dict()) == set(net.state_dict())
    load_backbone_state(net, base.state_dict())
    args = inputs()
    with torch.no_grad():
        expected = base(*args)
        torch.testing.assert_close(net(*args, disable_compression=True), expected, rtol=0, atol=0)
        torch.testing.assert_close(net(*args, force_dense=True), expected, rtol=0, atol=0)
        result, _, dense = net(*args, include_dense=True)
        torch.testing.assert_close(dense, expected, rtol=0, atol=0)
        assert not torch.allclose(result, expected)
        net.compression_config['enabled'] = False
        torch.testing.assert_close(net(*args), expected, rtol=0, atol=0)
        net.compression_config.update(enabled=True, factor=1)
        torch.testing.assert_close(net(*args), expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_checkpointed_kd_backward_and_export(tmp_path, dtype):
    torch.manual_seed(44)
    net = model(compression_config=compression())
    other = copy.deepcopy(net)
    other.use_checkpoint = True
    teacher = copy.deepcopy(net).eval().requires_grad_(False)
    args = inputs()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        target, target_stats = teacher(*args, return_stats=True, disable_compression=True, feature_layers=[1, 3])
    for candidate in (net, other):
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            out, stats, dense = candidate(*args, return_stats=True, feature_layers=[1, 3], include_dense=True)
        loss = F.mse_loss(out.float(), target.float()) + .1 * F.mse_loss(dense.float(), target.float())
        loss = loss + .1 * feature_distillation_loss(stats['features'], target_stats['features'])
        loss = loss + .1 * detail_distillation_loss(out, target)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())
        assert candidate.blocks[1].attn.qkv.weight.grad.abs().sum() > 0
    for p, q in zip(net.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    optimizer = torch.optim.AdamW(other.parameters(), lr=1e-4)
    optimizer.step()
    other.eval()
    save_export(other, tmp_path, {})
    restored = load_export(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(restored(*args), other(*args), rtol=0, atol=0)
    assert restored.compression_config == other.compression_config


def test_rope_unit_factor_matches_upstream_and_cell_centers():
    x = torch.randn(1, 4, 16, 16)
    reference = VisionRotaryEmbeddingFast(dim=8, pt_seq_len=4)
    torch.testing.assert_close(pooled_rope(4, 1, 4, 16, x.device)(x), reference(x), rtol=0, atol=0)
    # First coarse token at (0.5, 0.5), next at (0.5, 2.5); check unit-frequency channels.
    y = torch.zeros(1, 1, 4, 16)
    y[..., 0::2] = 1
    rotated = pooled_rope(4, 2, 4, 16, y.device)(y)
    torch.testing.assert_close(rotated[0, 0, :, 0], torch.tensor([.5, .5, 2.5, 2.5]).cos())
    torch.testing.assert_close(rotated[0, 0, :, 8], torch.tensor([.5, 2.5, .5, 2.5]).cos())


@pytest.mark.parametrize('cfg', [dict(factor=3), dict(factor=2.5), dict(start_block=-1),
    dict(start_block=3, end_block=2), dict(end_block=5), dict(enabled='true'), dict(typo=True)])
def test_reject_invalid_configs(cfg):
    with pytest.raises(ValueError):
        validate_compression_config(cfg, 4)


def test_bad_grid_fails_explicitly_but_fallback_works():
    net = model(compression_config=compression()).eval()
    with torch.no_grad():
        with pytest.raises(ValueError, match='compression factor'):
            net(*inputs(10))
        assert net(*inputs(10), disable_compression=True).shape[-1] == 10
        with pytest.raises(ValueError, match='feature_layers'):
            net(*inputs(), feature_layers=[4])


def test_sdt_alone_does_not_reduce_attention():
    net = model(router_config=dict(init_keep_prob=.5, routing_mode='capacity_topk')).eval()
    with torch.no_grad():
        _, stats = net(*inputs(), return_stats=True)
    assert stats['attention_tokens'].tolist() == [16]*4
    assert stats['attention_pair_ratio'].item() == 1
    assert stats['keep_fraction'].item() == .5


class TinyVAE(nn.Module):
    config = SimpleNamespace(latents_mean=[0, 0], latents_std=[1, 1])

    def encode(self, x):
        latent = F.avg_pool2d(x[:, :2], 8)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, x, **kwargs):
        return (F.interpolate(torch.cat((x, x[:, :1]), 1), scale_factor=8, mode='nearest'),)


def test_tiled_non_square_restore_and_fallback():
    net = model(compression_config=compression(), use_cross_attention=False).eval()
    source = Image.fromarray(np.random.default_rng(7).integers(0, 255, (13, 21, 3), dtype=np.uint8))
    pipeline = dict(precision='fp32')
    kwargs = dict(tile_size=64, tile_overlap=16, vae_tile_size=0, upscale=4, return_stats=True)
    image, stats = restore(net, TinyVAE(), None, source, pipeline, torch.device('cpu'), **kwargs)
    assert image.size == (84, 52)
    assert stats['tile_count'] > 1
    assert stats['attention_tokens_per_tile'] == [16, 4, 4, 16]
    _, fallback = restore(net, TinyVAE(), None, source, pipeline, torch.device('cpu'), disable_compression=True, **kwargs)
    assert fallback['attention_pair_ratio'] == 1


def test_shipped_compression_configs():
    cfg = load_config('configs/ablations/token_compression.yml')
    assert cfg['student']['router_config'] is None
    assert cfg['student']['use_cross_attention']
    assert validate_compression_config(cfg['student']['compression_config'], cfg['model']['depth'])['end_block'] == 32
    smoke = load_config('configs/ablations/token_compression_smoke.yml')
    assert smoke['training']['max_steps'] == 10
