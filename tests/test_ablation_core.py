import copy
import json
from pathlib import Path
import pytest
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from PIL import Image
from models.sdt_router import SDTRouter, AblationBlock, routed_mlp, linear_schedule, conditioning_key
from ablation_utils import PairedManifestDataset, load_config, read_weights, load_backbone_state


class TinyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.length = None

    def forward(self, x, rope=None):
        self.length = x.shape[1]
        return self.proj(x)


class TinyBlock(nn.Module):
    """Dependency-free fixture for the exact residual equations, not a VOSR checkpoint."""
    def __init__(self, dim=16, ca=False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = TinyAttention(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim))
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim))
        self.z_dims = dim if ca else None
        if ca:
            self.cross_attn = CrossAttention(dim)


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.calls = 0

    def forward(self, x, z):
        self.calls += 1
        return self.proj(z.mean(1, keepdim=True)).expand_as(x)


def test_router_ste_and_gradients():
    torch.manual_seed(7)
    router = SDTRouter(32, init_keep_prob=0.5).train()
    mask, p = router(torch.randn(2, 19, 32, requires_grad=True))
    assert set(mask.detach().flatten().tolist()) == {0.0, 1.0}
    (mask.sum() + p.sum()).backward()
    assert router.net[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(router.net[-1].weight.grad).all()


def test_eval_deterministic():
    router = SDTRouter(16, init_keep_prob=0.5).eval()
    x = torch.randn(2, 11, 16)
    assert torch.equal(router(x)[0], router(x)[0])


@pytest.mark.parametrize('kind', ['none', 'all', 'mixed'])
def test_sparse_matches_dense(kind):
    x = torch.randn(2, 13, 8)
    mlp = nn.Sequential(nn.Linear(8, 17), nn.GELU(), nn.Linear(17, 8)).eval()
    mask = torch.zeros(2, 13, 1) if kind == 'none' else torch.ones(2, 13, 1)
    if kind == 'mixed':
        mask[:, ::2] = 0
    with torch.no_grad():
        torch.testing.assert_close(routed_mlp(mlp, x, mask, True), routed_mlp(mlp, x, mask), rtol=1e-5, atol=1e-6)


def test_sparse_rejects_autograd():
    with pytest.raises(RuntimeError, match='inference-only'):
        routed_mlp(nn.Linear(2, 2), torch.randn(1, 3, 2), torch.ones(1, 3, 1), True)


@pytest.mark.parametrize('kwargs', [{'dim': 0}, {'dim': 16, 'init_keep_prob': 1.0},
                                    {'dim': 16, 'temperature': 0}, {'dim': 16, 'threshold': 1.0}])
def test_router_validation(kwargs):
    with pytest.raises(ValueError):
        SDTRouter(**kwargs)


def test_schedule():
    assert [linear_schedule(i, 1, 0, 10) for i in (0, 5, 10, 20)] == [1, .5, 0, 0]
    assert linear_schedule(0, 1, .75, 0) == .75
    with pytest.raises(ValueError):
        linear_schedule(-1, 1, 0, 10)


def test_attention_keeps_all_tokens_and_router_gradients():
    block = AblationBlock(TinyBlock(), {'init_keep_prob': .5})
    output, stats = block(torch.randn(2, 13, 16), torch.randn(2, 96), None, None, 0, False, False)
    (output.square().mean() + stats[0]).backward()
    assert output.shape == (2, 13, 16)
    assert block.attn.length == 13
    assert block.router.net[-1].weight.grad.abs().sum() > 0


def test_dense_warmup_keeps_ddp_gradient_edges():
    block = AblationBlock(TinyBlock(), {'init_keep_prob': .5})
    output, stats = block(torch.randn(2, 7, 16), torch.randn(2, 96), None, None, 0, True, False)
    output.sum().backward()
    assert stats[1] == 1
    assert all(p.grad is not None for p in block.router.parameters())


def test_zero_ca_does_not_call_or_need_conditioning():
    block = AblationBlock(TinyBlock(ca=True))
    block(torch.randn(2, 7, 16), torch.randn(2, 96), None, None, 0, False, False)
    assert block.cross_attn.calls == 0
    assert conditioning_key('blocks.0.cross_attn.q_linear.weight')
    assert not conditioning_key('blocks.0.norm1.weight')


def test_dense_block_equations_and_state_names():
    source = TinyBlock(ca=True)
    block = AblationBlock(source)
    x, c, z = torch.randn(2, 7, 16), torch.randn(2, 96), torch.randn(2, 3, 16)
    s1, a1, g1, s2, a2, g2 = (source.scale_shift_table[None] + c.reshape(2, 6, -1)).chunk(6, 1)
    expected = x + g1 * source.attn(source.norm1(x) * (1 + a1) + s1)
    expected = expected + source.cross_attn(expected, z)
    expected = expected + g2 * source.mlp(source.norm2(expected) * (1 + a2) + s2)
    actual, _ = block(x, c, z, None, 1, False, False)
    torch.testing.assert_close(actual, expected)
    assert set(source.state_dict()) == set(block.state_dict())


def test_checkpoint_recomputes_same_stochastic_routes():
    a = AblationBlock(TinyBlock(), {'init_keep_prob': .5})
    b = copy.deepcopy(a)
    x, c = torch.randn(2, 11, 16), torch.randn(2, 96)
    torch.manual_seed(19)
    y1, st1 = a(x, c, None, None, 0, False, False)
    (y1.square().mean() + st1[0]).backward()
    torch.manual_seed(19)
    y2, st2 = checkpoint(b, x, c, None, None, 0, False, False,
                         use_reentrant=False, preserve_rng_state=True)
    (y2.square().mean() + st2[0]).backward()
    torch.testing.assert_close(y1, y2)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_paired_crops_are_aligned(tmp_path):
    array = np.random.default_rng(8).integers(0, 255, (20, 24, 3), dtype=np.uint8)
    Image.fromarray(array).save(tmp_path / 'image.png')
    manifest = tmp_path / 'pairs.jsonl'
    manifest.write_text(json.dumps({'lq': 'image.png', 'gt': 'image.png'}) + '\n')
    item = PairedManifestDataset(manifest, resolution=16, upscale=1)[0]
    torch.testing.assert_close(item['lq'], item['gt'])
    assert item['lq'].shape == (3, 16, 16)


def test_mixed_missing_gt_rejected(tmp_path):
    p = tmp_path / 'pairs.jsonl'
    p.write_text('{"lq":"a","gt":"b"}\n{"lq":"c"}\n')
    with pytest.raises(ValueError, match='every row'):
        PairedManifestDataset(p)


def test_configs_resolve():
    root = Path(__file__).resolve().parents[1] / 'configs/ablations'
    no_ca = load_config(root / 'no_dino_no_ca.yml')
    routing = load_config(root / 'dydit_sdt.yml')
    assert no_ca['student']['use_cross_attention'] is False
    assert no_ca['student']['router_config'] is None
    assert routing['student']['router_config']['threshold'] == .5
    assert routing['training']['target_keep_ratio'] == .75
    assert no_ca['model'] == routing['model']


def test_circular_config_rejected(tmp_path):
    p = tmp_path / 'cycle.yml'
    p.write_text('_base_: cycle.yml\n')
    with pytest.raises(ValueError, match='Circular'):
        load_config(p)


def test_strict_loading_and_removed_conditioning():
    model = nn.Module()
    model.use_cross_attention = False
    model.proj = nn.Linear(3, 2)
    state = dict(model.state_dict())
    state['blocks.0.cross_attn.weight'] = torch.zeros(7, 7)
    report = load_backbone_state(model, state)
    assert len(report['discarded']) == 1
    with pytest.raises(ValueError, match='Shape mismatch'):
        load_backbone_state(model, dict(state, **{'proj.weight': torch.zeros(8, 9)}))
    with pytest.raises(ValueError, match='Missing'):
        load_backbone_state(model, {'proj.weight': state['proj.weight']})
    with pytest.raises(ValueError, match='Unexpected'):
        load_backbone_state(model, dict(state, unrelated=torch.zeros(1)))


def test_weight_directory_resolution(tmp_path):
    from safetensors.torch import save_file
    folder = tmp_path / 'checkpoints'
    folder.mkdir()
    save_file({'weight': torch.ones(3)}, str(folder / 'ema_model.safetensors'))
    assert torch.equal(read_weights(tmp_path)['weight'], torch.ones(3))
