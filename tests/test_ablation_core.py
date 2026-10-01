import copy
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from models.sdt_router import SDTRouter, AblationBlock, routed_mlp, linear_schedule, conditioning_key, mlp_budget_loss
from ablation_utils import load_config, read_weights, load_backbone_state


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


@pytest.mark.parametrize('mode', ['gumbel', 'capacity_topk'])
def test_router_ste_and_gradients(mode):
    torch.manual_seed(7)
    router = SDTRouter(32, init_keep_prob=0.5, routing_mode=mode).train()
    mask, p = router(torch.randn(2, 19, 32, requires_grad=True))
    assert set(mask.detach().flatten().tolist()) == {0.0, 1.0}
    # Task loss through the hard mask alone must train the router.
    mask.sum().backward()
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
                                    {'dim': 16, 'temperature': 0}, {'dim': 16, 'threshold': 1.0},
                                    {'dim': 16, 'routing_mode': 'invalid'},
                                    {'dim': 16, 'routing_mode': 'capacity_topk', 'temperature': 5},
                                    {'dim': 16, 'routing_mode': 'capacity_topk', 'threshold': .7}])
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


@pytest.mark.parametrize('mode', ['gumbel', 'capacity_topk'])
def test_checkpoint_recomputes_same_routes(mode):
    a = AblationBlock(TinyBlock(), {'init_keep_prob': .5, 'routing_mode': mode})
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


def test_configs_resolve():
    root = Path(__file__).resolve().parents[1] / 'configs/ablations'
    no_ca = load_config(root / 'no_dino_no_ca.yml')
    routing = load_config(root / 'dydit_sdt.yml')
    assert no_ca['student']['use_cross_attention'] is False
    assert no_ca['student']['router_config'] is None
    assert routing['student']['router_config']['threshold'] == .5
    assert routing['student']['router_config']['routing_mode'] == 'capacity_topk'
    assert routing['training']['budget_scope'] == 'global'
    assert routing['training']['target_keep_ratio'] == .75
    assert no_ca['model'] == routing['model']
    for name in ('no_dino_no_ca.yml', 'no_dino_fade.yml', 'dydit_sdt.yml'):
        data = load_config(root / name)['data']
        assert data['dataset_type'] == 'txt'
        assert data['train_dataset_config'] == 'configs/train_txt/train_dataset_txt.txt'
        assert 'manifest' not in data and 'upscale' not in data


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


def test_global_budget_allows_different_layer_capacities():
    # Old objective penalizes this valid allocation even though mean keep is .75.
    keep = torch.tensor([1., .5], requires_grad=True)
    loss = mlp_budget_loss(keep, .75)
    assert loss == 0
    loss.backward()
    assert torch.equal(keep.grad, torch.zeros(2))
    assert mlp_budget_loss(keep, .75, 'layer') == .0625
    keep = torch.tensor([.9, .7], requires_grad=True)
    mlp_budget_loss(keep, .75).backward()
    assert torch.all(keep.grad > 0)  # gradient descent reduces the aggregate budget


@pytest.mark.parametrize('tokens', [1, 7, 16, 1024])
def test_capacity_topk_fixes_zero_budget_full_inference_counterexample(tokens):
    router = SDTRouter(16, init_keep_prob=.75, routing_mode='capacity_topk')
    with torch.no_grad():
        router.net[-1].weight.zero_()  # exactly p=.75 for every token
    x = torch.randn(2, tokens, 16)
    rng = torch.get_rng_state()
    training_mask, p = router(x)
    eval_mask, _ = router.eval()(x)
    assert torch.equal(training_mask, eval_mask)
    assert torch.equal(rng, torch.get_rng_state())
    assert mlp_budget_loss(p.mean().view(1), .75).item() < 1e-12
    expected_count = int(.75 * tokens + .5)
    assert torch.all(eval_mask.sum(1) == expected_count)
    assert abs(eval_mask.mean().item() - .75) <= .5 / tokens + 1e-7
    # Equal scores break ties by original token position, including at BF16 ties.
    assert torch.all(eval_mask[:, :expected_count] == 1)
    assert torch.all(eval_mask[:, expected_count:] == 0)


def test_capacity_topk_selects_high_scores_and_independent_image_capacities():
    router = SDTRouter(1, routing_mode='capacity_topk')
    router.net = nn.Identity()  # known scores, without depending on learned weights
    x = torch.tensor([[[-3.], [2.], [-1.], [4.]], [[3.], [2.], [0.], [1.]]])
    mask, p = router(x)
    assert torch.equal(mask[0, :, 0], torch.tensor([0., 1., 0., 1.]))
    assert torch.equal(mask[1, :, 0], torch.tensor([1., 1., 0., 1.]))
    for sample in range(2):
        single, _ = router(x[sample:sample + 1])
        assert torch.equal(single, mask[sample:sample + 1])
    assert torch.all((mask.mean(1) - p.mean(1)).abs() <= .5 / x.shape[1])
    endpoints, _ = router(torch.tensor([[[-100.], [-100.]], [[100.], [100.]]]))
    assert torch.equal(endpoints[:, :, 0], torch.tensor([[0., 0.], [1., 1.]]))


def test_legacy_export_router_behavior_is_preserved():
    router = SDTRouter(16, init_keep_prob=.75)  # old exports have no routing_mode
    with torch.no_grad():
        router.net[-1].weight.zero_()
    x = torch.randn(1, 10000, 16)
    torch.manual_seed(13)
    mask, _ = router(x)
    assert abs(mask.mean().item() - .75) < .02
    assert torch.all(router.eval()(x)[0] == 1)


def test_capacity_topk_bf16_train_eval_match_and_rounding_bound():
    router = SDTRouter(32, init_keep_prob=.75, routing_mode='capacity_topk')
    x = torch.randn(2, 1024, 32)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        mask, p = router(x)
        evaluated, eval_p = router.eval()(x)
    assert torch.equal(mask, evaluated)
    assert torch.equal(p, eval_p)
    assert p.dtype == torch.float32
    assert torch.all((mask.mean(1) - p.mean(1)).abs() <= .5 / x.shape[1] + 1e-7)


def test_capacity_topk_budget_trains_real_router_toward_target():
    torch.manual_seed(37)
    router = SDTRouter(16, init_keep_prob=.9, routing_mode='capacity_topk')
    x = torch.randn(2, 64, 16)
    optimizer = torch.optim.AdamW(router.parameters(), lr=.02)
    initial = router(x)[1].mean().item()
    for _ in range(60):
        optimizer.zero_grad()
        _, p = router(x)
        mlp_budget_loss(p.mean().view(1), .6).backward()
        optimizer.step()
    mask, p = router(x)
    assert abs(p.mean().item() - .6) < abs(initial - .6) / 2
    assert abs(mask.mean().item() - p.mean().item()) <= .5 / x.shape[1]


def test_capacity_topk_block_train_eval_and_sparse_agree():
    torch.manual_seed(21)
    block = AblationBlock(TinyBlock(ca=True), {'init_keep_prob': .75, 'routing_mode': 'capacity_topk'})
    x, c, z = torch.randn(2, 16, 16), torch.randn(2, 96), torch.randn(2, 3, 16)
    with torch.no_grad():
        train, st = block(x, c, z, None, 1, False, False)
        block.eval()
        dense, sd = block(x, c, z, None, 1, False, False)
        sparse, ss = block(x, c, z, None, 1, False, True)
    torch.testing.assert_close(train, dense)
    torch.testing.assert_close(sparse, dense, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(st, sd)
    torch.testing.assert_close(ss, sd)
