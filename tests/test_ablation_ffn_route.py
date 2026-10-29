"""DyDiT-style FFN token routing (router_config.routing_mode: dydit) on the head split; needs the timm environment."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
pytest.importorskip('timm', reason='Full backbone requires the upstream timm environment')
from models.lightningdit_ablation import AblationLightningDiT
from models.sdt_router import AblationBlock, SDTRouter, mlp_budget_loss
from ablation_utils import load_backbone_state, load_config, load_export, save_export
from test_ablation_core import TinyBlock
from test_ablation_head_split import inputs, kwargs, paired_trainer_config, run_trainer, split, write_stats
from test_ablation_mlp_prune import flat

ROOT = Path(__file__).resolve().parents[1]
DYDIT = {'routing_mode': 'dydit', 'temperature': 5.0, 'init_keep_prob': .5}


def official_token_flops_loss(token_select, target):
    """DyDiT/loss.py DynamicLoss._get_token_loss with all heads and channel groups active and the attention FLOPs
    held out (fixed cost): token_select (B, layers, N, 1) are the TokenSelect masks."""
    mlp_flops = ((token_select.squeeze(3).sum(dim=2) / token_select.shape[2]) * 1.0).sum(dim=1)
    total_flops = mlp_flops / token_select.shape[1]
    return (total_flops.mean() - target) ** 2


def test_dydit_routes_like_gumbel_and_keeps_the_mask_gradient_in_its_keep():
    torch.manual_seed(3)
    x = torch.randn(2, 37, 32)
    dydit, gumbel = SDTRouter(32, **DYDIT), SDTRouter(32, **dict(DYDIT, routing_mode='gumbel'))
    gumbel.load_state_dict(dydit.state_dict())
    torch.manual_seed(4)
    mask, p = dydit(x)
    torch.manual_seed(4)
    reference, reference_p = gumbel(x)
    assert torch.equal(mask, reference) and torch.equal(p, reference_p)
    assert set(mask.flatten().tolist()) <= {0., 1.}
    with torch.no_grad():  # inference: sigmoid(logits) > 0.5, no noise
        logits = dydit.net(x)
        assert torch.equal(dydit.eval()(x)[0], (logits > 0).float())
    for mode, keeps_gradient in (('dydit', True), ('gumbel', False)):
        torch.manual_seed(5)
        block = AblationBlock(TinyBlock(), dict(DYDIT, routing_mode=mode))
        masks = []
        block.router.register_forward_hook(lambda module, args, output: masks.append(output[0]))
        _, stats = block(torch.randn(2, 13, 16), torch.randn(2, 96), None, None, 0, False, False)
        assert stats[1] == masks[0].float().mean()  # the realized keep of this pass
        gradients = torch.autograd.grad(stats[1], list(block.router.parameters()), allow_unused=True)
        assert any(g is not None and g.abs().sum() > 0 for g in gradients) == keeps_gradient


def test_dydit_budget_is_the_official_flops_loss():
    torch.manual_seed(6)
    blocks = nn.ModuleList(AblationBlock(TinyBlock(), DYDIT) for _ in range(3))
    masks = []
    for block in blocks:
        block.router.register_forward_hook(lambda module, args, output: masks.append(output[0]))
    x, c, fractions = torch.randn(4, 11, 16), torch.randn(4, 96), []
    for block in blocks:
        x, stats = block(x, c, None, None, 0, False, False)
        fractions.append(stats[1])
    ours = mlp_budget_loss(torch.stack(fractions), .25)
    official = official_token_flops_loss(torch.stack(masks, 1), .25)
    torch.testing.assert_close(ours, official)
    parameters = [p for block in blocks for p in block.router.parameters()]
    ours_grad = torch.autograd.grad(ours, parameters, retain_graph=True)
    official_grad = torch.autograd.grad(official, parameters)
    for a, b in zip(ours_grad, official_grad):
        torch.testing.assert_close(a, b)
    assert sum(g.abs().sum() for g in ours_grad) > 0


def routed_head_split(tmp_path, router_config=DYDIT):
    write_stats(tmp_path / 'stats.json')
    torch.manual_seed(0)
    teacher = AblationLightningDiT(**kwargs()).eval()
    torch.nn.init.normal_(teacher.final_layer.linear.weight, std=.02)
    student = AblationLightningDiT(**kwargs(), attn_compression=split(tmp_path / 'stats.json'),
                                   router_config=router_config)
    report = load_backbone_state(student, teacher.state_dict(), allow_new_modules=True)
    return student, report


def test_head_split_with_routed_ffn_trains_infers_sparse_and_exports(tmp_path):
    student, report = routed_head_split(tmp_path)
    assert len(report['new_router_parameters']) == 4 * 6  # two Linear layers per block, the backbone is loaded
    x, t, r, z = inputs()
    student.train()
    y, stats = student(x, t, r, z, return_stats=True)
    budget = mlp_budget_loss(stats['keep_fractions'], .25)
    (y.square().mean() + budget).backward()
    assert all(b.router.net[0].weight.grad.abs().sum() > 0 for b in student.blocks)
    assert all(b.attn.local_conv.weight.grad is not None for b in student.blocks if b.attn.local_heads)
    student.eval()
    with torch.no_grad():
        student.sparse_eval = False
        dense, dense_stats = student(x, t, r, z, return_stats=True)
        student.sparse_eval = True  # inference: only the kept tokens run the MLP
        sparse, sparse_stats = student(x, t, r, z, return_stats=True)
        torch.testing.assert_close(sparse, dense, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(sparse_stats['keep_fractions'], dense_stats['keep_fractions'])
        assert 0 < sparse_stats['keep_fraction'] < 1
        save_export(student, tmp_path / 'export', {})
        restored = load_export(tmp_path / 'export')
        torch.testing.assert_close(restored(x, t, r, z), sparse, rtol=1e-4, atol=1e-5)
    config = json.loads((tmp_path / 'export/model.json').read_text())
    assert config['router_config'] == DYDIT and config['attn_compression']['factor'] == 2
    assert all(b.router.routing_mode == 'dydit' for b in restored.blocks)


def test_shipped_ffn_route_config_changes_only_the_routing():
    root = ROOT / 'configs/ablations'
    cfg = load_config(root / 'ffn_route_k25.yml')
    config, reference = flat(cfg), flat(load_config(root / 'ffn_prune_ctrl.yml'))
    diff = {k for k in set(config) | set(reference) if config.get(k, '<absent>') != reference.get(k, '<absent>')}
    assert diff == {'student.router_config', 'student.router_config.routing_mode', 'student.router_config.temperature',
                    'student.router_config.threshold', 'student.router_config.init_keep_prob',
                    'training.output_dir', 'training.target_keep_ratio', 'training.budget_weight',
                    'training.budget_warmup_steps', 'training.dense_warmup_steps', 'training.dense_distill_weight'}
    assert reference['student.router_config'] is None
    tc = cfg['training']
    assert tc['output_dir'] == 'exp_vosr/ffn_route_k25' and tc['target_keep_ratio'] == .25
    assert (tc['budget_scope'], tc['budget_weight'], tc['router_lr_multiplier']) == ('global', 2., 10.)
    assert cfg['student_checkpoint'] == 'weights/head_split_f2' and cfg['student']['attn_compression']['factor'] == 2
    SDTRouter(cfg['model']['hidden_size'], **cfg['student']['router_config'])


def test_trainer_budgets_the_routed_ffn_on_the_head_split_and_inference_restores(tmp_path, monkeypatch):
    import yaml
    config_path = paired_trainer_config(tmp_path, token_merge_lr_multiplier=10., target_keep_ratio=.25,
                                        budget_scope='global', budget_weight=2., budget_warmup_steps=2,
                                        dense_warmup_steps=0, dense_distill_weight=0., router_lr_multiplier=10.)
    cfg = yaml.safe_load(config_path.read_text())
    cfg['student']['router_config'] = DYDIT
    config_path.write_text(yaml.safe_dump(cfg))
    run_trainer(config_path)
    output, = (tmp_path / 'run').iterdir()
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [row['target_keep'] for row in records] == pytest.approx([.99, .62, .25])
    for row in records:
        assert row['stochastic_mlp_keep'] == row['mlp_keep']
        assert row['budget'] == pytest.approx((row['mlp_keep'] - row['target_keep']) ** 2, abs=1e-6)
        assert row['router_learning_rate'] == pytest.approx(10 * row['learning_rate'])
        assert 'router/layer_03/actual_keep' in row and 'preview/deterministic_mlp_keep' in row
    run_trainer(config_path, resume=output / 'checkpoint-00000002')
    model = load_export(output / 'export')
    assert model.attn_compression['factor'] == 2 and model.blocks[0].router.routing_mode == 'dydit'

    import inference_vosr_ablation as inference
    from PIL import Image
    from ablation_hourglass_smoke import TinyVAE, fake_dino_features
    monkeypatch.setattr(inference, 'dino_features', fake_dino_features)
    pipeline = json.loads((output / 'export/pipeline.json').read_text())
    image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (40, 48, 3), dtype=np.uint8))
    restored, keep = inference.restore(model, TinyVAE(), nn.Identity(), image, pipeline, torch.device('cpu'),
                                       tile_size=64, tile_overlap=32, vae_tile_size=0, upscale=1)
    assert restored.size == (48, 40) and 0 <= keep <= 1
