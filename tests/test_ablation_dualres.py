"""Dual-resolution detail stream, LQ-detail routing, coarse DINO pooling and zero input noise.

Requires the upstream timm environment, like tests/test_ablation_hourglass.py, whose fixtures it reuses.
"""
import json
import os
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import numpy as np
import pytest
import torch
from torch import nn
pytest.importorskip('timm', reason='Full backbone requires the upstream timm environment')
from models.lightningdit_ablation import AblationLightningDiT
from models.token_hourglass import GroupRouter, LQDetailScore, detail_positions, hourglass_spec, pool_tokens
from ablation_utils import (input_noise, load_backbone_state, load_config, load_export, noise_input, save_export,
                            student_dino_config)
from test_ablation_hourglass import (HOURGLASS, ROOT, flat, inputs, kwargs, run_trainer, teacher_and_student,
                                     trainer_config)

SMALL_DETAIL = {'width': 8, 'depth': 2, 'kernel': 3, 'mlp_ratio': 2}
DETAIL = {'type': 'hourglass', 'detail': SMALL_DETAIL}  # 6 blocks: 0 and 5 full grid, 1-4 merged


def test_new_options_are_optional_and_validated():
    plain = hourglass_spec({'type': 'hourglass'}, 36)
    assert not {'detail', 'coarse_cond_pool'} & set(plain)  # existing specs and model.json files are unchanged
    assert hourglass_spec({'type': 'hourglass', 'detail': {}}, 36)['detail'] == {
        'width': 512, 'depth': 6, 'kernel': 7, 'mlp_ratio': 4}
    assert hourglass_spec({'type': 'hourglass', 'detail': None}, 36) == plain
    assert hourglass_spec({'type': 'hourglass', 'coarse_cond_pool': 2}, 36)['coarse_cond_pool'] == 2
    assert 'score' not in hourglass_spec({'type': 'hourglass', 'routing': {}}, 36)['routing']
    assert hourglass_spec({'type': 'hourglass', 'routing': {'score': 'lq_detail'}}, 36)['routing']['score'] == 'lq_detail'
    for bad, message in (({'detail': {'width': 0}}, 'detail.width'), ({'detail': {'kernel': 4}}, 'odd'),
                         ({'detail': {'depth': True}}, 'detail.depth'), ({'detail': {'widht': 8}}, 'Unknown'),
                         ({'detail': 8}, 'mapping'), ({'detail': {'depth': 35}}, 'cannot exceed'),
                         ({'detail': {}, 'inner_depth': 2}, 'inner_depth 0'), ({'detail': {}, 'routing': {}}, 'no routing'),
                         ({'coarse_cond_pool': 0}, 'coarse_cond_pool'), ({'coarse_cond_pool': True}, 'coarse_cond_pool'),
                         ({'routing': {'score': 'oracle'}}, 'routing.score')):
        with pytest.raises(ValueError, match=message):
            hourglass_spec({'type': 'hourglass', **bad}, 36)


def test_detail_blocks_spread_over_the_merged_blocks():
    assert detail_positions(list(range(2, 36)), 6) == {7: 0, 13: 1, 18: 2, 24: 3, 30: 4, 35: 5}
    assert detail_positions([*range(3, 9), *range(28, 34)], 6) == {4: 0, 6: 1, 8: 2, 29: 3, 31: 4, 33: 5}
    for name, blocks in (('dualres_min.yml', [4, 6, 8, 29, 31, 33]), ('dualres_2_34.yml', [7, 13, 18, 24, 30, 35])):
        cfg = load_config(ROOT / 'configs/ablations' / name)
        with torch.device('meta'):
            model = AblationLightningDiT(**dict(cfg['model'], auxiliary_time_cond=True, input_size=64),
                                         token_compression=cfg['student']['token_compression'])
        assert sorted(model.active_blocks[p] for p in model.detail_after) == blocks


@pytest.mark.parametrize('extra', [{}, {'fine_in': 0, 'fine_out': 0}, {'fine_blocks': [3]}])
def test_untrained_detail_stream_leaves_the_hourglass_unchanged(extra):
    teacher, plain = teacher_and_student({'type': 'hourglass', **extra})
    _, dual = teacher_and_student({**DETAIL, **extra})
    report = load_backbone_state(dual, teacher.state_dict(), allow_new_modules=True)
    assert any(k.startswith('detail_stream.') for k in report['new_token_merge_parameters'])
    with pytest.raises(ValueError, match='Missing backbone weights'):  # a resume must find the stream's weights
        load_backbone_state(dual, teacher.state_dict())
    x, t, r, z = inputs()
    with torch.no_grad():  # write and exit start at zero: bit for bit the plain hourglass
        torch.testing.assert_close(dual(x, t, r, z), plain(x, t, r, z), rtol=0, atol=0)


def test_detail_stream_gets_gradients_through_its_zero_initialized_exits():
    _, student = teacher_and_student({**DETAIL, 'fine_blocks': [3]}, use_checkpoint=True)
    assert sorted(student.active_blocks[p] for p in student.detail_after) == [2, 4]  # one per merged stretch
    student.train()
    student(*inputs()).square().mean().backward()
    assert [name for name, p in student.named_parameters() if p.grad is None] == []
    stream = student.detail_stream
    assert stream.exit.weight.grad.abs().sum() > 0
    assert all(block.write.weight.grad.abs().sum() > 0 for block in stream.blocks)


def test_detail_stream_needs_the_full_merged_span():
    _, student = teacher_and_student(DETAIL)
    with pytest.raises(ValueError, match='full merged span'):
        student.set_coarse_depth(2)
    with pytest.raises(ValueError, match='full merged span'):
        AblationLightningDiT(**kwargs(), token_compression=DETAIL, coarse_depth=2)


def test_export_roundtrip_keeps_detail_and_coarse_pooling(tmp_path):
    _, student = teacher_and_student({**DETAIL, 'coarse_cond_pool': 2})
    x, t, r, z = inputs()
    with torch.no_grad():
        for parameter in student.detail_stream.parameters():  # a stream that changes the output
            parameter.normal_(std=.02)
        expected = student(x, t, r, z)
    save_export(student, tmp_path, {})
    config = json.loads((tmp_path / 'model.json').read_text())['token_compression']
    assert config['detail'] == SMALL_DETAIL and config['coarse_cond_pool'] == 2
    restored = load_export(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(restored(x, t, r, z), expected)


def test_coarse_cond_pool_pools_dino_only_for_merged_blocks():
    _, student = teacher_and_student({'type': 'hourglass', 'coarse_cond_pool': 2})
    seen = {}
    for index in student.active_blocks:
        student.blocks[index].cross_attn.register_forward_hook(
            lambda module, args, output, i=index: seen.__setitem__(i, args[1]))
    x, t, r, z = inputs()
    with torch.no_grad():
        student(x, t, r, z)
        full = student.mlp_ca(student.layer_norm(z[0]))
    assert {i: tuple(cond.shape[1:2]) for i, cond in seen.items()} == {0: (16,), 1: (4,), 2: (4,), 3: (4,), 4: (4,),
                                                                       5: (16,)}
    torch.testing.assert_close(seen[0], full)
    torch.testing.assert_close(seen[2], pool_tokens(full, 2))
    _, plain = teacher_and_student({'type': 'hourglass'})
    _, one = teacher_and_student({'type': 'hourglass', 'coarse_cond_pool': 1})
    with torch.no_grad():
        torch.testing.assert_close(one(x, t, r, z), plain(x, t, r, z), rtol=0, atol=0)


def test_lq_detail_score_is_the_group_variance_of_the_lq_latent():
    torch.manual_seed(1)
    lq = torch.zeros(1, 2, 16, 16)
    lq[:, :, 4:8, 8:12] = torch.randn(1, 2, 4, 4)  # group row 1, column 2 of the 4x4 groups (4x4 latent pixels each)
    scores = LQDetailScore(2, 2)(lq)
    assert scores.shape == (1, 16) and scores.argmax().item() == 6
    torch.testing.assert_close(scores[0, 6], lq[0, :, 4:8, 8:12].var(dim=(1, 2), unbiased=False).sum())
    assert scores[0, [i for i in range(16) if i != 6]].abs().max() < 1e-6


def test_lq_detail_routing_keeps_the_detailed_groups_on_the_full_grid():
    teacher, student = teacher_and_student({'type': 'hourglass', 'routing': {'score': 'lq_detail'}})
    assert isinstance(student.group_router, LQDetailScore) and not list(student.group_router.parameters())
    kept, route_merge = [], student._route_merge

    def spy(x, tables, scores=None):
        merged, state = route_merge(x, tables, scores)
        kept.append(sorted(state['fine_idx'][0].tolist()))
        return merged, state
    student._route_merge = spy
    x, t, r, z = inputs()  # channels 0-1: LQ latent, 2-3: noise (ignored by the score)
    x[:, :2] = 0
    x[:, :2, 4:8, 8:12] = torch.randn(2, 2, 4, 4)
    x[:, :2, 12:16, 0:4] = 3 * torch.randn(2, 2, 4, 4)
    with torch.no_grad():
        student.set_keep_ratio(2 / 16)
        student(x, t, r, z)
        assert kept == [[6, 12]]
        student.set_keep_ratio(1)  # every group kept: the dense model
        torch.testing.assert_close(student(x, t, r, z), teacher(x, t, r, z), rtol=0, atol=0)
    student.train()
    student.set_train_keep_ratio(.5)
    student(x, t, r, z).square().mean().backward()
    assert [name for name, p in student.named_parameters() if p.grad is None] == []


def test_factor4_routing_mixes_full_grid_and_4x4_merged_tokens():
    teacher, student = teacher_and_student({'type': 'hourglass', 'factor': 4, 'routing': {'score': 'lq_detail'}})
    seen = {}
    for index in student.active_blocks:
        student.blocks[index].attn.register_forward_hook(
            lambda module, args, output, i=index: seen.__setitem__(i, args[0].shape[1]))
    with torch.no_grad():
        for ratio, routed in ((.25, 16 + 3), (.5, 2 * 16 + 2), (0, 4)):  # 8x8 tokens = 4 groups of 4x4
            student.set_keep_ratio(ratio)
            student(*inputs())
            assert seen == {0: 64, 1: routed, 2: routed, 3: routed, 4: routed, 5: 64}
        student.set_keep_ratio(.99)  # every group kept, through the routed gather/RoPE/scatter
        torch.testing.assert_close(student(*inputs()), teacher(*inputs()), rtol=1e-5, atol=1e-5)


def test_noise_input_validation():
    assert noise_input({}) == 'gaussian' and noise_input({'student': {'noise_input': 'zero'}}) == 'zero'
    with pytest.raises(ValueError, match='noise_input'):
        noise_input({'student': {'noise_input': 'none'}})
    latent = torch.randn(1, 2, 4, 4)
    assert not input_noise(latent, 'zero').any() and input_noise(latent, 'gaussian').shape == latent.shape


@pytest.mark.parametrize('noise', ['gaussian', 'zero'])
def test_trainer_exports_dualres_and_inference_follows_the_noise_input(tmp_path, monkeypatch, noise):
    compression = {'type': 'hourglass', 'coarse_cond_pool': 2, 'detail': dict(SMALL_DETAIL, kernel=3)}
    student = dict(HOURGLASS, token_compression=compression, **({'noise_input': noise} if noise == 'zero' else {}))
    config_path, _ = trainer_config(tmp_path, student, gt_weight=1., token_merge_lr_multiplier=10.)
    run_trainer(config_path)
    output, = (tmp_path / 'run').iterdir()
    pipeline = json.loads((output / 'export/pipeline.json').read_text())
    assert pipeline.get('noise_input') == ('zero' if noise == 'zero' else None)  # absent for gaussian, as before
    model = load_export(output / 'export')
    assert model.detail_stream.exit.weight.abs().sum() > 0  # the zero-initialized exit trained

    import inference_vosr_ablation as inference
    from PIL import Image
    from ablation_hourglass_smoke import TinyVAE, fake_dino_features
    monkeypatch.setattr(inference, 'dino_features', fake_dino_features)
    image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (40, 48, 3), dtype=np.uint8))
    outputs = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        restored, _ = inference.restore(model, TinyVAE(), nn.Identity(), image, pipeline, torch.device('cpu'),
                                        tile_size=64, tile_overlap=32, vae_tile_size=0, upscale=1)
        assert restored.size == (48, 40)
        outputs.append(np.asarray(restored))
    assert np.array_equal(*outputs) == (noise == 'zero')  # without noise the seed does not matter


CONFIG_CHANGES = {  # config: (its _base_, the only flattened keys it changes)
    'dualres_min.yml': ('hourglass_sft_min.yml', {'student.token_compression.detail.' + k for k in SMALL_DETAIL}),
    'dualres_2_34.yml': ('loss_all.yml', {*('student.token_compression.' + k for k in (
        'type', 'fine_in', 'fine_out', 'coarse_cond_pool')), *('student.token_compression.detail.' + k
                                                              for k in SMALL_DETAIL),
        'training.token_merge_lr_multiplier'}),
    'dualres_2_34_no_detail.yml': ('dualres_2_34.yml', {'student.token_compression.detail', *(
        'student.token_compression.detail.' + k for k in SMALL_DETAIL)}),
    'allmerged_sft.yml': ('loss_all.yml', {*('student.token_compression.' + k for k in (
        'type', 'fine_in', 'fine_out', 'coarse_cond_pool')), 'training.token_merge_lr_multiplier'}),
    'allmerged_sft_zero_noise.yml': ('allmerged_sft.yml', {'student.noise_input'}),
    'mixres_route_lq.yml': ('mixres_route.yml', {'student.token_compression.routing.score'}),
    'mixres_f4_lq.yml': ('mixres_route_lq.yml', {'student.token_compression.factor', *(
        'student.token_compression.routing.' + k for k in ('keep_ratios', 'eval_keep_ratio'))}),
}
CONFIG_COST = {  # config: (keep ratio or None, DiT MACs over the teacher's, over the dense no-DINO DiT)
    'dualres_min.yml': (None, .794, None), 'dualres_2_34.yml': (None, .288, None),
    'dualres_2_34_no_detail.yml': (None, .275, None), 'allmerged_sft.yml': (None, .232, None),
    'allmerged_sft_zero_noise.yml': (None, .232, None), 'mixres_route_lq.yml': (.75, .582, .821),
    'mixres_f4_lq.yml': (.2, .229, .323),
}


@pytest.mark.parametrize('name', sorted(CONFIG_CHANGES))
def test_shipped_configs_change_one_point_and_cost(name, monkeypatch):
    from torch.utils.flop_counter import FlopCounterMode
    root = ROOT / 'configs/ablations'
    base_name, changes = CONFIG_CHANGES[name]
    assert (root / name).read_text().startswith(f'_base_: {base_name}\n')
    cfg, base = load_config(root / name), load_config(root / base_name)
    config, flat_base = flat(cfg), flat(base)
    assert {k for k in set(config) | set(flat_base) if config.get(k, '<absent>') != flat_base.get(k, '<absent>')} \
        == changes | {'training.output_dir'}
    assert cfg['training']['output_dir'] == f'exp_vosr/{name[:-4]}' and cfg['training']['kd_weight'] == 0
    # autocast has no meta device; the router's FP32 Linear layers count the same either way
    monkeypatch.setattr(GroupRouter, 'forward', lambda self, groups: self.router.net(groups.float()).squeeze(-1))

    def macs(student, ratio=None):
        model = dict(cfg['model'], auxiliary_time_cond=True, input_size=64)
        with torch.device('meta'):
            network = AblationLightningDiT(**model, use_cross_attention=student['use_cross_attention'],
                                           token_compression=student.get('token_compression')).eval()
            x, t, r = torch.randn(1, 32, 64, 64), torch.ones(1), torch.zeros(1)
            z = ([torch.randn(1, (student_dino_config(cfg)['size'] // 14) ** 2, 1024)]
                 if student['use_cross_attention'] else None)
        if ratio is not None:
            network.set_keep_ratio(ratio)
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            network(x, t, r, z)
        return counter.get_total_flops() / 2
    ratio, of_teacher, of_no_dino = CONFIG_COST[name]
    cost = macs(cfg['student'], ratio)
    teacher = dict(use_cross_attention=True)
    assert abs(cost / macs(teacher) - of_teacher) < 1e-3
    if of_no_dino is not None:
        assert abs(cost / macs(dict(use_cross_attention=False)) - of_no_dino) < 1e-3
