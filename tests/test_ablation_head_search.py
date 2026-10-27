"""Learned head selection on head-split self-attention (attn_compression.search, kv_pool and dense heads);
requires the upstream timm environment."""
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import numpy as np
import pytest
import torch
import torch.nn.functional as F
pytest.importorskip('timm', reason='Full backbone requires the upstream timm environment')
from models.attn_compression import (HeadSearchAttention, HeadSplitAttention, attn_compression_spec, head_macs,
                                     search_price)
from models.lightningdit_ablation import AblationLightningDiT
from models.token_hourglass import new_module_key
from ablation_utils import load_backbone_state, load_config, load_export, read_weights, save_export
from test_ablation_head_split import (SHIPPED_STATS, attention_module, centroid_rope, inputs, kwargs,
                                      paired_trainer_config, reference_head_split, run_trainer, split,
                                      teacher_and_student, write_stats)

ROOT = Path(__file__).resolve().parents[1]
SEARCH = {'candidates': ['kv_pool', 'dense'], 'attn_budget': .6}


def reference_query_heads(attn, x, heads, factor, full_rope, pooled_rope, pooled_kv):
    """Per head: full-grid queries attend to the full-grid (dense) or factor x factor averaged (kv_pool) keys and
    values; each head projects with its columns of proj (no bias)."""
    b, n, c = x.shape
    side, hd = math.isqrt(n), c // attn.num_heads
    s = side // factor
    q = F.linear(x, attn.qkv.weight[:c], attn.qkv.bias[:c]).reshape(b, n, attn.num_heads, hd).transpose(1, 2)
    q = full_rope(attn.q_norm(q))
    source = x.reshape(b, s, factor, s, factor, c).mean((2, 4)).reshape(b, s * s, c) if pooled_kv else x
    k, v = F.linear(source, attn.qkv.weight[c:], attn.qkv.bias[c:]).chunk(2, -1)
    k, v = (t.reshape(b, -1, attn.num_heads, hd).transpose(1, 2) for t in (k, v))
    k = (pooled_rope if pooled_kv else full_rope)(attn.k_norm(k))
    out = torch.zeros(b, n, c)
    for h in heads:
        weights = (q[:, h] @ k[:, h].transpose(-1, -2) / math.sqrt(hd)).softmax(-1)
        out += (weights @ v[:, h]) @ attn.proj.weight[:, h * hd:(h + 1) * hd].T
    return out


@pytest.mark.parametrize('local, kv_pool, dense', [([0], [1], [2]), ([], [0, 1, 2, 3], []), ([], [], [0, 1, 2, 3]),
                                                    ([1], [], [0, 3]), ([2], [0, 1, 3], [])])
def test_kv_pool_and_dense_heads_match_per_head_reference(local, kv_pool, dense):
    attn = attention_module()
    module = HeadSplitAttention(attn, local, factor=2, kernel=3, upsample='bilinear', kv_pool_heads=kv_pool,
                                dense_heads=dense)
    kernels = torch.rand(len(local), 3, 3)
    module.set_local_kernels(kernels)
    x = torch.randn(2, 64, 64)  # an 8x8 token grid
    pooled, full = centroid_rope(8, 2), centroid_rope(8, 1)
    with torch.no_grad():
        out = module(x, rope=(pooled, full))
        # Query heads count as local heads with zero kernels in the head-split reference (no pooled output).
        typed = local + kv_pool + dense
        expected = reference_head_split(attn, x, typed, torch.cat((kernels, torch.zeros(len(typed) - len(local), 3, 3))),
                                        2, pooled)
        expected += reference_query_heads(attn, x, dense, 2, full, pooled, pooled_kv=False)
        expected += reference_query_heads(attn, x, kv_pool, 2, full, pooled, pooled_kv=True)
    torch.testing.assert_close(out, expected)
    assert module.needs_full_rope and module.global_heads == [h for h in range(4) if h not in typed]
    with pytest.raises(ValueError, match='need \\(pooled RoPE, full-grid RoPE\\)'):
        module(x, rope=pooled)
    with pytest.raises(ValueError, match='at most one of local, kv_pool and dense'):
        HeadSplitAttention(attn, [0], 2, 3, 'bilinear', kv_pool_heads=[0])


def fixed(local, kv_pool=None, dense=None, depth=6, **extra):
    spec = {'type': 'head_split', 'kernel': 3, 'local_heads': [local] * depth, **extra}
    if kv_pool is not None:
        spec['kv_pool_heads'] = [kv_pool] * depth
    if dense is not None:
        spec['dense_heads'] = [dense] * depth
    return spec


def test_dense_heads_everywhere_are_the_teacher_and_kv_pool_heads_differ():
    teacher, student = teacher_and_student(fixed([], dense=[0, 1, 2, 3]))
    with torch.no_grad():
        torch.testing.assert_close(student(*inputs()), teacher(*inputs()))
        _, kv = teacher_and_student(fixed([1], kv_pool=[0, 2], dense=[3]))
        assert not torch.allclose(kv(*inputs()), teacher(*inputs()))
    spec = kv.attn_compression
    assert spec['kv_pool_heads'] == [[0, 2]] * 6 and spec['dense_heads'] == [[3]] * 6
    assert attn_compression_spec(spec, 6, 4) == (spec, None)  # model.json passes through unchanged
    plain, _ = attn_compression_spec(fixed([1], kv_pool=[], dense=[]), 6, 4)
    assert 'kv_pool_heads' not in plain and 'dense_heads' not in plain  # absent when empty: f2 exports unchanged


def export_dir(tmp_path, spec, name='fixed'):
    _, model = teacher_and_student(spec)
    save_export(model, tmp_path / name, {})
    return tmp_path / name


@pytest.mark.parametrize('extra, message', [
    ({'local_heads': [[0]] * 6, 'kv_pool_heads': [[0]] + [[]] * 5}, 'a head is at most one of local, kv_pool'),
    ({'local_heads': [[]] * 6, 'blocks': [1], 'dense_heads': [[0]] + [[]] * 5}, r'dense_heads\[0\]: block 0 is not'),
    ({'local_heads': [[]] * 6, 'kv_pool_heads': [[4]] + [[]] * 5}, r'kv_pool_heads\[0\] must be unique heads'),
    ({'local_heads': [[]] * 6, 'dense_heads': [[0]] * 5}, 'dense_heads must hold one head list per block'),
    ({'local_heads': [[]] * 6, 'dense_heads': [[0]] * 6, 'search': SEARCH}, 'search starts from local and pooled'),
    ({'local_heads': [[]] * 6, 'search': dict(SEARCH, budget=.5)}, 'Unknown attn_compression.search options'),
    ({'local_heads': [[]] * 6, 'search': dict(SEARCH, candidates=['dense', 'dense'])}, 'candidates must be unique'),
    ({'local_heads': [[]] * 6, 'search': dict(SEARCH, candidates=['local'])}, 'candidates must be unique'),
    ({'local_heads': [[]] * 6, 'search': {'candidates': ['dense']}}, r'attn_budget .* must be in \(0, 1\]'),
    ({'local_heads': [[]] * 6, 'search': dict(SEARCH, attn_budget=1.5)}, r'attn_budget .* must be in \(0, 1\]'),
    ({'local_heads': [[]] * 6, 'search': dict(SEARCH, temperature=[1, 0])}, r'temperature must be \[start, end\]'),
    ({'local_heads': [[]] * 6, 'search': [.5]}, 'search must be a mapping'),
])
def test_spec_validation_of_query_heads_and_search(extra, message):
    with pytest.raises(ValueError, match=message):
        attn_compression_spec(dict({'type': 'head_split', 'kernel': 3}, **extra), 6, 4)


def test_heads_from_reads_the_fixed_lists_of_an_export(tmp_path):
    source = export_dir(tmp_path, fixed([1], kv_pool=[0], dense=[3]))
    spec, _ = attn_compression_spec({'type': 'head_split', 'kernel': 3, 'heads_from': str(source)}, 6, 4)
    assert spec == json.loads((source / 'model.json').read_text())['attn_compression']
    assert attn_compression_spec({'type': 'head_split', 'kernel': 3, 'heads_from': str(source / 'model.json')},
                                 6, 4)[0] == spec
    for extra, message in (({'local_heads': [[]] * 6}, 'heads_from replaces local_heads'),
                           ({'kernel': 5}, 'heads_from has kernel 3, this config 5'),
                           ({'factor': 4}, 'heads_from has factor 2, this config 4'),
                           ({'blocks': [0, 1]}, r'heads_from has blocks \[0, 1, 2, 3, 4, 5\], this config \[0, 1\]')):
        with pytest.raises(ValueError, match=message):
            attn_compression_spec(dict({'type': 'head_split', 'kernel': 3, 'heads_from': str(source)}, **extra), 6, 4)
    with pytest.raises(FileNotFoundError, match='heads_from: .* not found'):
        attn_compression_spec({'type': 'head_split', 'heads_from': str(tmp_path / 'missing')}, 6, 4)
    searching = export_dir(tmp_path, fixed([1], search=SEARCH), 'search')
    with pytest.raises(ValueError, match='is a head search state'):
        attn_compression_spec({'type': 'head_split', 'kernel': 3, 'heads_from': str(searching)}, 6, 4)
    plain = tmp_path / 'plain'
    plain.mkdir()
    (plain / 'model.json').write_text(json.dumps({'depth': 6}))
    with pytest.raises(ValueError, match='is not a head-split export'):
        attn_compression_spec({'type': 'head_split', 'heads_from': str(plain)}, 6, 4)


def test_search_price_meets_the_budget_with_the_smallest_price():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(40, 3, generator=generator, dtype=torch.float64) * 3
    costs = torch.tensor([.2, .5, 1.], dtype=torch.float64).repeat(40, 1)
    costs[::3, 0] = .4  # local-conv bases cost more than pooled ones
    for hard in (False, True):
        spend = lambda price: ((costs.gather(1, (logits - price * costs).argmax(1, keepdim=True)).sum()) if hard else
                               ((logits - price * costs).softmax(1) * costs).sum()).item()
        cheapest = costs.min(1).values.sum().item()
        for share in (.1, .5, .9):  # between the cheapest choice and the spend without a price
            target = cheapest + share * (spend(0.) - cheapest)
            price = search_price(logits, costs, target, hard=hard)
            assert price > 0 and spend(price) <= target + 1e-9 and spend(price * (1 - 1e-6)) > target
        assert search_price(logits, costs, spend(0.) + 1, hard=hard) == 0  # within budget without a price
    with pytest.raises(ValueError, match='does not exceed the cost of the cheapest option'):
        search_price(logits, costs, costs.min(1).values.sum().item())


def search_model(tmp_path, budget=.4, seed=1, **extra):
    write_stats(tmp_path / 'stats.json')
    _, model = teacher_and_student(split(tmp_path / 'stats.json', kernel=3,
                                         search=dict(SEARCH, attn_budget=budget)), **extra)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed)
        for block in model.blocks:  # trained-looking logits and convs
            block.attn.head_logits.copy_(torch.randn(block.attn.head_logits.shape, generator=generator) * 2)
            if block.attn.local_conv is not None:
                block.attn.local_conv.weight.normal_(generator=generator)
    return model


def test_search_model_structure_budget_and_new_logits(tmp_path):
    model = search_model(tmp_path)
    assert all(isinstance(block.attn, HeadSearchAttention) for block in model.blocks)
    logits = [k for k in model.state_dict() if k.endswith('head_logits')]
    assert len(logits) == 6 and all(new_module_key(k) for k in logits)
    assert model.head_search == {'candidates': ['kv_pool', 'dense'], 'attn_budget': .4, 'temperature': [1., .1]}
    # Costs in dense heads: pooled / local base, kv_pool, dense (8x8 tokens, hidden 64, head_dim 16, kernel 3).
    unit = lambda kind: head_macs(kind, 64, 64, 16, 2, 3) / head_macs('dense', 64, 64, 16, 2, 3)
    attn = model.blocks[0].attn
    for head in range(4):
        base = unit('local' if head in attn.local_heads else 'pooled')
        torch.testing.assert_close(attn.costs[head], torch.tensor([base, unit('kv_pool'), 1.]))
    report = model.head_search_report()
    assert report['selected_attn_ratio'] <= .4  # the fixed choice fits the budget
    assert report['expected_attn_ratio'] == pytest.approx(.4)
    assert report['kv_pool_heads'] + report['dense_heads'] > 0
    with pytest.raises(ValueError, match='must exceed the head-split base structure'):
        search_model(tmp_path, budget=.1)
    with pytest.raises(ValueError, match='must exceed the head-split base structure'):
        model.set_search_budget(.1)
    model.set_search_budget(1.)
    assert model.head_search['attn_budget'] == 1 and model.export_config['attn_compression']['search']['attn_budget'] == 1


def test_search_eval_is_the_selected_export_and_partial_blocks_count_as_dense(tmp_path):
    model = search_model(tmp_path, budget=.7)
    save_export(model, tmp_path / 'checkpoint', {})  # the search state: resumes and runs its eval choice
    save_export(model, tmp_path / 'selected', {}, selected_heads=True)
    config = json.loads((tmp_path / 'selected/model.json').read_text())['attn_compression']
    assert 'search' not in config and config == model.selected_attn_compression()
    state = read_weights(tmp_path / 'selected/model.safetensors')
    assert not any(k.endswith('head_logits') for k in state)
    for index, block in enumerate(model.blocks):  # local convs keep only the rows of heads that stay local
        kept = config['local_heads'][index]
        rows = [i * 16 + c for i, head in enumerate(block.attn.local_heads) if head in kept for c in range(16)]
        key = f'blocks.{index}.attn.local_conv.weight'
        assert (key in state) == bool(rows)
        if rows:
            assert torch.equal(state[key], block.attn.local_conv.weight[rows])
    selected, searching = load_export(tmp_path / 'selected'), load_export(tmp_path / 'checkpoint')
    assert not any(isinstance(block.attn, HeadSearchAttention) for block in selected.blocks)
    with torch.no_grad():
        torch.testing.assert_close(searching(*inputs()), selected(*inputs()), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(model.eval()(*inputs()), selected(*inputs()), rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match='selected_heads needs attn_compression.search'):
        selected.export_state(selected_heads=True)
    # Blocks outside attn_compression.blocks stay dense and count against the budget.
    _, partial = teacher_and_student(split(tmp_path / 'stats.json', kernel=3, blocks=[0, 2, 4],
                                           search=dict(SEARCH, attn_budget=.7)))
    report = partial.head_search_report()
    assert report['expected_attn_ratio'] == pytest.approx(.7) and report['selected_attn_ratio'] <= .7
    with pytest.raises(ValueError, match='must exceed the head-split base structure'):
        teacher_and_student(split(tmp_path / 'stats.json', kernel=3, blocks=[0, 2, 4],
                                  search=dict(SEARCH, attn_budget=.5)))


def test_training_samples_hard_choices_at_the_budget_and_only_the_logits_learn(tmp_path):
    model = search_model(tmp_path).train()
    torch.nn.init.normal_(model.final_layer.linear.weight, std=.02)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.endswith('head_logits'))
    model._set_search_price()
    modules = model._search_modules()
    torch.manual_seed(0)
    cost = 0
    for module in modules:
        choice = module.choice(4000)
        assert torch.equal(choice.detach().sum(-1), torch.ones(4000, 4)) and set(choice.detach().unique().tolist()) <= {0., 1.}
        cost = cost + (choice.detach() * module.costs).sum((1, 2))
    # Sampled self-attention MACs average to the budget (in dense heads: 0.4 x 6 blocks x 4 heads).
    assert cost.mean().item() == pytest.approx(.4 * 24, rel=.01)
    model.set_search_temperature(.3)
    assert all(module.temperature == .3 for module in modules)
    model(*inputs()).square().mean().backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert sorted(grads) == sorted(f'blocks.{i}.attn.head_logits' for i in range(6))
    assert all(grad.abs().sum() > 0 for grad in grads.values())


def test_vosr2_size_macs_of_query_heads_and_the_shipped_budget():
    from torch.utils.flop_counter import FlopCounterMode
    cfg = load_config(ROOT / 'configs/ablations/head_split_f2_search.yml')
    model = dict(cfg['model'], auxiliary_time_cond=False, input_size=64)
    release, _ = attn_compression_spec({'type': 'head_split', 'head_stats': str(SHIPPED_STATS)}, 36, 24)

    def macs(attn_compression):
        with torch.device('meta'):
            network = AblationLightningDiT(**model, attn_compression=attn_compression).eval()
            x, t, r, z = torch.randn(1, 32, 64, 64), torch.ones(1), torch.zeros(1), [torch.randn(1, 1024, 1024)]
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            network(x, t, r, z)
        self_attention = sum(sum(ops.values()) for name, ops in counter.get_flop_counts().items()
                             if name.count('.') == 3 and name.endswith('.attn'))
        return counter.get_total_flops() / 2, self_attention / 2
    # Upgrade some pooled and some local heads in every block.
    kv_pool = [[h for h in range(24) if h not in local][:3] for local in release['local_heads']]
    dense = [[h for h in range(24) if h not in local and h not in kv][:2] + local[:1]
             for local, kv in zip(release['local_heads'], kv_pool)]
    local = [[h for h in heads if h not in d] for heads, d in zip(release['local_heads'], dense)]
    spec = dict(release, local_heads=local, kv_pool_heads=kv_pool, dense_heads=dense)
    (dense_total, dense_attention), (f2_total, f2_attention), (_, attention) = macs(None), macs(release), macs(spec)
    per_head = lambda kind: head_macs(kind, 1024, 1536, 64, 2, 7)
    analytic = sum(per_head('local') * len(l) + per_head('kv_pool') * len(k) + per_head('dense') * len(d)
                   + per_head('pooled') * (24 - len(l) - len(k) - len(d)) for l, k, d in zip(local, kv_pool, dense))
    assert attention == analytic  # FlopCounterMode counts exactly the head_macs of the search's costs
    assert per_head('dense') * 864 == dense_attention
    # The shipped budget: self-attention 0.2947 of the dense MACs is DiT 80.0% of the teacher's.
    budget = cfg['student']['attn_compression']['search']['attn_budget']
    assert (f2_total - f2_attention + budget * dense_attention) / dense_total == pytest.approx(.800, abs=2e-4)
    assert f2_attention / dense_attention == pytest.approx(.2326, abs=1e-4)


def test_shipped_search_and_learned_configs():
    search = load_config(ROOT / 'configs/ablations/head_split_f2_search.yml')
    learned = load_config(ROOT / 'configs/ablations/head_split_f2_learned.yml')
    f2 = load_config(ROOT / 'configs/ablations/head_split_f2.yml')
    spec, _ = attn_compression_spec(search['student']['attn_compression'], 36, 24)
    release = json.loads((ROOT / 'weights/head_split_f2/model.json').read_text())['attn_compression']
    assert {k: v for k, v in spec.items() if k != 'search'} == release  # the release export's base structure
    assert spec['search'] == {'candidates': ['kv_pool', 'dense'], 'attn_budget': .2947, 'temperature': [1., .1]}
    assert search['student_checkpoint'] == 'weights/head_split_f2'
    tc = search['training']
    assert tc['trainable_parameters'] == [r'\.attn\.head_logits$'] and tc['token_merge_lr_multiplier'] == 1
    assert (tc['kd_weight'], tc['kd_text_weight'], tc['ema_decay'], tc['lr_schedule']) == (1, 3, 0, 'constant')
    assert tc['output_dir'] == 'exp_vosr/head_split_f2_search' and tc['max_steps'] == 2000
    attn = learned['student']['attn_compression']
    assert learned['student_checkpoint'] == attn['heads_from'] and attn['heads_from'].endswith('/export')
    assert attn.get('head_stats') is None and 'search' not in attn
    assert learned['training']['output_dir'] == 'exp_vosr/head_split_f2_learned'
    assert learned['training']['max_steps'] == 6000
    for key in ('kd_weight', 'learning_rate', 'lr_schedule', 'min_lr_ratio', 'ema_decay', 'token_merge_lr_multiplier'):
        assert learned['training'][key] == f2['training'][key]  # head_split_f2's recipe
    assert search['data'] == learned['data'] == f2['data']


def test_paired_text_masks_cover_the_box_rectangles(tmp_path):
    from ablation_utils import build_train_dataset
    from test_ablation_data import paired_folders
    config = paired_folders(tmp_path, sizes=(('a', 64, 64), ('b', 64, 64)))
    (tmp_path / 'boxes.json').write_text(json.dumps({'a': [[10, 12, 40, 10, 42, 30, 8, 28], [60, 60, 70, 60, 70, 70,
                                                                                               60, 70]]}))
    dataset = build_train_dataset(dict(config, resolution=64, text_boxes=str(tmp_path / 'boxes.json')),
                                  text_mask=True)
    mask = dataset.sample('a', random.Random(0))['text_mask']  # the crop is the whole image
    expected = torch.zeros(1, 64, 64)
    expected[:, 10:30, 8:42] = 1
    expected[:, 60:, 60:] = 1  # clipped to the crop
    assert torch.equal(mask, expected)
    assert torch.equal(dataset.sample('b', random.Random(0))['text_mask'], torch.zeros(1, 64, 64))  # no boxes
    shifted = build_train_dataset(dict(config, resolution=32, text_boxes=str(tmp_path / 'boxes.json')),
                                  text_mask=True)
    item = shifted.sample('a', random.Random(3))
    full = dataset.sample('a', random.Random(0))
    (y, x), = [(y, x) for y in range(33) for x in range(33) if torch.equal(full['hq'][:, y:y + 32, x:x + 32], item['hq'])]
    assert torch.equal(item['text_mask'], expected[:, y:y + 32, x:x + 32])
    assert 'text_mask' not in build_train_dataset(dict(config, text_boxes=str(tmp_path / 'boxes.json')))[0]
    with pytest.raises(ValueError, match='Text-box masks need data.text_boxes'):
        build_train_dataset(config, text_mask=True)
    with pytest.raises(ValueError, match='need data.dataset_type: paired'):
        build_train_dataset({'dataset_type': 'txt'}, text_mask=True)


def search_trainer_config(tmp_path, attn_budget=.4, **training):
    import yaml
    path = paired_trainer_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg['student']['attn_compression']['search'] = dict(SEARCH, attn_budget=attn_budget)
    cfg['training'].update(training)
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_trainer_searches_heads_with_text_weighted_kd_resumes_and_fine_tunes_the_choice(tmp_path):
    import yaml
    config_path = search_trainer_config(tmp_path, trainable_parameters=[r'\.attn\.head_logits$'], kd_text_weight=3.,
                                        learning_rate=.5)
    run_trainer(config_path)
    output, = (tmp_path / 'run').iterdir()
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [row['head_search/temperature'] for row in records] == pytest.approx([1., .7, .4])  # linear over 3 steps
    assert all(row['head_search/expected_attn_ratio'] == pytest.approx(.4) for row in records)
    assert all(row['head_search/selected_attn_ratio'] <= .4 and row['kd'] > 0 for row in records)
    teacher = torch.load(tmp_path / 'teacher.pt')
    last = read_weights(output / 'checkpoint-00000003/model.safetensors')
    for key, value in last.items():  # frozen: only the logits moved
        if key in teacher:
            assert torch.equal(value, teacher[key]), key
    assert any(last[f'blocks.{i}.attn.head_logits'].abs().sum() > 0 for i in range(4))
    selection = json.loads((output / 'checkpoint-00000002/head_selection.json').read_text())
    assert selection['options'] == ['base', 'kv_pool', 'dense'] and len(selection['probabilities']) == 4
    assert 'search' in json.loads((output / 'checkpoint-00000002/model.json').read_text())['attn_compression']
    final = json.loads((output / 'export/model.json').read_text())['attn_compression']
    searching = load_export(output / 'checkpoint-00000003')
    assert 'search' not in final and final == searching.selected_attn_compression()
    assert json.loads((output / 'export/head_selection.json').read_text())['choice'] == \
        searching.head_search_report(probabilities=True)['choice']
    with torch.no_grad():
        x, t, r = torch.randn(1, 6, 8, 8), torch.ones(1), torch.zeros(1)
        z = [torch.randn(1, 16, 6)]
        torch.testing.assert_close(load_export(output / 'export')(x, t, r, z), searching(x, t, r, z),
                                   rtol=1e-5, atol=1e-5)
    run_trainer(config_path, resume=output / 'checkpoint-00000002')  # rewrites checkpoint 3 and export/
    final = json.loads((output / 'export/model.json').read_text())['attn_compression']

    # Another budget for the same logits.
    tool = [sys.executable, 'tools/head_search_select.py', '--checkpoint', str(output / 'checkpoint-00000003'),
            '--output', str(tmp_path / 'select')]
    env = dict(os.environ, TORCHDYNAMO_DISABLE='1')
    run = subprocess.run(tool + ['--attn-budget', '.3'], cwd=ROOT, text=True, capture_output=True, timeout=300,
                         env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    summary = json.loads((tmp_path / 'select/head_selection.json').read_text())
    assert summary['attn_budget'] == .3 and summary['selected_attn_ratio'] <= .3
    assert json.loads((tmp_path / 'select/pipeline.json').read_text()) == \
        json.loads((output / 'checkpoint-00000003/pipeline.json').read_text())
    assert 'search' not in json.loads((tmp_path / 'select/model.json').read_text())['attn_compression']
    run = subprocess.run(tool + ['--attn-budget', '.01'], cwd=ROOT, text=True, capture_output=True, timeout=300,
                         env=env)
    assert run.returncode != 0 and 'must exceed the head-split base structure' in run.stderr
    run = subprocess.run([sys.executable, 'tools/head_search_select.py', '--checkpoint', str(output / 'export'),
                          '--output', str(tmp_path / 'again')], cwd=ROOT, text=True, capture_output=True,
                         timeout=300, env=env)
    assert run.returncode != 0 and 'has no attn_compression.search' in run.stderr

    # Stage 2: the fixed choice fine-tunes from the search export.
    cfg = yaml.safe_load(config_path.read_text())
    cfg['student_checkpoint'] = str(output / 'export')
    cfg['student']['attn_compression'] = {'type': 'head_split', 'kernel': 3, 'heads_from': str(output / 'export')}
    cfg['training'] = dict(cfg['training'], output_dir=str(tmp_path / 'learned'), learning_rate=1e-3,
                           token_merge_lr_multiplier=10.)
    for key in ('trainable_parameters', 'kd_text_weight'):
        cfg['training'].pop(key)
    config_path.write_text(yaml.safe_dump(cfg))
    run_trainer(config_path)
    learned, = (tmp_path / 'learned').iterdir()
    assert json.loads((learned / 'export/model.json').read_text())['attn_compression'] == final
    assert not (learned / 'export/head_selection.json').exists()


@pytest.mark.parametrize('training, message', [
    ({'kd_text_weight': -1}, 'kd_text_weight must be >= 0'),
    ({'kd_text_weight': 1, 'kd_weight': 0, 'gt_weight': 1}, 'it needs kd_weight > 0'),
])
def test_trainer_rejects_bad_text_weighted_kd(tmp_path, training, message):
    with pytest.raises(AssertionError, match=message):
        run_trainer(paired_trainer_config(tmp_path, **training))


def test_trainer_rejects_text_weighted_kd_without_boxes(tmp_path):
    import yaml
    config_path = paired_trainer_config(tmp_path, kd_text_weight=1.)
    cfg = yaml.safe_load(config_path.read_text())
    cfg['data'].pop('text_boxes')
    cfg['data'].pop('text_crop_prob')
    config_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(AssertionError, match='paired data with data.text_boxes'):
        run_trainer(config_path)
