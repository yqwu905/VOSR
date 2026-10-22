"""Generate budget-capped protected-head layouts plus equal-MAC random controls.

Feature rankings propose hypotheses; release-rescue rankings provide conditional
causal evidence. Neither guarantees OCR improvement or additive multi-head gains.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_utils import load_config, load_export, save_export
from models.attn_compression import attn_compression_spec
from models.head_sensitivity import assign_heads, attention_macs, head_modes, reconfigure
from tools.attn_head_sensitivity import checkpoint_fingerprint, kernels_from_config, write_json


def promotion_scores(report, modes, metric):
    scores = {}
    if report['kind'] == 'same_input_feature_screen':
        for row in report['rows']:
            l, h = row['layer'], row['head']
            if row['mode'] == modes[l][h] and metric in row:
                scores[l, h] = max(0., row[metric])
    elif report['kind'] == 'release_rescue':
        groups = {g['name']: g for g in report['groups']}
        for row in report['rows']:
            group = groups[row['group']]
            if len(group['heads']) == 1 and group.get('mode', 'dense') == 'dense' and metric in row:
                scores[tuple(group['heads'][0])] = max(0., -row[metric])
    else:
        raise ValueError('Use feature screen or per-head release_rescue rankings, not layer/group scores')
    if not scores:
        raise ValueError(f'No eligible scores for {metric}; do not silently fall back to another metric')
    return scores


def protect(spec, scores, depth, heads, dim, grid, ratio):
    base = attention_macs(spec, depth, heads, dim, grid)
    if not base['ratio'] <= ratio <= 1:
        raise ValueError(f'Budget ratio must be >= baseline {base["ratio"]:.6f} and <= 1')
    modes = head_modes(spec, depth, heads)
    cost = base['per_head']
    ordered = []
    for (layer, head), score in scores.items():
        mode = modes[layer][head]
        extra = cost['dense'] - cost[mode]
        if mode != 'dense' and score > 0 and extra > 0:
            ordered.append((score/extra, score, layer, head, extra))
    selected, total = [], base['macs']
    for _, score, layer, head, extra in sorted(ordered, key=lambda r: (-r[0], r[2], r[3])):
        if total + extra <= ratio * base['dense_macs'] + 1e-6:
            selected.append([layer, head, 'dense'])
            total += extra
    candidate = assign_heads(spec, selected, depth, heads)
    return candidate, selected


def random_control(spec, selected, depth, heads, seed):
    """Match the number of promoted conv/pool heads exactly: costs differ between those modes."""
    rng = random.Random(seed)
    modes, assignments = head_modes(spec, depth, heads), []
    for mode in ('pool', 'conv'):
        count = sum(modes[l][h] == mode for l, h, _ in selected)
        options = [(l, h) for l in range(depth) for h in range(heads) if modes[l][h] == mode]
        assignments.extend([l, h, 'dense'] for l, h in rng.sample(options, count))
    return assign_heads(spec, assignments, depth, heads)


def reclassify(spec, report, metric, depth, heads):
    if report['kind'] != 'same_input_feature_screen':
        raise ValueError('Reclassification needs matched conv/pool feature screening')
    rows = {(r['layer'], r['head'], r['mode']): r for r in report['rows']}
    assignments = []
    for l, modes in enumerate(head_modes(spec, depth, heads)):
        for h, current in enumerate(modes):
            if current == 'dense':
                continue
            a, b = rows.get((l, h, 'pool'), {}), rows.get((l, h, 'conv'), {})
            if metric not in a or metric not in b:
                raise ValueError(f'Missing matched scores at layer {l} head {h}')
            # Ties keep the original classification.
            if a[metric] != b[metric]:
                assignments.append([l, h, 'conv' if b[metric] < a[metric] else 'pool'])
    return assign_heads(spec, assignments, depth, heads)


def build(args):
    cfg = load_config(args.config)
    d, h, dim = (cfg['model'][k] for k in ('depth', 'num_heads', 'hidden_size'))
    grid = int(cfg['data'].get('crop_size') or cfg['data']['resolution']) // 8 // cfg['model']['patch_size']
    original, _ = attn_compression_spec(cfg['student']['attn_compression'], d, h)
    source_config = json.loads((Path(args.source_export)/'model.json').read_text())
    if any(source_config.get(k) != cfg['model'][k] for k in ('depth', 'num_heads', 'hidden_size', 'patch_size')):
        raise ValueError('Source export architecture differs from target config')
    source_spec, _ = attn_compression_spec(source_config['attn_compression'], d, h)
    if original != source_spec:
        raise ValueError('Config baseline layout differs from --source-export; use a matching config')
    report = json.loads(Path(args.ranking).read_text())
    if report['kind'] == 'release_rescue' and report.get('export_hashes') != checkpoint_fingerprint(args.source_export):
        raise ValueError('Rescue ranking was measured on different export weights')
    if report.get('architecture') and report['architecture'] != dict(depth=d, heads=h, dim=dim, grid=grid):
        raise ValueError('Ranking and target architecture/grid differ')
    if any(not 0 <= l < d or not 0 <= head < h for l, head in
           ((r['layer'], r['head']) for r in report['rows'] if 'layer' in r)):
        raise ValueError('Ranking head out of range')
    spec = reclassify(original, report, args.metric, d, h) if args.reclassify else original
    scores = promotion_scores(report, head_modes(spec, d, h), args.metric)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    experiments = []
    for budget in args.ratios:
        candidate, selected = protect(spec, scores, d, h, dim, grid, budget)
        tag = f'{budget:.4f}'.replace('.', 'p')
        variants = [(f'protect_{tag}', candidate)] + [
            (f'random_{tag}_s{seed}', random_control(spec, selected, d, h, seed)) for seed in args.random_seeds]
        if args.reclassify:
            variants.append((f'reclassified_{tag}', spec))
        for name, layout in variants:
            macs = attention_macs(layout, d, h, dim, grid)
            if macs['ratio'] > budget + 1e-12:
                raise ValueError('Reclassified baseline exceeds budget; choose a larger cap or disable --reclassify')
            write_json(out / f'{name}.json', layout)
            # Release conv channels must be migrated before feeding weights to the existing trainer.
            config = dict(_base_=str(Path(args.config).resolve()),
                          student_checkpoint=str(out.resolve()/name/'model.safetensors'),
                          student=dict(attn_compression=dict(layout, head_stats=None)),
                          training=dict(output_dir=f'exp_vosr/{name}', max_steps=args.steps,
                                        lr_warmup_steps=min(100, args.steps), save_every=args.steps,
                                        report_to='none', preview_every=0))
            (out / f'{name}.yml').write_text(yaml.safe_dump(config, sort_keys=False))
            experiments.append(dict(name=name, cap=budget, layout=f'{name}.json', config=f'{name}.yml',
                                    macs=macs, protected_heads=sum(map(len, layout.get('dense_heads', [])))))
    # Continuation control: same release, same number of steps, no architecture change.
    write_json(out / 'baseline.json', original)
    base_cfg = dict(_base_=str(Path(args.config).resolve()), student_checkpoint=args.source_export,
                    training=dict(output_dir='exp_vosr/head_budget_baseline', max_steps=args.steps,
                                  lr_warmup_steps=min(100, args.steps), save_every=args.steps,
                                  report_to='none', preview_every=0))
    (out / 'baseline.yml').write_text(yaml.safe_dump(base_cfg, sort_keys=False))
    write_json(out / 'experiments.json', dict(ranking=str(Path(args.ranking).resolve()),
               metric=args.metric, source_export=args.source_export, calibration_signature=report['cache_signature'],
               note='Proposals only; joint interventions and held-out OCR are required. No measured accuracy gain.',
               baseline=attention_macs(original, d, h, dim, grid), experiments=experiments))
    print(json.dumps([dict(name=e['name'], sa_ratio=e['macs']['ratio'], heads=e['protected_heads']) for e in experiments]))


def migrate(args):
    cfg = load_config(args.config)
    source, dest = Path(args.source), Path(args.output)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Export destination must be empty')
    model = load_export(source)
    spec = json.loads(Path(args.spec).read_text())
    # The proposed layouts produced by build cover all blocks. Reject partial migration of compressed exports.
    if spec.get('blocks') != list(range(len(model.blocks))):
        raise ValueError('Export migration requires an explicit all-block layout')
    reconfigure(model, spec, kernels_from_config(cfg))
    pipeline = json.loads((source/'pipeline.json').read_text())
    save_export(model, dest, pipeline)
    write_json(dest/'migration.json', dict(source=str(source.resolve()), spec=str(Path(args.spec).resolve()),
                                          rule='Learned local kernels copied by head ID; new heads use calibration kernels'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    p.add_argument('--config', required=True)
    p.add_argument('--ranking', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source-export', required=True, help='Existing head-split release for short recovery tuning')
    p.add_argument('--metric', default='small_mse', help='Feature: small_mse; rescue: delta_small_kd_mse')
    p.add_argument('--ratios', nargs='+', type=float, default=[.25, .27, .30])
    p.add_argument('--random-seeds', nargs='+', type=int, default=[11])
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--reclassify', action='store_true', help='Experimental conv/pool selection by matched output error')
    p = sub.add_parser('export')
    p.add_argument('--config', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'build' and args.steps < 1:
        parser.error('steps must be positive')
    with torch.no_grad():
        build(args) if args.command == 'build' else migrate(args)


if __name__ == '__main__':
    main()
