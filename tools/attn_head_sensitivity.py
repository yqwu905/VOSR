"""Reproducible VOSR2 head sensitivity: prepare -> profile -> ablate -> plan/export.

See docs/attn_head_sensitivity.md. Real model/data paths are required; there is no
synthetic fallback. --device cpu is supported for tiny fixtures, not a speed claim.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_utils import (dino_features, load_backbone_state, load_config, load_dino, load_export,
                            read_weights, save_export, vae_tiling)
from models.attn_compression import attn_compression_spec, load_head_stats, pooled_rope
from models.head_sensitivity import (assign_heads, attention_macs, head_modes, intervention,
                                     measure_heads, reconfigure, weighted_mse)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checkpoint_fingerprint(path):
    """Hash the actual weight bytes, including every shard; paths alone are not reproducible identities."""
    path = Path(path)
    if path.is_dir():
        choices = [path / sub / name for sub in ('clean_weights', 'checkpoints', '')
                   for name in ('ema_model.safetensors', 'model.safetensors', 'model.safetensors.index.json')]
        path = next((p for p in choices if p.is_file()), None)
        if path is None:
            raise FileNotFoundError('No weights for fingerprint')
    files = [path]
    if path.name.endswith('.safetensors.index.json'):
        files += [path.parent / name for name in sorted(set(json.loads(path.read_text())['weight_map'].values()))]
    return {p.name: sha256(p) for p in files}


def spearman(pairs):
    """Average ranks for ties, no scipy dependency; undefined correlations stay null."""
    if len(pairs) < 3:
        return None
    def ranks(values):
        order = np.argsort(values, kind='stable')
        ranked = np.empty(len(values), dtype=float)
        start = 0
        while start < len(order):
            end = start+1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            ranked[order[start:end]] = (start+end-1)/2
            start = end
        return ranked
    a, b = (ranks(v) for v in np.array(pairs).T)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None


def provenance():
    try:
        git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git, dirty = None, None
    return dict(git_commit=git, git_dirty=dirty, torch=torch.__version__, cuda=torch.version.cuda)


def amp(device, precision):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == 'bf16')


def kernels_from_config(cfg):
    compression = cfg['student'].get('attn_compression') or {}
    path = compression.get('head_stats')
    if not path:
        return None
    stats = load_head_stats(path)
    radius, kernel = stats['radius'], compression.get('kernel', 7)
    half = kernel // 2
    if half > radius:
        raise ValueError('head_stats window is smaller than the conv kernel')
    pattern = stats['pattern'].float().reshape(*stats['mass3x3'].shape, 2*radius+1, 2*radius+1)
    pattern = pattern[..., radius-half:radius+half+1, radius-half:radius+half+1]
    sums = pattern.sum((-2, -1), keepdim=True)
    if (sums <= 0).any() or not torch.isfinite(pattern).all():
        raise ValueError('Conv patterns need finite positive mass')
    return pattern / sums


def load_teacher(cfg, device):
    from models.lightningdit_ablation import AblationLightningDiT
    state = read_weights(cfg['teacher_checkpoint'])
    options = dict(cfg['model'])
    aux = options.pop('auxiliary_time_cond', 'auto')
    options['auxiliary_time_cond'] = any(k.startswith('r_embedder.') for k in state) if aux == 'auto' else bool(aux)
    options.update(input_size=int(cfg['data']['resolution']) // 8, use_checkpoint=False)
    model = AblationLightningDiT(**options, use_cross_attention=True)
    load_backbone_state(model, state)
    model.sensitivity_checkpoint_hashes = checkpoint_fingerprint(cfg['teacher_checkpoint'])
    return model.to(device).eval().requires_grad_(False)


def translated_boxes(quads, origin, size, complete=False):
    result = []
    offset = np.asarray(origin)
    for quad in quads:
        points = np.asarray(quad, dtype=np.float32).reshape(-1, 2) - offset
        if complete and (points.min() < 0 or points.max() > size):
            continue
        if points[:, 0].max() <= 0 or points[:, 1].max() <= 0 or \
                points[:, 0].min() >= size or points[:, 1].min() >= size:
            continue
        result.append(points)
    return result


def roi_masks(quads, size):
    from evaluate import text_size
    masks = {name: Image.new('L', (size, size)) for name in ('text', 'small')}
    for points in quads:
        polygon = [tuple(map(float, xy)) for xy in points]
        ImageDraw.Draw(masks['text']).polygon(polygon, fill=255)
        if text_size(points) < 64:  # SR/HQ pixels, never LR pixels
            ImageDraw.Draw(masks['small']).polygon(polygon, fill=255)
    return {name: torch.from_numpy(np.array(mask, copy=True)).float()[None, None] / 255
            for name, mask in masks.items()}


def token_masks(sample, side, device):
    return {name: F.adaptive_avg_pool2d(mask.to(device), (side, side)).flatten(2).transpose(1, 2)
            for name, mask in sample['masks'].items()}


@torch.no_grad()
def prepare(args, cfg, device):
    from dataloaders.paired_crops import PairedCropDataset
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    from tiled_vae import encode_dispatch
    data = cfg['data']
    if data.get('dataset_type') != 'paired' or not data.get('text_boxes'):
        raise ValueError('Prepare needs paired LQ/HQ data and text_boxes, preferably TRAIN calibration images')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Cache output must be empty; use a new directory to avoid stale/mixed samples')
    size = int(data.get('crop_size') or data['resolution'])
    dataset = PairedCropDataset(data['lq_dir'], data['hq_dir'], size, data['text_boxes'], 1.)
    names = dataset.names
    if args.names:
        names = [line.strip() for line in Path(args.names).read_text().splitlines() if line.strip()]
        if len(names) != len(set(names)) or not names or set(names) - set(dataset.names):
            raise ValueError('--names must contain unique valid image stems')
    if args.samples > len(names):
        raise ValueError('--samples is the number of distinct images; reduce it or extend the calibration split')
    chosen = random.Random(args.seed).sample(names, args.samples)
    annotations = {}
    if args.ann:
        from evaluate import load_annotations
        annotations = load_annotations(args.ann)
        if any(name not in annotations for name in chosen):
            raise ValueError('--ann must cover all selected calibration images, in HQ coordinates')
    vae = AutoencoderKLQwenImage2D.from_pretrained(cfg['vae_path']).to(device).eval()
    venc = load_dino(cfg['dino'], device)
    entries, labels, source_hashes = [], [], {}
    precision = args.precision
    for index, name in enumerate(chosen):
        batch = dataset.sample(name, random.Random(args.seed*1_000_003 + index), return_metadata=True)
        origin = batch['crop_origin']
        lq = batch['lq'][None].to(device) * 2 - 1
        latent, mean, std = encode_dispatch(vae, lq, vae_tiling(cfg)[1], device)
        with amp(device, precision):
            features = dino_features(venc, lq, cfg['dino'])
        quads = translated_boxes(dataset.boxes.get(name, []), origin, size)
        masks = roi_masks(quads, size)
        items = []
        for ann in annotations.get(name, []):
            points = translated_boxes([ann['points']], origin, size, complete=True)
            if points:
                items.append(dict(transcription=ann['text'], points=points[0].tolist(), difficult=ann['difficult']))
        source_hashes[name] = dict(lq=sha256(dataset.lq[name]), hq=sha256(dataset.hq[name]))
        for seed in args.noise_seeds:
            gen = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(latent.shape, generator=gen, device=device, dtype=latent.dtype)
            key = f'{index:04d}_s{seed}'
            payload = dict(inp=torch.cat((latent, noise), 1).cpu(), noise=noise.cpu(),
                           z=[z.cpu() for z in features], masks=masks,
                           mean=mean.cpu() if torch.is_tensor(mean) else mean,
                           std=std.cpu() if torch.is_tensor(std) else std)
            path = out / f'{key}.pt'
            torch.save(payload, path)
            entries.append(dict(id=key, image=name, origin=list(origin), noise_seed=seed,
                                file=path.name, sha256=sha256(path)))
            if args.ann:
                labels.append(f'{key}.png\t' + json.dumps(items, ensure_ascii=False))
        print(f'cached {index+1}/{len(chosen)}: {name}', flush=True)
    if args.ann:
        (out / 'Label.txt').write_text('\n'.join(labels)+'\n', encoding='utf-8')
    manifest = dict(version=1, kind='real_paired_crops', config=cfg, precision=precision, crop=size,
                    sampling_seed=args.seed, noise_seeds=args.noise_seeds, entries=entries,
                    sources=source_hashes, annotations=bool(args.ann), provenance=provenance())
    manifest['signature'] = signature(manifest)
    write_json(out / 'manifest.json', manifest)


def read_cache(path, cfg):
    root = Path(path)
    meta = json.loads((root / 'manifest.json').read_text())
    original_signature = meta.get('signature')
    if signature({k: v for k, v in meta.items() if k != 'signature'}) != original_signature:
        raise ValueError('Cache manifest signature mismatch')
    for key in ('vae_path', 'dino', 'model'):
        if meta['config'][key] != cfg[key]:
            raise ValueError(f'Cache/config mismatch: {key}; regenerate the cache')
    if meta['config']['data']['resolution'] != cfg['data']['resolution']:
        raise ValueError('Cache/config reference resolution differs (changes RoPE coordinates)')
    if not meta['entries']:
        raise ValueError('Empty cache')
    return root, meta


def samples(root, meta, device):
    for entry in meta['entries']:
        path = root / entry['file']
        if sha256(path) != entry['sha256']:
            raise ValueError(f'Cache checksum mismatch: {path}')
        sample = torch.load(path, map_location='cpu', weights_only=True)
        sample['inp'] = sample['inp'].to(device)
        sample['noise'] = sample['noise'].to(device)
        sample['z'] = [z.to(device) for z in sample['z']]
        yield entry, sample


def forward(model, sample):
    inp = sample['inp']
    return model(inp, inp.new_ones(inp.shape[0]), inp.new_zeros(inp.shape[0]), sample['z'])


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    excluded = {*keys, 'sample', 'image', 'noise_seed'}
    for key, items in sorted(groups.items()):
        result = dict(zip(keys, key), observations=len(items), images=len({r['image'] for r in items}))
        for name in sorted(set().union(*(row.keys() for row in items)) - excluded):
            values = [r[name] for r in items if isinstance(r.get(name), (int, float)) and not isinstance(r[name], bool)]
            if values:
                result[name] = float(np.mean(values))
                result[name+'_std'] = float(np.std(values))
                result[name+'_count'] = len(values)
        output.append(result)
    return output


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)


@torch.no_grad()
def profile(args, cfg, device):
    root, meta = read_cache(args.cache, cfg)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Profile output must be empty to avoid stale rankings')
    if args.precision != meta['precision']:
        raise ValueError('Use the cache precision to avoid mixing probe conditions')
    model = load_teacher(cfg, device)
    kernels = kernels_from_config(cfg)
    compression = cfg['student'].get('attn_compression') or {}
    factor = compression.get('factor', 2)
    rows, handles, state = [], [], {}
    layers = args.layers if args.layers is not None else list(range(len(model.blocks)))
    if len(set(layers)) != len(layers) or any(i < 0 or i >= len(model.blocks) for i in layers):
        raise ValueError('Invalid layer selection')
    reference = model.x_embedder.img_size[0] // model.patch_size
    for layer in layers:
        def block_hook(module, inputs, layer=layer):
            x, c = inputs[:2]
            state[layer] = (module.scale_shift_table[None] + c.reshape(x.shape[0], 6, -1)).chunk(6, 1)[2]
        def attn_hook(module, inputs, kwargs, layer=layer):
            x = inputs[0]
            full = kwargs.get('rope', inputs[1] if len(inputs) > 1 else None)
            side = math.isqrt(x.shape[1])
            positions = (torch.arange(side//factor, device=x.device).float()*factor + (factor-1)/2)
            if model.rope_mode == 'interpolate':
                positions = positions * reference/side
            coarse = pooled_rope(positions, module.head_dim//2, x.dtype) if model.use_rope else None
            for row in measure_heads(module, x, state[layer], full, coarse,
                    None if kernels is None else kernels[layer], token_masks(state['sample'], side, device),
                    factor, args.head_chunk, args.query_chunk, compression.get('upsample', 'bilinear')):
                rows.append(dict(layer=layer, sample=state['entry']['id'], image=state['entry']['image'],
                                 noise_seed=state['entry']['noise_seed'], **row))
        handles.append(model.blocks[layer].register_forward_pre_hook(block_hook))
        handles.append(model.blocks[layer].attn.register_forward_pre_hook(attn_hook, with_kwargs=True))
    try:
        for entry, sample in samples(root, meta, device):
            state.update(entry=entry, sample=sample)
            with amp(device, args.precision):
                forward(model, sample)
            print(f'profiled {entry["id"]}', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    ranked = aggregate(rows, ('layer', 'head', 'mode'))
    for mode in ('pool', 'conv'):
        for rank, row in enumerate(sorted((r for r in ranked if r['mode'] == mode),
                                        key=lambda r: r.get('small_mse', r['residual_mse']), reverse=True), 1):
            row['rank'] = rank
    layer_rows = []
    for layer in layers:
        for mode in ('pool', 'conv'):
            selected = [r for r in ranked if r['layer'] == layer and r['mode'] == mode]
            if selected:
                layer_rows.append(dict(layer=layer, mode=mode,
                    sum_head_residual_mse=sum(r['residual_mse'] for r in selected),
                    sum_head_small_mse=sum(r.get('small_mse', 0) for r in selected),
                    note='screening score; not a joint layer intervention'))
    result = dict(kind='same_input_feature_screen', cache_signature=meta['signature'], precision=args.precision,
                  teacher_checkpoint=cfg['teacher_checkpoint'], architecture=dict(depth=len(model.blocks),
                  heads=model.num_heads, dim=model.hidden_size, grid=meta['crop']//8//model.patch_size),
                  teacher_hashes=getattr(model, 'sensitivity_checkpoint_hashes', None),
                  head_stats_sha256=sha256(compression['head_stats']) if compression.get('head_stats') else None,
                  locality_spearman={mode: spearman([(r['mass3x3'], r['small_mse']) for r in ranked
                                       if r['mode'] == mode and 'small_mse' in r]) for mode in ('pool', 'conv')},
                  provenance=provenance(), rows=ranked, layer_scores=layer_rows)
    write_json(out / 'ranking.json', result)
    write_csv(out / 'heads.csv', ranked)
    write_csv(out / 'layers.csv', layer_rows)
    with (out / 'per_sample.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')
    spec, _ = attn_compression_spec(compression, len(model.blocks), model.num_heads)
    modes = head_modes(spec, len(model.blocks), model.num_heads)
    eligible = sorted((r for r in ranked if r['mode'] == modes[r['layer']][r['head']]
                       and 'small_mse' in r), key=lambda r: r['small_mse'], reverse=True)[:args.shortlist]
    for mode in ('pool', 'conv', 'dense'):
        groups = [dict(name=f'l{r["layer"]:02d}_h{r["head"]:02d}_{mode}', mode=mode,
                       heads=[[r['layer'], r['head']]]) for r in eligible]
        write_json(out / f'shortlist_{mode}.json', groups)
    if eligible:
        write_json(out / 'shortlist_joint.json', [dict(name='shortlist_joint', mode='dense',
                   heads=[[r['layer'], r['head']] for r in eligible])])


def make_groups(args, depth, heads):
    if args.groups:
        groups = json.loads(Path(args.groups).read_text())
    else:
        layers = args.layers if args.layers is not None else list(range(depth))
        chosen = args.heads if args.heads is not None else list(range(heads))
        groups = []
        for layer in layers:
            chunks = [[h] for h in chosen] if args.unit == 'head' else [chosen]
            for chunk in chunks:
                name = f'l{layer:02d}' + (f'_h{chunk[0]:02d}' if args.unit == 'head' else '')
                groups.append(dict(name=name, mode=args.mode, heads=[[layer, h] for h in chunk]))
    names = set()
    for group in groups:
        name = group['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name) \
                or name in names or name in ('baseline', 'teacher') or not group['heads']:
            raise ValueError('Groups need unique safe names (excluding baseline/teacher) and nonempty head lists')
        names.add(name)
        group['assignments'] = [[int(l), int(h), group.get('mode', args.mode)] for l, h in group['heads']]
        assign_heads(None, group['assignments'], depth, heads)  # validate before any expensive forward
    if not groups:
        raise ValueError('No groups selected')
    return groups


def velocity_metrics(value, reference, sample):
    delta = (value.float() - reference.float()).flatten(2).transpose(1, 2)
    result = {'kd_mse': weighted_mse(delta)}
    for name, mask in token_masks(sample, value.shape[-1], value.device).items():
        result[name+'_kd_mse'] = weighted_mse(delta, mask)
    return result


@torch.no_grad()
def ablate(args, cfg, device):
    root, meta = read_cache(args.cache, cfg)
    if args.precision != meta['precision']:
        raise ValueError('Use the cache precision to avoid mixing probe conditions')
    if args.ocr and not args.decode:
        raise ValueError('--ocr requires --decode')
    if args.ocr and not meta['annotations']:
        raise ValueError('OCR needs a cache prepared with --ann; boxes.json has no transcriptions')
    teacher = load_teacher(cfg, device)
    model = load_export(args.export, device) if args.export else teacher
    if args.export:
        pipeline = json.loads((Path(args.export) / 'pipeline.json').read_text())
        if pipeline.get('noise_input', 'gaussian') != 'gaussian' or pipeline['dino'] != cfg['dino'] \
                or model.token_compression is not None or not model.use_cross_attention:
            raise ValueError('Release rescue requires the same Gaussian input, DINO and ordinary VOSR2 backbone')
        if model.hidden_size != teacher.hidden_size or len(model.blocks) != len(teacher.blocks):
            raise ValueError('Teacher/release architectures do not match')
        if model.attn_compression is None:
            raise ValueError('--export must be a head-split export for rescue experiments')
        if model.attn_compression['kernel'] != (cfg['student']['attn_compression'].get('kernel', 7)):
            raise ValueError('Config kernel differs from release')
        if model.export_config.get('attn_type', 'full') != 'full' or model.ca_scale != 1.:
            raise ValueError('Rescue experiments require full self-attention and CA scale 1')
    groups = make_groups(args, len(model.blocks), model.num_heads)
    kernels = kernels_from_config(cfg)
    compression = cfg['student'].get('attn_compression') or {}
    if any(a[2] == 'conv' for g in groups for a in g['assignments']) and kernels is None:
        raise ValueError('Conv interventions require calibrated head_stats')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Ablation output must be empty to prevent stale OCR predictions')
    write_json(out / 'groups.json', groups)
    vae = None
    if args.decode:
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        from tiled_vae import decode_dispatch
        vae = AutoencoderKLQwenImage2D.from_pretrained(cfg['vae_path']).to(device).eval()
    def save_image(value, sample, name, key):
        if vae is None:
            return
        output = decode_dispatch(vae, sample['noise'] - value.float(), vae_tiling(cfg)[1],
                    sample['mean'].to(device) if torch.is_tensor(sample['mean']) else sample['mean'],
                    sample['std'].to(device) if torch.is_tensor(sample['std']) else sample['std'])
        pixels = ((output[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        dest = out / name / 'images'
        dest.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(dest / f'{key}.png')
    rows = []
    with (out / 'per_sample.jsonl').open('w', encoding='utf-8') as log:
        for entry, sample in samples(root, meta, device):
            with amp(device, args.precision):
                target = forward(teacher, sample)
                baseline = forward(model, sample) if args.export else target
            base = velocity_metrics(baseline, target, sample)
            save_image(target, sample, 'teacher', entry['id'])
            save_image(baseline, sample, 'baseline', entry['id'])
            for group in groups:
                with intervention(model, group['assignments'], compression.get('factor', 2),
                                  compression.get('kernel', 7), kernels), amp(device, args.precision):
                    value = forward(model, sample)
                metrics = velocity_metrics(value, target, sample)
                row = dict(group=group['name'], sample=entry['id'], image=entry['image'], noise_seed=entry['noise_seed'],
                           baseline_kd_mse=base['kd_mse'], output_delta_mse=velocity_metrics(value, baseline, sample)['kd_mse'],
                           **metrics)
                for key, score in metrics.items():
                    row['delta_'+key] = None if score is None else score - base[key]
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                save_image(value, sample, group['name'], entry['id'])
            print(f'ablated {entry["id"]} ({len(groups)} interventions)', flush=True)
    summary = aggregate(rows, ('group',))
    reverse = not bool(args.export)  # dense drop: largest harm; release rescue: most negative delta
    summary.sort(key=lambda r: r.get('delta_small_kd_mse', r['delta_kd_mse']), reverse=reverse)
    for rank, row in enumerate(summary, 1):
        row['rank'] = rank
    write_json(out / 'ranking.json', dict(kind='release_rescue' if args.export else 'dense_drop',
               cache_signature=meta['signature'], precision=args.precision, export=args.export,
               teacher_checkpoint=cfg['teacher_checkpoint'],
               teacher_hashes=getattr(teacher, 'sensitivity_checkpoint_hashes', None),
               export_hashes=checkpoint_fingerprint(args.export) if args.export else None,
               groups=groups, rows=summary, provenance=provenance()))
    write_csv(out / 'ranking.csv', summary)
    if args.ocr:
        from evaluate import main as evaluate_main
        from tools.head_ocr_compare import compare
        for name in ['teacher', 'baseline'] + [g['name'] for g in groups]:
            evaluate_main(['--pred', str(out/name/'images'), '--ann', str(root/'Label.txt'), '--ann-ref', 'pred',
                           '--size-buckets', '32', '48', '64', '96', '--ocr-model', args.ocr_model,
                           '--ocr-device', args.ocr_device, '--output', str(out/name/'eval')])
        comparisons = []
        for group in groups:
            comparison = compare(out/'baseline/eval/ocr_details.jsonl',
                                 out/group['name']/'eval/ocr_details.jsonl', args.bootstrap, args.seed,
                                 {e['id']+'.png': e['image'] for e in meta['entries']})
            comparisons.append(dict(group=group['name'], **comparison))
        comparisons.sort(key=lambda x: x['bins']['<64']['delta']['CER']
                         if x['bins']['<64']['delta'] is not None else (float('-inf') if reverse else float('inf')),
                         reverse=reverse)
        for rank, comparison in enumerate(comparisons, 1):
            comparison['small_cer_rank'] = rank if comparison['bins']['<64']['delta'] is not None else None
        write_json(out / 'ocr_deltas.json', comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'profile', 'ablate'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cuda')
        p.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
        if name != 'prepare':
            p.add_argument('--cache', required=True)
            p.add_argument('--layers', type=int, nargs='+')
    p = sub.choices['prepare']
    p.add_argument('--samples', type=int, default=64, help='Distinct images; one text-centered crop per image')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--noise-seeds', type=int, nargs='+', default=[42, 43])
    p.add_argument('--names', help='Calibration image stems, one per line; use disjoint splits for tuning/validation')
    p.add_argument('--ann', help='Optional Label.txt in HQ pixels, for downstream crop OCR; only complete boxes kept')
    p = sub.choices['profile']
    p.add_argument('--head-chunk', type=int, default=4)
    p.add_argument('--query-chunk', type=int, default=128)
    p.add_argument('--shortlist', type=int, default=24, help='Top heads by current-mode small-text feature error')
    p = sub.choices['ablate']
    p.add_argument('--unit', choices=('layer', 'head'), default='layer')
    p.add_argument('--heads', type=int, nargs='+')
    p.add_argument('--mode', choices=('pool', 'conv', 'dense'), default='pool')
    p.add_argument('--groups', help='JSON list of {name, heads: [[layer,head],...], mode}')
    p.add_argument('--export', help='Optional trained head-split export: rescue on the trained input distribution')
    p.add_argument('--decode', action='store_true')
    p.add_argument('--ocr', action='store_true')
    p.add_argument('--ocr-model', default='PP-OCRv5_server_rec')
    p.add_argument('--ocr-device', default='cpu')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'prepare' and (args.samples < 1 or not args.noise_seeds or
            len(set(args.noise_seeds)) != len(args.noise_seeds)):
        parser.error('Positive samples and unique noise seeds are required')
    if args.command == 'profile' and min(args.head_chunk, args.query_chunk, args.shortlist) < 1:
        parser.error('Chunk sizes must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable: real experiments were NOT run; use a GPU host or explicit --device cpu')
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_config(args.config)
    globals()[args.command](args, cfg, device)


if __name__ == '__main__':
    main()
