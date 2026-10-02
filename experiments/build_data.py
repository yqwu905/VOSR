"""Build explicit research ablations from the existing synthesis primitives.

Research candidates are separate from the production pipeline's accepted data.
They carry their true annotation/teacher status; no human glyph approval or
calibrated IQA approval is fabricated. Every crop shares its parent's split.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from data_pipeline.augmentation import sample_seed
from data_pipeline.geometry import (normalize_1k, regions, transform, text_crops,
                                    crop_regions, scale_matrix)
from data_pipeline.pipeline import atomic_json, file_digest, pixel_digest


def square_canvas(image, annotations, resolution):
    """Fit the whole crop without distorting or cutting characters; edge-pad."""
    ratio = resolution / max(image.size)
    width, height = [max(1, round(x * ratio)) for x in image.size]
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    left, top = (resolution - width) // 2, (resolution - height) // 2
    array = np.pad(np.asarray(resized), ((top, resolution-height-top),
                   (left, resolution-width-left), (0, 0)), mode='edge')
    matrix = scale_matrix(image.size, (width, height))
    matrix[0, 2], matrix[1, 2] = left, top
    return Image.fromarray(array), transform(annotations, matrix), matrix.tolist()


def build(manifests, output, target='original', resolution=512, long_edge=1024,
          max_crops=2, seed=42, validation_fraction=0.1):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(manifests={str(Path(p).resolve()): file_digest(p) for p in manifests},
                    target=target, resolution=resolution, long_edge=long_edge,
                    max_crops=max_crops, seed=seed, validation_fraction=validation_fraction,
                    code_sha256=file_digest(__file__))
    if target == 'vosr2':
        protocol['teacher_sha256'] = file_digest('preset/ckpts/VOSR2/checkpoints/ema_model.safetensors')
    if (output / 'protocol.json').exists():
        if json.loads((output / 'protocol.json').read_text()) != protocol:
            raise ValueError('Data construction changed; use a new directory')
    else:
        atomic_json(output / 'protocol.json', protocol)
    # Build components from both source group ids and exact pixel digests. This
    # prevents duplicate images across sources from entering different splits.
    source_rows, parent, pixel_owner = [], {}, {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for manifest in manifests:
        for line in Path(manifest).read_text().splitlines():
            row = json.loads(line)
            path = Path(row['image_path'])
            if not path.is_absolute():
                path = Path(manifest).parent / path
            with Image.open(path) as im:
                digest = pixel_digest(im.convert('RGB'))
            group = row.get('group_id', digest)
            find(group)
            if digest in pixel_owner:
                a, b = find(group), find(pixel_owner[digest])
                parent[max(a, b)] = min(a, b)
            else:
                pixel_owner[digest] = group
                source_rows.append(dict(row, image_path=str(path.resolve()), source_sha256=digest, group_id=group))
    teacher = None
    counts, rows, skipped = Counter(), [], []
    for index, row in enumerate(source_rows):
        digest = row['source_sha256']
        group = find(row['group_id'])
        split = 'validation' if sample_seed(seed, group, 'split') / 2**64 < validation_fraction else 'train'
        cache = output / 'records' / f'{digest}.json'
        if cache.exists():
            cached = json.loads(cache.read_text())
            if all(Path(r['hr_path']).is_file() and file_digest(r['hr_path']) == r['file_sha256'] for r in cached):
                rows.extend(cached)
                counts.update(f'{r["source"]}/{r["split"]}/{r["crop_mode"]}' for r in cached)
                continue
        with Image.open(row['image_path']) as im:
            original = im.convert('RGB')
        annotations = regions(row.get('annotations', []), original.size)
        if not annotations:
            skipped.append(dict(id=row['id'], source=row['source'], reason='no_valid_localization'))
            continue
        if target == 'vosr2':
            if teacher is None:
                from data_pipeline.vosr_teacher import VOSR2Teacher
                teacher = VOSR2Teacher('preset/ckpts/VOSR2', precision='bf16', max_output_edge=1024,
                                      overrides=dict(tile_size=512, tile_overlap=64, vae_tile_size=512))
            restored = teacher.restore(original, dict(seed=sample_seed(seed, digest, 'teacher')))
        else:
            restored = original
        full, _ = normalize_1k(restored, long_edge)
        annotations = transform(annotations, scale_matrix(original.size, full.size))
        complete = text_crops(annotations, full.size, width=resolution, height=resolution,
                              context=0.2, max_crops=max_crops)
        rng = random.Random(sample_seed(seed, digest, 'random_crop'))
        random_rects = []
        for _ in complete:
            width, height = min(resolution, full.width), min(resolution, full.height)
            x, y = rng.randint(0, full.width-width), rng.randint(0, full.height-height)
            random_rects.append((x, y, x+width, y+height))
        generated = []
        for mode, rectangles in [('text', complete), ('random', random_rects)]:
            for number, rectangle in enumerate(rectangles):
                crop, labels, matrix = square_canvas(full.crop(rectangle), crop_regions(annotations, rectangle), resolution)
                path = output / 'images' / mode / f'{digest}_{number:02}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(path)
                record = dict(id=f'{digest}_{mode}_{number:02}', source=row['source'], group_id=group,
                              source_id=row['id'], source_path=row['image_path'], source_sha256=digest,
                              split=split, crop_mode=mode, target=target, hr_path=str(path),
                              file_sha256=file_digest(path), crop_rect=list(rectangle),
                              canvas_transform=matrix, annotations=labels, dataset=row.get('dataset'),
                              pseudo_ground_truth=target != 'original', formal_glyph_verification=False,
                              use_policy='research_ablation_unreviewed', resolution=resolution)
                generated.append(record)
                counts[f'{row["source"]}/{split}/{mode}'] += 1
        cache.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(cache, generated)
        rows.extend(generated)
        if (index+1) % 50 == 0:
            print(json.dumps(dict(processed=index+1, total=len(source_rows), crops=len(rows))), flush=True)
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    atomic_json(output / 'summary.json', dict(counts=dict(counts), source_images=len(source_rows),
                                            skipped=skipped, candidates=len(rows), protocol=protocol))
    return dict(counts)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--target', choices=['original', 'vosr2'], default='original')
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--long-edge', type=int, default=1024)
    p.add_argument('--max-crops', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validation-fraction', type=float, default=0.1)
    print(json.dumps(build(**vars(p.parse_args())), indent=2))
