"""Strict paired OCR deltas and source-image bootstrap for evaluate.py outputs.

Multiple noise seeds are repeated measurements of the same image, not independent
bootstrap units. Positive delta CER / negative delta OCR-A and 1-NED are worse.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import normalize_text

BINS = {'all': (0, float('inf')), '<32': (0, 32), '32-48': (32, 48),
        '48-64': (48, 64), '<64': (0, 64), '64-96': (64, 96), '>=96': (96, float('inf'))}


def load(paths):
    paths = paths if isinstance(paths, (list, tuple)) else [paths]
    rows = {}
    for run, path in enumerate(paths):
        path = Path(path)
        if path.is_dir():
            path = path / 'ocr_details.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = (run, row['image'], row['index'])
            label = row.get('norm_label', normalize_text(row['label']))
            if not label:
                continue
            if key in rows:
                raise ValueError(f'Duplicate OCR row: {key}')
            row['norm_label'] = label
            rows[key] = row
    if not rows:
        raise ValueError('No valid OCR rows')
    return rows


def stats(row):
    label, prediction = row['norm_label'], normalize_text(row['pred']['text'])
    edits = row['pred']['edit_distance']
    return np.array([edits == 0, edits, len(label), edits / max(len(label), len(prediction)), 1], dtype=float)


def scores(counts):
    return np.array([counts[0] / counts[4], counts[1] / counts[2], 1-counts[3] / counts[4]])


def compare(reference, candidate, bootstrap=2000, seed=0, source_images=None):
    if bootstrap < 1:
        raise ValueError('bootstrap must be positive')
    ref, arm = load(reference), load(candidate)
    if ref.keys() != arm.keys():
        raise ValueError('Unpaired OCR inputs: missing/extra (seed, image, box) rows; refusing partial intersection')
    for key in ref:
        if ref[key]['norm_label'] != arm[key]['norm_label'] or not np.isclose(ref[key]['size'], arm[key]['size']):
            raise ValueError(f'OCR label/geometry mismatch for {key}')
        if source_images is not None and ref[key]['image'] not in source_images:
            raise ValueError(f'No source-image grouping for {ref[key]["image"]}')
    rng, result = np.random.default_rng(seed), {}
    names = ('OCR-A', 'CER', '1-NED')
    for name, (lo, hi) in BINS.items():
        grouped = defaultdict(lambda: np.zeros((2, 5)))
        for key, row in ref.items():
            if lo <= row['size'] < hi:
                image = source_images[row['image']] if source_images else row['image']
                grouped[image][0] += stats(row)
                grouped[image][1] += stats(arm[key])
        if not grouped:
            result[name] = dict(images=0, observations=0, reference=None, candidate=None, delta=None, ci95=None)
            continue
        values = np.stack([grouped[k] for k in sorted(grouped)])
        sums = values.sum(0)
        a, b = scores(sums[0]), scores(sums[1])
        interval = None
        if len(grouped) >= 10:
            boot = []
            for _ in range(bootstrap):
                sample = values[rng.integers(len(values), size=len(values))].sum(0)
                boot.append(scores(sample[1])-scores(sample[0]))
            low, high = np.quantile(boot, [.025, .975], axis=0)
            interval = {k: [float(l), float(h)] for k, l, h in zip(names, low, high)}
        result[name] = dict(images=len(grouped), observations=int(sums[0, 4]),
                            reference=dict(zip(names, a)), candidate=dict(zip(names, b)),
                            delta=dict(zip(names, b-a)), ci95=interval)
    return dict(sign='candidate minus reference', bootstrap_unit='source image (all boxes and seeds together)',
                bootstrap=bootstrap, seed=seed, bins=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', nargs='+', required=True, help='Eval directories in noise-seed order')
    p.add_argument('--candidate', nargs='+', required=True, help='Same noise-seed order as reference')
    p.add_argument('--output', required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if len(args.reference) != len(args.candidate):
        p.error('Reference/candidate seed counts differ')
    result = compare(args.reference, args.candidate, args.bootstrap, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['bins']['<64'], indent=2))


if __name__ == '__main__':
    main()
