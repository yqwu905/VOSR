"""Validate complete evaluations and compare them by paired source-group bootstrap.

Image metrics are means over images. OCR-A and 1-NED are means over text
instances; CER is total edits / total reference characters. Resampling original
source groups keeps overlapping crops together. These intervals describe the
evaluated scenes at one trained seed, not training-seed uncertainty.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data_pipeline.pipeline import atomic_json, file_digest
from evaluate import FR_METRICS, normalize_text, edit_distance


METRICS = (*FR_METRICS, 'OCR-A', 'CER', '1-NED')
LOWER = {'LPIPS', 'DISTS', 'CER'}
COMPARABLE_SETTINGS = ('rgb', 'crop_border', 'ignore_case', 'include_difficult',
                       'ocr_model', 'ocr_cls_model', 'ocr_batch_size', 'ann_ref')


def aggregate(values):
    """Columns: four image sums, image count, text count, matches, edits, chars, NED sum."""
    x = np.asarray(values)
    if np.any(x[..., [4, 5, 8]] <= 0):
        raise ValueError('Image, text and character denominators must be positive')
    return np.concatenate((x[..., :4] / x[..., 4:5],
                           x[..., 6:7] / x[..., 5:6],
                           x[..., 7:8] / x[..., 8:9],
                           1 - x[..., 9:10] / x[..., 5:6]), axis=-1)


def load_evaluation(directory, pairs, source='pred'):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    settings = summary['settings']
    group_map = {r['id']: r['group_id'] for r in pairs}
    if len(group_map) != len(pairs) or not pairs:
        raise ValueError('Pair manifest must contain unique images')
    group_names = sorted(set(group_map.values()))
    group_index = {g: i for i, g in enumerate(group_names)}
    totals = np.zeros((len(group_names), 10), dtype=np.float64)
    with (directory / 'per_image.csv').open() as f:
        image_rows = list(csv.DictReader(f))
    image_names = [Path(r['name']).stem for r in image_rows]
    if len(set(image_names)) != len(image_names) or set(image_names) != set(group_map):
        raise ValueError('Evaluation image coverage differs from pair manifest')
    for row, name in zip(image_rows, image_names):
        values = [float(row[m]) for m in FR_METRICS]
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite image metric: {name}')
        totals[group_index[group_map[name]], :4] += values
        totals[group_index[group_map[name]], 4] += 1
    labels, text_counts = {}, {name: 0 for name in image_names}
    for line in (directory / 'ocr_details.jsonl').read_text().splitlines():
        row = json.loads(line)
        name = Path(row['image']).stem
        if name not in group_map:
            raise ValueError('Unknown image in OCR details')
        key = (name, row['index'])
        if key in labels:
            raise ValueError('Duplicate OCR region')
        label = normalize_text(row['label'], settings.get('ignore_case', False))
        prediction = normalize_text(row[source]['text'], settings.get('ignore_case', False))
        if not label:
            raise ValueError('Empty reference label in evaluated OCR regions')
        distance = edit_distance(prediction, label)
        if distance != row[source]['edit_distance']:
            raise ValueError('OCR detail edit distance disagrees with raw text')
        labels[key] = label
        text_counts[name] += 1
        totals[group_index[group_map[name]], 5:] += [1, int(prediction == label), distance,
                                                     len(label), distance/max(len(label), len(prediction))]
    for row, name in zip(image_rows, image_names):
        if text_counts[name] != int(row['num_text']):
            raise ValueError('OCR region count differs between details and per-image rows')
    estimate = aggregate(totals.sum(axis=0))
    reported = np.array([summary['fr_iqa'][m] for m in FR_METRICS] +
                        [summary['ocr'][source][m] for m in METRICS[4:]])
    if not np.allclose(estimate, reported, rtol=0, atol=1e-7):
        raise ValueError('Summary metrics do not match independently aggregated details')
    if summary['num_images'] != len(pairs) or summary['ocr']['num_instances'] != len(labels):
        raise ValueError('Summary image/text counts disagree with artifacts')
    return dict(directory=str(directory.resolve()), totals=totals, estimate=estimate,
                groups=group_names, labels=labels, settings=settings,
                artifact_sha256={n: file_digest(directory/n) for n in
                                 ['summary.json', 'per_image.csv', 'ocr_details.jsonl']})


def paired_comparison(baseline, candidate, samples=5000, seed=2026):
    if candidate['groups'] != baseline['groups'] or candidate['labels'] != baseline['labels']:
        raise ValueError('Paired comparison requires identical source groups and OCR labels')
    for setting in COMPARABLE_SETTINGS:
        if candidate['settings'].get(setting) != baseline['settings'].get(setting):
            raise ValueError(f'Incompatible evaluation setting: {setting}')
    if samples < 100:
        raise ValueError('At least 100 bootstrap samples are required')
    n = len(baseline['groups'])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1/n), size=samples)
    reference = aggregate(counts @ baseline['totals'])
    current = aggregate(counts @ candidate['totals'])
    differences = current-reference
    result = {}
    for i, metric in enumerate(METRICS):
        interval = np.quantile(differences[:, i], [0.025, 0.975]).tolist()
        result[metric] = dict(baseline=float(baseline['estimate'][i]),
                              candidate=float(candidate['estimate'][i]),
                              delta=float(candidate['estimate'][i]-baseline['estimate'][i]),
                              delta_ci95=interval,
                              direction='lower' if metric in LOWER else 'higher',
                              improvement_interval_excludes_zero=interval[1] < 0 if metric in LOWER else interval[0] > 0)
    return dict(groups=n, text_instances=len(candidate['labels']), bootstrap_samples=samples,
                bootstrap_seed=seed, metrics=result,
                candidate_directory=candidate['directory'], candidate_artifact_sha256=candidate['artifact_sha256'],
                baseline_directory=baseline['directory'], baseline_artifact_sha256=baseline['artifact_sha256'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidates', nargs='+', required=True, help='Each argument is name=/path/to/metrics')
    p.add_argument('--pairs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=5000)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    pairs = [json.loads(x) for x in Path(args.pairs).read_text().splitlines()]
    baseline = load_evaluation(args.baseline, pairs)
    results = {}
    for entry in args.candidates:
        name, separator, path = entry.partition('=')
        if not name or not separator or name in results:
            raise ValueError('Candidate names must be unique and followed by =path')
        results[name] = paired_comparison(baseline, load_evaluation(path, pairs), args.samples, args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'comparison.json', dict(pairs_sha256=file_digest(args.pairs), comparisons=results,
                 interpretation='Paired source-group percentile bootstrap; fixed trained seed; selection uncertainty is separate.'))
    with (out/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'metric', 'baseline', 'candidate', 'delta', 'ci95_low', 'ci95_high'])
        writer.writeheader()
        for name, result in results.items():
            for metric, row in result['metrics'].items():
                writer.writerow(dict(model=name, metric=metric, baseline=row['baseline'], candidate=row['candidate'],
                                     delta=row['delta'], ci95_low=row['delta_ci95'][0], ci95_high=row['delta_ci95'][1]))
    print(json.dumps({n: {m: r['candidate'] for m, r in d['metrics'].items()} for n, d in results.items()}, indent=2))


if __name__ == '__main__':
    main()
