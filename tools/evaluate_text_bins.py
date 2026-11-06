"""Paired OCR evaluation by native SR text height, without loading an OCR model.

Consumes evaluate.py's ocr_details.jsonl.  Student and teacher must have been
recognized with the same frozen recognizer, normalization and annotations.
Bootstrap units are whole images, never individual (correlated) text boxes.
Repeated inference seeds are averaged inside each image before resampling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import unicodedata
from collections import defaultdict
from pathlib import Path


DEFAULT_HEIGHT_EDGES = (24, 32, 48, 64, 96)
OCR_SETTINGS = ('ocr_model', 'ocr_cls_model', 'ignore_case', 'include_difficult')
GEOMETRY_RTOL = 1e-6
GEOMETRY_ATOL = 1e-4


def normalized_label(record):
    value = record.get('norm_label')
    if value is None:
        value = ''.join(unicodedata.normalize('NFKC', record['label']).split())
    if not isinstance(value, str) or not value:
        raise ValueError('OCR details must contain a nonempty normalized label')
    return value


def image_key(name, strip_suffixes=()):
    path = Path(str(name).replace('\\', '/'))
    key = path.stem if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'} else path.name
    for suffix in sorted(strip_suffixes, key=len, reverse=True):
        if suffix and key.endswith(suffix) and key != suffix:
            return key[:-len(suffix)]
    return key


def record_geometry(item):
    """Validate optional native-SR geometry; missing legacy fields remain unverified."""
    quad, sr_size = None, None
    if 'quad' in item:
        value = item['quad']
        if not isinstance(value, list) or len(value) != 4 or any(
                not isinstance(point, list) or len(point) != 2 for point in value):
            raise ValueError('quad must contain four [x,y] points in native SR pixels')
        if any(isinstance(coordinate, bool) or not isinstance(coordinate, (int, float))
               or not math.isfinite(coordinate) for point in value for coordinate in point):
            raise ValueError('quad coordinates must be finite numeric values')
        quad = tuple(tuple(float(coordinate) for coordinate in point) for point in value)
        area2 = sum(quad[i][0] * quad[(i + 1) % 4][1] - quad[(i + 1) % 4][0] * quad[i][1]
                    for i in range(4))
        if not math.isfinite(area2) or abs(area2) <= 1e-8:
            raise ValueError('quad must have finite nonzero area')
    if 'sr_size' in item:
        value = item['sr_size']
        if not isinstance(value, list) or len(value) != 2 or any(
                isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size)
                or size <= 0 or not float(size).is_integer() for size in value):
            raise ValueError('sr_size must contain positive integer pixel dimensions [width,height]')
        sr_size = tuple(int(size) for size in value)
    return quad, sr_size


def load_details(path, strip_suffixes=()):
    path = Path(path)
    if path.is_dir():
        path = path / 'ocr_details.jsonl'
    if not path.is_file():
        raise FileNotFoundError(path)
    settings = {}
    summary_path = path.parent / 'summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        settings = {key: summary.get('settings', {})[key] for key in OCR_SETTINGS
                    if key in summary.get('settings', {})}
    records, names = {}, {}
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            name = item['image']
            stem = image_key(name, strip_suffixes)
            if stem in names and names[stem] != name:
                raise ValueError(f'ambiguous image stem {stem!r}')
            names[stem] = name
            index = item['index']
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError('index must be a nonnegative integer')
            key = (stem, index)
            if key in records:
                raise ValueError(f'duplicate image/index {key}')
            size = float(item['size'])
            edits = item['pred']['edit_distance']
            if not math.isfinite(size) or size <= 0:
                raise ValueError('size must be positive and finite, in native SR pixels')
            if isinstance(edits, bool) or not isinstance(edits, int) or edits < 0:
                raise ValueError('pred.edit_distance must be a nonnegative integer')
            label = normalized_label(item)
            prediction = item['pred'].get('norm_text')
            if prediction is None and isinstance(item['pred'].get('text'), str):
                prediction = ''.join(unicodedata.normalize('NFKC', item['pred']['text']).split())
                if settings.get('ignore_case', False):
                    prediction = prediction.lower()
            if prediction is not None and not isinstance(prediction, str):
                raise ValueError('pred.norm_text must be a string when present')
            ned = 1 - edits / max(len(prediction), len(label)) if prediction is not None else None
            if ned is not None and ned < 0:
                raise ValueError('edit_distance exceeds both normalized string lengths')
            quad, sr_size = record_geometry(item)
            records[key] = {'label': label, 'size': size, 'edits': edits, 'one_ned': ned,
                            'quad': quad, 'sr_size': sr_size}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'{path}:{line_number}: {error}') from error
    if not records:
        raise ValueError(f'{path}: no scored text instances')
    return records, {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'num_instances': len(records), 'ocr_settings': settings}


def align_runs(student_paths, teacher_paths, strip_suffixes=(), allow_intersection=False):
    if not student_paths or not teacher_paths:
        raise ValueError('at least one student and teacher details file is required')
    if len(teacher_paths) not in (1, len(student_paths)):
        raise ValueError('supply one teacher run, or one teacher run per student inference seed')
    students, teachers, provenance = [], [], {'student': [], 'teacher': []}
    for label, paths, destination in (('student', student_paths, students), ('teacher', teacher_paths, teachers)):
        for path in paths:
            records, meta = load_details(path, strip_suffixes)
            destination.append(records)
            provenance[label].append(meta)
    metas = provenance['student'] + provenance['teacher']
    for setting in OCR_SETTINGS:
        supplied = {json.dumps(meta['ocr_settings'][setting], sort_keys=True) for meta in metas
                    if setting in meta['ocr_settings']}
        if len(supplied) > 1:
            raise ValueError(f'OCR setting {setting} differs across runs; use identical recognizer settings')
    runs = students + teachers
    common = set(runs[0]).intersection(*(set(run) for run in runs[1:]))
    union = set().union(*(set(run) for run in runs))
    if common != union and not allow_intersection:
        raise ValueError(f'OCR instance keys differ across runs ({len(common)} shared / {len(union)} total); '
                         'fix missing predictions or explicitly use --allow-intersection')
    if not common:
        raise ValueError('no shared image/index pairs')
    if len(teachers) == 1:
        teachers *= len(students)
    paired = []
    unverified_geometry = 0
    for key in sorted(common):
        reference = runs[0][key]
        for run in runs[1:]:
            item = run[key]
            if item['label'] != reference['label']:
                raise ValueError(f'{key}: normalized labels differ')
            if not math.isclose(item['size'], reference['size'], rel_tol=1e-6, abs_tol=1e-4):
                raise ValueError(f'{key}: SR text heights differ; annotations/output scales must agree')
        # Compare every supplied geometry, even when the first run is a legacy
        # file without geometry. A missing field is not evidence of equality.
        quads = [run[key]['quad'] for run in runs if run[key]['quad'] is not None]
        sizes = [run[key]['sr_size'] for run in runs if run[key]['sr_size'] is not None]
        if any(any(not math.isclose(x, y, rel_tol=GEOMETRY_RTOL, abs_tol=GEOMETRY_ATOL)
                   for a, b in zip(quads[0], quad) for x, y in zip(a, b)) for quad in quads[1:]):
            raise ValueError(f'{key}: native SR quads differ; image/index/label/height do not identify the same crop')
        if any(size != sizes[0] for size in sizes[1:]):
            raise ValueError(f'{key}: SR output sizes differ; geometry uses different pixel frames')
        geometry_verified = len(quads) == len(runs) and len(sizes) == len(runs)
        unverified_geometry += not geometry_verified
        count = len(students)
        student_ned = [run[key]['one_ned'] for run in students]
        teacher_ned = [run[key]['one_ned'] for run in teachers]
        paired.append({
            'image': key[0], 'index': key[1], 'size': reference['size'], 'chars': len(reference['label']),
            'geometry_verified': geometry_verified,
            'student_edits': sum(run[key]['edits'] for run in students) / count,
            'teacher_edits': sum(run[key]['edits'] for run in teachers) / count,
            'student_correct': sum(run[key]['edits'] == 0 for run in students) / count,
            'teacher_correct': sum(run[key]['edits'] == 0 for run in teachers) / count,
            'teacher_correct_retained': sum(s[key]['edits'] == 0 and t[key]['edits'] == 0
                                            for s, t in zip(students, teachers)) / count,
            'student_ned_sum': sum(student_ned) / count if all(value is not None for value in student_ned) else 0.,
            'teacher_ned_sum': sum(teacher_ned) / count if all(value is not None for value in teacher_ned) else 0.,
            'student_ned_instances': int(all(value is not None for value in student_ned)),
            'teacher_ned_instances': int(all(value is not None for value in teacher_ned)),
        })
    provenance['alignment'] = {
        'strict': not allow_intersection, 'shared_instances': len(common), 'union_instances': len(union),
        'strict_scope': 'instance keys, labels and heights; geometric verification is reported separately',
        'geometry_verified': unverified_geometry == 0,
        'geometry_verified_instances': len(common) - unverified_geometry,
        'geometry_unverified_instances': unverified_geometry,
        'geometry_missing_instances_per_file': [sum(run[key]['quad'] is None or run[key]['sr_size'] is None
                                                    for key in common) for run in runs],
        'geometry_tolerance': {'quad_rtol': GEOMETRY_RTOL, 'quad_atol_px': GEOMETRY_ATOL,
                               'sr_size': 'exact integer pixel dimensions'},
        'excluded_instances_per_file': [len(run) - len(common) for run in runs],
        'student_inference_seeds': len(students), 'teacher_inference_seeds': len(teacher_paths),
    }
    return paired, provenance


def metrics(totals):
    chars, count = totals['chars'], totals['num_instances']
    if not count:
        return {name: None for name in ('student_cer', 'teacher_cer', 'cer_delta', 'student_ocr_a',
                                        'teacher_ocr_a', 'ocr_a_delta', 'teacher_correct_retention',
                                        'student_1_ned', 'teacher_1_ned', 'one_ned_delta')}
    s_cer, t_cer = totals['student_edits'] / chars, totals['teacher_edits'] / chars
    s_acc, t_acc = totals['student_correct'] / count, totals['teacher_correct'] / count
    s_ned = totals['student_ned_sum'] / count if totals['student_ned_instances'] == count else None
    t_ned = totals['teacher_ned_sum'] / count if totals['teacher_ned_instances'] == count else None
    return {'student_cer': s_cer, 'teacher_cer': t_cer, 'cer_delta': s_cer - t_cer,
            'student_ocr_a': s_acc, 'teacher_ocr_a': t_acc, 'ocr_a_delta': s_acc - t_acc,
            'student_1_ned': s_ned, 'teacher_1_ned': t_ned,
            'one_ned_delta': s_ned - t_ned if s_ned is not None and t_ned is not None else None,
            'teacher_correct_retention': (totals['teacher_correct_retained'] / totals['teacher_correct']
                                          if totals['teacher_correct'] else None)}


TOTAL_KEYS = ('chars', 'num_instances', 'student_edits', 'teacher_edits', 'student_correct',
              'teacher_correct', 'teacher_correct_retained', 'student_ned_sum', 'teacher_ned_sum',
              'student_ned_instances', 'teacher_ned_instances')


def sum_images(image_totals, selected):
    result = dict.fromkeys(TOTAL_KEYS, 0.0)
    for name in selected:
        for key in TOTAL_KEYS:
            result[key] += image_totals[name][key]
    return result


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize_bin(records, low, high, bootstrap=2000, seed=42, confidence=.95, min_images=10):
    images = {}
    for row in records:
        if low <= row['size'] < high:
            target = images.setdefault(row['image'], dict.fromkeys(TOTAL_KEYS, 0.0))
            for key in TOTAL_KEYS:
                target[key] += 1 if key == 'num_instances' else row[key]
    names = sorted(images)
    totals = sum_images(images, names)
    estimate = metrics(totals)
    result = {'height_px': [low, None if math.isinf(high) else high], 'num_images': len(names),
              'num_instances': int(totals['num_instances']), 'num_label_chars': int(totals['chars']),
              'teacher_correct_instances_seed_mean': totals['teacher_correct'], **estimate,
              'confidence_interval': {key: None for key in estimate}, 'bootstrap_valid_samples': {}}
    if not names:
        result['interval_status'] = 'empty_bin'
    elif bootstrap == 0:
        result['interval_status'] = 'disabled'
    elif len(names) < min_images:
        result['interval_status'] = f'insufficient_images:{len(names)}<{min_images}'
    else:
        rng = random.Random(seed)
        samples = defaultdict(list)
        for _ in range(bootstrap):
            draw = [names[rng.randrange(len(names))] for _ in names]
            for key, value in metrics(sum_images(images, draw)).items():
                if value is not None and estimate[key] is not None:
                    samples[key].append(value)
        tail = (1 - confidence) / 2
        result['confidence_interval'] = {
            key: ([percentile(samples[key], tail), percentile(samples[key], 1 - tail)]
                  if samples[key] else None) for key in estimate}
        result['bootstrap_valid_samples'] = {key: len(samples[key]) for key in estimate}
        result['interval_status'] = 'paired_image_percentile'
    return result


def evaluate(student_paths, teacher_paths, height_edges=DEFAULT_HEIGHT_EDGES, bootstrap=2000, seed=42,
             confidence=.95, min_images=10, strip_suffixes=(), allow_intersection=False):
    edges = list(map(float, height_edges))
    if not edges or any(not math.isfinite(x) or x <= 0 for x in edges) or edges != sorted(set(edges)):
        raise ValueError('height edges must be finite, positive and strictly increasing')
    if bootstrap < 0 or min_images < 2 or not 0 < confidence < 1:
        raise ValueError('bootstrap >= 0, min_images >= 2, and confidence in (0, 1) are required')
    rows, provenance = align_runs(student_paths, teacher_paths, strip_suffixes, allow_intersection)
    bounds = [0., *edges, float('inf')]
    bins = [summarize_bin(rows, a, b, bootstrap, seed + i, confidence, min_images)
            for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
    return {
        'schema_version': 1,
        'definition': {'height': 'recognition line height in native SR output pixels; [lo, hi)',
                       'cer': 'sum(edit_distance) / sum(normalized label characters)',
                       'delta': 'student minus teacher; positive CER delta is worse',
                       'teacher_correct_retention': 'P(student exact | teacher exact), paired by inference seed',
                       '1-NED': 'mean(1 - edits/max(predicted chars,label chars)); null if prediction text is missing',
                       'bootstrap_unit': 'image; all boxes and paired predictions stay together',
                       'seed_uncertainty': 'inference seeds averaged within image; CI does not model training-seed variance'},
        'settings': {'height_edges': edges, 'bootstrap': bootstrap, 'bootstrap_seed': seed,
                     'confidence': confidence, 'min_images': min_images},
        'inputs': provenance,
        'all': summarize_bin(rows, 0, float('inf'), bootstrap, seed + len(bins), confidence, min_images),
        'height_bins': bins,
        'small_under_64': summarize_bin(rows, 0, 64, bootstrap, seed + len(bins) + 1, confidence, min_images),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-details', nargs='+', required=True, help='JSONL file(s) or evaluate.py directories')
    parser.add_argument('--teacher-details', nargs='+', required=True)
    parser.add_argument('--output', required=True, help='New output directory; never overwrite existing reports')
    parser.add_argument('--height-buckets', nargs='+', type=float, default=list(DEFAULT_HEIGHT_EDGES))
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--min-images', type=int, default=10)
    parser.add_argument('--strip-suffix', nargs='+', default=[])
    parser.add_argument('--allow-intersection', action='store_true', help='Explicitly allow missing boxes; report all exclusions')
    args = parser.parse_args(argv)
    output = Path(args.output)
    targets = [output / 'text_bins.json', output / 'text_bins.csv']
    if any(path.exists() for path in targets):
        raise FileExistsError(f'{output}: report already exists; use a new evaluation output directory')
    report = evaluate(args.student_details, args.teacher_details, args.height_buckets, args.bootstrap, args.seed,
                      args.confidence, args.min_images, args.strip_suffix, args.allow_intersection)
    output.mkdir(parents=True, exist_ok=True)
    rows = [{'bin': 'all', **report['all']}, {'bin': 'small_under_64', **report['small_under_64']}]
    rows += [{'bin': f'{row["height_px"][0]:g}-{row["height_px"][1] or "inf"}', **row}
             for row in report['height_bins']]
    fields = ['bin', 'num_images', 'num_instances', 'num_label_chars', 'student_cer', 'teacher_cer', 'cer_delta',
              'student_ocr_a', 'teacher_ocr_a', 'student_1_ned', 'teacher_1_ned', 'one_ned_delta',
              'teacher_correct_retention', 'interval_status']
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    # Exclusive creation also protects two concurrent evaluators from silently replacing results.
    with targets[0].open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')
    with targets[1].open('x', encoding='utf-8', newline='') as file:
        file.write(buffer.getvalue())
    print(json.dumps({'output': str(output.resolve()), 'all': report['all'],
                      'small_under_64': report['small_under_64']}, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
