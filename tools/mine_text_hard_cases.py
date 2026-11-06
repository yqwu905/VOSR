"""Mine teacher-correct/student-wrong small text from an explicit TRAIN split.

The output is an annotation subset, not generated labels or copied images. New
evaluate.py details carry native-SR quads. For old details, --boxes must contain
rich line records with matching transcripts; bare indexed quads are ambiguous.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_text_bins import align_runs, image_key


def names_from_file(path):
    def stem(value):
        value = Path(value.replace('\\', '/')).name
        return Path(value).stem if Path(value).suffix.lower() in (
            '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff') else value
    values = {stem(line.split('\t', 1)[0].strip())
              for line in Path(path).read_text(encoding='utf-8').splitlines()
              if line.strip() and not line.lstrip().startswith('#')}
    if not values:
        raise ValueError(f'{path}: empty split manifest')
    return values


def normalize(text):
    return ''.join(unicodedata.normalize('NFKC', text).split())


def mine(student, teacher, train_names, exclude_names=(), boxes=None, max_height=64., min_confidence=0.,
         assume_aligned=False):
    if not math.isfinite(max_height) or max_height <= 0 or not 0 <= min_confidence <= 1:
        raise ValueError('max_height must be positive; min_confidence must be in [0, 1]')
    paired, provenance = align_runs([student], [teacher])
    train, excluded = set(train_names), set(exclude_names)
    if train & excluded:
        raise ValueError('Train and exclusion manifests overlap; fix the split before mining')
    evaluated = {item['image'] for item in paired}
    outside = evaluated - train
    if outside or evaluated & excluded:
        raise ValueError(f'Evaluation contains images outside the training split: {sorted(outside)[:8]}')
    raw_boxes = json.loads(Path(boxes).read_text(encoding='utf-8')) if boxes else {}
    rich = {}
    for name, value in raw_boxes.items():
        key = image_key(name)
        if key in rich:
            raise ValueError(f'Duplicate annotation image key: {key}')
        rich[key] = value
    path = Path(teacher)
    if path.is_dir():
        path = path / 'ocr_details.jsonl'
    rows = {(image_key(row['image']), row['index']): row
            for line in path.read_text(encoding='utf-8').splitlines() if line.strip()
            for row in [json.loads(line)]}
    selected = defaultdict(list)
    for item in paired:
        if not (item['teacher_correct'] == 1 and item['student_correct'] == 0 and item['size'] < max_height):
            continue
        row = rows[(item['image'], item['index'])]
        if float(row['pred'].get('score', 0.)) < min_confidence:
            continue
        label = row['label']
        quad = row.get('quad')
        record, image_aligned = None, None
        if item['image'] in rich:
            records = rich[item['image']]
            if isinstance(records, dict):
                image_aligned = records.get('alignment_valid')
                records = records.get('lines')
            if not isinstance(records, list) or item['index'] >= len(records):
                raise ValueError('OCR row index is absent from source annotations')
            record = records[item['index']]
        if image_aligned is False:
            continue
        if isinstance(record, dict):
            if record.get('valid', True) is False or record.get('ignore', False) or record.get('difficult', False):
                continue
            transcript = record.get('transcript', record.get('transcription', record.get('text', '')))
            if transcript and normalize(transcript) != normalize(label):
                raise ValueError('OCR row index and annotation transcript do not agree')
            if image_aligned is False or record.get('alignment_valid') is False:
                continue
        aligned = bool(assume_aligned or image_aligned is True or
                       (isinstance(record, dict) and record.get('alignment_valid') is True))
        if quad is None:
            if not isinstance(record, dict):
                raise ValueError('Legacy details need rich --boxes with matching transcripts, or rerun evaluate.py')
            transcript = record.get('transcript', record.get('transcription', record.get('text', '')))
            if not transcript or normalize(transcript) != normalize(label):
                raise ValueError('OCR row index and annotation transcript do not agree')
            quad = next((record[key] for key in ('quad', 'points', 'bbox', 'box') if key in record), None)
        from dataloaders.paired_crops import canonical_quad
        quad = canonical_quad(quad, 'hard-mining annotation').reshape(-1).tolist()
        if record is not None:
            source_quad = (next((record[key] for key in ('quad', 'points', 'bbox', 'box') if key in record), None)
                           if isinstance(record, dict) else record)
            source_quad = canonical_quad(source_quad).reshape(-1).tolist()
            if any(abs(a - b) > .01 for a, b in zip(quad, source_quad)):
                raise ValueError('OCR and training annotation coordinates differ; use the same native SR geometry')
        selected[item['image']].append(dict(quad=quad, transcript=label, teacher_correct=True,
                                           alignment_valid=aligned, mining_source_index=item['index']))
    if not selected:
        raise ValueError('No eligible hard cases; do not create an empty training source')
    report = dict(provenance, scope='training_only', criterion='teacher_correct_student_wrong',
                  max_height=max_height, min_teacher_confidence=min_confidence,
                  assume_aligned=bool(assume_aligned),
                  aligned_lines=sum(line['alignment_valid'] for lines in selected.values() for line in lines),
                  num_images=len(selected), num_lines=sum(map(len, selected.values())),
                  train_names_sha256=hashlib.sha256('\n'.join(sorted(train)).encode()).hexdigest())
    return dict(selected), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-details', required=True)
    parser.add_argument('--teacher-details', required=True)
    parser.add_argument('--train-names', required=True, help='Explicit train split: one stem/path per line')
    parser.add_argument('--exclude-names', action='append', default=[], help='Validation/test split manifest; repeatable')
    parser.add_argument('--boxes', help='Rich training boxes JSON, only needed for legacy OCR details')
    parser.add_argument('--max-height', type=float, default=64.)
    parser.add_argument('--min-confidence', type=float, default=0.)
    parser.add_argument('--assume-aligned', action='store_true',
                        help='Declare otherwise-unmarked pairs aligned; explicit invalid/unaligned records still skip')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    excluded = set().union(*(names_from_file(path) for path in args.exclude_names))
    boxes, report = mine(args.student_details, args.teacher_details, names_from_file(args.train_names),
                         excluded, args.boxes, args.max_height, args.min_confidence, args.assume_aligned)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'boxes.json').write_text(json.dumps(boxes, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'image_names.txt').write_text('\n'.join(sorted(boxes)) + '\n', encoding='utf-8')
    (output / 'mining.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
