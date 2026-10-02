"""Fix synthetic validation pairs from held-out original images, never pseudo GT."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data_pipeline.augmentation import degrade, sample_seed
from data_pipeline.pipeline import atomic_json, file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ['lq', 'gt', 'bicubic']:
        (out / name).mkdir(exist_ok=True)
    degradation = dict(scales=[4], blur_sigma=[0.2, 1.2], noise_std=[0, 5], jpeg_quality=[65, 95])
    records, labels = [], []
    for line in Path(args.manifest).read_text().splitlines():
        row = json.loads(line)
        if row['split'] != 'validation' or row['crop_mode'] != 'text' or row['source'] != 'anyword3m':
            continue
        if row['target'] != 'original' or row['pseudo_ground_truth']:
            raise ValueError('Validation targets must be original held-out pixels')
        annotations = []
        for a in row['annotations']:
            if not (a.get('trusted') and a.get('text') and a['text'] not in ('###', '*')):
                continue
            points = a['polygon']
            if len(points) < 4:
                points = cv2.boxPoints(cv2.minAreaRect(np.asarray(points, np.float32))).tolist()
            annotations.append(dict(transcription=a['text'], points=points, difficult=False))
        if not annotations:
            continue
        with Image.open(row['hr_path']) as im:
            gt = im.convert('RGB')
        seed = sample_seed(args.seed, row['id'], 'validation_degradation')
        lq, _, degradation_record = degrade(gt, [], degradation, seed)
        key = row['id']
        gt.save(out / 'gt' / f'{key}.png')
        lq.save(out / 'lq' / f'{key}.png')
        lq.resize(gt.size, Image.Resampling.BICUBIC).save(out / 'bicubic' / f'{key}.png')
        labels.append(f'{key}.png\t{json.dumps(annotations, ensure_ascii=False)}\n')
        records.append(dict(id=key, group_id=row['group_id'], scale=4, split='validation',
                            lq_path=str(out / 'lq' / f'{key}.png'), gt_path=str(out / 'gt' / f'{key}.png'),
                            gt_size=list(gt.size), annotations=annotations, degradation=degradation_record,
                            source_sha256=row['source_sha256']))
    (out / 'pairs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    (out / 'Label.txt').write_text(''.join(labels), encoding='utf-8')
    protocol = dict(images=len(records), groups=len({r['group_id'] for r in records}), seed=args.seed,
                    manifest_sha256=file_digest(args.manifest), source='AnyWord-ArT held-out original crops',
                    degradation=degradation, selection_metric='1-NED; OCR-A/CER and image quality also reported')
    atomic_json(out / 'protocol.json', protocol)
    print(json.dumps(protocol), flush=True)


if __name__ == '__main__':
    main()
