"""Convert the official Real-CE archive to explicit paired evaluation files.

Matches get_RealCE and PairedImageDatasetRealCEwRECwLabelMapwCannyv2:
valid_list.txt, GBK test annotations, OpenCV bicubic LQ downsampling,
and top-left GT modcrop. The archive's `val` split is the official test set.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def read_labels(path, encoding):
    result = []
    for number, line in enumerate(path.read_text(encoding=encoding).splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(',', 8)
        if len(fields) != 9:
            raise ValueError(f'{path}:{number}: expected eight coordinates and text')
        points = np.array([float(x) for x in fields[:8]]).reshape(4, 2)
        if not np.isfinite(points).all():
            raise ValueError(f'{path}:{number}: invalid coordinates')
        result.append(dict(points=points.tolist(), transcription=fields[8], difficult=False))
    return result


def prepare(root, output, split='val', scale=4, limit=None, seed=42):
    root, output = Path(root).resolve(), Path(output).resolve()
    folder = root / split
    names = (folder / 'valid_list.txt').read_text().splitlines()
    if len(set(names)) != len(names):
        raise ValueError('Duplicate name in official valid_list.txt')
    if limit:
        if split == 'val':
            raise ValueError('Do not subsample the official test set')
        names = sorted(names, key=lambda n: hashlib.sha256(f'{seed}:{n}'.encode()).digest())[:limit]
    names = sorted(names)
    pairs = [('13mm', '52mm')] if scale == 4 else [('26mm', '52mm'), ('13mm', '26mm')]
    if scale not in (2, 4):
        raise ValueError('RealCE supports scales 2 and 4')
    for sub in ['lq', 'gt', 'bicubic']:
        (output / sub).mkdir(parents=True, exist_ok=True)
    records, labels = [], []
    for name in names:
        label_path = folder / ('annos' if split == 'val' else 'trans_annos_52mm_renov') / (
            ('' if split == 'val' else 'res_') + Path(name).stem + '.txt')
        annotation = read_labels(label_path, 'gbk' if split == 'val' else 'utf-8')
        for lq_focal, gt_focal in pairs:
            lq_path, gt_path = folder / lq_focal / name, folder / gt_focal / name
            lq, gt = cv2.imread(str(lq_path)), cv2.imread(str(gt_path))
            if lq is None or gt is None or lq.shape != gt.shape:
                raise ValueError(f'Invalid aligned pair: {name}')
            height, width = gt.shape[:2]
            # Upstream interpolates float pixels before tensor conversion. Save
            # rounded 8-bit PNGs for reusable inputs and record this quantization.
            lq_float = cv2.resize(lq.astype(np.float32) / 255,
                                  (width // scale, height // scale), interpolation=cv2.INTER_CUBIC)
            lq = np.rint(np.clip(lq_float * 255, 0, 255)).astype(np.uint8)
            gt = gt[:(height // scale) * scale, :(width // scale) * scale]
            key = f'{Path(name).stem}_{lq_focal}_{gt_focal}'
            for sub, image in [('lq', lq), ('gt', gt), ('bicubic', cv2.resize(
                    lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC))]:
                if not cv2.imwrite(str(output / sub / f'{key}.png'), image):
                    raise OSError(f'Failed writing {key}')
            labels.append(f'{key}.png\t{json.dumps(annotation, ensure_ascii=False)}\n')
            records.append(dict(id=key, group_id=name, split='test' if split == 'val' else 'validation',
                                lq_path=str(output / 'lq' / f'{key}.png'),
                                gt_path=str(output / 'gt' / f'{key}.png'),
                                source_lq=str(lq_path), source_gt=str(gt_path),
                                source_annotation=str(label_path), scale=scale,
                                annotation_quality='official_manual' if split == 'val' else 'official_train_transcription',
                                gt_size=[gt.shape[1], gt.shape[0]], annotations=annotation))
    (output / 'Label.txt').write_text(''.join(labels), encoding='utf-8')
    (output / 'pairs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    summary = dict(images=len(records), groups=len(names), scale=scale, archive_split=split,
                   purpose='final_test' if split == 'val' else 'model_selection', seed=seed,
                   lq_quantization='round(clamp(OpenCV bicubic(float32 pixels / 255) * 255))',
                   annotations='GBK official val/annos' if split == 'val' else 'UTF-8 train/trans_annos_52mm_renov',
                   valid_list_sha256=hashlib.sha256((folder / 'valid_list.txt').read_bytes()).hexdigest(),
                   source_code='https://github.com/mjq11302010044/Real-CE/blob/main/basicsr/data/paired_image_dataset.py')
    (output / 'protocol.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='preset/datasets/RealCE/RealCE')
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--scale', type=int, choices=[2, 4], default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))
