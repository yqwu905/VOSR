"""Real-CE x4 in the official test protocol, for head-split training (dataset_type: paired) and OCR evaluation.

LR = the 13 mm capture resized to 1/4 of the 52 mm size with cv2.INTER_CUBIC; GT = the 52 mm capture cropped to 4x
the LR size. Writes, under --output:

  val/LR, val/HR, val/Label.txt   test set: LR inputs, GT, text boxes + transcriptions (PPOCRLabel format, GT
                                  pixels) for evaluate.py --ann ... --ann-ref gt
  train/LQ, train/HQ, train/boxes.json
                                  training pairs: LQ = PIL bicubic x4 of the LR (what the inference scripts feed the
                                  model), HQ = GT; boxes.json = text quads per image (52 mm annotations) for
                                  data.text_boxes

usage: python tools/prepare_realce.py --realce /path/to/RealCE --output preset/datasets/RealCE_x4
(--realce holds train/ and val/ of the Real-CE release, each with valid_list.txt, 13mm/ and 52mm/.)
"""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image

SCALE = 4


def read_pair(split, name):
    lq = cv2.imread(str(split / '13mm' / name), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(split / '52mm' / name), cv2.IMREAD_COLOR)
    if lq is None or gt is None:
        raise FileNotFoundError(f'{split}: cannot read 13mm/{name} or 52mm/{name}')
    h, w = gt.shape[:2]
    lr = cv2.resize(lq, (w // SCALE, h // SCALE), interpolation=cv2.INTER_CUBIC)
    return lr, gt[:lr.shape[0] * SCALE, :lr.shape[1] * SCALE]


def names(split):
    return [line.strip() for line in (split / 'valid_list.txt').read_text().splitlines() if line.strip()]


def build_val(split, out):
    (out / 'LR').mkdir(parents=True, exist_ok=True)
    (out / 'HR').mkdir(parents=True, exist_ok=True)
    lines, boxes, skipped = [], 0, 0
    for name in names(split):
        stem = name.rsplit('.', 1)[0]
        lr, gt = read_pair(split, name)
        cv2.imwrite(str(out / 'LR' / f'{stem}.png'), lr)
        cv2.imwrite(str(out / 'HR' / f'{stem}.png'), gt)
        items = []
        for raw in (split / 'annos' / f'{stem}.txt').read_text(encoding='gbk').splitlines():
            parts = raw.strip().split(',', 8)
            if not raw.strip():
                continue
            if len(parts) < 9:
                skipped += 1
                continue
            points = [[int(float(parts[2 * i])), int(float(parts[2 * i + 1]))] for i in range(4)]
            items.append({'transcription': parts[8], 'points': points, 'difficult': False})
        boxes += len(items)
        lines.append(f'{stem}.png\t' + json.dumps(items, ensure_ascii=False))
    (out / 'Label.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'val: {len(lines)} images, {boxes} text boxes, {skipped} malformed annotation lines skipped')
    return {line.split('\t', 1)[0].rsplit('.', 1)[0] for line in lines}


def build_train(split, out, val_stems):
    (out / 'LQ').mkdir(parents=True, exist_ok=True)
    (out / 'HQ').mkdir(parents=True, exist_ok=True)
    boxes = {}
    for name in names(split):
        stem = name.rsplit('.', 1)[0]
        if stem in val_stems:
            raise ValueError(f'{stem} is in both the train and the val split')
        lr, gt = read_pair(split, name)
        up = Image.fromarray(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)).resize((lr.shape[1] * SCALE, lr.shape[0] * SCALE),
                                                                         Image.Resampling.BICUBIC)
        up.save(out / 'LQ' / f'{stem}.png', compress_level=1)
        cv2.imwrite(str(out / 'HQ' / f'{stem}.png'), gt, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        quads = []
        annotation = split / 'trans_annos_52mm_renov' / f'res_{stem}.txt'
        if annotation.exists():
            for raw in annotation.read_text(encoding='utf-8', errors='replace').splitlines():
                parts = raw.strip().split(',', 8)
                if len(parts) >= 8:
                    try:
                        quads.append([float(v) for v in parts[:8]])
                    except ValueError:
                        pass
        boxes[stem] = quads
    (out / 'boxes.json').write_text(json.dumps(boxes))
    print(f'train: {len(boxes)} pairs, {sum(len(v) for v in boxes.values())} text boxes')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--realce', required=True, help='Real-CE root with train/ and val/')
    parser.add_argument('--output', required=True)
    parser.add_argument('--splits', nargs='+', choices=('val', 'train'), default=('val', 'train'))
    args = parser.parse_args()
    root, out = Path(args.realce), Path(args.output)
    val_stems = ({name.rsplit('.', 1)[0] for name in names(root / 'val')})
    if 'val' in args.splits:
        build_val(root / 'val', out / 'val')
    if 'train' in args.splits:
        build_train(root / 'train', out / 'train', val_stems)


if __name__ == '__main__':
    main()
