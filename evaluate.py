"""Evaluate SR outputs with full-reference IQA and OCR metrics on annotated text.

PSNR, SSIM, LPIPS and DISTS compare --pred with --gt and are computed only when --gt
is given. OCR-A, CER and 1-NED are computed when a text-box annotation file is given
(PaddleOCR / PPOCRLabel ``Label.txt`` format): every annotated text region is cropped
from pred, and from lq and gt when given, recognized with PaddleOCR and compared with
its transcription.

Images are matched across --pred/--gt/--lq and with the annotation file by file stem,
e.g. ``x.jpg`` (LQ), ``x.png`` (pred) and the annotation line of ``x.jpg``.
"""
import argparse
import csv
import json
import unicodedata
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
IS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
if IS_NPU:
    from torch_npu.contrib import transfer_to_npu

IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
IGNORED_TRANSCRIPTIONS = {'###', '*'}  # "don't care" boxes in PaddleOCR datasets
SOURCES = ('pred', 'lq', 'gt')
FR_METRICS = ('PSNR', 'SSIM', 'LPIPS', 'DISTS')
OCR_METRICS = ('OCR-A', 'CER', '1-NED')


def list_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f'Image file or directory not found: {path}')
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


def image_key(name, strip_suffixes=()):
    """File stem used for matching, without one of the optional ``strip_suffixes``."""
    stem = Path(name.replace('\\', '/')).stem
    for suffix in sorted(strip_suffixes, key=len, reverse=True):
        if suffix and stem.endswith(suffix) and stem != suffix:
            return stem[:-len(suffix)]
    return stem


def index_images(path, strip_suffixes=()):
    index = {}
    for file in list_images(path):
        key = image_key(file.name, strip_suffixes)
        if key in index:
            raise ValueError(f'Ambiguous image key {key!r}: {index[key]} and {file}')
        index[key] = file
    if not index:
        raise ValueError(f'No images found in {path}')
    return index


def load_annotations(path, strip_suffixes=()):
    """Read ``name<TAB>[{"transcription": str, "points": [[x, y], ...], "difficult": bool}, ...]`` lines."""
    annotations = {}
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        name, _, payload = line.partition('\t')
        try:
            items = json.loads(payload)
            if not name.strip() or not isinstance(items, list):
                raise ValueError
            instances = []
            for item in items:
                points = np.asarray(item['points'], dtype=np.float32)
                if not isinstance(item['transcription'], str) or points.ndim != 2 \
                        or points.shape[0] < 4 or points.shape[1] != 2:
                    raise ValueError
                instances.append({'text': item['transcription'], 'points': points,
                                  'difficult': bool(item.get('difficult', False))})
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'{path}:{number}: expected "image_name<TAB>[{{"transcription": str, '
                             f'"points": [[x, y], ...] (>= 4 points)}}, ...]"') from error
        key = image_key(name.strip(), strip_suffixes)
        if key in annotations:
            raise ValueError(f'{path}:{number}: duplicate annotation for image key {key!r}')
        annotations[key] = instances
    return annotations


def normalize_text(text, ignore_case=False):
    """NFKC (full-width to half-width), remove all whitespace, optionally lowercase."""
    text = ''.join(unicodedata.normalize('NFKC', text).split())
    return text.lower() if ignore_case else text


def edit_distance(a, b):
    """Levenshtein distance with unit insertion, deletion and substitution costs."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def ocr_scores(pairs):
    """OCR-A (exact-match rate), CER (sum of edits / label chars) and 1-NED (1 - mean of
    edits / max(len)) over ``(recognized, label)`` pairs with non-empty labels."""
    distances = [edit_distance(text, label) for text, label in pairs]
    return {'OCR-A': sum(text == label for text, label in pairs) / len(pairs),
            'CER': sum(distances) / sum(len(label) for _, label in pairs),
            '1-NED': 1 - sum(d / max(len(t), len(l)) for d, (t, l) in zip(distances, pairs)) / len(pairs)}


def crop_text_region(image, points):
    """Rectify one text region the way PPOCRLabel exports recognition crops.

    Quadrilaterals are perspective-warped in annotation order (made clockwise if drawn
    counter-clockwise); other polygons use their minimum-area rectangle like PaddleOCR.
    Crops at least 1.5x taller than wide are rotated 90 degrees counter-clockwise.
    """
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 4:
        # Shoelace signed area in image coordinates (y down): negative means counter-clockwise.
        if np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1]) < 0:
            points = points[[0, 3, 2, 1]]
    else:
        box = sorted(cv2.boxPoints(cv2.minAreaRect(points)).tolist())
        left, right = sorted(box[:2], key=lambda p: p[1]), sorted(box[2:], key=lambda p: p[1])
        points = np.float32([left[0], right[0], right[1], left[1]])
    width = max(1, int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))))
    height = max(1, int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))))
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (width, height),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(np.rot90(crop) if height / width >= 1.5 else crop)


def scale_points(points, src_size, dst_size):
    return points * np.float32([dst_size[0] / src_size[0], dst_size[1] / src_size[1]])


class PaddleTextRecognizer:
    """PaddleOCR 3.x text-line recognizer for BGR crops, with an optional 0/180-degree classifier."""

    def __init__(self, model_name, cls_model_name=None, device=None, batch_size=1):
        try:
            from paddleocr import TextLineOrientationClassification, TextRecognition
        except ImportError as error:
            raise ImportError('OCR metrics need PaddleOCR>=3.0: pip install paddleocr plus '
                              'paddlepaddle (or paddlepaddle-gpu)') from error
        options = {} if device is None else {'device': device}
        self.rec = TextRecognition(model_name=model_name, **options)
        self.cls = TextLineOrientationClassification(model_name=cls_model_name, **options) if cls_model_name else None
        self.batch_size = batch_size

    def __call__(self, crops):
        if not crops:
            return []
        if self.cls is not None:
            results = self.cls.predict(crops, batch_size=self.batch_size)
            crops = [np.ascontiguousarray(crop[::-1, ::-1]) if int(np.ravel(result['class_ids'])[0]) == 1 else crop
                     for crop, result in zip(crops, results)]
        return [(result['rec_text'], float(result['rec_score']))
                for result in self.rec.predict(crops, batch_size=self.batch_size)]


def evaluate_text(name, images, instances, recognizer, ann_ref=None, ignore_case=False):
    """Recognize the annotated boxes in every image; return detail records and (text, label) pairs."""
    sizes = {source: (image.shape[1], image.shape[0]) for source, image in images.items()}
    if ann_ref is None and len(set(sizes.values())) > 1:
        raise ValueError(f'{name}: image sizes differ ({", ".join(f"{s} {w}x{h}" for s, (w, h) in sizes.items())}); '
                         'set --ann-ref to the image whose pixel coordinates the annotation uses')
    ref_source = ann_ref or 'pred'
    width, height = sizes[ref_source]
    margin = 0.02 * max(width, height)
    outside = sum(bool((inst['points'] < -margin).any() or (inst['points'][:, 0] > width + margin).any()
                       or (inst['points'][:, 1] > height + margin).any()) for inst in instances)
    if outside:
        print(f'Warning: {name}: {outside} text boxes exceed the {width}x{height} {ref_source} image; check --ann-ref')
    records = [{'image': name, 'index': inst['index'], 'label': inst['text']} for inst in instances]
    pairs = {}
    for source, image in images.items():
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        crops = [crop_text_region(bgr, scale_points(inst['points'], sizes[ref_source], sizes[source]))
                 for inst in instances]
        pairs[source] = []
        for record, inst, (text, score) in zip(records, instances, recognizer(crops)):
            normalized = normalize_text(text, ignore_case)
            pairs[source].append((normalized, inst['label']))
            record[source] = {'text': text, 'score': score, 'edit_distance': edit_distance(normalized, inst['label'])}
    return records, pairs


def build_fr_metrics(device, rgb=False):
    import pyiqa
    color = {'test_y_channel': not rgb, 'color_space': 'ycbcr'}
    return {'PSNR': pyiqa.create_metric('psnr', device=device, **color),
            'SSIM': pyiqa.create_metric('ssim', device=device, **color),
            'LPIPS': pyiqa.create_metric('lpips', device=device),
            'DISTS': pyiqa.create_metric('dists', device=device)}


@torch.no_grad()
def compute_fr_metrics(metrics, pred, gt, device, crop_border=0):
    def to_tensor(image):
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
        if crop_border:
            tensor = tensor[..., crop_border:-crop_border, crop_border:-crop_border]
        return tensor.to(device)
    pred, gt = to_tensor(pred), to_tensor(gt)
    return {name: metric(pred, gt).item() for name, metric in metrics.items()}


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred', required=True, help='SR output image file or directory')
    parser.add_argument('--gt', help='GT image file or directory; enables PSNR/SSIM/LPIPS/DISTS')
    parser.add_argument('--lq', help='LQ image file or directory; adds OCR metrics of LQ')
    parser.add_argument('--ann', help='Text-box annotation file (PaddleOCR Label.txt format); enables OCR metrics')
    parser.add_argument('--output', help='Directory for summary.json, per_image.csv and ocr_details.jsonl')
    parser.add_argument('--strip-suffix', nargs='+', default=[], metavar='SUFFIX',
                        help='File-stem suffixes ignored when matching names, e.g. _LR4 _HR')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Torch device for PSNR/SSIM/LPIPS/DISTS')
    parser.add_argument('--crop-border', type=int, default=0,
                        help='Pixels removed from each border before PSNR/SSIM/LPIPS/DISTS')
    parser.add_argument('--rgb', action='store_true', help='PSNR/SSIM on RGB instead of the Y channel of YCbCr')
    parser.add_argument('--ann-ref', choices=SOURCES,
                        help='Image whose pixel coordinates the annotation uses (boxes are scaled to the '
                             'others); required when pred/lq/gt sizes differ')
    parser.add_argument('--include-difficult', action='store_true', help='Also evaluate boxes marked "difficult"')
    parser.add_argument('--ignore-case', action='store_true', help='Case-insensitive text comparison')
    parser.add_argument('--ocr-model', default='PP-OCRv5_server_rec', help='PaddleOCR text recognition model')
    parser.add_argument('--ocr-cls-model', default=None,
                        help='Optional PaddleOCR text-line orientation model, e.g. PP-LCNet_x1_0_textline_ori')
    parser.add_argument('--ocr-device', default=None, help='PaddleOCR device, e.g. gpu:0 or cpu (default: auto)')
    parser.add_argument('--ocr-batch-size', type=int, default=1,
                        help='PaddleOCR batch size; >1 is faster, but padding makes results depend on batching')
    args = parser.parse_args(argv)
    if not args.gt and not args.ann:
        parser.error('nothing to evaluate: give --gt and/or --ann')
    if args.ann_ref in ('lq', 'gt') and not getattr(args, args.ann_ref):
        parser.error(f'--ann-ref {args.ann_ref} requires --{args.ann_ref}')
    if args.crop_border < 0 or args.ocr_batch_size < 1:
        parser.error('--crop-border must be >= 0 and --ocr-batch-size >= 1')
    if args.lq and not args.ann:
        print('Warning: --lq is only used for OCR metrics (with --ann); ignoring it')
        args.lq = None
    return args


def main(argv=None):
    args = parse_args(argv)
    pred_index = index_images(args.pred, args.strip_suffix)
    references = {}
    for source in ('lq', 'gt'):
        if getattr(args, source):
            references[source] = index = index_images(getattr(args, source), args.strip_suffix)
            missing = [pred_index[key].name for key in pred_index if key not in index]
            if missing:
                raise FileNotFoundError(f'No {source} image for {len(missing)} predictions, e.g. {missing[0]}')
            if len(index) > len(pred_index):
                print(f'Warning: {len(index) - len(pred_index)} {source} images have no prediction and are skipped')
    annotations = load_annotations(args.ann, args.strip_suffix) if args.ann else {}
    if args.ann:
        unmatched = [key for key in annotations if key not in pred_index]
        if unmatched:
            print(f'Warning: {len(unmatched)} annotated images have no prediction, e.g. {unmatched[0]}')
        unannotated = [path.name for key, path in pred_index.items() if key not in annotations]
        if unannotated:
            print(f'Warning: {len(unannotated)} predictions have no text annotation, e.g. {unannotated[0]}')
    fr_metrics = build_fr_metrics(args.device, args.rgb) if 'gt' in references else None
    recognizer = PaddleTextRecognizer(args.ocr_model, args.ocr_cls_model, args.ocr_device, args.ocr_batch_size) \
        if any(key in annotations for key in pred_index) else None
    sources = [source for source in SOURCES if source == 'pred' or source in references]

    rows, details, fr_scores = [], [], {name: [] for name in FR_METRICS}
    all_pairs, ocr_images, ignored = {source: [] for source in sources}, 0, 0
    for key, pred_path in tqdm(pred_index.items(), desc='Evaluating'):
        images = {'pred': load_rgb(pred_path)}
        if 'gt' in references:
            images['gt'] = load_rgb(references['gt'][key])
        row = {'name': pred_path.name}
        if fr_metrics is not None:
            if images['pred'].shape != images['gt'].shape:
                raise ValueError(f'{pred_path.name}: pred size {images["pred"].shape[1::-1]} differs from '
                                 f'gt size {images["gt"].shape[1::-1]}')
            row.update(compute_fr_metrics(fr_metrics, images['pred'], images['gt'], args.device, args.crop_border))
            for name in FR_METRICS:
                fr_scores[name].append(row[name])
        if key in annotations:
            instances = []
            for index, inst in enumerate(annotations[key]):
                label = normalize_text(inst['text'], args.ignore_case)
                if not label or inst['text'].strip() in IGNORED_TRANSCRIPTIONS \
                        or (inst['difficult'] and not args.include_difficult):
                    ignored += 1
                else:
                    instances.append({**inst, 'index': index, 'label': label})
            row['num_text'] = len(instances)
            if instances:
                if 'lq' in references:
                    images['lq'] = load_rgb(references['lq'][key])
                records, pairs = evaluate_text(pred_path.name, {source: images[source] for source in sources},
                                               instances, recognizer, args.ann_ref, args.ignore_case)
                details.extend(records)
                ocr_images += 1
                for source in sources:
                    all_pairs[source].extend(pairs[source])
                    row.update((f'{source}_{name}', value) for name, value in ocr_scores(pairs[source]).items())
        rows.append(row)

    summary = {'num_images': len(rows)}
    if fr_metrics is not None:
        summary['fr_iqa'] = {'num_images': len(rows), **{name: float(np.mean(fr_scores[name])) for name in FR_METRICS}}
    if all_pairs['pred']:
        summary['ocr'] = {'num_images': ocr_images, 'num_instances': len(all_pairs['pred']),
                          'ignored_instances': ignored, **{source: ocr_scores(all_pairs[source]) for source in sources}}
    elif args.ann:
        print('Warning: no annotated text box was evaluated; OCR metrics are unavailable')
    summary['settings'] = vars(args)

    print(f'\nEvaluated {len(rows)} images')
    if 'fr_iqa' in summary:
        print(f'pred vs gt ({"RGB" if args.rgb else "Y channel"} PSNR/SSIM): '
              + '  '.join(f'{name} {summary["fr_iqa"][name]:.4f}' for name in FR_METRICS))
    if 'ocr' in summary:
        print(f'OCR on {summary["ocr"]["num_instances"]} text boxes in {ocr_images} images '
              f'({ignored} ignored), model {args.ocr_model}:')
        print(f'{"":6}' + ''.join(f'{name:>9}' for name in OCR_METRICS))
        for source in sources:
            print(f'{source:6}' + ''.join(f'{summary["ocr"][source][name]:9.4f}' for name in OCR_METRICS))
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        columns = ['name'] + (list(FR_METRICS) if fr_metrics is not None else [])
        if args.ann:
            columns += ['num_text'] + [f'{source}_{name}' for source in sources for name in OCR_METRICS]
        with open(output / 'per_image.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        if details:
            with open(output / 'ocr_details.jsonl', 'w', encoding='utf-8') as file:
                file.writelines(json.dumps(record, ensure_ascii=False) + '\n' for record in details)
        print(f'Results saved to {output}')
    return summary


if __name__ == '__main__':
    main()
