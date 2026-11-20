"""Run PP-OCR detection and recognition on HQ images and write VOSR single-character annotations."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image


def image_quad(box, width, height):
    points = np.asarray(box, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    if (points < 0).any() or (points > [width, height]).any():
        return None
    edges = np.roll(points, -1, axis=0) - points
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if (turns < -1e-8).all():
        points = points[[0, 3, 2, 1]]
    elif not (turns > 1e-8).all():
        return None
    return points.tolist()


def annotation_lines(result, width, height):
    """PaddleOCR 3.x returns CTC-aligned Chinese characters and grouped non-Chinese words."""
    lines = []
    skipped_groups = invalid_boxes = 0
    if not {'text_word', 'text_word_region'} <= result.keys():
        raise ValueError('PP-OCR must return character locations with return_word_box=True')
    for text, score, box, labels, regions in zip(
        result['rec_texts'], result['rec_scores'], result['rec_polys'],
        result['text_word'], result['text_word_region'], strict=True,
    ):
        line_quad = image_quad(box, width, height)
        if line_quad is None:
            invalid_boxes += 1
            continue
        glyphs = []
        for label, region in zip(labels, regions, strict=True):
            if len(label) != 1 or label.isspace():
                skipped_groups += 1
                continue
            quad = image_quad(region, width, height)
            if quad is None:
                invalid_boxes += 1
                continue
            glyphs.append({'char': label, 'quad': quad})
        lines.append({'transcript': text, 'quad': line_quad, 'rec_score': float(score), 'glyphs': glyphs})
    return lines, skipped_groups, invalid_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hq-dir', required=True, type=Path, help='HQ directory used by the paired dataset')
    parser.add_argument('--output', required=True, type=Path, help='JSON file for data.text_boxes')
    parser.add_argument('--device', default='gpu:0')
    parser.add_argument('--min-score', type=float, default=0.8, help='Minimum PP-OCR line recognition confidence')
    parser.add_argument('--det-model-dir', help='Local PP-OCRv5_server_det assets (default: PaddleOCR download)')
    parser.add_argument('--rec-model-dir', help='Local PP-OCRv5_server_rec assets (default: PaddleOCR download)')
    args = parser.parse_args()
    if not 0 <= args.min_score <= 1:
        parser.error('--min-score must be in [0, 1]')
    suffixes = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
    paths = sorted(p for p in args.hq_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    if not paths or len({p.stem for p in paths}) != len(paths):
        parser.error('--hq-dir needs images with unique filename stems')
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(text_detection_model_name='PP-OCRv5_server_det', text_detection_model_dir=args.det_model_dir,
                    text_recognition_model_name='PP-OCRv5_server_rec', text_recognition_model_dir=args.rec_model_dir,
                    device=args.device, use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)
    counts = dict(images=0, lines=0, glyphs=0, skipped_groups=0, invalid_boxes=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile('w', dir=args.output.parent, suffix='.tmp', delete=False, encoding='utf-8') as stream:
        temporary = Path(stream.name)
        try:
            stream.write('{')
            for path in paths:
                with Image.open(path) as image:
                    width, height = image.size
                    bgr = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
                result, = ocr.predict(bgr, return_word_box=True, text_rec_score_thresh=args.min_score)
                lines, skipped, invalid = annotation_lines(result, width, height)
                if counts['images']:
                    stream.write(',')
                stream.write('\n' + json.dumps(path.stem, ensure_ascii=False) + ': ')
                json.dump(lines, stream, ensure_ascii=False, allow_nan=False)
                counts['images'] += 1
                counts['lines'] += len(lines)
                counts['glyphs'] += sum(len(line['glyphs']) for line in lines)
                counts['skipped_groups'] += skipped
                counts['invalid_boxes'] += invalid
            stream.write('\n}\n')
            stream.close()
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
