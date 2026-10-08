"""Score frozen OCR predictions, or recognize fixed GT boxes with local Tesseract.

No weights are downloaded. For another OCR engine, provide the documented JSONL
predictions and keep its version, weights, language and preprocessing fixed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from token_compression_eval import read_manifest, read_jsonl, score_predictions, sha256_file


def recognize(manifest, images, *, executable='tesseract', language='chi_sim+eng', psm=7):
    from PIL import Image
    predictions = []
    with tempfile.TemporaryDirectory(prefix='vosr-ocr-') as temporary:
        crop_path = Path(temporary) / 'region.png'
        for row in manifest:
            with Image.open(Path(images) / (row['id'] + '.png')) as image:
                if image.size != tuple(row['size']):
                    raise ValueError(f"Unexpected output size: {row['id']}: {image.size} vs {row['size']}")
                regions = []
                for region in row['regions']:
                    image.convert('RGB').crop(region['bbox']).save(crop_path)
                    result = subprocess.run([executable, str(crop_path), 'stdout', '-l', language, '--psm', str(psm)],
                                            capture_output=True, text=True, encoding='utf-8', check=True, timeout=120)
                    regions.append(dict(id=region['id'], text=result.stdout.rstrip('\r\n')))
            predictions.append(dict(id=row['id'], regions=regions))
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--predictions', help='JSONL from any fixed recognizer')
    source.add_argument('--images', help='Directory containing <manifest id>.png')
    parser.add_argument('--output', required=True, help='Metrics JSON; must not exist')
    parser.add_argument('--engine-id', required=True, help='Recognizer version + weight hash + preprocessing identifier')
    parser.add_argument('--tesseract', default='tesseract')
    parser.add_argument('--language', default='chi_sim+eng')
    parser.add_argument('--psm', type=int, default=7)
    parser.add_argument('--strip-whitespace', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix('.predictions.jsonl').exists():
        parser.error('Output exists; use a new path')
    manifest = read_manifest(args.manifest)
    predictions = (read_jsonl(args.predictions) if args.predictions else
                   recognize(manifest, args.images, executable=args.tesseract, language=args.language, psm=args.psm))
    report = score_predictions(manifest, predictions, args.strip_whitespace)
    report.update(engine_id=args.engine_id, manifest_sha256=sha256_file(args.manifest))
    if args.images:
        report.update(language=args.language, psm=args.psm,
                      image_sha256={row['id']: sha256_file(Path(args.images) / (row['id'] + '.png')) for row in manifest},
                      engine_version=subprocess.run([args.tesseract, '--version'], capture_output=True, text=True,
                                                    check=True, timeout=30).stdout.splitlines()[0])
    else:
        report['predictions_sha256'] = sha256_file(args.predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.predictions.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in predictions), encoding='utf-8')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['groups'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
