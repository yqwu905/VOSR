"""Check synthesis source images against official test photographs and GT pixels.

Exact decoded-pixel matches are conclusive. DCT hashes only propose full-image
similarities for visual inspection; a negative screen cannot exclude all scene
overlap or cropped/rescaled copies.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data_pipeline.pipeline import atomic_json, file_digest, pixel_digest


def fingerprint(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        digest = pixel_digest(rgb)
        gray = np.asarray(rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        coefficients = cv2.dct(gray)[:8, :8].ravel()
        bits = coefficients > np.median(coefficients[1:])
        bits[0] = False
        hashed = sum(int(bit) << index for index, bit in enumerate(bits))
        return dict(path=str(path), sha256=digest, size=list(rgb.size), aspect=rgb.width/rgb.height, phash=hashed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', required=True, type=Path)
    p.add_argument('--test-pairs', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    cv2.setNumThreads(1)
    source_rows = {}
    for row in map(json.loads, args.sources.read_text().splitlines()):
        source_rows.setdefault(row['source_sha256'], row)
    sources = []
    for index, row in enumerate(source_rows.values()):
        item = fingerprint(row['source_path'])
        if item['sha256'] != row['source_sha256']:
            raise ValueError('A source image changed after synthesis')
        sources.append(dict(item, source=row['source'], split=row['split'], group_id=row['group_id']))
        if (index+1) % 200 == 0:
            print(json.dumps(dict(source_images_checked=index+1, total=len(source_rows))), flush=True)
    pairs = [json.loads(line) for line in args.test_pairs.read_text().splitlines()]
    if len({row['id'] for row in pairs}) != len(pairs):
        raise ValueError('Duplicate test scene identifiers')
    references = []
    for index, row in enumerate(pairs):
        for kind in ['source_lq', 'source_gt', 'gt_path']:
            references.append(dict(fingerprint(row[kind]), id=row['id'], kind=kind))
        if (index+1) % 50 == 0:
            print(json.dumps(dict(test_scenes_checked=index+1, total=len(pairs))), flush=True)
    source_by_hash = {row['sha256']: row for row in sources}
    exact, similar = [], []
    for reference in references:
        if reference['sha256'] in source_by_hash:
            exact.append(dict(source=source_by_hash[reference['sha256']], reference=reference))
        for source in sources:
            if abs(source['aspect']/reference['aspect']-1) > .03:
                continue
            distance = (source['phash'] ^ reference['phash']).bit_count()
            if distance <= 4:
                similar.append(dict(source=source, reference=reference, phash_distance=distance))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'source_fingerprints.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sources))
    (args.output/'test_fingerprints.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in references))
    counts = Counter(f'{r["source"]}/{r["split"]}' for r in sources)
    report = dict(source_images=len(sources), source_counts=dict(counts), test_scenes=len(pairs),
                  reference_images=len(references), exact_matches=exact, phash_candidates=similar,
                  protocol=dict(source_manifest_sha256=file_digest(args.sources),
                                test_pairs_sha256=file_digest(args.test_pairs), code_sha256=file_digest(__file__),
                                exact_method='SHA256 of RGB dimensions and decoded pixels',
                                similarity_method='32x32 grayscale, orthonormal DCT, top 8x8, median excluding DC, DC bit zero',
                                max_hamming=4, max_aspect_relative_difference=.03),
                  limitation='Full-image similarity screen; absence of candidates does not prove all near duplicates or scene overlap are absent.')
    atomic_json(args.output/'summary.json', report)
    print(json.dumps(dict(sources=len(sources), test_scenes=len(pairs), exact_matches=len(exact),
                          phash_candidates=len(similar))), flush=True)


if __name__ == '__main__':
    main()
