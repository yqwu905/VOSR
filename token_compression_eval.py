"""Strict OCR protocol and analytic attention accounting (no model dependencies)."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def read_manifest(path, require_images=False):
    """Boxes use the restored/HQ pixel grid; paths are relative to the manifest."""
    path = Path(path).resolve()
    rows = read_jsonl(path)
    if not rows:
        raise ValueError('Empty evaluation manifest')
    seen = set()
    for row in rows:
        identifier = row.get('id', '')
        if not isinstance(identifier, str) or not re.fullmatch(r'[\w-]+', identifier) or identifier in seen:
            raise ValueError(f'Invalid/duplicate image id: {identifier!r}')
        seen.add(identifier)
        size = row.get('size', [])
        if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError(f'{identifier}: size must be [output_width, output_height]')
        for key in ('lq', 'hq'):
            if key == 'lq' and not row.get(key):
                raise ValueError(f'{identifier}: lq path is required')
            if row.get(key):
                row[key] = str((path.parent / row[key]).resolve())
                if require_images and not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])
        if not row.get('regions'):
            raise ValueError(f'{identifier}: non-empty human-transcribed regions are required')
        region_ids = set()
        for region in row['regions']:
            rid = region.get('id')
            if not isinstance(rid, str) or not rid or rid in region_ids:
                raise ValueError(f'{identifier}: duplicate/invalid region id')
            region_ids.add(rid)
            if not isinstance(region.get('text'), str) or not region['text']:
                raise ValueError(f'{identifier}/{rid}: empty/non-string GT text')
            box = region.get('bbox', [])
            if (len(box) != 4 or any(type(v) is not int for v in box) or
                    not 0 <= box[0] < box[2] <= size[0] or not 0 <= box[1] < box[3] <= size[1]):
                raise ValueError(f'{identifier}/{rid}: bbox must be integer [x0,y0,x1,y1] in output pixels')
        for record in (row, *row['regions']):
            tags = record.get('tags', [])
            if not isinstance(tags, list) or any(not isinstance(t, str) or not t for t in tags):
                raise ValueError(f'{identifier}: tags must be a list of nonempty strings')
    return rows


def normalize_text(text, strip_whitespace=False):
    text = unicodedata.normalize('NFC', text)
    return ''.join(text.split()) if strip_whitespace else text


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, char in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def score_predictions(manifest, predictions, strip_whitespace=False):
    """Corpus CER includes substitutions, deletions AND insertions, without clipping.

    Recognition on fixed GT boxes; not a detector score. Missing predictions fail
    rather than silently improving scores by dropping hard/empty regions.
    """
    lookup = {}
    for row in predictions:
        identifier = row['id']
        if identifier in lookup:
            raise ValueError(f'Duplicate prediction: {identifier}')
        regions = {}
        for region in row['regions']:
            if region['id'] in regions or not isinstance(region['text'], str):
                raise ValueError(f'Duplicate/invalid prediction region: {identifier}')
            regions[region['id']] = region['text']
        lookup[identifier] = regions
    if set(lookup) != {row['id'] for row in manifest}:
        raise ValueError('Prediction image coverage does not match manifest exactly')
    totals, details = {}, []
    for row in manifest:
        hypotheses = lookup[row['id']]
        if set(hypotheses) != {region['id'] for region in row['regions']}:
            raise ValueError(f"Prediction region coverage mismatch: {row['id']}")
        for region in row['regions']:
            gt = normalize_text(region['text'], strip_whitespace)
            pred = normalize_text(hypotheses[region['id']], strip_whitespace)
            if not gt:
                raise ValueError('GT must not be empty after normalization')
            errors = edit_distance(gt, pred)
            groups = {'all', *row.get('tags', []), *region.get('tags', [])}
            for group in groups:
                total = totals.setdefault(group, dict(characters=0, errors=0, regions=0, exact=0))
                total['characters'] += len(gt)
                total['errors'] += errors
                total['regions'] += 1
                total['exact'] += int(gt == pred)
            details.append(dict(image_id=row['id'], region_id=region['id'], gt=gt, prediction=pred,
                                errors=errors, characters=len(gt), tags=sorted(groups - {'all'})))
    for total in totals.values():
        total['cer'] = total['errors'] / total['characters']
        total['character_accuracy'] = 1 - total['cer']  # can be negative when insertions dominate
        total['exact_match'] = total['exact'] / total['regions']
    return dict(protocol='fixed_gt_box_recognition', normalization='NFC' + ('+no_whitespace' if strip_whitespace else ''),
                groups=totals, regions=details)


def attention_cost(tokens_per_layer, hidden_size, *, batch_size=1, condition_tokens=0, dense_tokens=None):
    """Analytic multiply-add FLOPs from actual input shapes; excludes other ops.

    SA QK+AV: 4*B*N²*D; SA projections: 8*B*N*D². CA QK+AV:
    4*B*N*M*D; CA projections: 4*B*(N+M)*D². Fused kernels need
    not materialize N² scores; this is NOT a peak-memory prediction.
    """
    ns = list(tokens_per_layer)
    if not ns or min(ns) <= 0 or min(hidden_size, batch_size) <= 0 or condition_tokens < 0:
        raise ValueError('Invalid attention dimensions')
    d, b, m = hidden_size, batch_size, condition_tokens
    out = dict(self_qk_av_flops=4*b*d*sum(n*n for n in ns),
               self_projection_flops=8*b*d*d*sum(ns),
               cross_qk_av_flops=4*b*m*d*sum(ns),
               cross_projection_flops=4*b*d*d*sum(n+m for n in ns) if m else 0)
    out['attention_matmul_flops'] = sum(out.values())
    out['flops_convention'] = 'multiply+add=2; excludes norms/softmax/MLP/VAE/DINO/pool/restore'
    if dense_tokens is not None:
        out['self_pair_ratio'] = sum(n*n for n in ns) / (len(ns)*dense_tokens**2)
        out['self_projection_ratio'] = sum(ns) / (len(ns)*dense_tokens)
    return out
