"""Measure OCR evidence for data ablations without claiming glyph verification."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from data_pipeline.pipeline import atomic_json, file_digest
from evaluate import PaddleTextRecognizer, crop_text_region, normalize_text, edit_distance


def score(manifest, output, ocr_python, reference_manifest=None, model='PP-OCRv5_server_rec', iqa='musiq'):
    rows = [json.loads(x) for x in Path(manifest).read_text().splitlines()]
    references = {r['id']: r for r in (json.loads(x) for x in Path(reference_manifest).read_text().splitlines())} \
        if reference_manifest else {}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(manifest_sha256=file_digest(manifest), ocr_model=model,
                    reference_sha256=file_digest(reference_manifest) if reference_manifest else None,
                    ocr_python=str(Path(ocr_python).resolve()), batch_size=1, iqa=iqa,
                    code_sha256=file_digest(__file__), purpose='experimental_filter_evidence_not_glyph_approval')
    if (output / 'protocol.json').exists() and json.loads((output / 'protocol.json').read_text()) != protocol:
        raise ValueError('OCR scoring protocol changed')
    atomic_json(output / 'protocol.json', protocol)
    recognizer, metric = None, None
    if iqa != 'none':
        import torch
        import pyiqa
        metric = pyiqa.create_metric(iqa, device='cuda')
    scored, counts = [], Counter()
    try:
        for i, row in enumerate(rows):
            path = output / 'records' / f'{row["id"]}.json'
            if path.exists():
                scored.append(json.loads(path.read_text()))
                continue
            if file_digest(row['hr_path']) != row['file_sha256']:
                raise ValueError('Training image changed after manifest construction')
            image = cv2.imread(row['hr_path'])
            annotations = row['annotations']
            crops = [crop_text_region(image, np.array(a['polygon'], dtype=np.float32)) for a in annotations]
            if recognizer is None:
                recognizer = PaddleTextRecognizer(model, device='cpu', batch_size=1, python_executable=ocr_python)
            recognized = recognizer(crops)
            if len(recognized) != len(annotations):
                raise ValueError('OCR region count mismatch')
            prior = references.get(row['id'], {}).get('quality', {}).get('recognitions', [])
            evidence, total_edits, total_chars = [], 0, 0
            for index, (annotation, (text, confidence)) in enumerate(zip(annotations, recognized)):
                target, kind = None, 'unavailable'
                if annotation.get('trusted') and annotation.get('text'):
                    target, kind = annotation['text'], 'dataset_transcription'
                elif len(prior) == len(annotations) and prior[index]['confidence'] >= 0.95:
                    target, kind = prior[index]['text'], 'original_high_confidence_ocr_consistency'
                if target is not None:
                    label = normalize_text(target)
                    edits = edit_distance(normalize_text(text), label)
                    total_edits += edits
                    total_chars += len(label)
                else:
                    edits = None
                evidence.append(dict(text=text, confidence=confidence, reference=target,
                                     reference_kind=kind, edit_distance=edits))
            quality = dict(recognitions=evidence,
                           min_confidence=min((r['confidence'] for r in evidence), default=0),
                           mean_confidence=float(np.mean([r['confidence'] for r in evidence])) if evidence else 0,
                           reference_cer=total_edits/total_chars if total_chars else None,
                           reference_match=bool(evidence) and all(r['edit_distance'] == 0 for r in evidence),
                           glyph_verified=False, iqa_calibrated=False)
            if metric is not None:
                tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1])).permute(2, 0, 1)[None].float().cuda()/255
                with torch.no_grad():
                    quality['iqa'] = {iqa: float(metric(tensor).item())}
            result = dict(row, quality=quality)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, result)
            scored.append(result)
            if (i+1) % 50 == 0:
                print(json.dumps(dict(processed=i+1, total=len(rows))), flush=True)
        for row in scored:
            counts[f'{row["source"]}/{row["crop_mode"]}/all'] += 1
            if row['quality']['min_confidence'] >= 0.9:
                counts[f'{row["source"]}/{row["crop_mode"]}/confidence_0.9'] += 1
                if row['quality']['reference_match']:
                    counts[f'{row["source"]}/{row["crop_mode"]}/fidelity'] += 1
        (output / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in scored))
        atomic_json(output / 'summary.json', dict(counts=dict(counts), protocol=protocol))
    finally:
        if recognizer is not None:
            recognizer.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ocr-python', default='.venv-ocr/bin/python')
    p.add_argument('--reference-manifest')
    p.add_argument('--model', default='PP-OCRv5_server_rec')
    p.add_argument('--iqa', choices=['musiq', 'none'], default='musiq')
    score(**vars(p.parse_args()))
