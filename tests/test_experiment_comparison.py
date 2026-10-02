import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.compare import aggregate, load_evaluation, paired_comparison, METRICS
from evaluate import edit_distance, ocr_scores


def fixture_evaluation(directory, texts):
    directory.mkdir()
    labels = ['ABCD', 'A', 'XYZ']
    pairs = [dict(id=f'image{i}', group_id='shared' if i < 2 else 'independent') for i in range(3)]
    image_rows, details = [], []
    for i, (prediction, label) in enumerate(zip(texts, labels)):
        image_rows.append(dict(name=f'image{i}.png', PSNR=30+i, SSIM=0.9, LPIPS=0.1, DISTS=0.2, num_text=1))
        details.append(dict(image=f'image{i}.png', index=0, label=label,
                            pred=dict(text=prediction, score=0.9, edit_distance=edit_distance(prediction, label))))
    with (directory/'per_image.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)
    (directory/'ocr_details.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details))
    summary = dict(num_images=3, fr_iqa=dict(PSNR=31, SSIM=0.9, LPIPS=0.1, DISTS=0.2),
                   ocr=dict(num_instances=3, pred=ocr_scores(list(zip(texts, labels)))),
                   settings=dict(rgb=True, crop_border=2, ignore_case=False, include_difficult=False,
                                 ocr_model='fixed_model', ocr_batch_size=1, ocr_cls_model=None, ann_ref='gt'))
    (directory/'summary.json').write_text(json.dumps(summary))
    return pairs


def test_comparison_uses_source_groups_and_correct_ocr_denominators(tmp_path):
    pairs = fixture_evaluation(tmp_path/'baseline', ['ABCD', '', 'XY'])
    fixture_evaluation(tmp_path/'candidate', ['ABCD', 'A', 'XYZ'])
    baseline = load_evaluation(tmp_path/'baseline', pairs)
    candidate = load_evaluation(tmp_path/'candidate', pairs)
    assert len(baseline['totals']) == 2  # Three crops from only two independent sources.
    assert baseline['estimate'][METRICS.index('CER')] == pytest.approx(2/8)
    assert baseline['estimate'][METRICS.index('1-NED')] == pytest.approx(1-(0+1+1/3)/3)
    result = paired_comparison(baseline, candidate, samples=1000)
    assert result['groups'] == 2
    assert result['metrics']['CER']['delta'] == pytest.approx(-0.25)
    assert result['metrics']['PSNR']['delta_ci95'] == [0.0, 0.0]
    assert result['metrics']['CER']['improvement_interval_excludes_zero']


def test_compare_rejects_changed_labels_settings_and_inconsistent_summary(tmp_path):
    pairs = fixture_evaluation(tmp_path/'baseline', ['ABCD', 'A', 'XYZ'])
    fixture_evaluation(tmp_path/'candidate', ['ABCD', 'A', 'XYZ'])
    baseline = load_evaluation(tmp_path/'baseline', pairs)
    candidate = load_evaluation(tmp_path/'candidate', pairs)
    candidate['labels'][('image0', 0)] = 'different reference'
    with pytest.raises(ValueError, match='identical'):
        paired_comparison(baseline, candidate, samples=100)
    candidate = load_evaluation(tmp_path/'candidate', pairs)
    candidate['settings']['ignore_case'] = True
    with pytest.raises(ValueError, match='ignore_case'):
        paired_comparison(baseline, candidate, samples=100)
    path = tmp_path/'candidate/summary.json'
    summary = json.loads(path.read_text())
    summary['ocr']['pred']['CER'] = 0.1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='independently aggregated'):
        load_evaluation(tmp_path/'candidate', pairs)


def test_duplicate_images_and_regions_cannot_change_denominators(tmp_path):
    pairs = fixture_evaluation(tmp_path/'baseline', ['ABCD', 'A', 'XYZ'])
    with pytest.raises(ValueError, match='unique'):
        load_evaluation(tmp_path/'baseline', pairs + [pairs[0]])
    path = tmp_path/'baseline/ocr_details.jsonl'
    path.write_text(path.read_text()+path.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError, match='Duplicate OCR'):
        load_evaluation(tmp_path/'baseline', pairs)


def test_bootstrap_for_constant_paired_difference_is_constant(tmp_path):
    pairs = fixture_evaluation(tmp_path/'baseline', ['ABCD', 'A', 'XYZ'])
    base = load_evaluation(tmp_path/'baseline', pairs)
    candidate = dict(base, totals=base['totals'].copy())
    # Every image improves by exactly 2dB; unequal group sizes must not affect it.
    candidate['totals'][:, 0] += 2*candidate['totals'][:, 4]
    candidate['estimate'] = aggregate(candidate['totals'].sum(0))
    result = paired_comparison(base, candidate, samples=100)
    assert result['metrics']['PSNR']['delta_ci95'] == pytest.approx([2, 2])
