import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.build_data import square_canvas
from experiments.prepare_realce import prepare, read_labels
from experiments.training_data import ResearchDataset
from evaluate import evaluate_text, ocr_scores, normalize_text


def test_square_canvas_keeps_whole_text_and_aspect():
    image = Image.new('RGB', (800, 200), 'white')
    annotations = [dict(polygon=[[0, 0], [800, 0], [800, 200], [0, 200]], bbox=[0, 0, 800, 200])]
    out, labels, matrix = square_canvas(image, annotations, 512)
    assert out.size == (512, 512)
    assert labels[0]['bbox'] == [0.0, 192.0, 512.0, 320.0]
    assert matrix[0][0] == matrix[1][1]


def test_realce_uses_valid_list_gbk_and_modcrop(tmp_path):
    root = tmp_path / 'RealCE'
    for sub in ['13mm', '52mm', 'annos']:
        (root / 'val' / sub).mkdir(parents=True)
    (root / 'val/valid_list.txt').write_text('valid.JPG\n')
    for name in ['valid', 'excluded']:
        for focal in ['13mm', '52mm']:
            Image.new('RGB', (35, 43), 'white').save(root / 'val' / focal / f'{name}.JPG')
    (root / 'val/annos/valid.txt').write_text('1,2,20,2,20,15,1,15,中文,A\n', encoding='gbk')
    out = tmp_path / 'prepared'
    summary = prepare(root, out)
    assert summary['images'] == 1
    assert Image.open(next((out / 'gt').glob('*.png'))).size == (32, 40)
    assert Image.open(next((out / 'lq').glob('*.png'))).size == (8, 10)
    assert '中文,A' in (out / 'Label.txt').read_text()
    with pytest.raises(ValueError, match='subsample'):
        prepare(root, tmp_path / 'bad', limit=1)


def make_dataset(tmp_path):
    rows = []
    for source, count in [('a', 1), ('b', 20)]:
        for i in range(count):
            image = tmp_path / f'{source}{i}.png'
            Image.new('RGB', (32, 32), (30+i, 100, 200)).save(image)
            rows.append(dict(id=f'{source}{i}', group_id=f'{source}{i}', source=source,
                             split='train', crop_mode='text', target='original',
                             hr_path=str(image), use_policy='research_ablation_unreviewed'))
    # A validation-only source must not enter training.
    rows.append(dict(rows[0], id='heldout', group_id='heldout', split='validation', source='val'))
    path = tmp_path / 'manifest.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return dict(manifest=str(path), split='train', crop_mode='text', target='original',
                use_policy='research_ablation_unreviewed', resolution=32,
                source_weights={'a': 0.5, 'b': 0.5},
                degradation=dict(scales=[4], blur_sigma=[0.2, 1.2], noise_std=[0, 5], jpeg_quality=[65, 95]))


def test_source_probabilities_independent_of_source_size_and_resume(tmp_path):
    config = make_dataset(tmp_path)
    ds = ResearchDataset(config, 1000, 42)
    count_a = sum(ds[i]['source'] == 'a' for i in range(1000))
    assert 450 < count_a < 550  # Would be ~48 for sampling by concatenated image count.
    resumed = ResearchDataset(config, 1000, 42, start=500)
    assert ds[500]['id'] == resumed[0]['id']
    assert (ds[500]['lq'] == resumed[0]['lq']).all()
    assert 'val' not in ds.groups
    config['quality_filter'] = 'confidence'
    with pytest.raises(ValueError, match='evidence'):
        ResearchDataset(config, 1000, 42)


def test_ocr_rejects_silent_dropped_regions():
    image = np.zeros((32, 64, 3), np.uint8)
    instance = dict(index=0, text='ABC', label='ABC', points=np.array([[0,0],[60,0],[60,20],[0,20]]))
    with pytest.raises(ValueError, match='every text region'):
        evaluate_text('test', {'pred': image}, [instance], lambda crops: [])


def test_ocr_metrics_use_micro_cer_and_macro_ned():
    score = ocr_scores([('ABC', 'ABC'), ('', 'D'), ('XYZ', 'X')])
    assert score['OCR-A'] == pytest.approx(1/3)
    assert score['CER'] == pytest.approx(3/5)
    assert score['1-NED'] == pytest.approx(1-(0+1+2/3)/3)
    assert normalize_text('Ａ b\n中', True) == 'ab中'


def test_iqa_quantile_excludes_validation(tmp_path):
    config = make_dataset(tmp_path)
    path = Path(config['manifest'])
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    for i, row in enumerate(rows):
        row['quality'] = dict(iqa={'musiq': float(i)})
    rows[-1].update(source='b', quality=dict(iqa={'musiq': 1e9}))
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config.update(iqa_keep_fraction=0.5)
    ds = ResearchDataset(config, 100, 42)
    assert ds.summary['b']['iqa_threshold'] == 10.5
    assert ds.summary['b']['crops'] == 10


def test_prediction_only_ocr_uses_gt_coordinates():
    images = dict(pred=np.zeros((16, 32, 3), np.uint8), gt=np.zeros((32, 64, 3), np.uint8))
    inst = dict(index=0, text='A', label='A', points=np.array([[0,0],[40,0],[40,20],[0,20]]))
    seen = []
    def recognize(crops):
        seen.extend(crops)
        return [('A', 1.0)]
    records, pairs = evaluate_text('test', images, [inst], recognize, ann_ref='gt', recognition_sources=['pred'])
    assert list(pairs) == ['pred']
    assert len(seen) == 1 and seen[0].shape[:2] == (10, 20)
    assert 'gt' not in records[0]
