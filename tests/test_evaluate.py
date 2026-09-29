"""evaluate.py tests; OCR and IQA backends are stubbed, so no PaddleOCR or pyiqa weights are needed."""
import csv
import json
import numpy as np
import pytest
from PIL import Image

import evaluate
from evaluate import (crop_text_region, edit_distance, image_key, load_annotations, normalize_text,
                      ocr_scores, scale_points)

EXAMPLE = ('19700107_112359_541_iso400_20.0X_CMM_05_AIGC_INPUT.jpg\t[{"transcription": "手機報在线", '
           '"points": [[1179, 551], [1652, 544], [1654, 666], [1181, 673]], "difficult": false}, '
           '{"transcription": "SCARFACE", "points": [[2859, 1292], [2993, 1299], [3004, 2186], [2866, 2186]], '
           '"difficult": false}]')


def test_load_ppocrlabel_file(tmp_path):
    path = tmp_path / 'Label.txt'
    path.write_text('﻿' + EXAMPLE + '\n\nimages/b.png\t[]\n', encoding='utf-8')
    annotations = load_annotations(path, ['_AIGC_INPUT'])
    assert set(annotations) == {'19700107_112359_541_iso400_20.0X_CMM_05', 'b'}
    instances = annotations['19700107_112359_541_iso400_20.0X_CMM_05']
    assert [inst['text'] for inst in instances] == ['手機報在线', 'SCARFACE']
    assert [inst['difficult'] for inst in instances] == [False, False]
    assert instances[1]['points'].dtype == np.float32
    np.testing.assert_array_equal(instances[1]['points'], [[2859, 1292], [2993, 1299], [3004, 2186], [2866, 2186]])
    assert annotations['b'] == []


@pytest.mark.parametrize('line', ['no_tab.jpg', '\t[]', 'a.jpg\t{}',
                                  'a.jpg\t[{"points": [[0, 0], [1, 0], [1, 1], [0, 1]]}]',
                                  'a.jpg\t[{"transcription": null, "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}]',
                                  'a.jpg\t[{"transcription": "x", "points": [[0, 0], [1, 1]]}]'])
def test_invalid_annotation_reports_line(tmp_path, line):
    path = tmp_path / 'Label.txt'
    path.write_text('first.jpg\t[]\n' + line + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match=r'Label.txt:2: expected'):
        load_annotations(path)


def test_duplicate_annotation_key_rejected(tmp_path):
    path = tmp_path / 'Label.txt'
    path.write_text('train/a.jpg\t[]\ntest/a.png\t[]\n', encoding='utf-8')
    with pytest.raises(ValueError, match="Label.txt:2: duplicate annotation for image key 'a'"):
        load_annotations(path)


def test_image_key():
    assert image_key('dir/Nikon_024_LR4.png', ['_LR4', '_HR']) == 'Nikon_024'
    assert image_key('Nikon_024_HR.png', ['_LR4', '_HR']) == 'Nikon_024'
    assert image_key('a\\b\\x_AIGC_INPUT.jpg', ['_INPUT', '_AIGC_INPUT']) == 'x'
    assert image_key('_HR.png', ['_HR']) == '_HR'
    assert image_key('19700107_112359_541_iso400_20.0X_CMM_05_AIGC_INPUT.jpg') == \
        '19700107_112359_541_iso400_20.0X_CMM_05_AIGC_INPUT'


def test_normalize_text():
    assert normalize_text('国内统一刊号：ＣＮ61 -1027/N　') == '国内统一刊号:CN61-1027/N'
    assert normalize_text('Shoujibao.CN') == 'Shoujibao.CN'
    assert normalize_text('Shoujibao.CN', ignore_case=True) == 'shoujibao.cn'


@pytest.mark.parametrize('a, b, distance', [('', '', 0), ('abc', '', 3), ('kitten', 'sitting', 3),
                                            ('ab', 'ba', 2), ('手機報在线', '手机報在线', 1)])
def test_edit_distance(a, b, distance):
    assert edit_distance(a, b) == edit_distance(b, a) == distance


def test_ocr_scores():
    # Edit distances 0, 1, 2, 4 over 12 label characters; NED terms 0/4, 1/4, 2/2, 4/4.
    scores = ocr_scores([('abcd', 'abcd'), ('abc', 'abcd'), ('', 'xy'), ('xyzw', 'ab')])
    assert scores['OCR-A'] == pytest.approx(1 / 4)
    assert scores['CER'] == pytest.approx(7 / 12)
    assert scores['1-NED'] == pytest.approx(1 - (0 + 0.25 + 1 + 1) / 4)


def test_crop_axis_aligned_box_in_either_order():
    image = np.random.default_rng(0).integers(0, 256, (100, 200, 3), dtype=np.uint8)
    box = np.float32([[50, 20], [150, 20], [150, 40], [50, 40]])
    crop = crop_text_region(image, box)
    assert crop.flags['C_CONTIGUOUS']
    np.testing.assert_array_equal(crop, image[20:40, 50:150])
    np.testing.assert_array_equal(crop_text_region(image, box[[0, 3, 2, 1]]), crop)  # counter-clockwise


def test_crop_tall_box_and_polygon():
    image = np.random.default_rng(1).integers(0, 256, (100, 200, 3), dtype=np.uint8)
    # Vertical text reading top to bottom becomes a left-to-right line.
    np.testing.assert_array_equal(crop_text_region(image, [[40, 10], [60, 10], [60, 90], [40, 90]]),
                                  np.rot90(image[10:90, 40:60]))
    polygon = [[50, 20], [100, 20], [150, 20], [150, 40], [100, 40], [50, 40]]
    np.testing.assert_array_equal(crop_text_region(image, polygon), image[20:40, 50:150])
    assert crop_text_region(image, [[10, 10]] * 4).shape == (1, 1, 3)


def test_scale_points():
    np.testing.assert_allclose(scale_points(np.float32([[40, 20], [400, 200]]), (400, 200), (100, 50)),
                               [[10, 5], [100, 50]])


class StubRecognizer:
    """Reads a crop as its label when bright, and as "wrong" when mostly dark."""

    def __init__(self, labels):
        self.labels = labels
        self.calls = []

    def __call__(self, crops):
        self.calls.append([crop.shape for crop in crops])
        return [(label if crop.mean() > 128 else 'wrong', 0.9) for label, crop in zip(self.labels, crops)]


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    for name in ('gt', 'lq', 'pred'):
        (tmp_path / name).mkdir()
    for key in ('a', 'b'):
        gt = np.full((40, 80, 3), 255, np.uint8)
        pred = gt.copy()
        pred[20:30, 40:70] = 0  # the second text box of image "a" is unreadable in pred
        Image.fromarray(gt).save(tmp_path / 'gt' / f'{key}_HR.png')
        Image.fromarray(pred if key == 'a' else gt).save(tmp_path / 'pred' / f'{key}_LR4.png')
        Image.fromarray(gt[::4, ::4]).save(tmp_path / 'lq' / f'{key}_LR4.png')
    boxes = [{'transcription': '手機報', 'points': [[4, 4], [36, 4], [36, 14], [4, 14]], 'difficult': False},
             {'transcription': 'CN61-1027', 'points': [[40, 20], [70, 20], [70, 30], [40, 30]], 'difficult': False},
             {'transcription': 'hard', 'points': [[0, 0], [8, 0], [8, 8], [0, 8]], 'difficult': True},
             {'transcription': '###', 'points': [[0, 30], [8, 30], [8, 38], [0, 38]], 'difficult': False}]
    (tmp_path / 'Label.txt').write_text(f'imgs/a_LR4.jpg\t{json.dumps(boxes, ensure_ascii=False)}\n', encoding='utf-8')
    stub = StubRecognizer(['手機報', 'CN61-1027', 'hard'])
    monkeypatch.setattr(evaluate, 'PaddleTextRecognizer', lambda *args: stub)
    fake = {name: (lambda offset: lambda pred, gt: (pred - gt).abs().mean() + offset)(i)
            for i, name in enumerate(evaluate.FR_METRICS)}
    monkeypatch.setattr(evaluate, 'build_fr_metrics', lambda device, rgb=False: fake)
    monkeypatch.chdir(tmp_path)
    return stub


ARGS = ['--pred', 'pred', '--gt', 'gt', '--lq', 'lq', '--ann', 'Label.txt', '--strip-suffix', '_LR4', '_HR',
        '--device', 'cpu']


def test_end_to_end_with_all_inputs(dataset):
    summary = evaluate.main(ARGS + ['--ann-ref', 'gt', '--output', 'out'])
    assert summary['num_images'] == 2
    dark = 10 * 30 * 255 / (40 * 80 * 255)  # mean |pred - gt| of image "a" in [0, 1]
    assert summary['fr_iqa']['PSNR'] == pytest.approx(dark / 2)
    assert summary['fr_iqa']['DISTS'] == pytest.approx(dark / 2 + 3)
    ocr = summary['ocr']
    assert (ocr['num_images'], ocr['num_instances'], ocr['ignored_instances']) == (1, 2, 2)
    assert ocr['pred'] == pytest.approx({'OCR-A': 0.5, 'CER': 9 / 12, '1-NED': 0.5})
    assert ocr['lq'] == ocr['gt'] == {'OCR-A': 1.0, 'CER': 0.0, '1-NED': 1.0}
    # pred, lq, gt crops of the two evaluated boxes; lq boxes are scaled down by 4.
    assert dataset.calls == [[(10, 32, 3), (10, 30, 3)], [(2, 8, 3), (2, 7, 3)], [(10, 32, 3), (10, 30, 3)]]

    with open('out/per_image.csv', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    assert [row['name'] for row in rows] == ['a_LR4.png', 'b_LR4.png']
    assert float(rows[0]['pred_OCR-A']) == 0.5 and rows[0]['num_text'] == '2'
    assert rows[1]['num_text'] == '' and rows[1]['pred_OCR-A'] == '' and float(rows[1]['PSNR']) == 0
    details = [json.loads(line) for line in open('out/ocr_details.jsonl', encoding='utf-8')]
    assert [(d['index'], d['label'], d['pred']['text'], d['pred']['edit_distance']) for d in details] == \
        [(0, '手機報', '手機報', 0), (1, 'CN61-1027', 'wrong', 9)]
    assert json.loads(open('out/summary.json', encoding='utf-8').read())['ocr'] == ocr


def test_ocr_only_and_options(dataset):
    summary = evaluate.main(['--pred', 'pred', '--ann', 'Label.txt', '--strip-suffix', '_LR4', '_HR',
                             '--include-difficult'])
    assert 'fr_iqa' not in summary and set(summary['ocr']) >= {'pred'} and 'lq' not in summary['ocr']
    assert (summary['ocr']['num_instances'], summary['ocr']['ignored_instances']) == (3, 1)


def test_different_sizes_need_ann_ref(dataset):
    with pytest.raises(ValueError, match=r'image sizes differ \(pred 80x40, lq 20x10, gt 80x40\); set --ann-ref'):
        evaluate.main(ARGS)


def test_missing_reference_and_size_mismatch(dataset, tmp_path):
    (tmp_path / 'gt' / 'b_HR.png').unlink()
    with pytest.raises(FileNotFoundError, match='No gt image for 1 predictions, e.g. b_LR4.png'):
        evaluate.main(ARGS + ['--ann-ref', 'gt'])
    Image.new('RGB', (81, 40)).save(tmp_path / 'gt' / 'b_HR.png')
    with pytest.raises(ValueError, match=r'b_LR4.png: pred size \(80, 40\) differs from gt size \(81, 40\)'):
        evaluate.main(ARGS + ['--ann-ref', 'gt'])


@pytest.mark.parametrize('argv, message', [([], 'nothing to evaluate'),
                                           (['--ann', 'x', '--ann-ref', 'lq'], 'requires --lq'),
                                           (['--gt', 'x', '--crop-border', '-1'], '--crop-border')])
def test_invalid_arguments(argv, message, capsys):
    with pytest.raises(SystemExit):
        evaluate.parse_args(['--pred', 'p'] + argv)
    assert message in capsys.readouterr().err
