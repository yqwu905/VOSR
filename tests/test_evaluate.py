"""evaluate.py OCR text-size buckets; PaddleOCR is replaced by a fake recognizer."""
import json
import numpy as np
import pytest
from PIL import Image
import evaluate


def test_text_size_is_the_shorter_side_of_the_box():
    assert evaluate.text_size([[0, 0], [100, 0], [100, 20], [0, 20]]) == pytest.approx(20)  # horizontal line
    assert evaluate.text_size([[0, 0], [0, 100], [20, 100], [20, 0]]) == pytest.approx(20)  # vertical, counter-clockwise
    assert evaluate.text_size([[0, 0], [50, 0], [100, 0], [100, 30], [50, 30], [0, 30]]) == pytest.approx(30)
    assert evaluate.size_buckets([16, 32]) == [(0., 16.), (16., 32.), (32., float('inf'))]


class FakeRecognizer:
    """Reads every crop as 'ab'."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, crops):
        return [('ab', 1.) for _ in crops]


def box(x, y, width, height, text):
    return {'transcription': text, 'points': [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]}


def test_ocr_metrics_per_text_size(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluate, 'PaddleTextRecognizer', FakeRecognizer)
    (tmp_path / 'pred').mkdir()
    (tmp_path / 'lq').mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (160, 200, 3), dtype=np.uint8)).save(tmp_path / 'pred/a.png')
    Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(tmp_path / 'lq/a.jpg')
    # LQ coordinates, 4x smaller than pred: pred line heights 12, 20, 28 and 40.
    boxes = [box(1, 1, 30, 3, 'ab'), box(1, 6, 30, 5, 'ac'), box(1, 13, 30, 7, 'ab'), box(1, 22, 30, 10, 'xy')]
    (tmp_path / 'Label.txt').write_text('a.jpg\t' + json.dumps(boxes) + '\n', encoding='utf-8')
    summary = evaluate.main(['--pred', str(tmp_path / 'pred'), '--lq', str(tmp_path / 'lq'),
                             '--ann', str(tmp_path / 'Label.txt'), '--ann-ref', 'lq',
                             '--size-buckets', '16', '32', '64', '--output', str(tmp_path / 'eval')])
    buckets = summary['ocr_by_size']
    assert [b['size'] for b in buckets] == [[0, 16], [16, 32], [32, 64]]  # empty >=64 bucket left out
    assert [b['num_instances'] for b in buckets] == [1, 2, 1]
    assert [b['pred']['OCR-A'] for b in buckets] == [1., .5, 0.]
    assert [b['pred']['CER'] for b in buckets] == [0., .25, 1.]
    assert buckets[1]['lq'] == buckets[1]['pred']  # every source is bucketed by the same boxes
    assert summary['ocr']['num_instances'] == 4 and summary['ocr']['pred']['CER'] == pytest.approx(3 / 8)
    sizes = [json.loads(line)['size'] for line in (tmp_path / 'eval/ocr_details.jsonl').read_text().splitlines()]
    assert sizes == pytest.approx([12, 20, 28, 40])
    assert 'text size 16-32 px, 2 boxes' in capsys.readouterr().out


@pytest.mark.parametrize('edges', [['32', '16'], ['0', '16'], ['16', '16']])
def test_size_buckets_need_ascending_positive_edges(edges):
    with pytest.raises(SystemExit):
        evaluate.parse_args(['--pred', 'p', '--ann', 'a', '--size-buckets', *edges])
    with pytest.raises(SystemExit):
        evaluate.parse_args(['--pred', 'p', '--gt', 'g', '--size-buckets', '16'])  # no --ann
