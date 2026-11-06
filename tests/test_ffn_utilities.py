import json

import pytest
from torch import nn

from tools.benchmark_ffn_export import check_widths
from tools.mine_text_hard_cases import mine, names_from_file


def test_physical_width_check_rejects_wrong_width():
    model = nn.Module()
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.w12, block.mlp.w3 = nn.Linear(8, 8), nn.Linear(4, 8)
    model.blocks = nn.ModuleList([block])
    assert check_widths(model, 4)[0]['hidden'] == 4
    with pytest.raises(ValueError, match='expected 8'):
        check_widths(model, 8)
    block.mlp.fade_scale = 0
    with pytest.raises(ValueError, match='finished narrow'):
        check_widths(model, 4)


def test_mining_only_training_and_exact_geometry(tmp_path):
    row = dict(image='page.02.png', index=0, label='测试', norm_label='测试', size=40.,
               quad=[[0., 0.], [80., 0.], [80., 40.], [0., 40.]],
               pred={'edit_distance': 0, 'score': .9})
    teacher, student = tmp_path / 'teacher.jsonl', tmp_path / 'student.jsonl'
    teacher.write_text(json.dumps(row, ensure_ascii=False) + '\n')
    student.write_text(json.dumps(dict(row, pred={'edit_distance': 1, 'score': .8}), ensure_ascii=False) + '\n')
    names = tmp_path / 'train.txt'
    names.write_text('page.02\n')
    train = names_from_file(names)
    assert train == {'page.02'}
    boxes, report = mine(student, teacher, train)
    assert report['num_lines'] == 1
    assert boxes['page.02'][0]['teacher_correct'] is True
    assert boxes['page.02'][0]['alignment_valid'] is False  # OCR correctness does not establish alignment
    assert boxes['page.02'][0]['quad'] == [0., 0., 80., 0., 80., 40., 0., 40.]
    with pytest.raises(ValueError, match='outside the training split'):
        mine(student, teacher, {'train_other'})
    with pytest.raises(ValueError, match='overlap'):
        mine(student, teacher, train, train)
    annotation = tmp_path / 'boxes.json'
    annotation.write_text(json.dumps({'page.02': [dict(bbox=[0, 0, 80, 40], transcription='测试',
                                                     alignment_valid=True)]}, ensure_ascii=False))
    legacy = dict(row)
    legacy.pop('quad')
    teacher.write_text(json.dumps(legacy, ensure_ascii=False) + '\n')
    mined, _ = mine(student, teacher, train, boxes=annotation)
    assert mined['page.02'][0]['alignment_valid'] is True
    annotation.write_text(json.dumps({'page.02': [dict(bbox=[0, 0, 80, 40], transcription='测试',
                                                     alignment_valid=False)]}, ensure_ascii=False))
    with pytest.raises(ValueError, match='No eligible hard cases'):
        mine(student, teacher, train, boxes=annotation, assume_aligned=True)
    teacher.write_text(json.dumps(row, ensure_ascii=False) + '\n')
    annotation.write_text(json.dumps({'page.02': {'alignment_valid': False, 'lines': [row['quad']]}}))
    with pytest.raises(ValueError, match='No eligible hard cases'):
        mine(student, teacher, train, boxes=annotation, assume_aligned=True)
