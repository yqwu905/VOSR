import json
from pathlib import Path

import pytest

from tools import evaluate_text_bins as evaluator


def write_rows(tmp_path, name, rows, settings=None):
    directory = tmp_path / name
    directory.mkdir()
    (directory / 'ocr_details.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    if settings is not None:
        (directory / 'summary.json').write_text(json.dumps({'settings': settings}), encoding='utf-8')
    return directory


def row(image='page.png', index=0, label='ab', height=32, edits=0):
    return {'image': image, 'index': index, 'label': label, 'norm_label': label,
            'size': height, 'pred': {'text': '', 'edit_distance': edits}}


def test_native_height_bins_pooled_cer_and_teacher_correct_retention(tmp_path):
    teacher = [row('a.png', label='abcdefghij', height=23, edits=0),
               row('b.png', label='x', height=24, edits=0),
               row('c.png', label='yy', height=64, edits=1)]
    student = [row('a.png', label='abcdefghij', height=23, edits=1),
               row('b.png', label='x', height=24, edits=0),
               row('c.png', label='yy', height=64, edits=0)]
    result = evaluator.evaluate([write_rows(tmp_path, 's', student)], [write_rows(tmp_path, 't', teacher)], bootstrap=0)
    assert result['all']['student_cer'] == pytest.approx(1 / 13)
    assert result['all']['teacher_correct_retention'] == .5
    assert result['all']['student_ocr_a'] == pytest.approx(2 / 3)
    assert [entry['num_instances'] for entry in result['height_bins']] == [1, 1, 0, 0, 1, 0]
    assert result['small_under_64']['num_instances'] == 2
    assert result['height_bins'][2]['student_cer'] is None


def test_bootstrap_resamples_images_not_boxes_and_repeated_seeds_are_averaged(tmp_path):
    teacher = [row(f'{i}.png', edits=0) for i in range(12)]
    student = [row(f'{i}.png', edits=i % 3) for i in range(12)]
    teacher_many = [{**value, 'index': j} for value in teacher for j in range(9)]
    student_many = [{**value, 'index': j} for value in student for j in range(9)]
    t, s = write_rows(tmp_path, 't', teacher), write_rows(tmp_path, 's', student)
    report = evaluator.evaluate([s], [t], bootstrap=300)
    duplicate = evaluator.evaluate([write_rows(tmp_path, 's9', student_many)],
                                   [write_rows(tmp_path, 't9', teacher_many)], bootstrap=300)
    assert report['all']['confidence_interval']['cer_delta'] == duplicate['all']['confidence_interval']['cer_delta']
    perfect = write_rows(tmp_path, 's_perfect', teacher)
    paired = evaluator.evaluate([s, perfect], [t], bootstrap=0)
    assert paired['all']['student_cer'] == pytest.approx(report['all']['student_cer'] / 2)
    assert paired['all']['teacher_correct_retention'] == pytest.approx((report['all']['teacher_correct_retention'] + 1) / 2)


def test_alignment_fails_closed_and_explicit_intersection_reports_exclusions(tmp_path):
    s = write_rows(tmp_path, 's', [row('a.png'), row('b.png')])
    t = write_rows(tmp_path, 't', [row('a.jpg')])
    with pytest.raises(ValueError, match='instance keys differ'):
        evaluator.evaluate([s], [t])
    result = evaluator.evaluate([s], [t], allow_intersection=True, bootstrap=0)
    assert result['inputs']['alignment']['excluded_instances_per_file'] == [1, 0]
    assert result['all']['num_instances'] == 1


@pytest.mark.parametrize('replacement, message', [({'norm_label': 'other'}, 'labels differ'),
                                                  ({'size': 31.0}, 'heights differ')])
def test_pair_labels_and_height_must_match(tmp_path, replacement, message):
    s = write_rows(tmp_path, 's', [row()])
    t = write_rows(tmp_path, 't', [{**row(), **replacement}])
    with pytest.raises(ValueError, match=message):
        evaluator.evaluate([s], [t])


@pytest.mark.parametrize('replacement,message', [
    ({'quad': [[10, 0], [90, 0], [90, 32], [10, 32]]}, 'quads differ'),
    ({'sr_size': [201, 120]}, 'SR output sizes differ'),
])
def test_geometry_mismatch_rejected_even_with_equal_label_height_and_legacy_first_run(tmp_path, replacement, message):
    annotated = {**row(), 'quad': [[0, 0], [80, 0], [80, 32], [0, 32]], 'sr_size': [200, 120]}
    legacy = write_rows(tmp_path, 'legacy', [row()])
    student = write_rows(tmp_path, 's', [annotated])
    teacher = write_rows(tmp_path, 't', [{**annotated, **replacement}])
    with pytest.raises(ValueError, match=message):
        evaluator.evaluate([student], [teacher])
    with pytest.raises(ValueError, match=message):
        evaluator.evaluate([legacy, student], [teacher])


def test_geometry_tolerance_and_missing_legacy_geometry_are_reported(tmp_path):
    annotated = {**row(), 'quad': [[0, 0], [80, 0], [80, 32], [0, 32]], 'sr_size': [200, 120]}
    rounded = {**annotated, 'quad': [[.00001, 0], [80.00001, 0], [80.00001, 32], [.00001, 32]]}
    s = write_rows(tmp_path, 's', [annotated])
    t = write_rows(tmp_path, 't', [rounded])
    report = evaluator.evaluate([s], [t], bootstrap=0)
    assert report['inputs']['alignment']['geometry_verified'] is True
    assert report['inputs']['alignment']['geometry_unverified_instances'] == 0
    legacy = write_rows(tmp_path, 'legacy', [row()])
    report = evaluator.evaluate([legacy], [t], bootstrap=0)
    assert report['inputs']['alignment']['geometry_verified'] is False
    assert report['inputs']['alignment']['geometry_unverified_instances'] == 1
    assert report['inputs']['alignment']['geometry_verified_instances'] == 0
    assert report['inputs']['alignment']['geometry_missing_instances_per_file'] == [1, 0]


@pytest.mark.parametrize('geometry', [
    {'quad': [[0, 0], [80, 0], [80, float('nan')], [0, 32]]},
    {'quad': [[0, 0], [80, 0], [80, 32]]},
    {'sr_size': [200, float('inf')]},
    {'sr_size': [200, -1]},
    {'sr_size': [200, True]},
])
def test_present_geometry_must_be_numerically_valid(tmp_path, geometry):
    details = write_rows(tmp_path, 'bad', [{**row(), **geometry}])
    with pytest.raises(ValueError, match='quad|sr_size'):
        evaluator.load_details(details)


def test_recognizer_settings_and_duplicate_keys_are_checked(tmp_path):
    s = write_rows(tmp_path, 's', [row()], {'ocr_model': 'recognizer-A'})
    t = write_rows(tmp_path, 't', [row()], {'ocr_model': 'recognizer-B'})
    with pytest.raises(ValueError, match='ocr_model differs'):
        evaluator.evaluate([s], [t])
    duplicate = write_rows(tmp_path, 'duplicate', [row(), row()])
    with pytest.raises(ValueError, match='duplicate image/index'):
        evaluator.load_details(duplicate)


def test_ci_unavailable_for_insufficient_images_or_no_teacher_correct(tmp_path):
    s = write_rows(tmp_path, 's', [row(edits=0)])
    t = write_rows(tmp_path, 't', [row(edits=1)])
    result = evaluator.evaluate([s], [t])
    assert result['all']['teacher_correct_retention'] is None
    assert result['all']['confidence_interval']['cer_delta'] is None
    assert result['all']['interval_status'].startswith('insufficient_images')


def test_one_ned_uses_prediction_length_and_legacy_normalization(tmp_path):
    teacher = row()
    teacher['pred'] = {'edit_distance': 0, 'norm_text': 'ab'}
    student = row(edits=1)
    student['pred'] = {'edit_distance': 1, 'text': 'Ａ BＣ '}
    s = write_rows(tmp_path, 's', [student], {'ignore_case': True})
    t = write_rows(tmp_path, 't', [teacher], {'ignore_case': True})
    result = evaluator.evaluate([s], [t], bootstrap=0)
    assert result['all']['student_cer'] == .5
    assert result['all']['student_1_ned'] == pytest.approx(2 / 3)
    assert result['all']['teacher_1_ned'] == 1
    assert result['all']['one_ned_delta'] == pytest.approx(-1 / 3)


def test_one_ned_missing_prediction_is_unavailable_not_fabricated(tmp_path):
    missing = row()
    missing['pred'] = {'edit_distance': 0}
    s = write_rows(tmp_path, 's', [missing, row('other.png')])
    t = write_rows(tmp_path, 't', [row(), row('other.png')])
    result = evaluator.evaluate([s], [t], bootstrap=50, min_images=2)
    assert result['all']['student_1_ned'] is None
    assert result['all']['one_ned_delta'] is None
    assert result['all']['confidence_interval']['student_1_ned'] is None
    assert result['all']['student_cer'] == 0


def test_cli_no_overwrite_and_dotted_stems(tmp_path, capsys):
    assert evaluator.image_key('page.02') == 'page.02'
    assert evaluator.image_key('page.02.png') == 'page.02'
    s = write_rows(tmp_path, 's', [row()])
    t = write_rows(tmp_path, 't', [row()])
    output = tmp_path / 'report'
    args = ['--student-details', str(s), '--teacher-details', str(t), '--output', str(output), '--bootstrap', '0']
    evaluator.main(args)
    assert (output / 'text_bins.json').is_file()
    assert (output / 'text_bins.csv').is_file()
    with pytest.raises(FileExistsError):
        evaluator.main(args)


@pytest.mark.parametrize('edges', [[0, 24], [32, 24], [24, 24], [float('inf')]])
def test_invalid_bins_rejected(edges):
    with pytest.raises(ValueError, match='height edges'):
        evaluator.evaluate([], [], height_edges=edges)
