import json
import subprocess
import sys

import pytest

from experiments.run_study import (confirmation_groups, derived_config, freeze,
                                   rank_key, require_selection_validation, validate_paired_targets,
                                   wait_ready)


def result(name, ned=0.8, cer=0.2, lpips=0.1, psnr=30, seed=42):
    return dict(name=name, seed=seed, combination='recipe_aw075',
                metrics={'PSNR': psnr, 'SSIM': 0.9, 'LPIPS': lpips, 'DISTS': 0.1,
                         'OCR-A': 0.7, 'CER': cer, '1-NED': ned})


def test_selection_follows_text_priority_and_precommitted_ties():
    entries = [result('better_psnr', ned=0.79, psnr=40),
               result('better_cer', cer=0.15), result('better_ned', ned=0.81),
               result('better_lpips', cer=0.15, lpips=0.08),
               result('last_tiebreak', cer=0.15, lpips=0.08, psnr=31)]
    assert [r['name'] for r in sorted(entries, key=rank_key)] == [
        'better_ned', 'last_tiebreak', 'better_lpips', 'better_cer', 'better_psnr']
    invalid = result('invalid', ned=float('nan'))
    with pytest.raises(ValueError, match='finite'):
        rank_key(invalid)


def test_ratio_and_longer_runs_preserve_initialization_and_training_recipe():
    base = {'checkpoint': 'released_initial',
            'data': {'source_weights': {'anyword3m': 0.5, 'easytext': 0.5}, 'target': 'original'},
            'training': {'seed': 42, 'max_steps': 500, 'output_dir': 'old',
                         'learning_rate': 2e-6, 'teacher_weight': 0.1}}
    derived = derived_config(base, 0, 123, 2000, 'new_confirm')
    assert derived['checkpoint'] == base['checkpoint']
    assert derived['data']['source_weights'] == {'anyword3m': 0, 'easytext': 1}
    assert derived['training']['learning_rate'] == base['training']['learning_rate']
    assert derived['training']['teacher_weight'] == base['training']['teacher_weight']
    assert derived['training']['max_steps'] == 2000
    assert derived['training']['seed'] == 123
    assert base['data']['source_weights']['anyword3m'] == 0.5
    assert base['training']['max_steps'] == 500
    with pytest.raises(ValueError, match='probability'):
        derived_config(base, -0.1, 42, 500, 'invalid')


def test_confirmation_uses_all_seeds_and_sample_standard_deviation():
    rows = [result('a', ned=0.7, seed=42), result('b', ned=0.8, seed=123),
            result('c', ned=0.9, seed=2026)]
    group = confirmation_groups(rows)[0]
    assert group['metrics']['1-NED'] == pytest.approx(0.8)
    assert group['seed_std']['1-NED'] == pytest.approx(0.1)
    with pytest.raises(ValueError, match='exactly seeds'):
        confirmation_groups(rows[:2])
    with pytest.raises(ValueError, match='exactly seeds'):
        confirmation_groups([rows[0], rows[0], rows[2]])


def test_test_data_and_changed_frozen_decisions_are_rejected(tmp_path):
    (tmp_path/'protocol.json').write_text(json.dumps({'archive_split': 'val'}))
    with pytest.raises(ValueError, match='Official test'):
        require_selection_validation(tmp_path)
    (tmp_path/'protocol.json').write_text(json.dumps({'purpose': 'final_test'}))
    with pytest.raises(ValueError, match='Official test'):
        require_selection_validation(tmp_path)
    path = tmp_path/'selection.json'
    freeze(path, {'selected': 'a'})
    freeze(path, {'selected': 'a'})
    with pytest.raises(ValueError, match='Frozen study artifact changed'):
        freeze(path, {'selected': 'b'})
    assert json.loads(path.read_text()) == {'selected': 'a'}


def test_teacher_reference_requires_identical_crop_and_annotation_correspondence(tmp_path):
    row = dict(id='image_text_00', source='anyword3m', group_id='parent', source_sha256='hash',
               split='train', crop_mode='text', crop_rect=[0, 0, 512, 512], canvas_transform=[],
               annotations=[{'text': 'EXAMPLE'}], resolution=512, target='original')
    original, teacher = tmp_path/'original.jsonl', tmp_path/'teacher.jsonl'
    original.write_text(json.dumps(row)+'\n')
    other = dict(row, target='vosr2')
    teacher.write_text(json.dumps(other)+'\n')
    assert validate_paired_targets(original, teacher)['crops'] == 1
    other['annotations'] = [{'text': 'DIFFERENT'}]
    teacher.write_text(json.dumps(other)+'\n')
    with pytest.raises(ValueError, match='annotations'):
        validate_paired_targets(original, teacher)
    teacher.write_text('')
    with pytest.raises(ValueError, match='nonempty'):
        validate_paired_targets(original, teacher)


def test_failed_producer_does_not_leave_an_infinite_wait(tmp_path):
    producer = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(1)'])
    producer.wait()
    with pytest.raises(RuntimeError, match='exited before completing'):
        wait_ready([tmp_path/'never_written.json'], producer.pid)
    with pytest.raises(RuntimeError, match='no producer PID'):
        wait_ready([tmp_path/'never_written.json'])
