from copy import deepcopy
import fcntl
import json
import subprocess
import sys
import threading
import time

import pytest

from experiments.compare import METRICS
from experiments.run_study import confirmation_groups, rank_key
from experiments.run_final_test import collect_frozen, evaluate_parallel, load_frozen_study


def entry(name, recipe, probability, ned, stage='processing', seed=42, steps=500):
    return dict(name=name, recipe=recipe, anyword_probability=probability, seed=seed, steps=steps,
                stage=stage, checkpoint=f'/fixture_exports/{name}',
                metrics=dict(zip(METRICS, [30, .9, .1, .1, .7, .2, ned])))


def frozen_fixture():
    protocol = dict(processing_screen=dict(recipes=['a', 'b'], seed=42, anyword_probability=.5, steps=500),
                    mixture_screen=dict(anyword_probabilities=[0, .25, .5, .75, 1], steps=500),
                    confirmation=dict(steps=2000, seeds=[42, 123, 2026]))
    processing = dict(ranking=[entry('a_s42', 'a', .5, .8), entry('b_s42', 'b', .5, .79)],
                      selected=['a_s42', 'b_s42'])
    mixtures = []
    for i, base in enumerate(processing['ranking']):
        for probability in protocol['mixture_screen']['anyword_probabilities']:
            if probability == .5:
                mixtures.append(deepcopy(base))
            else:
                name = f'{base["recipe"]}_aw{round(probability*100):03}_s42'
                mixtures.append(entry(name, base['recipe'], probability, .805-i*.02+probability*.001, 'mixture'))
    mixtures.sort(key=rank_key)
    mixture = dict(ranking=mixtures, selected=[r['name'] for r in mixtures[:2]])
    confirmed = []
    for index, base in enumerate(mixtures[:2]):
        combination = f'{base["recipe"]}_aw{round(base["anyword_probability"]*100):03}'
        for seed in protocol['confirmation']['seeds']:
            row = entry(f'{combination}_confirm_s{seed}', base['recipe'], base['anyword_probability'],
                        .82-index*.01, 'confirmation', seed, 2000)
            confirmed.append(dict(row, combination=combination))
    final = dict(all_runs=confirmed, ranking_by_seed_mean=confirmation_groups(confirmed),
                 best_trained_checkpoint=confirmed[0]['checkpoint'], canonical_export_seed=42,
                 official_test_used_for_selection=False)
    return protocol, processing, mixture, final


def test_frozen_collection_covers_every_source_ratio_and_confirmation_seed():
    protocol, processing, mixture, final = frozen_fixture()
    rows, confirmed = collect_frozen(protocol, processing, mixture, final)
    assert len(rows) == 16  # Two processing + eight new mixtures + six confirmations.
    assert len({r['name'] for r in rows}) == 16
    assert len(confirmed) == 6
    for base in processing['ranking']:
        assert sum(r['name'] == base['name'] for r in rows) == 1
    bad = deepcopy(mixture)
    bad['ranking'].pop()
    with pytest.raises(ValueError, match='missing, extra or duplicate'):
        collect_frozen(protocol, processing, bad, final)


def test_frozen_collection_rejects_budget_seed_or_selection_changes():
    protocol, processing, mixture, final = frozen_fixture()
    bad = deepcopy(processing)
    bad['ranking'][0]['steps'] = 200
    with pytest.raises(ValueError, match='training budget'):
        collect_frozen(protocol, bad, mixture, final)
    bad = deepcopy(final)
    bad['all_runs'].pop()
    with pytest.raises(ValueError, match='confirmation seed coverage'):
        collect_frozen(protocol, processing, mixture, bad)
    bad = deepcopy(final)
    bad['official_test_used_for_selection'] = True
    with pytest.raises(ValueError, match='validation-only'):
        collect_frozen(protocol, processing, mixture, bad)
    bad = deepcopy(final)
    bad['best_trained_checkpoint'] = bad['all_runs'][1]['checkpoint']
    with pytest.raises(ValueError, match='representative'):
        collect_frozen(protocol, processing, mixture, bad)


def test_reused_half_half_run_cannot_be_replaced_between_stages():
    protocol, processing, mixture, final = frozen_fixture()
    reused = next(r for r in mixture['ranking'] if r['name'] == processing['ranking'][0]['name'])
    reused['checkpoint'] = '/different_export'
    with pytest.raises(ValueError, match='changed between selection stages'):
        collect_frozen(protocol, processing, mixture, final)


def test_final_test_requires_all_frozen_selection_stages(tmp_path):
    state = tmp_path/'orchestration'
    state.mkdir()
    (state/'processing_selection.json').write_text('{}')
    with pytest.raises(RuntimeError, match='must be frozen before final testing'):
        load_frozen_study(tmp_path)
    for name in ['mixture_selection.json', 'final_validation_selection.json']:
        (state/name).write_text('{}')
    with pytest.raises(RuntimeError, match='complete validation reports'):
        load_frozen_study(tmp_path)


def test_active_coordinator_lock_prevents_a_second_test_queue(tmp_path):
    state = tmp_path/'orchestration'
    state.mkdir()
    with (state/'.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, '-m', 'experiments.run_final_test',
                                 '--root', str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'coordinator is still active' in result.stderr
    assert not (state/'parallel_test_execution.json').exists()


def test_parallel_evaluation_is_bounded_and_records_every_result_in_input_order(tmp_path):
    rows = [entry(name, 'fixture', .5, .8) for name in ['b', 'a', 'c']]
    lock, barrier = threading.Lock(), threading.Barrier(2)
    active = maximum = 0
    destinations = []

    def evaluate(row, dataset, destination):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            destinations.append(destination)
        if row['name'] in ['b', 'a']:
            barrier.wait(timeout=5)
        time.sleep(.02)
        with lock:
            active -= 1
        return dict(row)

    result = evaluate_parallel(rows, tmp_path/'dataset', tmp_path/'outputs', tmp_path/'state',
                               'test', workers=2, evaluator=evaluate)
    assert maximum == 2
    assert [r['name'] for r in result] == ['b', 'a', 'c']
    assert len(set(destinations)) == 3
    assert json.loads((tmp_path/'state/test_results.json').read_text()) == result
    with pytest.raises(ValueError, match='own output directory'):
        evaluate_parallel(rows[:1]*2, tmp_path, tmp_path, tmp_path, 'bad', evaluator=evaluate)
