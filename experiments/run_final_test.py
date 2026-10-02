"""Evaluate a fully frozen study with at most two concurrent GPU/CPU jobs.

This is an optional handoff from run_study after all training/validation ends.
It takes the same coordinator lock and verifies the frozen choices and their
underlying artifacts before reading any final test results. Training is never
started here. Each model keeps its original inference and evaluation commands.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import time

from data_pipeline.pipeline import atomic_json, file_digest
from experiments.compare import METRICS, load_evaluation
from experiments.run_study import (read, rank_key, screen_entry, confirmation_groups,
                                   evaluate_checkpoint, compare_entries, write_table,
                                   require_selection_validation, process_identity)


def ranked_stage(stage, expected):
    rows = stage['ranking']
    names = [r['name'] for r in rows]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError('Frozen stage has missing, extra or duplicate experiments')
    if names != [r['name'] for r in sorted(rows, key=rank_key)]:
        raise ValueError('Frozen stage is not ranked by the committed validation rule')
    if stage['selected'] != names[:2]:
        raise ValueError('Frozen stage selection disagrees with its ranking')
    for row in rows:
        recipe, probability, steps, seed = expected[row['name']]
        if (row['recipe'], row['anyword_probability'], row['steps'], row['seed']) != \
                (recipe, probability, steps, seed):
            raise ValueError('Frozen stage changed recipe, source ratio or training budget')
    return rows


def collect_frozen(protocol, processing, mixture, final):
    """Validate complete recipe/ratio/seed coverage before permitting testing."""
    p = protocol['processing_screen']
    processing_rows = ranked_stage(processing, {
        f'{recipe}_s{p["seed"]}': (recipe, p['anyword_probability'], p['steps'], p['seed'])
        for recipe in p['recipes']})
    m = protocol['mixture_screen']
    expected = {}
    for entry in processing_rows[:2]:
        for probability in m['anyword_probabilities']:
            name = entry['name'] if probability == 0.5 else f'{entry["recipe"]}_aw{round(probability*100):03}_s42'
            expected[name] = (entry['recipe'], probability, m['steps'], p['seed'])
    mixture_rows = ranked_stage(mixture, expected)
    c = protocol['confirmation']
    expected_confirmed = {}
    for entry in mixture_rows[:2]:
        combination = f'{entry["recipe"]}_aw{round(entry["anyword_probability"]*100):03}'
        for seed in c['seeds']:
            expected_confirmed[f'{combination}_confirm_s{seed}'] = (
                entry['recipe'], entry['anyword_probability'], c['steps'], seed, combination)
    confirmed = final['all_runs']
    names = [r['name'] for r in confirmed]
    if len(names) != len(set(names)) or set(names) != set(expected_confirmed):
        raise ValueError('Final selection lacks complete confirmation seed coverage')
    for row in confirmed:
        if (row['recipe'], row['anyword_probability'], row['steps'], row['seed'], row['combination']) != \
                expected_confirmed[row['name']]:
            raise ValueError('Confirmation configuration differs from the selected mixture')
    groups = confirmation_groups(confirmed)
    if groups != final['ranking_by_seed_mean']:
        raise ValueError('Confirmation seed means disagree with frozen ranking')
    if final['official_test_used_for_selection'] is not False or final['canonical_export_seed'] != 42:
        raise ValueError('Final evaluation requires validation-only selection and fixed export seed')
    representative = next(r for r in confirmed if r['combination'] == groups[0]['name'] and r['seed'] == 42)
    if representative['checkpoint'] != final['best_trained_checkpoint']:
        raise ValueError('Chosen export differs from the frozen recipe representative')
    all_rows = {}
    for row in processing_rows + mixture_rows + confirmed:
        if row['name'] in all_rows and all_rows[row['name']] != row:
            raise ValueError('A reused screening run changed between selection stages')
        all_rows[row['name']] = row
    return list(all_rows.values()), confirmed


def load_frozen_study(root):
    state = root/'orchestration'
    required = ['processing_selection.json', 'mixture_selection.json', 'final_validation_selection.json']
    reports = [f'{stage}_{kind}' for stage in ['processing', 'mixture', 'confirmation']
               for kind in ['comparisons.json', 'validation.csv']]
    if not all((state/name).is_file() for name in required+reports):
        raise RuntimeError('All choices must be frozen before final testing, with complete validation reports')
    protocol = read('configs/experiments/study_protocol.json')
    version = int(protocol.get('version', 1))
    snapshot = 'protocol_snapshot.json' if version == 1 else f'protocol_snapshot_v{version}.json'
    if read(state/snapshot) != protocol:
        raise ValueError('Current study protocol differs from its frozen version')
    final = read(state/'final_validation_selection.json')
    entries, confirmed = collect_frozen(protocol, read(state/required[0]), read(state/required[1]), final)
    pairs = require_selection_validation(root/'synthetic_validation')
    for entry in entries:
        actual = screen_entry(entry['config'], Path(entry['queue']), pairs)
        for key, value in actual.items():
            if entry[key] != value:
                raise ValueError(f'Frozen validation evidence changed: {entry["name"]}, {key}')
    baseline = load_evaluation(root/'metrics/vosr2_initial_synthetic_validation', pairs)
    baseline_metrics = dict(zip(METRICS, map(float, baseline['estimate'])))
    if baseline_metrics != final['baseline_metrics']:
        raise ValueError('Released-model validation baseline changed after selection')
    winner = final['ranking_by_seed_mean'][0]
    baseline_row = dict(name='vosr2_initial', metrics=baseline_metrics)
    recommended = final['best_trained_checkpoint'] if rank_key(winner) < rank_key(baseline_row) else 'preset/ckpts/VOSR2'
    if recommended != final['recommended_checkpoint_by_validation']:
        raise ValueError('Frozen recommendation is inconsistent with validation evidence')
    expected_test = Path(protocol['final_test']).resolve()
    if (root/'realce_test_x4/pairs.jsonl').resolve() != expected_test:
        raise ValueError('Final test inputs differ from the study protocol')
    return entries, confirmed, final


def evaluate_parallel(entries, dataset, output, state, prefix, workers=2, evaluator=evaluate_checkpoint):
    if workers not in (1, 2):
        raise ValueError('The observed 96GB GPU supports one or two evaluation jobs')
    names = [r['name'] for r in entries]
    if len(names) != len(set(names)):
        raise ValueError('Each evaluation job needs its own output directory')
    results = {}
    state.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(evaluator, entry, dataset, output/entry['name']): entry['name'] for entry in entries}
        try:
            for future in as_completed(jobs):
                name = jobs[future]
                results[name] = future.result()
                ordered = [results[n] for n in names if n in results]
                atomic_json(state/f'{prefix}_results.json', ordered)
                write_table(state/f'{prefix}_progress.csv', ordered)
                print(json.dumps(dict(event='evaluation_verified', dataset=str(dataset), model=name,
                                      completed=len(results), total=len(entries))), flush=True)
        except BaseException:
            for future in jobs:
                future.cancel()
            raise
    return [results[name] for name in names]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/vosr2_data_ablation'))
    p.add_argument('--workers', type=int, choices=[1, 2], default=2)
    p.add_argument('--wait-for-pid', type=int, help='Await an existing evaluation child during coordinator handoff')
    args = p.parse_args()
    root, state = args.root, args.root/'orchestration'
    lock = (state/'.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('The training/study coordinator is still active; no duplicate test queue started') from error
    if args.wait_for_pid == os.getpid():
        raise ValueError('Cannot await the current process')
    if args.wait_for_pid:
        identity = process_identity(args.wait_for_pid)
        while identity is not None and identity['state'] != 'Z':
            print(json.dumps(dict(event='awaiting_previous_evaluation', pid=args.wait_for_pid)), flush=True)
            time.sleep(30)
            current = process_identity(args.wait_for_pid)
            if current is not None and current['start_ticks'] != identity['start_ticks']:
                break
            identity = current
    entries, confirmed, final = load_frozen_study(root)
    selection_sha = file_digest(state/'final_validation_selection.json')
    atomic_json(state/'parallel_test_execution.json', dict(
        workers=args.workers, selection_sha256=selection_sha, model_names=[r['name'] for r in entries],
        code_sha256=file_digest(__file__), no_training_started=True))
    secondary = evaluate_parallel(confirmed, root/'realce_validation_x4', root/'secondary_validation',
                                  state, 'secondary', args.workers)
    write_table(state/'secondary_validation.csv', secondary)
    secondary_pairs = [json.loads(line) for line in (root/'realce_validation_x4/pairs.jsonl').read_text().splitlines()]
    secondary_base = load_evaluation(root/'metrics/vosr2_initial_realce_validation', secondary_pairs)
    compare_entries(secondary, secondary_base, secondary_pairs, state/'secondary_comparisons.json')
    tests = evaluate_parallel(entries, root/'realce_test_x4', root/'final_test', state, 'test', args.workers)
    write_table(state/'realce_test_trained.csv', tests)
    pairs = [json.loads(line) for line in (root/'realce_test_x4/pairs.jsonl').read_text().splitlines()]
    baseline = load_evaluation(root/'metrics/vosr2_initial_test_x4', pairs)
    compare_entries(tests, baseline, pairs, state/'test_comparisons.json')
    confirmed_test = [row for row in tests if row['stage'] == 'confirmation']
    test_groups = {row['name']: row for row in confirmation_groups(confirmed_test)}
    # Preserve validation-selected order; test metrics cannot redefine the winner.
    atomic_json(state/'test_confirmation_seed_summary.json',
                [test_groups[row['name']] for row in final['ranking_by_seed_mean']])
    references = []
    for name, directory in [('bicubic', 'bicubic_test_x4'), ('vosr2_initial', 'vosr2_initial_test_x4')]:
        evaluation = load_evaluation(root/'metrics'/directory, pairs)
        references.append(dict(name=name, stage='reference',
                          metrics=dict(zip(METRICS, map(float, evaluation['estimate'])))))
    write_table(state/'realce_test_all.csv', references+tests)
    if file_digest(state/'final_validation_selection.json') != selection_sha:
        raise ValueError('Validation selection changed during final testing')
    atomic_json(state/'complete.json', dict(trained_models=len(entries), test_images=len(pairs),
                test_groups=len(baseline['groups']), test_text_instances=len(baseline['labels']),
                best_trained_checkpoint=final['best_trained_checkpoint'], selection_sha256=selection_sha,
                report_table=str(state/'realce_test_all.csv'), evaluation_workers=args.workers))


if __name__ == '__main__':
    main()
