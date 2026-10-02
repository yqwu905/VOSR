"""Resume the precommitted data study, freeze validation choices, then test.

One training job runs at a time. Existing producer processes may be awaited by
PID; their files are consumed only after their atomic completion summaries.
This runner does not modify the training objective or use test scores to rank.
"""
import argparse
from copy import deepcopy
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from data_pipeline.pipeline import atomic_json, file_digest
from experiments.compare import METRICS, load_evaluation, paired_comparison
from experiments.run_screen import run


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, payload):
    path = Path(path)
    if path.exists() and read(path) != payload:
        raise ValueError(f'Frozen study artifact changed: {path}')
    atomic_json(path, payload)


def rank_key(record):
    metrics = record['metrics']
    if not all(np.isfinite(metrics[m]) for m in METRICS):
        raise ValueError('Selection requires finite values for every metric')
    return (-metrics['1-NED'], metrics['CER'], metrics['LPIPS'], -metrics['PSNR'], record['name'])


def require_selection_validation(directory):
    directory = Path(directory)
    protocol = read(directory / 'protocol.json')
    if protocol.get('purpose') == 'final_test' or protocol.get('archive_split') == 'val':
        raise ValueError('Official test data cannot select study configurations')
    return [json.loads(line) for line in (directory / 'pairs.jsonl').read_text().splitlines()]


def derived_config(base, probability, seed, steps, name):
    if not 0 <= probability <= 1:
        raise ValueError('AnyWord probability must lie in [0, 1]')
    cfg = deepcopy(base)
    cfg['data']['source_weights'] = dict(anyword3m=probability, easytext=1-probability)
    cfg['training'].update(seed=seed, max_steps=steps,
                           output_dir=f'exp_vosr/vosr2_data/{name}')
    # Every run starts from the released checkpoint. A longer confirmation is
    # an independent run, never a continuation of its 500-step screening run.
    return cfg


def wait_ready(paths, pid=None):
    paths = [Path(p) for p in paths]
    while not all(p.is_file() for p in paths):
        if pid is None:
            raise RuntimeError(f'Prerequisite incomplete and no producer PID supplied: {paths}')
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise RuntimeError(f'Producer {pid} exited before completing {paths}') from error
        identity = process_identity(pid)
        if identity is None or identity['state'] == 'Z':
            raise RuntimeError(f'Producer {pid} exited before completing {paths}')
        print(json.dumps(dict(event='awaiting_producer', pid=pid,
                              missing=[str(p) for p in paths if not p.is_file()])), flush=True)
        time.sleep(30)


def process_identity(pid):
    """Linux process start ticks disambiguate a stale PID after a resumed run."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(pid=pid, start_ticks=fields[19], state=fields[0])


def prepare_teacher_quality(root, teacher_pid, original_quality_pid):
    state = root/'orchestration'
    state.mkdir(parents=True, exist_ok=True)
    lock = (state/'teacher_quality.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    wait_ready([root/'quality_original/manifest.jsonl', root/'quality_original/summary.json'], original_quality_pid)
    wait_ready([root/'data_teacher/manifest.jsonl', root/'data_teacher/summary.json'], teacher_pid)
    freeze(state/'paired_targets_audit.json', validate_paired_targets(
           root/'data_original/manifest.jsonl', root/'data_teacher/manifest.jsonl'))
    if not (root/'quality_teacher/summary.json').is_file():
        run([sys.executable, '-u', '-m', 'experiments.score_data',
             '--manifest', str(root/'data_teacher/manifest.jsonl'), '--output', str(root/'quality_teacher'),
             '--reference-manifest', str(root/'quality_original/manifest.jsonl')], root/'logs/score_teacher.log')
    wait_ready([root/'quality_teacher/manifest.jsonl', root/'quality_teacher/summary.json'])
    quality = read(root/'quality_teacher/summary.json')['protocol']
    if quality['manifest_sha256'] != file_digest(root/'data_teacher/manifest.jsonl') or \
            quality['reference_sha256'] != file_digest(root/'quality_original/manifest.jsonl'):
        raise ValueError('Teacher quality evidence uses different candidates or references')


def launch_teacher_quality(root, teacher_pid, original_quality_pid):
    if (root/'quality_teacher/summary.json').is_file():
        return None
    record_path = root/'orchestration/teacher_score_process.json'
    if record_path.exists():
        old = read(record_path)
        current = process_identity(old['pid'])
        if current is not None and current['state'] != 'Z' and current['start_ticks'] == old.get('start_ticks'):
            return old['pid']
    command = [sys.executable, '-u', '-m', 'experiments.run_study', '--root', str(root), '--prepare-teacher-quality']
    for flag, pid in [('--teacher-pid', teacher_pid), ('--original-quality-pid', original_quality_pid)]:
        if pid is not None:
            command.extend([flag, str(pid)])
    logfile = root/'logs/teacher_quality_waiter.log'
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(process.pid)
    if identity is None:
        raise RuntimeError('Teacher quality supervisor exited during launch; inspect its log')
    atomic_json(record_path, dict(identity, log=str(logfile)))
    return process.pid


def check_disk(directory, minimum_gib=35):
    free = shutil.disk_usage(directory).free / 2**30
    if free < minimum_gib:
        raise RuntimeError(f'Insufficient checkpoint space: {free:.1f} GiB free; need {minimum_gib}')


def validate_paired_targets(original_path, teacher_path):
    def records(path):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        indexed = {row['id']: row for row in rows}
        if len(indexed) != len(rows) or not rows:
            raise ValueError('Matched data require unique, nonempty crop identifiers')
        return indexed
    original, teacher = records(original_path), records(teacher_path)
    if original.keys() != teacher.keys():
        raise ValueError('Original and teacher candidate coverage differs')
    fields = ['source', 'group_id', 'source_sha256', 'split', 'crop_mode',
              'crop_rect', 'canvas_transform', 'annotations', 'resolution']
    for key, row in original.items():
        other = teacher[key]
        if row['target'] != 'original' or other['target'] != 'vosr2':
            raise ValueError('Matched data have incorrect target provenance')
        for field in fields:
            if row[field] != other[field]:
                raise ValueError(f'Original/teacher correspondence changed: {key}, {field}')
    return dict(crops=len(original), identical_correspondence_fields=fields,
                original_sha256=file_digest(original_path), teacher_sha256=file_digest(teacher_path))


def screen(configs, validation, output):
    from experiments.training_data import ResearchDataset
    for path in configs:
        cfg = read(path)
        dataset = ResearchDataset(cfg['data'], length=1, seed=cfg['training']['seed'])
        print(json.dumps(dict(event='data_ready', config=str(path), sources=dataset.summary)), flush=True)
    if any(not (Path(read(p)['training']['output_dir'])/'complete.json').is_file() for p in configs):
        check_disk('.')
    run([sys.executable, '-u', '-m', 'experiments.run_screen', '--configs', *map(str, configs),
         '--validation', str(validation), '--output', str(output)], output/'queue.log')


def screen_entry(config_path, queue, pairs):
    cfg = read(config_path)
    training = Path(cfg['training']['output_dir'])
    complete, provenance = read(training/'complete.json'), read(training/'provenance.json')
    if provenance['config'] != cfg or complete['steps'] != cfg['training']['max_steps']:
        raise ValueError('Training configuration/completion mismatch')
    if (complete['parameters'], complete['cross_attention_blocks'], complete['router_count']) != (1393943616, 36, 0):
        raise ValueError('Architecture changed during this data-only study')
    if provenance['manifest_sha256'] != file_digest(cfg['data']['manifest']):
        raise ValueError('Training data changed after training')
    name = training.name
    directory = queue/name/'validation_metrics'
    evaluation = load_evaluation(directory, pairs)
    inference = read(queue/name/'validation_sr/complete.json')
    if inference['images'] != len(pairs):
        raise ValueError('Incomplete validation inference')
    weights = cfg['data']['source_weights']
    return dict(name=name, config=str(config_path), queue=str(queue), checkpoint=str(training/'export'),
                metrics_directory=str(directory), metrics=dict(zip(METRICS, map(float, evaluation['estimate']))),
                metrics_sha256=evaluation['artifact_sha256'], config_sha256=file_digest(config_path),
                weight_sha256=inference['protocol']['weight_sha256'], seed=cfg['training']['seed'],
                steps=complete['steps'], sampled_sources=complete['sampled_sources'],
                anyword_probability=weights.get('anyword3m', 0)/sum(weights.values()),
                data_summary=read(training/'data_summary.json'))


def compare_entries(entries, baseline, pairs, destination):
    results = {}
    for entry in entries:
        result = load_evaluation(entry['metrics_directory'], pairs)
        results[entry['name']] = paired_comparison(baseline, result)
    atomic_json(destination, dict(comparisons=results,
                interpretation='Paired source-group intervals, fixed trained seed; selection and seed uncertainty are separate.'))


def confirmation_groups(entries):
    groups = {}
    for entry in entries:
        groups.setdefault(entry['combination'], []).append(entry)
    result = []
    for name, rows in groups.items():
        seeds = [r['seed'] for r in rows]
        if sorted(seeds) != [42, 123, 2026]:
            raise ValueError('Confirmation requires exactly seeds 42, 123 and 2026')
        values = np.array([[r['metrics'][m] for m in METRICS] for r in rows])
        result.append(dict(name=name, metrics=dict(zip(METRICS, map(float, values.mean(0)))),
                           seed_std=dict(zip(METRICS, map(float, values.std(0, ddof=1)))),
                           members=[r['name'] for r in rows]))
    return sorted(result, key=rank_key)


def evaluate_checkpoint(entry, dataset, destination):
    prediction, metrics = destination/'sr', destination/'metrics'
    check_disk('.', 5)
    run([sys.executable, '-u', '-m', 'experiments.infer', '--checkpoint', entry['checkpoint'],
         '--pairs', str(dataset/'pairs.jsonl'), '--output', str(prediction)], destination/'infer.log')
    if not (metrics/'summary.json').exists():
        run([sys.executable, '-u', 'evaluate.py', '--pred', str(prediction), '--gt', str(dataset/'gt'),
             '--ann', str(dataset/'Label.txt'), '--ann-ref', 'gt', '--ocr-python', '.venv-ocr/bin/python',
             '--ocr-device', 'cpu', '--ocr-pred-only', '--strict', '--rgb', '--crop-border', '2',
             '--output', str(metrics)], destination/'evaluate.log')
    pairs = [json.loads(line) for line in (dataset/'pairs.jsonl').read_text().splitlines()]
    evaluation = load_evaluation(metrics, pairs)
    result = dict(entry, metrics_directory=str(metrics),
                  metrics=dict(zip(METRICS, map(float, evaluation['estimate']))),
                  metrics_sha256=evaluation['artifact_sha256'])
    atomic_json(destination/'verified.json', result)
    return result


def write_table(path, rows):
    fields = ['name', 'stage', 'recipe', 'anyword_probability', 'steps', 'seed', *METRICS, 'checkpoint']
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k, '') for k in fields if k not in METRICS}, **row['metrics']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/vosr2_data_ablation'))
    p.add_argument('--raw-pid', type=int)
    p.add_argument('--original-quality-pid', type=int)
    p.add_argument('--teacher-pid', type=int)
    p.add_argument('--prepare-teacher-quality', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--stop-after', choices=['processing', 'mixtures', 'confirmation', 'test'], default='test')
    args = p.parse_args()
    root, config_root = args.root, Path('configs/experiments')
    if args.prepare_teacher_quality:
        prepare_teacher_quality(root, args.teacher_pid, args.original_quality_pid)
        return
    state = root/'orchestration'
    state.mkdir(parents=True, exist_ok=True)
    lock = (state/'.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(config_root/'study_protocol.json')
    version = int(protocol.get('version', 1))
    snapshot = 'protocol_snapshot.json' if version == 1 else f'protocol_snapshot_v{version}.json'
    freeze(state/snapshot, protocol)
    validation = root/'synthetic_validation'
    pairs = require_selection_validation(validation)
    if file_digest(validation/'pairs.jsonl') != file_digest(protocol['selection_validation']):
        raise ValueError('Selection data differ from the committed protocol')
    baseline = load_evaluation(root/'metrics/vosr2_initial_synthetic_validation', pairs)
    quality_pid = launch_teacher_quality(root, args.teacher_pid, args.original_quality_pid)

    wait_ready([root/'screen_raw/queue_complete.json'], args.raw_pid)
    wait_ready([root/'quality_original/manifest.jsonl', root/'quality_original/summary.json'], args.original_quality_pid)
    original_configs = [config_root/f'vosr2_{r}.json' for r in ['original_ocr', 'original_ocr_iqa']]
    screen(original_configs, validation, root/'screen_original_filtered')

    wait_ready([root/'data_teacher/manifest.jsonl', root/'data_teacher/summary.json'], args.teacher_pid)
    freeze(state/'paired_targets_audit.json', validate_paired_targets(
           root/'data_original/manifest.jsonl', root/'data_teacher/manifest.jsonl'))
    screen([config_root/'vosr2_teacher_text.json'], validation, root/'screen_teacher_raw')
    wait_ready([root/'quality_teacher/manifest.jsonl', root/'quality_teacher/summary.json'], quality_pid)
    teacher_configs = [config_root/f'vosr2_{r}.json' for r in protocol['processing_screen']['recipes']
                       if r.startswith('teacher_') and r != 'teacher_text']
    screen(teacher_configs, validation, root/'screen_teacher_filtered')

    processing = []
    for recipe in protocol['processing_screen']['recipes']:
        queue = ('screen_raw' if recipe in ['original_text', 'original_random', 'original_strong'] else
                 'screen_original_filtered' if recipe.startswith('original_') else
                 'screen_teacher_raw' if recipe == 'teacher_text' else 'screen_teacher_filtered')
        entry = screen_entry(config_root/f'vosr2_{recipe}.json', root/queue, pairs)
        processing.append(dict(entry, recipe=recipe, stage='processing'))
    processing.sort(key=rank_key)
    compare_entries(processing, baseline, pairs, state/'processing_comparisons.json')
    freeze(state/'processing_selection.json', dict(ranking=processing, selected=[r['name'] for r in processing[:2]]))
    write_table(state/'processing_validation.csv', processing)
    if args.stop_after == 'processing':
        return

    generated = state/'configs'
    generated.mkdir(exist_ok=True)
    mixture_configs, mixture_meta = [], []
    mixtures = list(processing[:2])
    for entry in processing[:2]:
        for probability in protocol['mixture_screen']['anyword_probabilities']:
            if probability == 0.5:
                continue
            name = f'{entry["recipe"]}_aw{round(probability*100):03}_s42'
            cfg = derived_config(read(entry['config']), probability, 42, 500, name)
            path = generated/f'{name}.json'
            freeze(path, cfg)
            mixture_configs.append(path)
            mixture_meta.append((path, entry['recipe']))
    screen(mixture_configs, validation, root/'screen_mixtures')
    for path, recipe in mixture_meta:
        entry = screen_entry(path, root/'screen_mixtures', pairs)
        mixtures.append(dict(entry, recipe=recipe, stage='mixture'))
    mixtures.sort(key=rank_key)
    compare_entries(mixtures, baseline, pairs, state/'mixture_comparisons.json')
    freeze(state/'mixture_selection.json', dict(ranking=mixtures, selected=[r['name'] for r in mixtures[:2]]))
    write_table(state/'mixture_validation.csv', mixtures)
    if args.stop_after == 'mixtures':
        return

    confirmation_configs, confirmation_meta = [], []
    for entry in mixtures[:2]:
        combination = f'{entry["recipe"]}_aw{round(entry["anyword_probability"]*100):03}'
        for seed in protocol['confirmation']['seeds']:
            name = f'{combination}_confirm_s{seed}'
            cfg = derived_config(read(entry['config']), entry['anyword_probability'], seed,
                                 protocol['confirmation']['steps'], name)
            path = generated/f'{name}.json'
            freeze(path, cfg)
            confirmation_configs.append(path)
            confirmation_meta.append((path, entry['recipe'], combination))
    screen(confirmation_configs, validation, root/'confirmation')
    confirmed = []
    for path, recipe, combination in confirmation_meta:
        entry = screen_entry(path, root/'confirmation', pairs)
        confirmed.append(dict(entry, recipe=recipe, combination=combination, stage='confirmation'))
    groups = confirmation_groups(confirmed)
    selected = next(r for r in confirmed if r['combination'] == groups[0]['name'] and r['seed'] == 42)
    baseline_rank = dict(name='vosr2_initial', metrics=dict(zip(METRICS, map(float, baseline['estimate']))))
    recommended = selected['checkpoint'] if rank_key(groups[0]) < rank_key(baseline_rank) else 'preset/ckpts/VOSR2'
    freeze(state/'final_validation_selection.json', dict(
           ranking_by_seed_mean=groups, all_runs=confirmed, best_trained_checkpoint=selected['checkpoint'],
           canonical_export_seed=42, recommended_checkpoint_by_validation=recommended,
           baseline_metrics=baseline_rank['metrics'], official_test_used_for_selection=False))
    compare_entries(confirmed, baseline, pairs, state/'confirmation_comparisons.json')
    write_table(state/'confirmation_validation.csv', confirmed)
    if args.stop_after == 'confirmation':
        return

    # All model choices are now immutable. Neither secondary nor official test
    # results are read by rank_key or used to construct any subsequent training.
    secondary = []
    for entry in confirmed:
        secondary.append(evaluate_checkpoint(entry, root/'realce_validation_x4',
                                             root/'secondary_validation'/entry['name']))
        atomic_json(state/'secondary_results.json', secondary)
    write_table(state/'secondary_validation.csv', secondary)
    all_trained = {r['name']: r for r in processing + mixtures + confirmed}
    test_entries = []
    test_pairs = [json.loads(line) for line in (root/'realce_test_x4/pairs.jsonl').read_text().splitlines()]
    test_baseline = load_evaluation(root/'metrics/vosr2_initial_test_x4', test_pairs)
    for entry in all_trained.values():
        test_entries.append(evaluate_checkpoint(entry, root/'realce_test_x4', root/'final_test'/entry['name']))
        atomic_json(state/'test_results.json', test_entries)
        write_table(state/'realce_test_trained.csv', test_entries)
    compare_entries(test_entries, test_baseline, test_pairs, state/'test_comparisons.json')
    for name, directory in [('vosr2_initial', 'vosr2_initial_test_x4'), ('bicubic', 'bicubic_test_x4')]:
        evaluation = load_evaluation(root/'metrics'/directory, test_pairs)
        test_entries.insert(0, dict(name=name, stage='reference',
                            metrics=dict(zip(METRICS, map(float, evaluation['estimate'])))))
    write_table(state/'realce_test_all.csv', test_entries)
    atomic_json(state/'complete.json', dict(trained_models=len(all_trained), test_images=len(test_pairs),
                test_groups=len(test_baseline['groups']), test_text_instances=len(test_baseline['labels']),
                best_trained_checkpoint=selected['checkpoint'],
                selection_sha256=file_digest(state/'final_validation_selection.json'),
                report_table=str(state/'realce_test_all.csv')))


if __name__ == '__main__':
    main()
