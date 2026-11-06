"""Materialize and run the FFN half-width experiment DAG.

This planner imports no torch and does not download weights.  External paths must
be supplied in a site YAML.  Training is an explicit `run` action; `list`,
`preflight`, `status`, and `run --dry-run` never start training.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO / 'configs/ffn_halfwidth/matrix.yml'
DEFAULT_BASE = REPO / 'configs/ffn_halfwidth/base.yml'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
HEIGHT_EDGES = [24, 32, 48, 64, 96]
SMALL_SAMPLING = {'mode': 'height_buckets', 'height_edges': HEIGHT_EDGES,
                  'bucket_weights': [.05, .15, .35, .25, .10, .10],
                  'jitter_fraction': .25, 'min_visible_fraction': .5}
INSTRUMENTATION_KEYS = {'report_to', 'log_every', 'save_every', 'preview_every', 'preview_num_images',
                        'preview_seed', 'tracker_project_name', 'wandb_mode'}
SITE_KEYS = {'schema_version', 'teacher_checkpoint', 'vae_path', 'dino', 'headsplit_checkpoint', 'head_stats',
             'ocr', 'runtime', 'train_data', 'calibration_data', 'hard_data', 'split_manifests',
             'calibration', 'reconstruction', 'training_overrides', 'budgets', 'eval'}


def merge(old, new):
    result = copy.deepcopy(old)
    for key, value in new.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) \
            else copy.deepcopy(value)
    return result


def read_yaml(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f'cyclic YAML inheritance: {path}')
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected YAML mapping')
    parent = value.pop('_base_', None)
    return merge(read_yaml(path.parent / parent, (*seen, path)), value) if parent else value


def digest(value, length=12):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:length]


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def input_snapshot(site, cached=None):
    """Strong hashes for weights/code/labels; images use a documented metadata manifest.

    A cached snapshot may reuse hashes ONLY for matching path/size/mtime during
    per-node checks. The initial run preflight re-hashes all strong files.
    """
    files, images, problems = {}, {}, []
    previous = (cached or {}).get('files', {})

    def add_file(value):
        if not value:
            return
        path = Path(value).resolve()
        key = str(path)
        if key in files:
            return
        if not path.is_file():
            files[key] = {'missing': True}
            return
        stat = path.stat()
        record = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        old = previous.get(key, {})
        record['sha256'] = (old['sha256'] if old.get('bytes') == record['bytes']
                            and old.get('mtime_ns') == record['mtime_ns'] and old.get('sha256') else file_sha256(path))
        files[key] = record

    for value in (site.get('teacher_checkpoint'), site.get('headsplit_checkpoint'),
                  site.get('ocr', {}).get('checkpoint')):
        if value:
            try:
                for file in weight_files(value):
                    add_file(file)
                if Path(value).is_dir() and (Path(value) / 'config.json').is_file():
                    add_file(Path(value) / 'config.json')
            except (OSError, ValueError, KeyError) as error:
                problems.append(str(error))
    for root, patterns in ((site['vae_path'], ('*.safetensors', '*.json')),
                           (site['dino']['cache_dir'], ('*.pth',)),
                           (site['dino']['local_repo'], ('*.py',))):
        path = Path(root)
        if not path.is_dir():
            problems.append(f'missing source directory: {path}')
        for pattern in patterns:
            for file in sorted(path.rglob(pattern)) if path.is_dir() else []:
                add_file(file)
    add_file(site.get('head_stats'))
    for value in site.get('split_manifests', {}).values():
        add_file(value)
    for data_key in ('train_data', 'calibration_data', 'hard_data'):
        data = site.get(data_key)
        if not data:
            continue
        for source in data.get('sources', [data]):
            add_file(source.get('text_boxes', data.get('text_boxes')))
            image_names = source.get('image_names', data.get('image_names'))
            if isinstance(image_names, str):
                add_file(image_names)
            for key in ('hq_dir', 'lq_dir'):
                directory = Path(source[key])
                if str(directory) in images:
                    continue
                manifest = []
                if directory.is_dir():
                    for path in sorted(directory.iterdir()):
                        if path.suffix.lower() in IMAGE_SUFFIXES:
                            stat = path.stat()
                            manifest.append((path.name, stat.st_size, stat.st_mtime_ns))
                    images[str(directory)] = {'images': len(manifest), 'metadata_sha256': digest(manifest, 64)}
                else:
                    images[str(directory)] = {'missing': True}
    code_paths = list(REPO.glob('*.py'))
    for directory in ('models', 'tools', 'dataloaders'):
        code_paths.extend((REPO / directory).rglob('*.py'))
    for path in sorted(set(code_paths)):
        add_file(path)
    return {'policy': 'weights/code/annotations: SHA256 bytes; images: names/bytes/mtime_ns, not image bytes',
            'files': files, 'image_directories': images, 'problems': sorted(problems)}


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def absolute_path(value, directory, required=False, name='path'):
    if value is None or value == '':
        if required:
            raise ValueError(f'{name} is unset; supply the actual local path in --site')
        return None
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a local path string')
    value = os.path.expandvars(os.path.expanduser(value))
    if re.search(r'\$\{?\w', value) or '://' in value:
        raise ValueError(f'{name} must resolve to a local path, not {value!r}')
    path = Path(value)
    return str((path if path.is_absolute() else directory / path).resolve())


def resolve_data(data, directory, name):
    if not isinstance(data, dict) or data.get('dataset_type', 'paired') != 'paired':
        raise ValueError(f'{name} must be a paired dataset mapping')
    result = merge({'dataset_type': 'paired', 'upscale': 4, 'resolution': 512, 'text_crop_prob': .8,
                    'text_metadata': True, 'small_text_height': 64, 'teacher_error_weight': .1,
                    'deterministic_crops': True}, data)
    for key in ('upscale', 'resolution'):
        positive_integer(result[key], f'{name}.{key}')
    if 'sources' in result:
        for key in ('text_boxes', 'image_names'):
            if key in result and not (key == 'image_names' and isinstance(result[key], list)):
                result[key] = absolute_path(result[key], directory, name=f'{name}.{key}')
    sources = result.get('sources', [result])
    if not isinstance(sources, list) or not sources:
        raise ValueError(f'{name}.sources must be a nonempty list')
    names = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f'{name}.sources[{index}] must be a mapping')
        source_name = source.get('name', f'source_{index}')
        if source_name in names:
            raise ValueError(f'{name}: duplicate source name {source_name}')
        names.add(source_name)
        if 'sources' in result:
            if not isinstance(source.get('name'), str) or not source['name']:
                raise ValueError(f'{name}.sources[{index}] needs a nonempty name')
            weight = source.get('weight')
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                raise ValueError(f'{name}.{source_name}.weight must be positive and finite')
        for key in ('lq_dir', 'hq_dir', 'text_boxes', 'image_names'):
            if key in source or key in ('lq_dir', 'hq_dir'):
                if key == 'image_names' and isinstance(source.get(key), list):
                    continue
                source[key] = absolute_path(source.get(key), directory, key in ('lq_dir', 'hq_dir'),
                                            f'{name}.{source_name}.{key}')
    return result


def load_site(path, nproc=None, global_batch=None):
    path = Path(path).resolve()
    site = read_yaml(path)
    if site.get('schema_version', 1) != 1 or set(site) - SITE_KEYS:
        raise ValueError(f'unsupported site keys/schema: {sorted(set(site) - SITE_KEYS)}')
    directory = path.parent
    for key in ('teacher_checkpoint', 'vae_path', 'headsplit_checkpoint', 'head_stats'):
        site[key] = absolute_path(site.get(key), directory, key in ('teacher_checkpoint', 'vae_path'), key)
    site['dino'] = merge({'local_repo': None, 'cache_dir': None}, site.get('dino') or {})
    for key in ('local_repo', 'cache_dir'):
        site['dino'][key] = absolute_path(site['dino'].get(key), directory, True, f'dino.{key}')
    site['ocr'] = site.get('ocr') or {'type': 'ppocr', 'checkpoint': None}
    site['ocr']['checkpoint'] = absolute_path(site['ocr'].get('checkpoint'), directory, name='ocr.checkpoint')
    site['train_data'] = resolve_data(site.get('train_data'), directory, 'train_data')
    for key in ('calibration_data', 'hard_data'):
        if site.get(key) is not None:
            site[key] = resolve_data(site[key], directory, key)
    manifests = site.setdefault('split_manifests', {})
    for key in ('train', 'validation', 'test'):
        manifests[key] = absolute_path(manifests.get(key), directory, name=f'split_manifests.{key}')
    runtime = merge({'python': sys.executable, 'nproc_per_node': 1, 'global_batch_size': 16,
                     'batch_size_per_gpu': 1, 'num_workers': 4, 'device': 'auto', 'precision': 'bf16',
                     'gradient_checkpointing': True, 'zero_optimizer': False}, site.get('runtime') or {})
    runtime['python'] = runtime.get('python') or sys.executable
    if not Path(runtime['python']).is_absolute():
        runtime['python'] = shutil.which(runtime['python']) or absolute_path(runtime['python'], directory)
    if nproc is not None:
        runtime['nproc_per_node'] = nproc
    if global_batch is not None:
        runtime['global_batch_size'] = global_batch
    for key in ('nproc_per_node', 'global_batch_size', 'batch_size_per_gpu'):
        positive_integer(runtime[key], f'runtime.{key}')
    divisor = runtime['nproc_per_node'] * runtime['batch_size_per_gpu']
    if runtime['global_batch_size'] % divisor:
        raise ValueError('global_batch_size must divide exactly by nproc_per_node * batch_size_per_gpu')
    if runtime['device'] not in ('auto', 'cuda', 'npu', 'cpu') or runtime['precision'] not in ('bf16', 'fp32'):
        raise ValueError('invalid runtime.device or precision')
    runtime['gradient_accumulation_steps'] = runtime['global_batch_size'] // divisor
    if runtime.get('worker_start_method') not in (None, 'spawn', 'fork', 'forkserver'):
        raise ValueError('runtime.worker_start_method must be spawn, fork, forkserver or null')
    site['runtime'] = runtime
    overrides = site.get('training_overrides') or {}
    if set(overrides) - INSTRUMENTATION_KEYS:
        raise ValueError('training_overrides permits instrumentation only; change algorithm settings in an explicit '
                         f'matrix arm: {sorted(set(overrides) - INSTRUMENTATION_KEYS)}')
    evaluation = site.setdefault('eval', {})
    for split in ('validation', 'test'):
        spec = evaluation.setdefault(split, {})
        for key in ('native_lq_dir', 'hq_dir', 'annotations', 'teacher_details'):
            spec[key] = absolute_path(spec.get(key), directory, name=f'eval.{split}.{key}')
    site['_site_file'] = str(path)
    return site


def load_matrix(path=DEFAULT_MATRIX):
    raw = read_yaml(path)
    if raw.get('schema_version') != 1 or not isinstance(raw.get('arms'), list):
        raise ValueError('matrix requires schema_version:1 and arms list')
    definitions = {}
    for item in raw['arms']:
        name = item.get('id')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name) or name in definitions:
            raise ValueError(f'invalid or duplicate arm id {name!r}')
        definitions[name] = item
    resolved = {}

    def resolve(name, seen=()):
        if name in resolved:
            return resolved[name]
        if name in seen or name not in definitions:
            raise ValueError(f'cyclic/unknown arm inheritance: {(*seen, name)}')
        item = definitions[name]
        parent = resolve(item['extends'], (*seen, name)) if item.get('extends') else raw.get('defaults', {})
        result = merge(parent, {key: value for key, value in item.items() if key != 'extends'})
        result['id'] = name
        if not 0 < float(result['keep_ratio']) <= 1:
            raise ValueError(f'{name}: keep_ratio must be in (0,1]')
        if result['source'] not in ('dense', 'headsplit') or result['calibration'] not in (None, 'uniform', 'text', 'balanced'):
            raise ValueError(f'{name}: invalid source/calibration')
        if result.get('fade_steps_full') and (result.get('reconstruction') is not None or result['schedule'] != 'single'):
            raise ValueError(f'{name}: fading is an independent single-stage arm, without local reconstruction')
        resolved[name] = result
        return result

    for name in definitions:
        resolve(name)
    return raw, resolved


def select_arms(arms, patterns=None, exclude=()):
    def matches(name, pattern):
        return pattern == 'all' or (pattern.startswith('group:') and pattern[6:] in arms[name].get('groups', [])) \
            or fnmatch.fnmatchcase(name, pattern)
    selected = [name for name in arms if any(matches(name, p) for p in (patterns or ['all']))
                and not any(matches(name, p) for p in exclude)]
    if not selected:
        raise ValueError('selection matches no experiment arms')
    for pattern in patterns or []:
        if not any(matches(name, pattern) for name in arms):
            raise ValueError(f'selection pattern matches nothing: {pattern}')
    return selected


def training_data(site, seed, sampling, stage):
    key = 'hard_data' if stage == 'D' and site.get('hard_data') else 'train_data'
    data = copy.deepcopy(site[key])
    data.update(seed=seed, deterministic_crops=True, text_metadata=True)
    if sampling == 'small_text':
        data['text_sampling'] = copy.deepcopy(SMALL_SAMPLING)
    elif sampling != 'standard':
        raise ValueError(f'unknown sampling mode {sampling}')
    return data, key


def model_config(site, arm, seed, base):
    cfg = copy.deepcopy(base)
    cfg.update(teacher_checkpoint=site['teacher_checkpoint'], vae_path=site['vae_path'])
    cfg['dino'] = merge(cfg['dino'], site['dino'])
    cfg['student_checkpoint'] = None
    if arm['source'] == 'headsplit':
        # None is kept explicit until preflight; no fabricated checkpoint path.
        cfg['student_checkpoint'] = site.get('headsplit_checkpoint')
        cfg['student']['attn_compression'] = {'type': 'head_split', 'factor': 2, 'kernel': 7,
                                            'upsample': 'bilinear', 'head_stats': site.get('head_stats'),
                                            'local_threshold': .5}
    else:
        cfg['student']['attn_compression'] = None
    if float(arm['keep_ratio']) < 1:
        cfg['student']['mlp_compression'] = {
            'type': 'neuron_prune', 'keep_ratio': float(arm['keep_ratio']), 'score': arm['score'],
            'compensation': arm['compensation'], 'seed': seed}
    else:
        cfg['student']['mlp_compression'] = None
    runtime = site['runtime']
    cfg['training'].update({key: runtime[key] for key in (
        'device', 'precision', 'batch_size_per_gpu', 'gradient_accumulation_steps', 'num_workers',
        'gradient_checkpointing', 'zero_optimizer')})
    cfg['training'].update(seed=seed)
    if site['runtime'].get('worker_start_method') is not None:
        cfg['training']['worker_start_method'] = site['runtime']['worker_start_method']
    cfg['training'].update(site.get('training_overrides') or {})
    cfg['ocr'] = copy.deepcopy(site['ocr'])
    return cfg


def build_plan(site, matrix_path=DEFAULT_MATRIX, base_path=DEFAULT_BASE, output='.', profile='pilot',
               selected=None, exclude=(), seeds=None):
    matrix, arms = load_matrix(matrix_path)
    chosen = select_arms(arms, selected, exclude)
    if profile not in matrix['profiles']:
        raise ValueError(f'unknown profile {profile}')
    budgets = merge(matrix['profiles'][profile], site.get('budgets') or {})
    if set(budgets) != {'reconstruction', 'B', 'C', 'D'}:
        raise ValueError('budgets supports reconstruction, B, C, D only')
    for key, value in budgets.items():
        positive_integer(value, f'budgets.{key}')
    base, output = read_yaml(base_path), Path(output).resolve()
    nodes, configs, endpoints = {}, {}, []

    def add_node(node, cfg, owner):
        name = node['id']
        if name in nodes:
            if configs[name] != cfg:
                raise ValueError(f'conflicting shared node: {name}')
            if owner not in nodes[name]['owners']:
                nodes[name]['owners'].append(owner)
            return nodes[name]
        node.update(owners=[owner], config=str(output / 'configs' / f'{name}.yml'))
        nodes[name], configs[name] = node, cfg
        return node

    for name in chosen:
        arm = arms[name]
        arm_seeds = seeds if seeds is not None else arm.get('seeds', [42])
        if not arm_seeds or len(set(arm_seeds)) != len(arm_seeds) or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in arm_seeds):
            raise ValueError(f'{name}: seeds must be distinct nonnegative integers')
        for seed in arm_seeds:
            owner, dependency, stats_path = f'{name}.s{seed}', None, None
            initial = model_config(site, arm, seed, base)
            if arm['calibration'] is not None and float(arm['keep_ratio']) < 1:
                calibration = merge({'samples': 1024, 'seed': 11, 'noise_seed': 23, 'small_text_height': 64,
                                     'strata_weights': [.5, .3, .2], 'layer_group_size': 2, 'moment_device': 'cpu',
                                     'moment_dtype': 'float64', 'chunk_tokens': 1024}, site.get('calibration') or {})
                calibration['weighting'] = arm['calibration']
                if profile == 'smoke':
                    calibration['samples'] = min(int(calibration['samples']), 8)
                cal_cfg = model_config(site, {**arm, 'score': 'obs', 'compensation': 'lstsq', 'keep_ratio': .5},
                                       int(calibration['seed']), base)
                # Statistics always see the SAME source crops for weighting ablations.
                cal_cfg['data'] = copy.deepcopy(site.get('calibration_data') or site['train_data'])
                cal_cfg['data'].update(seed=int(calibration['seed']), deterministic_crops=True, text_metadata=True)
                cal_cfg['calibration'] = calibration
                cal_cfg['training']['output_dir'] = None
                identifier = f'cal.{arm["source"]}.{arm["calibration"]}.{digest(cal_cfg)}'
                stats_path = str(output / 'nodes' / identifier / 'neuron_stats.safetensors')
                cal_cfg['student']['mlp_compression']['neuron_stats'] = stats_path
                cal_cfg['training']['output_dir'] = str(output / 'nodes' / identifier / 'unused_training_output')
                node = add_node({'id': identifier, 'kind': 'calibration', 'deps': [], 'artifact': stats_path,
                                 'source': arm['source'], 'requires': ['headsplit'] if arm['source'] == 'headsplit' else [],
                                 'samples': calibration['samples']}, cal_cfg, owner)
                dependency = node['id']
            if initial['student'].get('mlp_compression') is not None and stats_path:
                initial['student']['mlp_compression']['neuron_stats'] = stats_path
            if arm.get('reconstruction') is not None:
                recon = merge({'learning_rate': 1e-4, 'weight_decay': 0., 'lr_schedule': 'cosine', 'min_lr_ratio': .1,
                               'lr_warmup_steps': 100, 'max_grad_norm': 1., 'precision': site['runtime']['precision'],
                               'gated': True, 'energy_floor': 1e-4, 'weighting': 'balanced',
                               'small_text_height': 64, 'strata_weights': [.5, .3, .2], 'log_every': 10,
                               'layers': None, 'seed': seed}, site.get('reconstruction') or {})
                recon = merge(recon, arm['reconstruction'])
                multiplier = float(recon.pop('steps_multiplier', 1))
                if not math.isfinite(multiplier) or multiplier <= 0:
                    raise ValueError('reconstruction.steps_multiplier must be finite and positive')
                recon.update(max_steps=max(1, round(budgets['reconstruction'] * multiplier)), seed=seed,
                             batch_size_per_gpu=site['runtime']['batch_size_per_gpu'],
                             gradient_accumulation_steps=site['runtime']['gradient_accumulation_steps'])
                recon['save_every'] = min(int(recon.get('save_every', 500)), recon['max_steps'])
                recon['lr_warmup_steps'] = min(int(recon['lr_warmup_steps']), max(0, recon['max_steps'] // 10))
                cfg = copy.deepcopy(initial)
                cfg['data'], data_key = training_data(site, seed, arm['sampling'], 'A')
                cfg['reconstruction'] = recon
                cfg['training']['output_dir'] = None
                identifier = f'recon.{arm["source"]}.s{seed}.{digest(cfg)}'
                cfg['training']['output_dir'] = str(output / 'nodes' / identifier / 'unused_training_output')
                node = add_node({'id': identifier, 'kind': 'reconstruction', 'deps': [dependency] if dependency else [],
                                 'artifact': str(output / 'nodes' / identifier / 'export'), 'source': arm['source'],
                                 'requires': ['headsplit'] if arm['source'] == 'headsplit' else [],
                                 'steps': recon['max_steps'], 'data_key': data_key}, cfg, owner)
                dependency = node['id']
            stages = ['E'] if arm['schedule'] == 'single' else ['B', 'C', 'D']
            stage_ids = []
            for stage in stages:
                identifier = f'{owner}.{stage}'
                cfg = copy.deepcopy(initial)
                cfg['data'], data_key = training_data(site, seed, arm['sampling'], stage)
                total_steps = sum(budgets[key] for key in ('B', 'C', 'D')) if stage == 'E' else budgets[stage]
                tc = cfg['training']
                tc.update(max_steps=total_steps, trainable_parameters=[r'^blocks\.\d+\.mlp\.'] if stage == 'B' else None,
                          learning_rate={'B': 5e-5, 'C': 2e-6, 'D': 5e-7, 'E': 1e-5}[stage],
                          ffn_lr_multiplier={'B': 1., 'C': 10., 'D': 10., 'E': 1.}[stage])
                tc.update(copy.deepcopy(arm.get('training') or {}))
                if stage == 'B':
                    # Pixel/GT ablations start in C, after FFN-only recovery.
                    tc.update(gt_weight=0., roi_charbonnier_weight=0., roi_edge_weight=0., roi_ocr_weight=0.,
                              lpips_weight=0., ocr_kl_weight=0.)
                tc.update(copy.deepcopy((arm.get('stages') or {}).get(stage, {})))
                tc['lr_warmup_steps'] = min(int(tc.get('lr_warmup_steps', 100)), max(0, total_steps // 10))
                tc['save_every'] = min(positive_integer(int(tc['save_every']), 'save_every'), total_steps)
                tc['pixel_loss_warmup_steps'] = min(int(tc.get('pixel_loss_warmup_steps', 2000)), max(1, total_steps // 4))
                tc['output_dir'] = str(output / 'nodes' / identifier / 'runs')
                if arm.get('fade_steps_full'):
                    full_total = sum(matrix['profiles']['full'][key] for key in ('B', 'C', 'D'))
                    fade = max(1, round(int(arm['fade_steps_full']) * total_steps / full_total))
                    if fade > total_steps:
                        raise ValueError(f'{name}: fade exceeds training budget')
                    cfg['student']['mlp_compression']['fade_steps'] = fade
                init_from = None
                if dependency and nodes[dependency]['kind'] != 'calibration':
                    init_from = nodes[dependency]['artifact']
                    cfg['student_checkpoint'] = init_from
                requirements = ['headsplit'] if arm['source'] == 'headsplit' else []
                if tc.get('roi_ocr_weight', 0) or tc.get('ocr_kl_weight', 0):
                    requirements.append('ocr')
                node = add_node({'id': identifier, 'kind': 'train', 'stage': stage,
                                 'deps': [dependency] if dependency else [], 'init_from': init_from,
                                 'artifact': str(output / 'nodes' / identifier / 'export'), 'source': arm['source'],
                                 'requires': requirements, 'steps': total_steps, 'data_key': data_key}, cfg, owner)
                dependency = node['id']
                stage_ids.append(dependency)
            endpoints.append({'arm': name, 'seed': seed, 'final_node': dependency, 'stage_nodes': stage_ids,
                              'keep_ratio': arm['keep_ratio'], 'all_layers_half_width': float(arm['keep_ratio']) == .5,
                              'source': arm['source'], 'compare_to': arm.get('compare_to'),
                              'e2e_optimizer_steps': sum(nodes[node]['steps'] for node in stage_ids),
                              'local_reconstruction_steps': (recon['max_steps'] if arm.get('reconstruction') is not None else 0)})
    plan = {'schema_version': 1, 'repo': str(REPO), 'output': str(output), 'profile': profile, 'budgets': budgets,
            'site': copy.deepcopy(site), 'matrix_file': str(Path(matrix_path).resolve()),
            'matrix_sha256': hashlib.sha256(Path(matrix_path).read_bytes()).hexdigest(),
            'base_sha256': digest(base, 64), 'arms': {name: arms[name] for name in chosen},
            'nodes': list(nodes.values()), 'endpoints': endpoints,
            'budget_note': 'E2E optimizer steps and global batch are matched; local reconstruction is extra compute. '
                           'Wall time, memory, teacher passes, and full-width fade costs must be measured separately.'}
    for node in plan['nodes']:
        node['config_sha256'] = hashlib.sha256(yaml.safe_dump(configs[node['id']], allow_unicode=True,
                                                            sort_keys=False).encode()).hexdigest()
    plan['input_snapshot'] = input_snapshot(site)
    plan['plan_id'] = digest(plan, 20)
    return plan, configs


def write_immutable(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as file:
            file.write(text)
    except FileExistsError:
        if path.read_text(encoding='utf-8') != text:
            raise FileExistsError(f'{path}: different content already exists; choose a new output root')


def materialize(plan, configs):
    output = Path(plan['output'])
    # Verify all conflicts BEFORE making any changes to an existing plan.
    files = {output / 'manifest.json': json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True) + '\n'}
    files.update({Path(node['config']): yaml.safe_dump(configs[node['id']], allow_unicode=True, sort_keys=False)
                  for node in plan['nodes']})
    for path, text in files.items():
        if path.exists() and path.read_text(encoding='utf-8') != text:
            raise FileExistsError(f'{path}: immutable plan/config differs; use a new output root')
    for path, text in files.items():
        write_immutable(path, text)
    return output / 'manifest.json'


def load_plan(output):
    path = Path(output).resolve() / 'manifest.json'
    plan = json.loads(path.read_text(encoding='utf-8'))
    if plan.get('schema_version') != 1 or plan['output'] != str(path.parent):
        raise ValueError('unsupported manifest or moved output root; rematerialize at the actual path')
    check = dict(plan)
    expected = check.pop('plan_id')
    if digest(check, 20) != expected:
        raise ValueError('manifest was modified; materialize a new immutable plan')
    return plan


def selected_nodes(plan, patterns=None, exclude=()):
    selected = set(select_arms(plan['arms'], patterns, exclude))
    needed = {point['final_node'] for point in plan['endpoints'] if point['arm'] in selected}
    lookup = {node['id']: node for node in plan['nodes']}
    stack = list(needed)
    while stack:
        for dependency in lookup[stack.pop()]['deps']:
            if dependency not in needed:
                needed.add(dependency)
                stack.append(dependency)
    return [node for node in plan['nodes'] if node['id'] in needed]


def weight_files(value):
    if not value:
        raise ValueError('checkpoint is unset')
    path = Path(value)
    if path.is_dir():
        candidates = [path / sub / name for sub in ('clean_weights', 'checkpoints', '')
                      for name in ('ema_model.safetensors', 'model.safetensors', 'model.safetensors.index.json')]
        path = next((item for item in candidates if item.is_file()), path / 'model.safetensors')
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'missing/empty checkpoint: {path}')
    with path.open('rb') as file:
        if file.read(50).startswith(b'version https://git-lfs.github.com/spec'):
            raise ValueError(f'Git LFS pointer, not downloaded weights: {path}')
    if path.name.endswith('.index.json'):
        index = json.loads(path.read_text(encoding='utf-8'))
        shards = [path.parent / name for name in sorted(set(index['weight_map'].values()))]
        for shard in shards:
            weight_files(str(shard))
        return [path, *shards]
    return [path]


def data_errors(data, label):
    errors, records = [], []
    for i, source in enumerate(data.get('sources', [data])):
        name = f'{label}.{source.get("name", i)}'
        paths = {key: Path(source[key]) if source.get(key) else None for key in ('lq_dir', 'hq_dir')}
        for key, path in paths.items():
            if path is None or not path.is_dir():
                errors.append(f'{name}.{key}: missing directory {path}')
        if all(path is not None and path.is_dir() for path in paths.values()):
            images = {key: {p.stem for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
                      for key, path in paths.items()}
            selected = images['hq_dir']
            if source.get('image_names', data.get('image_names')):
                supplied_names = source.get('image_names', data.get('image_names'))
                names_path = Path(supplied_names) if isinstance(supplied_names, str) else None
                if names_path is not None and not names_path.is_file():
                    errors.append(f'{name}.image_names: missing {names_path}')
                else:
                    lines = names_path.read_text(encoding='utf-8').splitlines() if names_path else supplied_names
                    names = [Path(line.strip().replace('\\', '/')) for line in lines
                             if line.strip() and not line.lstrip().startswith('#')]
                    selected = {path.stem if path.suffix.lower() in IMAGE_SUFFIXES else path.name for path in names}
            if not selected or not selected <= images['hq_dir'] or not selected <= images['lq_dir']:
                errors.append(f'{name}: no pairs or selected filenames are not present in both LQ and HQ')
            records.append({'source': name, 'hq_dir': str(paths['hq_dir']), 'lq_dir': str(paths['lq_dir']),
                            'num_selected_images': len(selected)})
        boxes = source.get('text_boxes', data.get('text_boxes'))
        allow_missing = source.get('allow_missing_text', data.get('allow_missing_text', False))
        if not boxes and not allow_missing:
            errors.append(f'{name}: text_boxes required, or explicitly declare allow_missing_text:true')
        elif not boxes and source.get('text_crop_prob', data.get('text_crop_prob', 0)):
            errors.append(f'{name}: a natural source without boxes must explicitly set text_crop_prob:0')
        elif boxes and not Path(boxes).is_file():
            errors.append(f'{name}: missing text_boxes {boxes}')
    return errors, records


def runtime_probe(runtime):
    python = runtime['python']
    if not Path(python).is_file():
        return [f'Python interpreter not found: {python}'], None
    code = ("import json,sys; sys.path.insert(0, sys.argv[1]); import torch; "
            "from training_runtime import select_device; d=select_device(sys.argv[2],0); "
            "n=(torch.cuda.device_count() if d.type=='cuda' else "
            "torch.npu.device_count() if d.type=='npu' else 1); "
            "print(json.dumps({'torch':torch.__version__,'device':str(d),'device_count':n}))")
    run = subprocess.run([python, '-c', code, str(REPO), runtime['device']], text=True, capture_output=True,
                         env=os.environ.copy(), timeout=60)
    if run.returncode:
        return [f'runtime probe failed: {run.stderr[-2000:]}'], None
    try:
        info = json.loads(run.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return ['runtime probe returned no JSON device report'], None
    errors = []
    if info['device_count'] < runtime['nproc_per_node'] and not info['device'].startswith('cpu'):
        errors.append(f'{runtime["nproc_per_node"]} processes requested but only {info["device_count"]} visible devices')
    return errors, info


def preflight(plan, nodes=None, check_runtime=False, verify_artifacts=False):
    site, errors, warnings, data_report = plan['site'], [], [], []
    nodes = nodes or plan['nodes']
    current_snapshot = input_snapshot(site)
    if current_snapshot != plan.get('input_snapshot'):
        errors.append('source weights, annotations, image manifest or training code changed since materialization; '
                      'do not reuse old calibration/reconstruction, materialize a new output root')
    try:
        weight_files(site['teacher_checkpoint'])
    except (ValueError, OSError, KeyError) as error:
        errors.append(f'teacher: {error}')
    vae = Path(site['vae_path'])
    if not vae.is_dir() or not (vae / 'config.json').is_file() or not any(vae.glob('*.safetensors')):
        errors.append(f'vae_path needs config.json and actual safetensors weights: {vae}')
    dino = site['dino']
    if not (Path(dino['local_repo']) / 'hubconf.py').is_file():
        errors.append(f'DINO local_repo lacks hubconf.py: {dino["local_repo"]}')
    if not Path(dino['cache_dir']).is_dir() or not any(Path(dino['cache_dir']).rglob('*.pth')):
        errors.append(f'DINO cache has no .pth weights (implicit downloads are not a preflight substitute): {dino["cache_dir"]}')
    for key in ('train_data', 'calibration_data', 'hard_data'):
        if site.get(key):
            problems, counts = data_errors(site[key], key)
            errors.extend(problems)
            data_report.extend(counts)
    manifests = site.get('split_manifests', {})
    supplied = [bool(manifests.get(key)) for key in ('train', 'validation', 'test')]
    if any(supplied) and not all(supplied):
        errors.append('supply all three split_manifests (train/validation/test), or leave all unset')
    elif all(supplied):
        ids = {}
        for key, value in manifests.items():
            path = Path(value)
            if not path.is_file():
                errors.append(f'missing {key} split manifest: {path}')
                continue
            lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
                     if line.strip() and not line.lstrip().startswith('#')]
            if len(set(lines)) != len(lines) or not lines:
                errors.append(f'{key} split manifest must contain unique, nonempty sample IDs')
            ids[key] = set(lines)
        for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            overlap = ids.get(a, set()) & ids.get(b, set())
            if overlap:
                errors.append(f'data leakage: {a}/{b} share {len(overlap)} IDs, e.g. {sorted(overlap)[0]}')
    else:
        warnings.append('split_manifests absent: directory checks cannot prove content/source disjointness')
    training_dirs = {row['hq_dir'] for row in data_report}
    for split in ('validation', 'test'):
        directory = (site.get('eval') or {}).get(split, {}).get('hq_dir')
        if directory and directory in training_dirs:
            errors.append(f'data leakage: eval.{split}.hq_dir is also a training/calibration/hard-data directory')
    if not site.get('hard_data'):
        warnings.append('hard_data absent: stage D explicitly reuses train_data; no mined data is invented')
    if not (site.get('eval') or {}).get('enabled'):
        warnings.append('eval.enabled is false: training completion is not an OCR/quality result')
    node_errors = {}
    for node in nodes:
        issues = []
        for requirement in node['requires']:
            if requirement == 'headsplit':
                try:
                    weight_files(site.get('headsplit_checkpoint'))
                except (ValueError, OSError, KeyError) as error:
                    issues.append(f'headsplit checkpoint: {error}')
                if not site.get('head_stats') or not Path(site['head_stats']).is_file():
                    issues.append('headsplit requires a real head_stats JSON file')
            elif requirement == 'ocr':
                checkpoint = site.get('ocr', {}).get('checkpoint')
                if not checkpoint or not Path(checkpoint).is_dir() or not (Path(checkpoint) / 'config.json').is_file():
                    issues.append('OCR loss requires local ocr.checkpoint directory with config.json')
                else:
                    try:
                        weight_files(checkpoint)
                    except (ValueError, OSError, KeyError) as error:
                        issues.append(f'OCR weights: {error}')
        if not Path(node['config']).is_file():
            issues.append(f'materialized config missing: {node["config"]}')
        elif hashlib.sha256(Path(node['config']).read_bytes()).hexdigest() != node['config_sha256']:
            issues.append('materialized config was modified; create a new immutable plan')
        state = read_state(plan, node)
        if state['status'] == 'completed':
            try:
                check_artifact(node)
                verify_artifact_snapshot(state, strong=verify_artifacts)
            except (ValueError, OSError) as error:
                issues.append(str(error))
        if issues:
            node_errors[node['id']] = issues
    runtime = None
    if check_runtime:
        problems, runtime = runtime_probe(site['runtime'])
        errors.extend(problems)
    return {'ok': not errors and not node_errors, 'errors': errors, 'node_errors': node_errors,
            'warnings': warnings, 'data': data_report, 'runtime': runtime,
            'selected_nodes': len(nodes), 'plan_id': plan['plan_id'], 'artifact_sha256_rechecked': verify_artifacts}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as file:
            json.dump(value, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_state(plan, node):
    path = Path(plan['output']) / 'state' / f'{node["id"]}.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'status': 'pending', 'attempts': []}


def save_state(plan, node, state):
    state.update(node=node['id'], plan_id=plan['plan_id'], updated_at=time.time())
    atomic_json(Path(plan['output']) / 'state' / f'{node["id"]}.json', state)


def claim_path(plan, node):
    return Path(plan['output']) / 'claims' / f'{node["id"]}.json'


def claim_node(plan, node, worker_id):
    path = claim_path(plan, node)
    path.parent.mkdir(parents=True, exist_ok=True)
    claim = {'token': uuid.uuid4().hex, 'node': node['id'], 'plan_id': plan['plan_id'],
             'worker_id': worker_id, 'hostname': socket.gethostname(), 'pid': os.getpid(), 'started_at': time.time()}
    try:
        with path.open('x', encoding='utf-8') as file:
            json.dump(claim, file, indent=2)
            file.write('\n')
    except FileExistsError:
        return None
    return claim


def release_claim(plan, node, token):
    path = claim_path(plan, node)
    if path.is_file() and json.loads(path.read_text(encoding='utf-8')).get('token') == token:
        path.unlink()


def check_artifact(node, path=None):
    path = Path(path or node['artifact'])
    if node['kind'] == 'calibration':
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'missing/empty statistics artifact: {path}')
        sidecar = path.with_suffix('.provenance.json')
        if not sidecar.is_file():
            raise ValueError(f'missing statistics provenance: {sidecar}')
        json.loads(sidecar.read_text(encoding='utf-8'))
    else:
        for name in ('model.json', 'pipeline.json', 'export.complete'):
            if not (path / name).is_file():
                raise ValueError(f'incomplete export: {path / name}')
        weight_files(str(path))
        json.loads((path / 'model.json').read_text(encoding='utf-8'))
    return str(path.resolve())


def completed(plan, node):
    state = read_state(plan, node)
    if state['status'] != 'completed':
        return False
    try:
        check_artifact(node)
        verify_artifact_snapshot(state)
    except (ValueError, OSError):
        return False
    return True


def artifact_snapshot(node, path):
    path = Path(path).resolve()
    files = [path, path.with_suffix('.provenance.json')] if node['kind'] == 'calibration' else \
        [*weight_files(str(path)), path / 'model.json', path / 'pipeline.json', path / 'export.complete']
    return {str(file): {'bytes': file.stat().st_size, 'mtime_ns': file.stat().st_mtime_ns,
                        'sha256': file_sha256(file)} for file in files}


def verify_artifact_snapshot(state, strong=False):
    snapshot = state.get('artifact_snapshot')
    if not snapshot:
        raise ValueError('completed state has no artifact hash manifest')
    for name, record in snapshot.items():
        path = Path(name)
        if not path.is_file() or path.stat().st_size != record['bytes'] or path.stat().st_mtime_ns != record['mtime_ns']:
            raise ValueError(f'artifact metadata changed: {path}')
        if strong and file_sha256(path) != record['sha256']:
            raise ValueError(f'artifact SHA256 changed: {path}')


def publish_artifact(node, source):
    source, destination = Path(source).resolve(), Path(node['artifact'])
    check_artifact(node, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    links = [(source, destination)]
    if node['kind'] == 'calibration':
        links.append((source.with_suffix('.provenance.json'), destination.with_suffix('.provenance.json')))
    for actual, link in links:
        if link.exists() or link.is_symlink():
            if link.resolve() != actual:
                raise FileExistsError(f'refusing to replace artifact {link}')
        else:
            # Exclusive symlink creation; all downstream configs use this stable absolute path.
            link.symlink_to(actual, target_is_directory=actual.is_dir())
    check_artifact(node)


def latest_checkpoint(plan, node, state):
    candidates = []
    for attempt in state.get('attempts', []):
        directory = attempt.get('run_dir') if node['kind'] == 'train' else attempt.get('reconstruction_dir')
        if node['kind'] == 'train' and not directory and attempt.get('run_dir_file'):
            file = Path(attempt['run_dir_file'])
            if file.is_file():
                directory = file.read_text(encoding='utf-8').strip()
        if not directory:
            continue
        for checkpoint in Path(directory).glob('checkpoint-*'):
            state_name = 'training_state.pt' if node['kind'] == 'train' else 'reconstruction_state.pt'
            if (checkpoint / 'checkpoint.complete').is_file() and (checkpoint / state_name).is_file():
                match = re.fullmatch(r'checkpoint-(\d+)', checkpoint.name)
                if match:
                    candidates.append((int(match[1]), checkpoint.stat().st_mtime_ns, checkpoint))
    return str(max(candidates)[2].resolve()) if candidates else None


def python_prefix(plan, distributed=True):
    runtime = plan['site']['runtime']
    python, world = runtime['python'], runtime['nproc_per_node']
    return ([python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f'--nproc_per_node={world}']
            if distributed and world > 1 else [python])


def command_for_node(plan, node, attempt_directory, checkpoint=None):
    attempt_directory = Path(attempt_directory)
    device = plan['site']['runtime']['device']
    if node['kind'] == 'calibration':
        return [*python_prefix(plan, False), str(REPO / 'tools/mlp_neuron_stats.py'), '--config', node['config'],
                '--output', str(attempt_directory / 'neuron_stats.safetensors'), '--covariance', '--device', device]
    if node['kind'] == 'reconstruction':
        return [*python_prefix(plan), str(REPO / 'tools/reconstruct_ffn.py'), '--config', node['config'],
                '--output', str(attempt_directory / 'reconstruction'), '--device', device,
                *(['--resume', checkpoint] if checkpoint else [])]
    if node['kind'] != 'train':
        raise ValueError(f'unknown node kind {node["kind"]}')
    initialization = ['--resume', checkpoint] if checkpoint else \
        ['--init-from', node['init_from']] if node.get('init_from') else []
    return [*python_prefix(plan), str(REPO / 'train_vosr_ablation.py'), '--config', node['config'],
            '--run-dir-file', str(attempt_directory / 'run_dir.txt'), *initialization]


def run_process(command, logfile):
    """No shell, no environment replacement; CUDA/NPU visibility and vendor env survive."""
    with Path(logfile).open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=REPO, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait()
        except BaseException:
            # Stopping a worker must not leave an unclaimed torchrun process using the devices.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def execute_node(plan, node, claim, resume=False):
    if input_snapshot(plan['site'], cached=plan['input_snapshot']) != plan['input_snapshot']:
        raise ValueError('inputs/code changed before this node; stop and materialize a new plan')
    state = read_state(plan, node)
    if state['status'] == 'completed':
        check_artifact(node)
        return 0
    if state['status'] != 'pending' and not resume:
        raise ValueError(f'{node["id"]}: previous attempt exists; inspect logs and explicitly pass --resume')
    checkpoint = latest_checkpoint(plan, node, state) if resume and node['kind'] != 'calibration' else None
    attempt_directory = Path(plan['output']) / 'nodes' / node['id'] / f'attempt-{len(state["attempts"]) + 1:04d}'
    attempt_directory.mkdir(parents=True, exist_ok=False)
    command = command_for_node(plan, node, attempt_directory, checkpoint)
    attempt = {'directory': str(attempt_directory), 'started_at': time.time(), 'worker_id': claim['worker_id'],
               'command': command, 'checkpoint': checkpoint, 'log': str(attempt_directory / 'process.log'),
               'run_dir_file': str(attempt_directory / 'run_dir.txt'),
               'reconstruction_dir': str(attempt_directory / 'reconstruction'),
               'restart_mode': 'checkpoint_resume' if checkpoint else 'fresh_initialization'}
    state['attempts'].append(attempt)
    state['status'] = 'running'
    save_state(plan, node, state)
    print(f'START {node["id"]} {shlex.join(command)}', flush=True)
    try:
        code = run_process(command, attempt['log'])
        attempt['exit_code'], attempt['finished_at'] = code, time.time()
        if Path(attempt['run_dir_file']).is_file():
            attempt['run_dir'] = Path(attempt['run_dir_file']).read_text(encoding='utf-8').strip()
        if code:
            raise RuntimeError(f'process exited {code}; see {attempt["log"]}')
        artifact = (attempt_directory / 'neuron_stats.safetensors' if node['kind'] == 'calibration' else
                    attempt_directory / 'reconstruction' / 'export' if node['kind'] == 'reconstruction' else
                    Path(attempt['run_dir']) / 'export')
        publish_artifact(node, artifact)
        state.update(status='completed', artifact=str(Path(node['artifact']).resolve()), completed_at=time.time(),
                     artifact_snapshot=artifact_snapshot(node, artifact))
        save_state(plan, node, state)
        print(f'DONE  {node["id"]}', flush=True)
        return 0
    except BaseException as error:
        attempt.update(finished_at=time.time(), error=f'{type(error).__name__}: {error}')
        state['status'] = 'failed'
        save_state(plan, node, state)
        print(f'FAILED {node["id"]}: {error}', flush=True)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return 1


def shard_for(node_id, count):
    return int(hashlib.sha256(node_id.encode()).hexdigest(), 16) % count


def queue_status(plan, nodes=None):
    lookup = {node['id']: node for node in plan['nodes']}
    rows = []
    for node in nodes or plan['nodes']:
        state = read_state(plan, node)
        status = state['status']
        if status == 'completed' and not completed(plan, node):
            status = 'invalid_artifact'
        if claim_path(plan, node).exists():
            status = 'claimed'
        unmet = [dependency for dependency in node['deps'] if not completed(plan, lookup[dependency])]
        rows.append({'id': node['id'], 'kind': node['kind'], 'status': status, 'unmet_dependencies': unmet,
                     'attempts': len(state.get('attempts', [])), 'artifact': node['artifact']})
    return rows


def run_queue(plan, nodes, resume=False, dry_run=False, worker_id=None, shards=1, shard_index=0,
              max_jobs=0, max_idle_seconds=0, stop_on_error=False):
    positive_integer(shards, 'shards')
    if not 0 <= shard_index < shards or max_jobs < 0 or max_idle_seconds < 0:
        raise ValueError('invalid shard-index, max-jobs or max-idle-seconds')
    eligible = [node for node in nodes if shard_for(node['id'], shards) == shard_index]
    if dry_run:
        rows = []
        for node in eligible:
            state = read_state(plan, node)
            checkpoint = latest_checkpoint(plan, node, state) if resume and node['kind'] != 'calibration' else None
            directory = Path(plan['output']) / 'nodes' / node['id'] / f'attempt-{len(state["attempts"]) + 1:04d}'
            rows.append({'id': node['id'], 'deps': node['deps'], 'status': state['status'],
                         'command': command_for_node(plan, node, directory, checkpoint)})
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return {'dry_run': True, 'commands': len(rows)}
    report = preflight(plan, nodes, check_runtime=True)
    if report['errors']:
        raise ValueError('preflight failed:\n' + '\n'.join(report['errors']))
    for warning in report['warnings']:
        print(f'WARNING {warning}', flush=True)
    worker_id = worker_id or f'{socket.gethostname()}:{os.getpid()}'
    lookup = {node['id']: node for node in plan['nodes']}
    attempted, failed, idle_since = set(), [], time.monotonic()
    while not max_jobs or len(attempted) < max_jobs:
        selected, claim = None, None
        for node in eligible:
            if node['id'] in attempted or node['id'] in report['node_errors'] or completed(plan, node):
                continue
            state = read_state(plan, node)
            if state['status'] != 'pending' and not resume:
                continue
            if any(not completed(plan, lookup[dependency]) for dependency in node['deps']):
                continue
            claim = claim_node(plan, node, worker_id)
            if claim is not None:
                # Another worker may have completed it between our read and claim.
                if completed(plan, node):
                    release_claim(plan, node, claim['token'])
                    continue
                selected = node
                break
        if selected is None:
            if max_idle_seconds and time.monotonic() - idle_since < max_idle_seconds:
                time.sleep(min(5., max_idle_seconds - (time.monotonic() - idle_since)))
                continue
            break
        attempted.add(selected['id'])
        try:
            if execute_node(plan, selected, claim, resume):
                failed.append(selected['id'])
                if stop_on_error:
                    break
        finally:
            release_claim(plan, selected, claim['token'])
        idle_since = time.monotonic()
    result = {'worker_id': worker_id, 'attempted': sorted(attempted), 'failed': failed,
              'blocked_requirements': report['node_errors'], 'nodes': queue_status(plan, nodes)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def manually_release(plan, node_id, reason, force_remote=False):
    if not reason.strip():
        raise ValueError('a reason is required for releasing a stale claim')
    node = next((node for node in plan['nodes'] if node['id'] == node_id), None)
    if node is None:
        raise ValueError(f'unknown node {node_id}')
    path = claim_path(plan, node)
    claim = json.loads(path.read_text(encoding='utf-8'))
    if claim['hostname'] == socket.gethostname():
        try:
            os.kill(int(claim['pid']), 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('claim owner PID is still alive; stop the worker and its children first')
    elif not force_remote:
        raise ValueError('remote claim: verify its worker/children stopped, then explicitly use --force-remote')
    archive = Path(plan['output']) / 'released_claims' / f'{node_id}.{claim["token"]}.json'
    write_immutable(archive, json.dumps({**claim, 'release_reason': reason, 'released_at': time.time()}, indent=2) + '\n')
    release_claim(plan, node, claim['token'])
    print(f'Released stale claim for {node_id}; next execution requires --resume if an attempt exists')


def evaluation_commands(plan, arm, seed, split, destination):
    points = [point for point in plan['endpoints'] if point['arm'] == arm and point['seed'] == seed]
    if len(points) != 1:
        raise ValueError('arm/seed does not identify one materialized endpoint')
    evaluation = plan['site'].get('eval') or {}
    if not evaluation.get('enabled'):
        raise ValueError('evaluation is disabled; fill site.eval paths and enable it in a new plan')
    spec = evaluation.get(split) or {}
    for key in ('native_lq_dir', 'hq_dir', 'annotations', 'teacher_details'):
        if not spec.get(key) or not Path(spec[key]).exists():
            raise ValueError(f'eval.{split}.{key} needs an existing real path')
    node = next(node for node in plan['nodes'] if node['id'] == points[0]['final_node'])
    destination = Path(destination)
    predictions, metrics_dir = destination / 'predictions', destination / 'ocr'
    runtime = plan['site']['runtime']
    commands = [
        [*python_prefix(plan), str(REPO / 'inference_vosr_ablation.py'), '--export', node['artifact'],
         '--input', spec['native_lq_dir'], '--output', str(predictions), '--device', runtime['device'],
         '--seed', str(evaluation.get('inference_seed', 42))],
        [*python_prefix(plan, False), str(REPO / 'evaluate.py'), '--pred', str(predictions), '--gt', spec['hq_dir'],
         '--lq', spec['native_lq_dir'], '--ann', spec['annotations'], '--ann-ref', evaluation.get('ann_ref', 'gt'),
         '--output', str(metrics_dir), '--ocr-model', evaluation.get('ocr_model', 'PP-OCRv5_server_rec'),
         '--size-buckets', *map(str, HEIGHT_EDGES),
         *(['--ocr-device', evaluation['ocr_device']] if evaluation.get('ocr_device') else [])],
        [*python_prefix(plan, False), str(REPO / 'tools/evaluate_text_bins.py'), '--student-details', str(metrics_dir),
         '--teacher-details', spec['teacher_details'], '--output', str(destination / 'paired'),
         '--height-buckets', *map(str, HEIGHT_EDGES), '--bootstrap', str(evaluation.get('bootstrap', 2000)),
         '--seed', str(evaluation.get('bootstrap_seed', 42))],
    ]
    return node, commands


def run_evaluation(plan, arm, seed, split, tag='v1', dry_run=False):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', tag):
        raise ValueError('evaluation tag must be a simple directory name')
    destination = Path(plan['output']) / 'evaluations' / f'{arm}.s{seed}' / split / tag
    node, commands = evaluation_commands(plan, arm, seed, split, destination)
    if dry_run:
        print(json.dumps({'destination': str(destination), 'commands': commands}, indent=2))
        return {'dry_run': True}
    if not completed(plan, node):
        raise ValueError('final training export is not completed and verified')
    destination.mkdir(parents=True, exist_ok=False)
    write_immutable(destination / 'commands.json', json.dumps(commands, indent=2) + '\n')
    for i, command in enumerate(commands):
        code = run_process(command, destination / f'{i + 1:02d}.log')
        if code:
            raise RuntimeError(f'evaluation command {i + 1} exited {code}; inspect {destination}, rerun with a new --tag')
    write_immutable(destination / 'evaluation.complete', 'completed\n')
    print(f'Evaluation saved: {destination}')
    return {'output': str(destination)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    listing = sub.add_parser('list', help='List named arms without machine paths or torch')
    listing.add_argument('--matrix', default=str(DEFAULT_MATRIX))
    listing.add_argument('--select', nargs='+')
    listing.add_argument('--exclude', nargs='+', default=[])
    make = sub.add_parser('materialize', help='Write an immutable manifest and fully resolved configs')
    make.add_argument('--site', required=True)
    make.add_argument('--output', required=True)
    make.add_argument('--matrix', default=str(DEFAULT_MATRIX))
    make.add_argument('--base', default=str(DEFAULT_BASE))
    make.add_argument('--profile', choices=('smoke', 'pilot', 'full'), default='pilot')
    make.add_argument('--select', nargs='+')
    make.add_argument('--exclude', nargs='+', default=[])
    make.add_argument('--seeds', nargs='+', type=int)
    make.add_argument('--nproc-per-node', type=int)
    make.add_argument('--global-batch-size', type=int)
    make.add_argument('--dry-run', action='store_true')
    for action in ('preflight', 'run', 'status'):
        command = sub.add_parser(action)
        command.add_argument('--output', required=True)
        command.add_argument('--select', nargs='+')
        command.add_argument('--exclude', nargs='+', default=[])
        if action == 'preflight':
            command.add_argument('--check-runtime', action='store_true')
            command.add_argument('--verify-artifacts', action='store_true', help='Re-read every completed artifact and verify SHA256')
        if action == 'run':
            command.add_argument('--resume', action='store_true')
            command.add_argument('--dry-run', action='store_true')
            command.add_argument('--worker-id')
            command.add_argument('--nproc-per-node', type=int, help='Must match materialized world size')
            command.add_argument('--shards', type=int, default=1)
            command.add_argument('--shard-index', type=int, default=0)
            command.add_argument('--max-jobs', type=int, default=0, help='0 = all currently ready jobs')
            command.add_argument('--max-idle-seconds', type=float, default=0,
                                 help='Wait for other workers to finish dependencies; 0 exits when no work is ready')
            command.add_argument('--stop-on-error', action='store_true')
    release = sub.add_parser('release', help='Manually release a verified stale claim, never delete outputs')
    release.add_argument('--output', required=True)
    release.add_argument('--node', required=True)
    release.add_argument('--reason', required=True)
    release.add_argument('--force-remote', action='store_true')
    evaluate = sub.add_parser('evaluate', help='Infer, recognize and compute paired-image confidence intervals')
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--arm', required=True)
    evaluate.add_argument('--seed', type=int, default=42, help='Training seed identifying the endpoint')
    evaluate.add_argument('--split', choices=('validation', 'test'), default='validation')
    evaluate.add_argument('--tag', default='v1')
    evaluate.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.action == 'list':
        _, arms = load_matrix(args.matrix)
        selected = select_arms(arms, args.select, args.exclude)
        print(json.dumps([{'id': name, 'description': arms[name]['description'], 'groups': arms[name]['groups'],
                           'compare_to': arms[name].get('compare_to'), 'seeds': arms[name]['seeds'],
                           'keep_ratio': arms[name]['keep_ratio']} for name in selected], indent=2, ensure_ascii=False))
        return 0
    if args.action == 'materialize':
        site = load_site(args.site, args.nproc_per_node, args.global_batch_size)
        plan, configs = build_plan(site, args.matrix, args.base, args.output, args.profile, args.select,
                                   args.exclude, args.seeds)
        if not args.dry_run:
            materialize(plan, configs)
        print(json.dumps({'plan_id': plan['plan_id'], 'output': plan['output'], 'profile': plan['profile'],
                           'arms': len(plan['arms']), 'endpoints': len(plan['endpoints']), 'nodes': len(plan['nodes']),
                           'calibration_nodes': sum(node['kind'] == 'calibration' for node in plan['nodes']),
                           'reconstruction_nodes': sum(node['kind'] == 'reconstruction' for node in plan['nodes']),
                           'budgets': plan['budgets'], 'dry_run': args.dry_run}, indent=2))
        return 0
    plan = load_plan(args.output)
    if args.action in ('preflight', 'run', 'status'):
        nodes = selected_nodes(plan, args.select, args.exclude)
    if args.action == 'preflight':
        report = preflight(plan, nodes, args.check_runtime, args.verify_artifacts)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report['ok'] else 2
    if args.action == 'status':
        print(json.dumps(queue_status(plan, nodes), indent=2, ensure_ascii=False))
    elif args.action == 'run':
        if args.nproc_per_node is not None and args.nproc_per_node != plan['site']['runtime']['nproc_per_node']:
            raise ValueError('nproc differs from immutable plan; rematerialize to preserve global batch and resume semantics')
        result = run_queue(plan, nodes, args.resume, args.dry_run, args.worker_id, args.shards, args.shard_index,
                           args.max_jobs, args.max_idle_seconds, args.stop_on_error)
        if result.get('failed') or result.get('blocked_requirements'):
            return 2
    elif args.action == 'release':
        manually_release(plan, args.node, args.reason, args.force_remote)
    elif args.action == 'evaluate':
        run_evaluation(plan, args.arm, args.seed, args.split, args.tag, args.dry_run)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(2)
