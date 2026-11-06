import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from tools import ffn_experiments as runner


@pytest.fixture
def site_file(tmp_path):
    # These are intentional CPU unit-test fixtures. They are never used to train.
    teacher = tmp_path / 'teacher.safetensors'
    teacher.write_bytes(b'fixture, not model weights')
    vae = tmp_path / 'vae'
    vae.mkdir()
    (vae / 'config.json').write_text('{}')
    (vae / 'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
    dino_repo, dino_cache = tmp_path / 'dinov2', tmp_path / 'torch_cache'
    dino_repo.mkdir()
    dino_cache.mkdir()
    (dino_repo / 'hubconf.py').write_text('# fixture\n')
    (dino_cache / 'dinov2_fixture.pth').write_bytes(b'fixture')
    hq, lq = tmp_path / 'HQ', tmp_path / 'LQ'
    hq.mkdir()
    lq.mkdir()
    (hq / 'page.02.png').write_bytes(b'fixture')
    (lq / 'page.02.jpg').write_bytes(b'fixture')
    boxes = tmp_path / 'boxes.json'
    boxes.write_text(json.dumps({'page.02': [[0, 0, 100, 0, 100, 40, 0, 40]]}))
    config = {'schema_version': 1, 'teacher_checkpoint': str(teacher), 'vae_path': str(vae),
              'dino': {'local_repo': str(dino_repo), 'cache_dir': str(dino_cache)},
              'runtime': {'python': sys.executable, 'device': 'cpu', 'precision': 'fp32', 'nproc_per_node': 1,
                          'global_batch_size': 16, 'batch_size_per_gpu': 1},
              'train_data': {'dataset_type': 'paired', 'upscale': 4, 'resolution': 512,
                             'sources': [{'name': 'text', 'weight': 1., 'hq_dir': str(hq), 'lq_dir': str(lq),
                                          'text_boxes': str(boxes)}]}}
    path = tmp_path / 'site.yml'
    path.write_text(yaml.safe_dump(config))
    return path


def plan_for(site_file, tmp_path, selected, **kwargs):
    site = runner.load_site(site_file)
    return runner.build_plan(site, output=tmp_path / 'plan', selected=selected, **kwargs)


def test_matrix_has_meaningful_comparisons_and_no_invalid_loss_combination():
    _, arms = runner.load_matrix()
    assert len(arms) >= 35
    assert arms['main_recipe']['seeds'] == [11, 42, 2026]
    assert arms['main_recipe']['training'].get('gt_weight', 0) == 0
    assert arms['main_no_internal']['training']['internal_small_text_weight'] == 0
    assert arms['dense_recipe_control']['keep_ratio'] == 1
    assert arms['k75_recipe_control']['keep_ratio'] == .75
    assert all(arm['compare_to'] in arms for arm in arms.values() if arm.get('compare_to'))


def test_shared_calibration_reconstruction_and_fixed_stage_budgets(site_file, tmp_path):
    plan, configs = plan_for(site_file, tmp_path, ['recon_balanced', 'update_kd', 'feature_kd'], profile='full')
    assert sum(node['kind'] == 'calibration' for node in plan['nodes']) == 1
    assert sum(node['kind'] == 'reconstruction' for node in plan['nodes']) == 1
    assert all(point['e2e_optimizer_steps'] == 34000 for point in plan['endpoints'])
    assert all(point['local_reconstruction_steps'] == 2000 for point in plan['endpoints'])
    for node in plan['nodes']:
        config = configs[node['id']]
        assert '_base_' not in config
        assert Path(config['teacher_checkpoint']).is_absolute()
        if node['kind'] == 'train':
            assert config['training']['gradient_accumulation_steps'] == 16
            assert config['student']['mlp_compression']['keep_ratio'] == .5
            if node['stage'] == 'B':
                assert config['training']['learning_rate'] == 5e-5
                assert config['training']['trainable_parameters'] == [r'^blocks\.\d+\.mlp\.']
            if node['stage'] == 'C':
                assert config['training']['learning_rate'] * config['training']['ffn_lr_multiplier'] == pytest.approx(2e-5)
            if node['stage'] == 'D':
                assert config['training']['learning_rate'] * config['training']['ffn_lr_multiplier'] == pytest.approx(5e-6)
                assert node['data_key'] == 'train_data'


def test_dense_and_headsplit_calibration_never_share(site_file, tmp_path):
    plan, _ = plan_for(site_file, tmp_path, ['obs_balanced', 'headsplit_direct'])
    nodes = [node for node in plan['nodes'] if node['kind'] == 'calibration']
    assert len(nodes) == 2
    assert {node['source'] for node in nodes} == {'dense', 'headsplit'}


def test_sampling_calibration_and_loss_weights_are_independent(site_file, tmp_path):
    plan, configs = plan_for(site_file, tmp_path, ['obs_balanced', 'obs_smallcrop', 'obs_text_kd'])
    assert sum(node['kind'] == 'calibration' for node in plan['nodes']) == 1
    baseline = configs['obs_balanced.s42.C']
    crop = configs['obs_smallcrop.s42.C']
    weighted = configs['obs_text_kd.s42.C']
    assert 'text_sampling' not in baseline['data']
    assert crop['data']['text_sampling']['height_edges'] == [24, 32, 48, 64, 96]
    assert crop['training']['kd_small_text_weight'] == 0
    assert 'text_sampling' not in weighted['data']
    assert weighted['training']['kd_small_text_weight'] == 4


def test_fade_arms_have_equal_total_budget_and_no_reconstruction(site_file, tmp_path):
    plan, configs = plan_for(site_file, tmp_path, ['group:fade'], profile='full')
    assert not any(node['kind'] == 'reconstruction' for node in plan['nodes'])
    assert {point['e2e_optimizer_steps'] for point in plan['endpoints']} == {34000}
    assert configs['fade6k.s42.E']['student']['mlp_compression']['fade_steps'] == 6000
    assert configs['fade12k.s42.E']['student']['mlp_compression']['fade_steps'] == 12000


def test_materialization_immutable_and_tampering_detected(site_file, tmp_path):
    plan, configs = plan_for(site_file, tmp_path, ['obs_uniform'])
    runner.materialize(plan, configs)
    manifest = Path(plan['output']) / 'manifest.json'
    before = manifest.stat().st_mtime_ns
    runner.materialize(plan, configs)
    assert manifest.stat().st_mtime_ns == before
    assert runner.load_plan(plan['output'])['plan_id'] == plan['plan_id']
    changed = copy.deepcopy(configs)
    changed[plan['nodes'][0]['id']]['training']['seed'] = 999
    with pytest.raises(FileExistsError):
        runner.materialize(plan, changed)
    cfg_file = Path(plan['nodes'][0]['config'])
    cfg_file.write_text(cfg_file.read_text() + '\n# altered\n')
    assert plan['nodes'][0]['id'] in runner.preflight(plan)['node_errors']


def test_site_no_fake_paths_and_batch_arithmetic(site_file):
    site = runner.load_site(site_file, nproc=4)
    assert site['runtime']['gradient_accumulation_steps'] == 4
    with pytest.raises(ValueError, match='divide exactly'):
        runner.load_site(site_file, nproc=3)
    config = yaml.safe_load(site_file.read_text())
    config['teacher_checkpoint'] = None
    site_file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='actual local path'):
        runner.load_site(site_file)


def test_preflight_blocked_optional_assets_and_lfs_pointer(site_file, tmp_path):
    plan, configs = plan_for(site_file, tmp_path, ['obs_uniform', 'roi_ocr', 'headsplit_direct'])
    runner.materialize(plan, configs)
    report = runner.preflight(plan)
    assert report['errors'] == []
    assert 'roi_ocr.s42.C' in report['node_errors']
    assert 'roi_ocr.s42.B' not in report['node_errors']
    assert 'obs_uniform.s42.B' not in report['node_errors']
    Path(plan['site']['teacher_checkpoint']).write_text('version https://git-lfs.github.com/spec/v1\n')
    assert any('LFS pointer' in error for error in runner.preflight(plan)['errors'])


def test_split_manifest_overlap_blocks_training(site_file, tmp_path):
    config = yaml.safe_load(site_file.read_text())
    manifests = {}
    for split, content in [('train', 'same\ntrain2\n'), ('validation', 'same\n'), ('test', 'test1\n')]:
        path = tmp_path / f'{split}.txt'
        path.write_text(content)
        manifests[split] = str(path)
    config['split_manifests'] = manifests
    site_file.write_text(yaml.safe_dump(config))
    plan, configs = plan_for(site_file, tmp_path, ['random_none'])
    runner.materialize(plan, configs)
    assert any('data leakage' in error for error in runner.preflight(plan)['errors'])


def test_atomic_claim_one_winner_and_release_does_not_delete_artifact(site_file, tmp_path):
    plan, _ = plan_for(site_file, tmp_path, ['random_none'])
    node = plan['nodes'][0]
    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = list(executor.map(lambda index: runner.claim_node(plan, node, str(index)), range(8)))
    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    runner.release_claim(plan, node, 'wrong-token')
    assert runner.claim_path(plan, node).exists()
    runner.release_claim(plan, node, winners[0]['token'])
    assert not runner.claim_path(plan, node).exists()


def test_dry_run_no_claims_states_or_processes_and_resume_command_correct(site_file, tmp_path, capsys):
    plan, configs = plan_for(site_file, tmp_path, ['random_none'])
    runner.materialize(plan, configs)
    before = sorted(str(path) for path in Path(plan['output']).rglob('*'))
    runner.run_queue(plan, plan['nodes'], dry_run=True)
    assert sorted(str(path) for path in Path(plan['output']).rglob('*')) == before
    node = plan['nodes'][1]
    command = runner.command_for_node(plan, node, tmp_path / 'attempt', checkpoint='/actual/completed/checkpoint')
    assert '--resume' in command and '--init-from' not in command
    fresh = runner.command_for_node(plan, node, tmp_path / 'attempt')
    assert '--init-from' in fresh and '--resume' not in fresh


def test_latest_checkpoint_requires_complete_marker(site_file, tmp_path):
    plan, _ = plan_for(site_file, tmp_path, ['random_none'])
    run_dir = tmp_path / 'actual_run'
    for step in (10, 20):
        checkpoint = run_dir / f'checkpoint-{step:08d}'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'training_state.pt').write_bytes(b'unit fixture')
    (run_dir / 'checkpoint-00000010/checkpoint.complete').write_text('10')
    state = {'attempts': [{'run_dir': str(run_dir)}]}
    assert runner.latest_checkpoint(plan, plan['nodes'][0], state).endswith('checkpoint-00000010')


def test_ready_queue_skips_blocked_dependencies_and_runs_independent_arm(site_file, tmp_path, monkeypatch, capsys):
    plan, _ = plan_for(site_file, tmp_path, ['random_none', 'dense_control'])
    done, executed = set(), []
    monkeypatch.setattr(runner, 'preflight', lambda *args, **kwargs: {'errors': [], 'node_errors': {}, 'warnings': []})
    monkeypatch.setattr(runner, 'completed', lambda plan, node: node['id'] in done)

    def execute(plan, node, claim, resume=False):
        executed.append(node['id'])
        if node['id'].startswith('dense_control'):
            done.add(node['id'])
            return 0
        return 1

    # Matrix order places dense_control first. Limit to one failure plus independent work;
    # force the random arm first to exercise the blocked-dependency path.
    nodes = sorted(plan['nodes'], key=lambda node: (not node['id'].startswith('random_none'), node['id']))
    monkeypatch.setattr(runner, 'execute_node', execute)
    result = runner.run_queue(plan, nodes, max_jobs=2)
    assert executed == ['random_none.s42.B', 'dense_control.s42.B']
    assert result['failed'] == ['random_none.s42.B']
    assert not list((Path(plan['output']) / 'claims').glob('*.json'))


def test_process_environment_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv('FFN_UNIT_ENV_MARKER', 'preserved')
    logfile = tmp_path / 'process.log'
    code = runner.run_process([sys.executable, '-c', 'import os; print(os.environ["FFN_UNIT_ENV_MARKER"])'], logfile)
    assert code == 0 and logfile.read_text().strip() == 'preserved'


def test_dynamic_queue_default_and_explicit_shards_cover_nodes_once(site_file, tmp_path):
    plan, _ = plan_for(site_file, tmp_path, ['obs_balanced', 'recon_balanced'])
    partitions = [{node['id'] for node in plan['nodes'] if runner.shard_for(node['id'], 3) == index}
                  for index in range(3)]
    assert set.union(*partitions) == {node['id'] for node in plan['nodes']}
    assert sum(map(len, partitions)) == len(plan['nodes'])


def test_global_annotation_and_image_subset_paths_are_resolved_and_checked(site_file, tmp_path):
    config = yaml.safe_load(site_file.read_text())
    source = config['train_data']['sources'][0]
    source.pop('text_boxes')
    config['train_data']['text_boxes'] = 'boxes.json'
    config['train_data']['image_names'] = 'selected.txt'
    (tmp_path / 'selected.txt').write_text('page.02\n')
    site_file.write_text(yaml.safe_dump(config))
    plan, configs = plan_for(site_file, tmp_path, ['obs_uniform'], profile='smoke')
    runner.materialize(plan, configs)
    assert plan['site']['train_data']['text_boxes'] == str(tmp_path / 'boxes.json')
    assert plan['site']['train_data']['image_names'] == str(tmp_path / 'selected.txt')
    report = runner.preflight(plan)
    assert report['ok'] and report['data'][0]['num_selected_images'] == 1
    calibration = next(configs[node['id']] for node in plan['nodes'] if node['kind'] == 'calibration')
    assert calibration['calibration']['samples'] == 8
    (tmp_path / 'selected.txt').write_text('does-not-exist\n')
    errors = runner.preflight(plan)['errors']
    assert any('filenames are not present' in error for error in errors)
    assert any('changed since materialization' in error for error in errors)


def test_input_strong_sha_detects_same_size_same_mtime_changes(site_file, tmp_path):
    plan, configs = plan_for(site_file, tmp_path, ['random_none'])
    runner.materialize(plan, configs)
    checkpoint = Path(plan['site']['teacher_checkpoint'])
    stat = checkpoint.stat()
    data = checkpoint.read_bytes()
    checkpoint.write_bytes(b'X' + data[1:])
    os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert checkpoint.stat().st_size == stat.st_size
    assert any('changed since materialization' in error for error in runner.preflight(plan)['errors'])


def test_artifact_metadata_and_explicit_strong_hash_check(tmp_path):
    artifact = tmp_path / 'stats.safetensors'
    artifact.write_bytes(b'original-statistics')
    artifact.with_suffix('.provenance.json').write_text('{}')
    state = {'artifact_snapshot': runner.artifact_snapshot({'kind': 'calibration'}, artifact)}
    runner.verify_artifact_snapshot(state, strong=True)
    stat = artifact.stat()
    artifact.write_bytes(b'X' + artifact.read_bytes()[1:])
    os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    runner.verify_artifact_snapshot(state)  # documented fast metadata check
    with pytest.raises(ValueError, match='SHA256 changed'):
        runner.verify_artifact_snapshot(state, strong=True)


def test_every_generated_config_matches_actual_training_and_reconstruction_contracts(site_file, tmp_path):
    from argparse import Namespace
    from models.ffn_distillation import text_distillation_spec, validate_text_data
    from models.mlp_compression import mlp_compression_spec
    from tools.mlp_neuron_stats import calibration_options
    from tools.reconstruct_ffn import reconstruction_options
    plan, configs = plan_for(site_file, tmp_path, ['all'], profile='smoke')
    for node in plan['nodes']:
        config = configs[node['id']]
        text_distillation_spec(config)
        validate_text_data(config['data'])
        if config['student']['mlp_compression'] is not None:
            mlp_compression_spec(config['student']['mlp_compression'], 36, 4096)
        if node['kind'] == 'calibration':
            calibration_options(config, Namespace())
        if node['kind'] == 'reconstruction':
            reconstruction_options(config)
