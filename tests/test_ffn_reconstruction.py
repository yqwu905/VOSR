"""Weighted calibration, real nonlinear updates, and deterministic local-training resume."""
import json
import os
from pathlib import Path
import sys
from types import ModuleType

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import pytest
import torch
from torch import nn

from models.ffn_reconstruction import (WeightedMoments, local_reconstruction_loss, reconstruction_microstep,
                                      select_reconstruction_parameters, subsample_token_weights, token_weights)


def test_balanced_weights_allocate_mass_and_redistribute_empty_strata():
    text = torch.zeros(2, 1, 4, 4)
    small = torch.zeros_like(text)
    text[0, :, :2] = 1
    small[0, :, :2, :2] = 1
    weights = token_weights(text, small, tokens=4)
    torch.testing.assert_close(weights[0], torch.tensor([2.0, 1.2, .4, .4]))
    torch.testing.assert_close(weights[1], torch.ones(4))  # natural image, no annotation
    torch.testing.assert_close(weights.mean(1), torch.ones(2))
    # One sub-token character is retained by area pooling, not thresholded away.
    text.zero_()
    small.zero_()
    text[0, 0, 0, 0] = small[0, 0, 0, 0] = 1
    weights = token_weights(text, small, tokens=4)
    assert weights[0, 0] > weights[0, 1]
    assert torch.isfinite(weights).all()


def test_weighted_welford_matches_centered_reference_across_chunks():
    generator = torch.Generator().manual_seed(12)
    activation = 1e6 + torch.randn(31, 5, generator=generator, dtype=torch.float64)
    weights = torch.rand(31, generator=generator, dtype=torch.float64)
    weights[::5] = 0
    mean = (activation * weights[:, None]).sum(0) / weights.sum()
    centered = activation - mean
    reference = centered.T @ (centered * weights[:, None]) / weights.sum()
    chunked = WeightedMoments(5, covariance=True, chunk_tokens=3)
    chunked.update(activation[:11], weights[:11])
    chunked.update(activation[11:], weights[11:])
    actual = chunked.result()
    torch.testing.assert_close(actual['mean'], mean.float())
    torch.testing.assert_close(actual['covariance'], reference.float(), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual['variance'], reference.diag().float(), atol=2e-6, rtol=2e-6)
    assert torch.linalg.eigvalsh(actual['covariance']).min() > -1e-6
    diagonal = WeightedMoments(5, covariance=False, chunk_tokens=4)
    diagonal.update(activation, weights)
    torch.testing.assert_close(diagonal.result()['variance'], actual['variance'])


def test_padding_is_never_given_background_mass():
    text, valid = torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)
    valid[:, :, :, :2] = 1
    for mode in ('uniform', 'text', 'balanced'):
        weights = token_weights(text, text, tokens=4, mode=mode, valid_mask=valid)
        torch.testing.assert_close(weights, torch.tensor([[2., 0., 2., 0.]]))
    with pytest.raises(ValueError, match='unpadded'):
        token_weights(text, text, tokens=4, valid_mask=valid * 0)


def test_token_subsampling_is_deterministic_and_keeps_inclusion_weights():
    weights = torch.tensor([[1., 0., 2., 0., 3., 4.], [0., 2., 0., 0., 0., 0.]])
    selected, indices = subsample_token_weights(weights, 2, seed=14)
    state = torch.get_rng_state()
    repeated, repeated_indices = subsample_token_weights(weights, 2, seed=14)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    torch.testing.assert_close(indices, repeated_indices, rtol=0, atol=0)
    torch.testing.assert_close(selected, repeated, rtol=0, atol=0)
    assert indices.numel() == indices.unique().numel() == 3
    assert (indices[:2] < 6).all() and indices[-1] == 7
    assert (weights.flatten()[indices] > 0).all()
    torch.testing.assert_close(selected, weights.flatten()[indices] * torch.tensor([2., 2., 1.]))
    for limit in (0, 4, 100):
        original, all_indices = subsample_token_weights(weights, limit, seed=14)
        assert original is weights and all_indices is None
    with pytest.raises(ValueError, match='positive-weight'):
        subsample_token_weights(torch.zeros(1, 5), 2)


@pytest.mark.parametrize('weights', [torch.tensor([-1., 1.]), torch.tensor([float('nan'), 1.])])
def test_moments_reject_invalid_weights(weights):
    with pytest.raises(ValueError, match='finite'):
        WeightedMoments(3).update(torch.ones(2, 3), weights)


def tiny_models():
    pytest.importorskip('timm')
    from models.lightningdit_ablation import AblationLightningDiT
    from ablation_utils import load_backbone_state
    torch.manual_seed(6)
    config = dict(input_size=8, patch_size=2, in_channels=4, out_channels=2, hidden_size=16,
                  depth=2, num_heads=2, mlp_ratio=2, z_dims=8, auxiliary_time_cond=True,
                  use_rope=True, use_qknorm=True, use_swiglu=True, use_rmsnorm=True)
    teacher = AblationLightningDiT(**config).eval().requires_grad_(False)
    student = AblationLightningDiT(**config, mlp_compression=dict(type='neuron_prune', keep_ratio=.5,
                                                                                score='random', compensation='none'))
    load_backbone_state(student, teacher.state_dict(), allow_new_modules=True)
    inp = torch.randn(1, 4, 8, 8)
    features = [torch.randn(1, 4, 8)]
    return teacher, student, inp, features


def test_grouped_weighted_statistics_replay_same_samples():
    from tools.mlp_neuron_stats import neuron_statistics
    teacher, _, inp, features = tiny_models()
    weights = torch.linspace(.2, 2, 16)[None]
    samples = [(inp, features, weights), (inp * .5, features, weights.flip(1))]
    grouped = neuron_statistics(teacher, lambda: iter(samples), covariance=True, layer_group_size=1, chunk_tokens=5)
    together = neuron_statistics(teacher, samples, covariance=True, chunk_tokens=5)
    for key in grouped:
        torch.testing.assert_close(grouped[key], together[key])
    with pytest.raises(ValueError, match='re-iterable'):
        neuron_statistics(teacher, iter(samples), covariance=True, layer_group_size=1)


def test_subsampled_statistics_equal_explicit_selected_activations():
    from tools.mlp_neuron_stats import neuron_statistics
    teacher, _, inp, features = tiny_models()
    weights = torch.linspace(.2, 2, 16)[None]
    selected_weights, indices = subsample_token_weights(weights, 5, seed=29)
    activations = []
    handle = teacher.blocks[0].mlp.w3.register_forward_pre_hook(lambda module, args: activations.append(args[0].clone()))
    try:
        with torch.no_grad():
            teacher(inp, inp.new_ones(1), inp.new_zeros(1), features)
    finally:
        handle.remove()
    selected = activations[0].flatten(0, 1)[indices]
    mean = (selected.double() * selected_weights.double()[:, None]).sum(0) / selected_weights.double().sum()
    centered = selected.double() - mean
    covariance = centered.T @ (centered * selected_weights.double()[:, None]) / selected_weights.double().sum()
    samples = [(inp, features, selected_weights, indices)]
    grouped = neuron_statistics(teacher, samples, covariance=True, layer_group_size=1, chunk_tokens=2)
    all_layers = neuron_statistics(teacher, samples, covariance=True, chunk_tokens=2)
    torch.testing.assert_close(grouped['mean'][0], mean.float())
    torch.testing.assert_close(grouped['covariance'][0], covariance.float())
    for key in grouped:
        torch.testing.assert_close(grouped[key], all_layers[key])


def test_every_layer_updates_both_matrices_and_teacher_is_unchanged():
    teacher, student, inp, features = tiny_models()
    teacher_state = {key: value.clone() for key, value in teacher.state_dict().items()}
    before = {key: value.clone() for key, value in student.state_dict().items()}
    layers, parameters = select_reconstruction_parameters(student)
    optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=0)
    weights = torch.ones(1, 16)
    first = last = None
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        losses = reconstruction_microstep(teacher, student, inp, features, weights, layers=layers)
        assert set(losses) == set(layers)
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
        first = sum(losses.values()) if first is None else first
        last = sum(losses.values())
        optimizer.step()
    assert last < first * .6
    for index in layers:
        for matrix in ('w12', 'w3'):
            key = f'blocks.{index}.mlp.{matrix}.weight'
            assert not torch.equal(student.state_dict()[key], before[key])
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_state[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    for key, value in student.state_dict().items():
        if '.mlp.' not in key:
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_w3_only_ablation_and_hook_cleanup_after_failure():
    teacher, student, inp, features = tiny_models()
    layers, parameters = select_reconstruction_parameters(student, patterns=[r'\.mlp\.w3\.'])
    assert all(not block.mlp.w12.weight.requires_grad for block in student.blocks)
    with pytest.raises(ValueError, match='positive token weight'):
        reconstruction_microstep(teacher, student, inp, features, torch.zeros(1, 16), layers=layers)
    assert not any(block.mlp._forward_hooks for block in teacher.blocks)
    losses = reconstruction_microstep(teacher, student, inp, features, torch.ones(1, 16), layers=layers)
    assert set(losses) == set(layers)
    assert all(parameter.grad is not None for parameter in parameters)


def test_local_loss_gate_and_energy_floor_are_real_not_detached():
    prediction = torch.tensor([[[1., 4.], [2., 7.]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    gate = torch.tensor([[1., 0.]])
    loss = local_reconstruction_loss(prediction, target, gate, torch.ones(1, 2), energy_floor=1.)
    loss.backward()
    assert prediction.grad[..., 0].abs().sum() > 0
    assert prediction.grad[..., 1].abs().sum() == 0


def _gradient_worker(rank, directory):
    import torch.distributed as dist
    from tools.reconstruct_ffn import synchronize_gradients
    dist.init_process_group('gloo', init_method='file://' + str(Path(directory) / 'rendezvous'), rank=rank, world_size=2)
    try:
        parameters = [nn.Parameter(torch.zeros(5)), nn.Parameter(torch.zeros(3))]
        for index, parameter in enumerate(parameters):
            parameter.grad = torch.full_like(parameter, float(rank + index + 1))
        synchronize_gradients(parameters, 2, bucket_bytes=8)
        for index, parameter in enumerate(parameters):
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, index + 1.5))
        if rank == 1:
            parameters[0].grad = None
        try:
            synchronize_gradients(parameters, 2, bucket_bytes=8)
        except FloatingPointError:
            (Path(directory) / f'rank-{rank}.passed').write_text('averaged and rejected a missing rank gradient')
        else:
            raise AssertionError('All ranks must reject a missing gradient on any rank')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('VOSR_TEST_DDP') != '1',
                    reason='Set VOSR_TEST_DDP=1 on a host permitting local Gloo sockets')
def test_two_rank_gradient_average_and_collective_failure(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_gradient_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert all((tmp_path / f'rank-{rank}.passed').is_file() for rank in range(2))


def test_mixed_calibration_and_reconstruction_resume_export_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip('timm')
    import yaml
    from ablation_utils import load_export, read_weights
    from test_ablation_head_split import paired_trainer_config
    from ablation_hourglass_smoke import TinyVAE, fake_dino_features
    import tools.mlp_neuron_stats as calibrate
    import tools.reconstruct_ffn as reconstruct

    fake_vae = ModuleType('models.qwenimage_vae2d')
    fake_vae.AutoencoderKLQwenImage2D = TinyVAE
    monkeypatch.setitem(sys.modules, 'models.qwenimage_vae2d', fake_vae)
    for module in (calibrate, reconstruct):
        monkeypatch.setattr(module, 'load_dino', lambda *args: nn.Identity())
        monkeypatch.setattr(module, 'dino_features', fake_dino_features)
    path = paired_trainer_config(tmp_path)
    config = yaml.safe_load(path.read_text())
    config['student']['attn_compression'] = None  # Dense attention baseline
    stats_path = tmp_path / 'weighted.safetensors'
    config['student']['mlp_compression'] = dict(type='neuron_prune', keep_ratio=.5, score='obs',
                                               compensation='lstsq', neuron_stats=str(stats_path), fade_steps=0)
    data = config['data']
    lq, hq, boxes = (data.pop(key) for key in ('lq_dir', 'hq_dir', 'text_boxes'))
    data['sources'] = [dict(name='text', weight=.7, lq_dir=lq, hq_dir=hq, text_boxes=boxes),
                       dict(name='natural', weight=.3, lq_dir=lq, hq_dir=hq, allow_missing_text=True,
                            text_crop_prob=0)]
    config['calibration'] = dict(samples=3, weighting='balanced', covariance=True, layer_group_size=1,
                                  chunk_tokens=4, moment_device='cpu', seed=5, max_tokens_per_sample=4)
    config['reconstruction'] = dict(max_steps=3, batch_size_per_gpu=1, gradient_accumulation_steps=1,
        learning_rate=.003, lr_warmup_steps=0, precision='fp32', save_every=1, log_every=1, seed=17)
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, 'argv', ['calibrate', '--config', str(path), '--output', str(stats_path), '--device', 'cpu'])
    calibrate.main()
    assert stats_path.is_file()
    provenance = json.loads(stats_path.with_suffix('.provenance.json').read_text())
    assert len(provenance['source']['fingerprint']) == 64
    assert len(provenance['data']['sources']) == 2
    assert len(provenance['samples']) == 3
    assert provenance['calibration']['weighting'] == 'balanced'
    assert provenance['calibration']['max_tokens_per_sample'] == 4
    assert provenance['token_sampling']['approximate']
    assert provenance['token_sampling']['sampled_tokens'] == 12
    assert provenance['token_sampling']['crop_draws'] == 3
    assert provenance['layer_group_passes'] == 4

    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    monkeypatch.setattr(sys, 'argv', ['reconstruct', '--config', str(path), '--output', str(full), '--device', 'cpu'])
    reconstruct.main()
    record = json.loads((full / 'metrics.jsonl').read_text().splitlines()[-1])
    assert record['updates_per_layer'] == {str(i): 3 for i in range(4)}
    assert (full / 'export/export.complete').is_file()
    assert (full / 'checkpoint-00000001/checkpoint.complete').is_file()
    stats_path.unlink()  # a narrow reconstruction checkpoint resumes without pruning statistics
    monkeypatch.setattr(sys, 'argv', ['reconstruct', '--config', str(path), '--output', str(resumed),
                                     '--resume', str(full / 'checkpoint-00000001'), '--device', 'cpu'])
    reconstruct.main()
    for key, value in read_weights(full / 'export').items():
        torch.testing.assert_close(value, read_weights(resumed / 'export')[key], rtol=0, atol=0)
    restored = load_export(resumed / 'export')
    assert all(block.mlp.w3.in_features == 21 for block in restored.blocks)
    monkeypatch.setattr(sys, 'argv', ['reconstruct', '--config', str(path), '--output', str(full), '--device', 'cpu'])
    with pytest.raises(RuntimeError, match='not empty'):
        reconstruct.main()
    monkeypatch.setattr(sys, 'argv', ['reconstruct', '--config', str(path), '--output', str(full),
                                     '--resume', str(full / 'checkpoint-00000003'), '--device', 'cpu'])
    with pytest.raises(RuntimeError, match='already has an export'):
        reconstruct.main()
