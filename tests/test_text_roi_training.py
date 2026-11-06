"""Native crop geometry, balanced sampling and gradients for the half-width text recipe."""
import json
import random
import socket
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from dataloaders.paired_crops import (PairedCropDataset, PairedMixtureDataset, canonical_quad,
                                     collate_paired_samples, dataset_options_from_config, polygon_mask)
from models.pixel_losses import KLOCRLoss
from models.text_losses import TextROILoss, rectify_text_roi


def pair_dirs(tmp_path, names=('page.02',), size=(64, 64)):
    lq, hq = tmp_path / 'lq', tmp_path / 'hq'
    lq.mkdir()
    hq.mkdir()
    rng = np.random.default_rng(7)
    for name in names:
        pixels = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(hq / f'{name}.png')
        Image.fromarray(255 - pixels).save(lq / f'{name}.png')
    return str(lq), str(hq)


def annotations(tmp_path, value):
    path = tmp_path / 'boxes.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return str(path)


def test_native_polygon_geometry_metadata_and_no_resizing(tmp_path):
    lq, hq = pair_dirs(tmp_path)
    # Diamond: AABB area 32*32, actual text polygon area 512; the two must not be confused.
    quad = [[16, 32], [32, 16], [48, 32], [32, 48]]
    boxes = annotations(tmp_path, {'page.02.png': [
        dict(points=quad, transcription='小字', teacher_correct=False),
        dict(bbox=[2, 2, 10, 8], transcript='AB', alignment_valid=False, teacher_correct=False),
    ]})
    ds = PairedCropDataset(lq, hq, 64, boxes, text_metadata=True, small_text_height=24)
    item = ds.sample('page.02', random.Random(0))
    assert item['hq'].shape == (3, 64, 64)
    torch.testing.assert_close(item['hq'] + item['lq'], torch.ones_like(item['hq']))
    assert item['sample_name'] == 'page.02'
    first, second = item['text_lines']
    assert first['quad'] == quad and first['bbox'] == [16., 16., 48., 48.]
    assert first['native_height'] == pytest.approx(np.sqrt(512))
    assert first['fully_visible'] and first['visible_fraction'] == 1
    assert first['transcript'] == '小字'
    assert 480 <= item['small_text_mask'].sum() < 650
    assert item['text_mask'][0, 16, 16] == 0  # Inside AABB but outside the polygon.
    assert item['kd_trust_mask'][0, 32, 32] == pytest.approx(.1)
    assert item['kd_trust_mask'][0, 4, 4] == 1  # Unaligned GT must not override teacher trust.
    assert second['alignment_valid'] is False


def test_cropped_line_keeps_original_height_and_disables_ocr_for_partial_line(tmp_path):
    lq, hq = pair_dirs(tmp_path, size=(80, 80))
    boxes = annotations(tmp_path, {'page.02': [dict(bbox=[8, 8, 72, 72], transcript='not a full line')]})
    ds = PairedCropDataset(lq, hq, 32, boxes, text_crop_prob=1, text_metadata=True,
                           text_sampling=dict(jitter_fraction=0, min_visible_fraction=.1))
    item = ds.sample('page.02', random.Random(3))
    line, = item['text_lines']
    assert item['crop_origin'].tolist() == [24, 24]
    assert line['quad'] == [[-16., -16.], [48., -16.], [48., 48.], [-16., 48.]]
    assert line['bbox'] == [0., 0., 32., 32.]
    assert line['native_height'] == 64 and line['visible_fraction'] == .25
    assert line['valid'] and not line['fully_visible']
    assert item['small_text_mask'].sum() == 0  # Clipping a 64px line must not relabel it as 32px small text.


def test_unreliable_annotations_do_not_change_kd_trust_and_padding_is_invalid(tmp_path):
    lq, hq = pair_dirs(tmp_path, size=(16, 20))
    boxes = annotations(tmp_path, {'page.02': [dict(bbox=[2, 2, 8, 8], valid=False, teacher_correct=False)]})
    ds = PairedCropDataset(lq, hq, 32, boxes, text_metadata=True)
    item = ds[0]
    assert item['kd_trust_mask'].min() == 1
    assert item['text_mask'].sum() == 0
    assert item['valid_mask'].sum() == 16 * 20
    assert item['valid_mask'][0, 20, 20] == 0


def test_legacy_tensor_contract_and_variable_line_collation(tmp_path):
    lq, hq = pair_dirs(tmp_path, names=('a', 'b'))
    boxes = annotations(tmp_path, {'a': [[2, 2, 10, 8]], 'b': []})
    old = PairedCropDataset(lq, hq, 64, boxes, text_mask=True)
    assert set(old[0]) == {'hq', 'lq', 'text_mask'}
    assert old[0]['text_mask'].sum() == 8 * 6
    ds = PairedCropDataset(lq, hq, 64, boxes, text_metadata=True)
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_paired_samples)))
    assert batch['text_mask'].shape == (2, 1, 64, 64)
    assert [len(lines) for lines in batch['text_lines']] == [1, 0]
    assert batch['text_lines'][0][0]['quad'][0] == [2., 2.]
    assert batch['sample_name'] == ['a', 'b']


@pytest.mark.parametrize('bad', [
    [0, 0, 0, 5], [0, 0, float('nan'), 5],
    [[0, 0], [5, 5], [0, 5], [5, 0]],  # Crossed corners cannot be silently sorted.
    [[0, 0], [5, 0], [2, 2], [0, 5]],  # Degenerate/concave.
])
def test_malformed_geometry_rejected(bad):
    with pytest.raises(ValueError):
        canonical_quad(bad)


def test_out_of_bounds_annotations_report_coordinate_system(tmp_path):
    lq, hq = pair_dirs(tmp_path)
    boxes = annotations(tmp_path, {'page.02': [[0, 0, 500, 500]]})
    ds = PairedCropDataset(lq, hq, 64, boxes, text_metadata=True)
    with pytest.raises(ValueError, match='HQ/output pixels'):
        ds[0]


def test_global_line_balancing_and_height_buckets(tmp_path):
    lq, hq = pair_dirs(tmp_path, names=('few', 'many'), size=(16, 16))
    boxes = annotations(tmp_path, {'few': [[0, 0, 8, 4]],
                                   'many': [[i, 0, i + 4, 12] for i in range(9)]})
    # 1 vs 9 lines: each line has equal probability, not each image.
    ds = PairedCropDataset(lq, hq, 16, boxes, text_crop_prob=1, text_metadata=True,
                           text_sampling=dict(mode='line_balanced'))
    rng = random.Random(7)
    names = [ds.sample_index(0, rng)['sample_name'] for _ in range(200)]
    assert .84 < names.count('many') / len(names) < .96
    # "many" vertical boxes rotate to native height 4, so use non-tall boxes for the bucket test.
    boxes2 = annotations(tmp_path, {'few': [[0, 0, 12, 4]],
                                    'many': [[0, 0, 12, 12]]})
    ds2 = PairedCropDataset(lq, hq, 16, boxes2, text_crop_prob=1, text_metadata=True,
                            text_sampling=dict(mode='height_buckets', height_edges=[8], bucket_weights=[1, 0]))
    assert {ds2.sample_index(i, random.Random(i))['sample_name'] for i in range(30)} == {'few'}


def test_mixture_overrides_natural_source_and_reproducible_resume(tmp_path):
    lq, hq = pair_dirs(tmp_path, names=('a', 'b'), size=(80, 80))
    boxes = annotations(tmp_path, {'a': [[4, 4, 20, 12]], 'b': [[30, 30, 60, 60]]})
    config = dict(resolution=32, text_metadata=True, text_boxes=boxes, text_crop_prob=1,
                  deterministic_crops=True, seed=51,
                  text_sampling=dict(mode='height_buckets', height_edges=[16], bucket_weights=[1, 0]),
                  sources=[
                      dict(name='small', weight=1, lq_dir=lq, hq_dir=hq, image_names=['a']),
                      dict(name='natural', weight=1, lq_dir=lq, hq_dir=hq, text_boxes=None, text_crop_prob=0,
                           allow_missing_text=True, text_sampling=dict(mode='image'), image_names=['b']),
                  ])
    ds = PairedMixtureDataset(config)
    first = [ds[i] for i in range(12)]
    # Ambient Python RNG and sample access order do not affect index+epoch deterministic crops.
    random.seed(999)
    assert torch.equal(ds[4]['hq'], first[4]['hq'])
    ds.set_epoch(2)
    assert any(not torch.equal(ds[i]['hq'], first[i]['hq']) for i in range(12))
    ds.set_epoch(0)
    for i, expected in enumerate(first):
        actual = ds[i]
        assert actual['source_id'] == expected['source_id']
        torch.testing.assert_close(actual['hq'], expected['hq'])
        if actual['source_id'] == 'natural':
            assert actual['text_lines'] == [] and actual['text_mask'].sum() == 0
    batched = collate_paired_samples(first)
    assert batched['small_text_mask'].shape == (12, 1, 32, 32)
    assert set(batched['source_id']) == {'small', 'natural'}


def test_restricted_image_names_file_and_validation(tmp_path):
    lq, hq = pair_dirs(tmp_path, names=('a', 'page.02'))
    selected = tmp_path / 'selected.txt'
    selected.write_text('# mined training images\npage.02.png\n')
    ds = PairedCropDataset(lq, hq, 32, image_names=str(selected))
    assert ds.names == ['page.02']
    assert dataset_options_from_config({'image_names': ['a'], 'not_an_option': 1}) == {'image_names': ['a']}
    with pytest.raises(FileNotFoundError, match='HQ images not found'):
        PairedCropDataset(lq, hq, 32, image_names=['absent'])
    with pytest.raises(ValueError, match='nonempty'):
        PairedCropDataset(lq, hq, 32, image_names=[])


def test_deterministic_crops_do_not_depend_on_worker_count(tmp_path):
    # torch's tensor FD sharing needs a Unix socket even with spawn. Restricted
    # CI may prohibit socket creation; fail-fast/skip instead of hanging its queue.
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.close()
    except (PermissionError, OSError):
        pytest.skip('Host does not permit Unix sockets required by DataLoader tensor sharing')
    lq, hq = pair_dirs(tmp_path, names=('a', 'b', 'c'), size=(80, 80))
    boxes = annotations(tmp_path, {name: [[8, 8, 64, 24]] for name in ('a', 'b', 'c')})
    ds = PairedCropDataset(lq, hq, 32, boxes, text_metadata=True, deterministic_crops=True,
                           text_crop_prob=.8, seed=19)
    loader = DataLoader(ds, batch_size=2, num_workers=2, multiprocessing_context='spawn',
                        collate_fn=collate_paired_samples, timeout=20)
    for epoch in (0, 3):
        ds.set_epoch(epoch)
        expected = torch.stack([ds[i]['hq'] for i in range(len(ds))])
        actual = torch.cat([batch['hq'] for batch in loader])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def line(box, **kwargs):
    return dict(quad=canonical_quad(box).tolist(), valid=True, alignment_valid=True,
                fully_visible=True, visible_fraction=1., **kwargs)


def test_roi_charbonnier_is_line_balanced_not_area_balanced_and_masks_background():
    target = torch.zeros(1, 3, 32, 40, requires_grad=True)
    pred = torch.zeros_like(target)
    pred[..., 2:6, 2:6] = .2
    pred[..., 8:24, 12:36] = .8
    pred[..., 28:, :] = 1  # Outside both polygons.
    pred.requires_grad_(True)
    lines = [[line([2, 2, 6, 6]), line([12, 8, 36, 24])]]
    losses = TextROILoss()(pred, target, lines)
    expected = (np.sqrt(.2 ** 2 + 1e-6) - .001 + np.sqrt(.8 ** 2 + 1e-6) - .001) / 2
    assert losses['charbonnier'].item() == pytest.approx(expected, rel=1e-5)
    losses['charbonnier'].backward()
    assert pred.grad[..., 2:6, 2:6].abs().sum() > 0
    assert pred.grad[..., 28:, :].abs().sum() == 0
    assert target.grad is None


def test_sobel_polygon_mask_ignores_aabb_corners_and_remains_differentiable():
    target = torch.zeros(1, 3, 32, 32)
    # Large error in the AABB corner, far enough from the diamond to avoid the 3x3 Sobel support.
    pred = torch.zeros_like(target)
    pred[..., 4:7, 4:7] = 1
    pred.requires_grad_(True)
    quad = [[4, 16], [16, 4], [28, 16], [16, 28]]
    metadata = [[line(quad)]]
    losses = TextROILoss()(pred, target, metadata)
    assert losses['charbonnier'] == 0 and losses['edge'] == 0
    changed = pred.detach().clone()
    changed[..., 15:17, 12:20] = .5
    changed.requires_grad_(True)
    losses = TextROILoss()(changed, target, metadata)
    assert losses['edge'] > 0
    losses['edge'].backward()
    assert changed.grad.abs().sum() > 0


def test_invalid_unaligned_and_empty_roi_losses_are_connected_zero():
    pred = torch.rand(2, 3, 16, 16, requires_grad=True)
    target = torch.zeros_like(pred)
    invalid = line([0, 0, 8, 8])
    invalid['valid'] = False
    unaligned = line([2, 2, 10, 10])
    unaligned['alignment_valid'] = False
    losses = TextROILoss()(pred, target, [[invalid, unaligned], []])
    sum(losses.values()).backward()
    assert all(value == 0 for value in losses.values())
    assert pred.grad is not None and pred.grad.count_nonzero() == 0
    # A spatial validity mask can also exclude a nominally aligned ROI.
    losses = TextROILoss()(pred, target, [[line([0, 0, 8, 8])], []], torch.zeros(2, 1, 16, 16))
    assert losses['charbonnier'] == 0


def test_perspective_identity_and_input_gradient():
    image = torch.arange(24 * 32, dtype=torch.float32).view(1, 1, 24, 32).requires_grad_(True)
    cropped = rectify_text_roi(image, [4, 3, 20, 15], (12, 16))
    torch.testing.assert_close(cropped, image[..., 3:15, 4:20], rtol=0, atol=1e-4)
    cropped.sum().backward()
    assert image.grad[..., 3:15, 4:20].min() > .999
    # A non-affine quadrilateral must also propagate finite image gradients.
    variable = image.detach().clone().requires_grad_()
    warped = rectify_text_roi(variable, [[3, 2], [25, 4], [22, 21], [7, 19]], (12, 24))
    warped.square().mean().backward()
    assert torch.isfinite(variable.grad).all() and variable.grad.abs().sum() > 0


class TinyOCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 7, (8, 4), stride=(8, 4))

    def forward(self, images):
        return self.conv(images).squeeze(2).transpose(1, 2) * 4


def test_roi_ocr_kl_gradients_line_normalization_and_partial_filtering():
    torch.manual_seed(9)
    ocr = KLOCRLoss(TinyOCR(), input_height=8, temperature=2)
    loss = TextROILoss(dict(ocr_max_width=128), ocr).train()
    assert not ocr.recognizer.training
    target = torch.rand(1, 3, 40, 64, requires_grad=True)
    pred = (target.detach() + .15 * torch.randn_like(target)).requires_grad_(True)
    a, b = line([2, 2, 26, 14]), line([8, 20, 56, 32])
    together = loss(pred, target, [[a, b]])['ocr']
    individual = (loss(pred, target, [[a]])['ocr'] + loss(pred, target, [[b]])['ocr']) / 2
    torch.testing.assert_close(together, individual)
    assert together.item() > 0
    together.backward()
    assert pred.grad.abs().sum() > 0 and target.grad is None
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in ocr.parameters())
    assert loss(target.detach(), target.detach(), [[a]])['ocr'].abs() < 1e-7
    partial = {**a, 'fully_visible': False, 'visible_fraction': .8}
    result = loss(pred, target, [[partial]])
    assert result['ocr'] == 0 and result['charbonnier'] > 0
    assert loss.last_counts['ocr_lines'] == 0


def test_polygon_center_rasterization_matches_aabb_slice():
    mask = polygon_mask(canonical_quad([2, 3, 6, 8]), 12, 12)
    expected = np.zeros((12, 12), dtype=np.float32)
    expected[3:8, 2:6] = 1
    np.testing.assert_array_equal(mask, expected)


class LocalDecoder(nn.Module):
    """A local differentiable decoder with deliberately unbounded RGB outputs."""
    config = SimpleNamespace(scaling_factor=1.)

    def forward(self, latent):
        return F.interpolate(latent, scale_factor=8, mode='nearest')

    def decode(self, latent, return_dict=False):
        return (self(latent),)


@pytest.mark.parametrize('ae_type', ['qwen', 'sd2'])
@pytest.mark.parametrize('tile_size', [0, 16, 64])  # Untiled, overlapping tiles, tiled fast path.
def test_unclamped_decode_restores_roi_gradients_for_saturated_pixels(ae_type, tile_size):
    from tiled_vae import decode_dispatch

    latent = torch.tensor([[1.25, -1.5, 1.75, -2.]]).repeat(4, 1)[None, None].repeat(1, 3, 1, 1)
    latent.requires_grad_(True)
    decoder = LocalDecoder()
    args = SimpleNamespace(ae_type=ae_type, vae_tile_size=tile_size, vae_tile_overlap=8, verbose=False)
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    bounded = decode_dispatch(decoder, latent, args, mean, std, decoder)
    explicit = decode_dispatch(decoder, latent, args, mean, std, decoder, clamp_output=True)
    torch.testing.assert_close(bounded, explicit, rtol=0, atol=0)
    assert bounded.abs().max() <= 1
    raw = decode_dispatch(decoder, latent, args, mean, std, decoder, clamp_output=False)
    torch.testing.assert_close(raw, decoder(latent))
    assert raw.abs().min() > 1
    target = torch.zeros_like(raw)
    lines = [[line([0, 0, 32, 32])]]
    criterion = TextROILoss()
    old = criterion(bounded, target, lines)
    old_gradient, = torch.autograd.grad(old['charbonnier'] + old['edge'], latent)
    assert old_gradient.count_nonzero() == 0  # The old decode clipping hid both useful gradients.
    losses = criterion(raw, target, lines)
    for name in ('charbonnier', 'edge'):
        gradient, = torch.autograd.grad(losses[name], latent, retain_graph=name == 'charbonnier')
        assert losses[name] > 0
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_raw_pixel_supervision_and_bounded_ocr_use_distinct_predictions():
    torch.manual_seed(51)
    criterion = TextROILoss(ocr_kl=KLOCRLoss(TinyOCR(), input_height=8, temperature=2))
    raw = torch.full((1, 3, 8, 16), 2., requires_grad=True)
    bounded, target = raw.clamp(-1, 1), torch.zeros_like(raw)
    metadata = [[line([0, 0, 16, 8])]]
    actual = criterion(raw, target, metadata, ocr_prediction=bounded)
    expected_ocr = criterion(bounded, target, metadata)['ocr']
    torch.testing.assert_close(actual['ocr'], expected_ocr)
    assert actual['charbonnier'] > 1.99
    gradient, = torch.autograd.grad(actual['charbonnier'], raw)
    assert gradient.min() > 0
    with pytest.raises(ValueError, match='shape and device'):
        criterion(raw, target, metadata, ocr_prediction=bounded[..., :4])
