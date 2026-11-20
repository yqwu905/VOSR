import json
import random
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F

from ablation_utils import load_config
from dataloaders.paired_crops import PairedCropDataset, canonical_quad, glyph_annotation_chars
from models.ffn_distillation import text_distillation_spec, validate_text_data
from models.glyph_loss import TextVerifierGlyphLoss
from tools.annotate_glyphs import annotation_lines


def glyph(char, box):
    return {'char': char, 'quad': canonical_quad(box).tolist()}


def test_paired_glyph_recipe_is_opt_in():
    configs = Path(__file__).resolve().parents[1] / 'configs' / 'ablations'
    assert text_distillation_spec(load_config(configs / 'base' / 'base_sft.yml'))['glyph_loss_weight'] == 0
    recipe = load_config(configs / 'loss' / 'glyph_sft.yml')
    validate_text_data(recipe['data'])
    assert text_distillation_spec(recipe)['glyph_loss_weight'] == 1
    assert recipe['training']['gt_weight'] == 1 and recipe['training']['kd_weight'] == 0


def test_ppocr_boxes_are_individual_and_crop_relative(tmp_path):
    Image.new('RGB', (80, 80)).save(tmp_path / 'page.png')
    quad = canonical_quad([8, 8, 72, 72])
    regions = [canonical_quad(box) for box in ([14, 27, 26, 40], [29, 27, 40, 40], [44, 27, 55, 40])]
    result = dict(rec_texts=['AB中文'], rec_scores=[.98], rec_polys=[quad],
                  text_word=[['AB', '中', '文']], text_word_region=[regions])
    lines, groups, invalid = annotation_lines(result, 80, 80)
    assert (groups, invalid) == (1, 0)
    boxes = tmp_path / 'boxes.json'
    boxes.write_text(json.dumps({'page': lines}), encoding='utf-8')
    ds = PairedCropDataset(tmp_path, tmp_path, 32, str(boxes), 1., text_metadata=True,
                           text_sampling={'jitter_fraction': 0})
    assert glyph_annotation_chars(ds) == {'中', '文'}
    item = ds.sample('page', random.Random(0))
    assert item['crop_origin'].tolist() == [24, 24]
    assert item['text_lines'][0]['glyphs'] == [glyph('中', [5, 3, 16, 16]), glyph('文', [20, 3, 31, 16])]
    # Clip one real character box without moving the other or inferring missing positions.
    lines[0]['glyphs'][0]['quad'] = canonical_quad([20, 27, 40, 40]).tolist()
    boxes.write_text(json.dumps({'page': lines}), encoding='utf-8')
    ds = PairedCropDataset(tmp_path, tmp_path, 32, str(boxes), 1., text_metadata=True,
                           text_sampling={'jitter_fraction': 0})
    assert ds[0]['text_lines'][0]['glyphs'] == [glyph('文', [20, 3, 31, 16])]


def test_line_only_annotations_rejected_when_glyphs_required(tmp_path):
    Image.new('RGB', (16, 16)).save(tmp_path / 'page.png')
    boxes = tmp_path / 'boxes.json'
    boxes.write_text(json.dumps({'page': [{'transcript': '中文', 'bbox': [0, 0, 16, 16]}]}))
    ds = PairedCropDataset(tmp_path, tmp_path, 16, str(boxes), text_metadata=True)
    assert ds[0]['text_lines'][0]['transcript'] == '中文'
    with pytest.raises(ValueError, match='explicit glyphs'):
        glyph_annotation_chars(ds)


def test_glyph_mean_nearest_target_and_backward():
    torch.manual_seed(5)
    encoder = nn.Sequential(nn.Conv2d(3, 3, 1, bias=False), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    prototypes = torch.randn(2, 3, 3)
    loss_fn = TextVerifierGlyphLoss(encoder, prototypes, {'中': 0, '文': 1}, batch_size=2).train()
    assert not encoder.training
    sr = torch.rand(2, 3, 16, 16, requires_grad=True)
    boxes = [glyph('中', [0, 0, 8, 8]), glyph('文', [8, 0, 16, 8]), glyph('?', [0, 8, 8, 16])]
    lines = [[{'glyphs': boxes}], [{'glyphs': [boxes[1]]}]]
    loss = loss_fn(sr, lines)
    assert loss_fn.last_counts == (3, 1)
    with torch.no_grad():
        crops = torch.stack([sr[0, :, :8, :8], sr[0, :, :8, 8:], sr[1, :, :8, 8:]])
        features = F.normalize(encoder(crops))
        selected = F.normalize(prototypes, dim=-1)[[0, 1, 1]]
        expected = (1 - (features[:, None] * selected).sum(-1).max(-1).values).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(sr.grad).all() and sr.grad.abs().sum() > 0
    assert torch.count_nonzero(sr.grad[:, :, 8:]) == 0
    assert all(p.grad is None for p in encoder.parameters())
    sr.grad = None
    empty = loss_fn(sr, [[{'glyphs': [boxes[-1]]}], []])
    assert loss_fn.last_counts == (0, 1) and empty.item() == 0
    empty.backward()
    assert torch.count_nonzero(sr.grad) == 0
