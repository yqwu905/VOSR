"""Synthetic CPU protocol integration; VAE/weights are fixtures, not VOSR2."""
import json
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest
import torch
import torch.nn.functional as F
import yaml

from ablation_utils import save_export
from benchmark_token_compression import read_suite, run_variant
from models.lightningdit_ablation import AblationLightningDiT
from summarize_token_compression import summarize
from token_compression_eval import sha256_file, score_predictions


class TinyVAE(torch.nn.Module):
    config = SimpleNamespace(latents_mean=[0, 0], latents_std=[1, 1])

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def encode(self, x):
        latent = F.avg_pool2d(x[:, :2], 8)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, x, **kwargs):
        return (F.interpolate(torch.cat((x, x[:, :1]), 1), scale_factor=8, mode='nearest'),)


def test_benchmark_three_way_protocol_and_ocr_summary(tmp_path, monkeypatch):
    module = ModuleType('models.qwenimage_vae2d')
    module.AutoencoderKLQwenImage2D = TinyVAE
    monkeypatch.setitem(sys.modules, 'models.qwenimage_vae2d', module)
    Image.new('RGB', (16, 16), 'gray').save(tmp_path/'lq.png')
    manifest = tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps(dict(id='test', lq='lq.png', size=[64, 64], tags=['dense'], regions=[
        dict(id='line', bbox=[0, 0, 60, 20], text='测试', tags=['small', 'chinese'])])), encoding='utf-8')
    pipeline = dict(precision='fp32', vae_path='synthetic', resolution=64, upscale=4, dino={})
    variants = {}
    for name in ('original', 'sdt', 'compressed'):
        net = AblationLightningDiT(input_size=8, patch_size=2, in_channels=4, out_channels=2,
            hidden_size=32, depth=2, num_heads=4, use_cross_attention=False,
            router_config=dict(init_keep_prob=.5, routing_mode='capacity_topk') if name == 'sdt' else None,
            compression_config=dict(factor=2, start_block=0, end_block=2) if name == 'compressed' else None)
        export = tmp_path/name
        save_export(net, export, pipeline)
        variants[name] = dict(export=str(export))
    cfg = dict(manifest=str(manifest), output=str(tmp_path/'run'), device='cpu', seed=42,
               warmup=1, repeats=2, tile_size=64, tile_overlap=16, vae_tile_size=0,
               required_tags=['small', 'chinese', 'dense'], variants=variants)
    config_path = tmp_path/'suite.yml'
    config_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    cfg, rows = read_suite(config_path)
    for name in variants:
        run_variant(cfg, rows, name)
    reports = summarize(cfg['output'])
    assert all(r['ocr_status'] == 'not_measured' and r['max_peak_allocated_bytes'] is None for r in reports)
    assert [r['mean_self_pair_ratio'] for r in reports] == [1, 1, 1/16]
    predictions = [dict(id='test', regions=[dict(id='line', text='测试')])]
    for name in variants:
        ocr = score_predictions(rows, predictions)
        ocr.update(engine_id='synthetic_fixture', manifest_sha256=sha256_file(manifest))
        (tmp_path/'run'/name/'ocr.json').write_text(json.dumps(ocr), encoding='utf-8')
    assert all(r['ocr']['small']['delta_cer_vs_original'] == 0 for r in summarize(cfg['output']))
    # A score made from a different recognizer must not be silently compared.
    ocr['engine_id'] = 'different'
    (tmp_path/'run'/'compressed'/'ocr.json').write_text(json.dumps(ocr), encoding='utf-8')
    with pytest.raises(ValueError, match='OCR engine'):
        summarize(cfg['output'])
    with pytest.raises(FileExistsError):
        run_variant(cfg, rows, 'original')
