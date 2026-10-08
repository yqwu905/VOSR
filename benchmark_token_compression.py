"""Run original/SDT/compressed exports in isolated processes on the same inputs.

Latency is synchronized, warmed end-to-end restore() wall time: resize, VAE,
DINO, tiled DiT, blend, decode and CPU image conversion. Disk I/O, loading,
hashing and OCR are excluded. FLOPs are analytic counts from observed shapes.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from token_compression_eval import read_manifest, sha256_file, attention_cost


def read_suite(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(cfg, dict) or not isinstance(cfg.get('variants'), dict):
        raise ValueError('Suite must contain a variants mapping')
    if not {'original', 'sdt', 'compressed'} <= set(cfg['variants']):
        raise ValueError('Suite must include original, sdt and compressed')
    if type(cfg.get('warmup')) is not int or cfg['warmup'] < 1 or type(cfg.get('repeats')) is not int or cfg['repeats'] < 2:
        raise ValueError('Use warmup >= 1 and repeats >= 2')
    manifest = read_manifest(cfg['manifest'], require_images=True)
    tags = {tag for row in manifest for record in (row, *row['regions']) for tag in record.get('tags', [])}
    if not set(cfg.get('required_tags', [])) <= tags:
        raise ValueError('Manifest is missing required OCR strata')
    shared_pipeline = shared_architecture = None
    for name, variant in cfg['variants'].items():
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
            raise ValueError('Variant names must use lowercase letters, digits, _ or -')
        export = Path(variant['export'])
        for filename in ('model.json', 'pipeline.json', 'model.safetensors'):
            if not (export / filename).is_file():
                raise FileNotFoundError(export / filename)
        pipeline = json.loads((export / 'pipeline.json').read_text(encoding='utf-8'))
        model = json.loads((export / 'model.json').read_text(encoding='utf-8'))
        architecture = {k: v for k, v in model.items() if k not in
                        ('router_config', 'compression_config', 'use_checkpoint', 'sparse_eval')}
        if shared_pipeline is not None and (pipeline != shared_pipeline or architecture != shared_architecture):
            raise ValueError('All variants must share backbone, CA, VAE, DINO, precision, resolution and upscale')
        shared_pipeline, shared_architecture = pipeline, architecture
        compressor = model.get('compression_config') or {}
        compressed = compressor.get('enabled', bool(compressor)) and compressor.get('factor', 2) > 1
        routed = model.get('router_config') is not None
        if name == 'original' and (routed or compressed):
            raise ValueError('original must have no router or compression')
        if name == 'sdt' and (not routed or compressed):
            raise ValueError('sdt must use routing without compression')
        if name == 'compressed' and (routed or not compressed or variant.get('disable_compression', False)):
            raise ValueError('compressed must enable compression without SDT')
    return cfg, manifest


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


class AttentionAudit:
    """Hooks observe real SA/CA inputs in an untimed pass, across every tile."""
    def __init__(self, model):
        self.records, self.handles = [], []
        for index, block in enumerate(model.blocks):
            def self_hook(module, inputs, index=index):
                b, n, d = inputs[0].shape
                self.records.append(dict(kind='self', block=index, batch=b, n=n, m=n, d=d))
            self.handles.append(block.attn.register_forward_pre_hook(self_hook))
            if hasattr(block, 'cross_attn'):
                def cross_hook(module, inputs, index=index):
                    b, n, d = inputs[0].shape
                    self.records.append(dict(kind='cross', block=index, batch=b, n=n, m=inputs[1].shape[1], d=d))
                self.handles.append(block.cross_attn.register_forward_pre_hook(cross_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()

    def summary(self):
        totals = dict(self_qk_av_flops=0, self_projection_flops=0, cross_qk_av_flops=0, cross_projection_flops=0)
        for record in self.records:
            b, n, m, d = (record[k] for k in ('batch', 'n', 'm', 'd'))
            cost = attention_cost([n], d, batch_size=b, condition_tokens=m if record['kind'] == 'cross' else 0)
            for key in totals:
                if key.startswith(record['kind'] + '_'):
                    totals[key] += cost[key]
        totals['attention_matmul_flops'] = sum(totals.values())
        return dict(observed_shapes=self.records, analytic_flops=totals,
                    convention='multiply+add=2; shapes observed; excludes softmax/norms/MLP/pool/lift/VAE/DINO')


def run_variant(cfg, manifest, name):
    import torch
    from PIL import Image
    from ablation_utils import load_export, load_dino
    from inference_vosr_ablation import restore
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D

    requested = cfg.get('device', 'cuda:0')
    if requested.startswith('npu'):
        import torch_npu  # noqa: F401; optional, installed by the operator
    device = torch.device(requested)
    backend = getattr(torch, device.type) if device.type in ('cuda', 'npu') else None
    if backend is not None:
        if not backend.is_available():
            raise RuntimeError(f'Requested device unavailable: {device}')
        backend.set_device(device)
    variant = cfg['variants'][name]
    directory = Path(cfg['output']) / name
    if directory.exists():
        raise FileExistsError(f'Refusing to mix runs: {directory}')
    directory.mkdir(parents=True)
    images = directory / 'images'
    images.mkdir()
    export = Path(variant['export'])
    pipeline = json.loads((export / 'pipeline.json').read_text(encoding='utf-8'))
    model = load_export(export, device)
    model.sparse_eval = not variant.get('dense_mlp', False)
    vae = AutoencoderKLQwenImage2D.from_pretrained(pipeline['vae_path']).to(device).eval().requires_grad_(False)
    venc = load_dino(pipeline['dino'], device) if model.use_cross_attention and model.ca_scale else None
    synchronize = backend.synchronize if backend is not None else lambda: None
    records = []
    for row in manifest:
        with Image.open(row['lq']) as opened:
            source = opened.convert('RGB')
        if [source.width * pipeline['upscale'], source.height * pipeline['upscale']] != row['size']:
            raise ValueError(f"Manifest size must equal LQ size * upscale: {row['id']}")
        def run():
            return restore(model, vae, venc, source, pipeline, device,
                           tile_size=cfg['tile_size'], tile_overlap=cfg['tile_overlap'],
                           vae_tile_size=cfg['vae_tile_size'], upscale=pipeline['upscale'],
                           disable_compression=variant.get('disable_compression', False), return_stats=True)
        # Shape observation is outside timing and all hooks are removed before measurement.
        audit = AttentionAudit(model)
        try:
            torch.manual_seed(cfg['seed'])
            output, stats = run()
        finally:
            audit.close()
        costs = audit.summary()
        for _ in range(cfg['warmup']):
            torch.manual_seed(cfg['seed'])
            run()
        elapsed, memory = [], []
        for _ in range(cfg['repeats']):
            torch.manual_seed(cfg['seed'])
            synchronize()
            before = backend.memory_allocated(device) if backend is not None else None
            if backend is not None:
                backend.reset_peak_memory_stats(device)
            start = time.perf_counter()
            output, measured_stats = run()
            synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
            memory.append(dict(allocated_before_bytes=before,
                peak_allocated_bytes=backend.max_memory_allocated(device) if backend is not None else None,
                peak_reserved_bytes=backend.max_memory_reserved(device) if backend is not None else None,
                peak_incremental_allocated_bytes=(backend.max_memory_allocated(device) - before) if backend is not None else None))
        if measured_stats != stats:
            raise RuntimeError('Non-deterministic shape/routing statistics with the same seed')
        output.save(images / (row['id'] + '.png'))
        record = dict(id=row['id'], lq_sha256=sha256_file(row['lq']),
                      output_sha256=sha256_file(images / (row['id'] + '.png')), **stats, attention=costs,
                      latency_ms=elapsed, latency_median_ms=statistics.median(elapsed),
                      latency_p95_ms=percentile(elapsed, .95), memory=memory)
        records.append(record)
        print(json.dumps(dict(variant=name, id=row['id'], median_ms=record['latency_median_ms'], **stats)), flush=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=False).stdout.strip()
    metadata = dict(variant=name, config=cfg, pipeline=pipeline,
                    model_config=json.loads((export / 'model.json').read_text(encoding='utf-8')),
                    weight_sha256=sha256_file(export / 'model.safetensors'), manifest_sha256=sha256_file(cfg['manifest']),
                    git_commit=commit, git_dirty=bool(dirty), python=platform.python_version(), torch=torch.__version__,
                    platform=platform.platform(), device=str(device),
                    device_name=backend.get_device_name(device) if backend is not None else platform.processor(),
                    cuda=torch.version.cuda, torchdynamo_disable=os.environ.get('TORCHDYNAMO_DISABLE'),
                    timing='warm synchronized restore(), excludes disk I/O, loading, hashes and OCR',
                    memory_scope='total allocated/reserved pipeline, plus peak minus pre-call allocation; CPU=null',
                    samples=records)
    (directory / 'performance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='configs/eval/token_compression.yml')
    parser.add_argument('--check-only', action='store_true', help='Validate local exports, protocol and input files')
    parser.add_argument('--variant', help='Internal worker; runs one variant in a fresh process')
    args = parser.parse_args()
    cfg, manifest = read_suite(args.suite)
    if args.check_only:
        print(f"Validated {len(manifest)} images, {len(cfg['variants'])} variants; no inference performed")
        return
    if args.variant:
        if args.variant not in cfg['variants']:
            parser.error('Unknown variant')
        run_variant(cfg, manifest, args.variant)
    else:
        if Path(cfg['output']).exists():
            raise FileExistsError('Use a new suite.output for every comparison')
        for name in cfg['variants']:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--suite', args.suite,
                            '--variant', name], check=True)


if __name__ == '__main__':
    main()
