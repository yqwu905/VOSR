"""Validate physical FFN widths and time exported end-to-end SR on the target device.

Loads the real VAE/DINO pipeline and excludes file I/O/model loading. Includes
image preprocessing, encode, DINO, DiT, decode and image conversion; no synthetic
MAC number is reported as measured speed. Run one process per benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch

from ablation_utils import load_dino, load_export
from inference_vosr_ablation import restore
from training_runtime import select_device


def check_widths(model, expected_width=None):
    records = []
    for index, block in enumerate(model.blocks):
        mlp = getattr(block, 'mlp', None)
        if mlp is None or not hasattr(mlp, 'w12') or not hasattr(mlp, 'w3'):
            raise ValueError(f'Block {index} is not a physical SwiGLU FFN')
        if hasattr(mlp, 'fade_scale') or mlp.__class__.__name__ == 'FadingSwiGLU':
            raise ValueError('Benchmark a finished narrow export, not a full-width fade checkpoint')
        width, dim = mlp.w3.weight.shape[1], mlp.w3.weight.shape[0]
        if tuple(mlp.w12.weight.shape) != (2 * width, dim):
            raise ValueError(f'Block {index}: incompatible packed SwiGLU weight shapes')
        if expected_width is not None and width != expected_width:
            raise ValueError(f'Block {index}: hidden={width}, expected {expected_width}')
        records.append({'block': index, 'hidden': width, 'residual_dim': dim,
                        'parameters': sum(parameter.numel() for parameter in mlp.parameters())})
    if not records:
        raise ValueError('No FFN blocks found')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', required=True)
    parser.add_argument('--input', required=True, help='Native low-resolution image, as for normal inference')
    parser.add_argument('--output', required=True, help='Benchmark JSON (must not already exist)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--expected-width', type=int)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tile-size', type=int)
    parser.add_argument('--tile-overlap', type=int, default=64)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise ValueError('Run one process for synchronized latency measurement')
    if args.warmup < 0 or args.iterations < 1 or (args.expected_width is not None and args.expected_width <= 0):
        raise ValueError('Warmup must be nonnegative; iterations and width must be positive')
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    device = select_device(args.device)
    pipeline = json.loads((Path(args.export) / 'pipeline.json').read_text(encoding='utf-8'))
    model = load_export(args.export, device)
    widths = check_widths(model, args.expected_width)
    vae = None
    if not getattr(model, 'pixel_space', False):
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        vae = AutoencoderKLQwenImage2D.from_pretrained(pipeline['vae_path']).to(device).eval()
    venc = load_dino(pipeline['dino'], device) if model.use_cross_attention and model.ca_scale else None
    with Image.open(args.input) as image:
        image = image.convert('RGB').copy()
    backend = getattr(torch, device.type) if device.type != 'cpu' else None
    def synchronize():
        if backend is not None:
            backend.synchronize(device)
    def run():
        torch.manual_seed(args.seed)
        return restore(model, vae, venc, image, pipeline, device,
                       args.tile_size or pipeline['resolution'], args.tile_overlap,
                       pipeline.get('vae_tile_size', 512), pipeline['upscale'])
    with torch.inference_mode():
        for _ in range(args.warmup):
            run()
        synchronize()
        if backend is not None and hasattr(backend, 'reset_peak_memory_stats'):
            backend.reset_peak_memory_stats(device)
        durations = []
        for _ in range(args.iterations):
            synchronize()
            start = time.perf_counter()
            output, _ = run()
            synchronize()
            durations.append((time.perf_counter() - start) * 1000)
    report = dict(export=str(Path(args.export).resolve()), input=str(Path(args.input).resolve()),
                  input_size=list(image.size), output_size=list(output.size), device=str(device),
                  device_name=backend.get_device_name(device) if backend else platform.processor(),
                  torch_version=torch.__version__, python=platform.python_version(), pipeline=pipeline,
                  tile_size=args.tile_size or pipeline['resolution'], tile_overlap=args.tile_overlap,
                  warmup=args.warmup, iterations=args.iterations, seed=args.seed,
                  latency_ms=dict(mean=float(np.mean(durations)), p50=float(np.percentile(durations, 50)),
                                  p95=float(np.percentile(durations, 95)), samples=durations), ffn=widths,
                  ffn_parameters=sum(row['parameters'] for row in widths))
    if backend is not None and hasattr(backend, 'max_memory_allocated'):
        report['peak_memory_allocated_bytes'] = backend.max_memory_allocated(device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != 'ffn'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
