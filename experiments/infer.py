"""Resumable, fixed-seed inference using the original one-step VOSR2 backend."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image

from data_pipeline.augmentation import sample_seed
from data_pipeline.pipeline import atomic_json, file_digest
from data_pipeline.vosr_teacher import VOSR2Teacher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='preset/ckpts/VOSR2')
    p.add_argument('--pairs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tile-size', type=int, default=512)
    args = p.parse_args()
    records = [json.loads(line) for line in Path(args.pairs).read_text().splitlines()]
    if len({r['id'] for r in records}) != len(records) or not records:
        raise ValueError('Expected nonempty, unique image ids')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    weights = next((Path(args.checkpoint) / 'checkpoints' / name for name in
                    ['ema_model.safetensors', 'model.safetensors']
                    if (Path(args.checkpoint) / 'checkpoints' / name).is_file()))
    protocol = dict(checkpoint=str(Path(args.checkpoint).resolve()), weight_sha256=file_digest(weights),
                    pairs_sha256=file_digest(args.pairs), seed=args.seed, precision='bf16',
                    tile_size=args.tile_size, tile_overlap=64, vae_tile_size=512,
                    color_alignment='wavelet', infer_steps=1)
    if (out / 'protocol.json').exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Inference inputs/configuration changed; use a fresh output directory')
    else:
        atomic_json(out / 'protocol.json', protocol)
    completed = {}
    if (out / 'progress.jsonl').exists():
        for line in (out / 'progress.jsonl').read_text().splitlines():
            try:
                record = json.loads(line)
                completed[record['id']] = record
            except json.JSONDecodeError:
                pass  # An interrupted final write is regenerated below.
    model = None
    for i, record in enumerate(records):
        path = out / f'{record["id"]}.png'
        prior = completed.get(record['id'])
        if prior and path.is_file() and file_digest(path) == prior['sha256']:
            continue
        if model is None:
            model = VOSR2Teacher(args.checkpoint, precision='bf16', upscale=record['scale'],
                                overrides=dict(tile_size=args.tile_size, tile_overlap=64,
                                               vae_tile_size=512, vae_tile_overlap=64))
        if model.upscale != record['scale']:
            raise ValueError('One inference manifest must use one scale')
        start = time.monotonic()
        with Image.open(record['lq_path']) as im:
            image = model.restore(im.convert('RGB'), dict(seed=sample_seed(args.seed, record['id'], 'inference')))
        if list(image.size) != record['gt_size']:
            raise ValueError(f'{record["id"]}: unexpected output size {image.size}')
        temp = path.with_suffix('.tmp.png')
        image.save(temp)
        temp.replace(path)
        status = dict(id=record['id'], index=i+1, total=len(records),
                      seconds=time.monotonic()-start, sha256=file_digest(path))
        with (out / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(status) + '\n')
        print(json.dumps(status), flush=True)
    atomic_json(out / 'complete.json', dict(images=len(records), protocol=protocol))


if __name__ == '__main__':
    main()
