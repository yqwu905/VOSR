"""One-step tiled inference for exports from train_vosr_ablation.py."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ablation_utils import load_export, load_dino, dino_features


@torch.no_grad()
def restore(model, vae, venc, image, pipeline, device, tile_size=512,
            tile_overlap=64, vae_tile_size=512, upscale=4):
    from tiled_vae import encode_dispatch, decode_dispatch, _make_tile_grid, _gaussian_weights
    quantum = 8 * model.patch_size
    if tile_size <= 0 or tile_size % quantum or not 0 <= tile_overlap < tile_size or tile_overlap % quantum:
        raise ValueError('Tile/overlap must align to 8*patch_size, with 0 <= overlap < tile')
    if upscale <= 0 or (vae_tile_size and (vae_tile_size % 8 or vae_tile_size < 16)):
        raise ValueError('upscale must be positive; VAE tile size must be 0 or a multiple of 8 >= 16')
    image = image.convert('RGB').resize((image.width * upscale, image.height * upscale), Image.Resampling.BICUBIC)
    width, height = image.size
    pixels = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1
    pad_h = max(tile_size, ((height + quantum - 1) // quantum) * quantum) - height
    pad_w = max(tile_size, ((width + quantum - 1) // quantum) * quantum) - width
    pixels = F.pad(pixels, (0, pad_w, 0, pad_h), mode='replicate')
    vae_overlap = min(64, max(8, vae_tile_size // 8)) if vae_tile_size else 0
    args = SimpleNamespace(ae_type='qwen', posterior_mode=True, vae_tile_size=vae_tile_size,
                           vae_tile_overlap=vae_overlap, tile_overlap=tile_overlap)
    lq, mean, std = encode_dispatch(vae, pixels, args, device)
    noise = torch.randn_like(lq)
    _, channels, h, w = lq.shape
    tile, overlap = tile_size // 8, tile_overlap // 8
    ys, xs = _make_tile_grid(h, tile, overlap), _make_tile_grid(w, tile, overlap)
    gaussian = _gaussian_weights(tile, tile, channels, device)
    velocity, weights = torch.zeros_like(lq), torch.zeros_like(lq)
    keep_rates = []
    for y in ys:
        for x in xs:
            features = None
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=pipeline['precision'] == 'bf16'):
                if venc is not None:
                    features = dino_features(venc, pixels[:, :, y*8:(y+tile)*8, x*8:(x+tile)*8], pipeline['dino'])
                inp = torch.cat((lq[:, :, y:y+tile, x:x+tile], noise[:, :, y:y+tile, x:x+tile]), 1)
                result, stats = model(inp, lq.new_ones(1), lq.new_zeros(1), features, return_stats=True)
            velocity[:, :, y:y+tile, x:x+tile] += result.float() * gaussian
            weights[:, :, y:y+tile, x:x+tile] += gaussian
            keep_rates.append(stats['keep_fraction'].item())
    output = decode_dispatch(vae, noise - velocity / weights.clamp_min(1e-12), args, mean, std)
    output = ((output[0, :, :height, :width].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(output), float(np.mean(keep_rates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', required=True)
    parser.add_argument('--input', required=True, help='Image file or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--upscale', type=int, default=None)
    parser.add_argument('--tile-size', type=int, default=None)
    parser.add_argument('--tile-overlap', type=int, default=64)
    parser.add_argument('--vae-tile-size', type=int, default=512)
    parser.add_argument('--vae-path', default=None)
    parser.add_argument('--dense-mlp', action='store_true', help='Dense masked MLP fallback; same routing decisions')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = json.loads((Path(args.export) / 'pipeline.json').read_text(encoding='utf-8'))
    model = load_export(args.export, device)
    model.sparse_eval = not args.dense_mlp
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    vae = AutoencoderKLQwenImage2D.from_pretrained(args.vae_path or pipeline['vae_path']).to(device).eval()
    # A no-CA export does not load DINO, download its weights, or project features.
    venc = load_dino(pipeline['dino'], device) if model.use_cross_attention and model.ca_scale else None
    source = Path(args.input)
    files = [source] if source.is_file() else sorted(p for p in source.iterdir()
              if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'))
    if not files:
        raise ValueError('No input images found')
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        with Image.open(path) as image:
            output, keep = restore(model, vae, venc, image, pipeline, device,
                                   pipeline['resolution'] if args.tile_size is None else args.tile_size, args.tile_overlap,
                                   args.vae_tile_size, pipeline['upscale'] if args.upscale is None else args.upscale)
        output.save(destination / f'{path.stem}.png')
        print(json.dumps({'file': path.name, 'deterministic_mlp_keep': keep, 'dino_loaded': venc is not None}))


if __name__ == '__main__':
    main()
