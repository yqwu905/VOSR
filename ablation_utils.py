"""Checkpoint, TXT-data and DINO helpers for VOSR2 ablation fine-tuning."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch


def load_config(path, seen=()):
    import yaml
    path = Path(path).resolve()
    if path in seen:
        raise ValueError('Circular _base_ configuration')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a mapping')
    base = config.pop('_base_', None)
    if base is None:
        return config
    def merge(old, new):
        result = dict(old)
        for key, value in new.items():
            result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
        return result
    return merge(load_config(path.parent / base, (*seen, path)), config)


def build_txt_dataset(config):
    """Use upstream TXT lists/repeat counts and TxtPairDataset's HQ crops.

    Like the original trainers, relative paths use the working directory.
    Each dataset-config row is ``image_list.txt[, integer_repeat]``; each
    image-list row is one HQ image path. LQ is synthesized in the training loop.
    """
    if 'manifest' in config:
        raise ValueError('Replace data.manifest with data.train_dataset_config '
                         '(upstream TXT image lists and repeat counts)')
    if config.get('dataset_type', 'txt') != 'txt':
        raise ValueError('Ablation training requires data.dataset_type: txt')
    resolution = int(config['resolution'])
    if resolution <= 0:
        raise ValueError('resolution must be positive')
    path = Path(config['train_dataset_config'])
    txt_paths, repeats = [], []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [part.strip() for part in line.split(',')]
        try:
            if len(parts) not in (1, 2) or not parts[0]:
                raise ValueError
            repeat = int(parts[1]) if len(parts) == 2 else 1
            if repeat < 0:
                raise ValueError
        except ValueError as error:
            raise ValueError(f'{path}:{number}: expected image_list.txt[, nonnegative integer repeat]') from error
        if not Path(parts[0]).is_file():
            raise FileNotFoundError(f'{path}:{number}: HQ image-list TXT not found: {parts[0]}')
        txt_paths.append(parts[0])
        repeats.append(repeat)
    if not txt_paths or not any(repeats):
        raise ValueError(f'{path}: empty training dataset')

    from dataloaders.realsr_dataset import TxtPairDataset
    args = SimpleNamespace(resolution=resolution, train_dataset_txt_paths_list=txt_paths,
                           train_dataset_prob_paths_list=repeats)
    dataset = TxtPairDataset(split='train', args=args)
    if not len(dataset):
        raise ValueError(f'{path}: empty training dataset')
    return dataset


@torch.no_grad()
def prepare_training_batch(batch, degradation, device):
    """Match upstream online degradation and normalize HQ/LQ to [-1, 1]."""
    hq = batch['hq'].to(device, non_blocking=True)
    _, lq = degradation.degrade_process(hq, resize_bak=True)
    return hq * 2 - 1, lq * 2 - 1


def read_weights(path):
    path = Path(path)
    if path.is_dir():
        candidates = [path / sub / name for sub in ('clean_weights', 'checkpoints', '')
                      for name in ('ema_model.safetensors', 'model.safetensors')]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError('Pass the exact clean .safetensors file or its containing directory')
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a clean tensor state_dict, not an optimizer checkpoint')
    return state


def load_backbone_state(model, state, allow_new_modules=False):
    """Strict backbone load. New routers / token-merge layers may be missing only when allowed."""
    from models.sdt_router import conditioning_key
    from models.token_hourglass import new_module_key
    removed = getattr(model, 'is_removed_weight', None)
    expected, filtered, discarded = model.state_dict(), {}, []
    for key, value in state.items():
        if key.startswith('feat_rope.') or (not model.use_cross_attention and conditioning_key(key)) \
                or (removed is not None and removed(key)):
            discarded.append(key)
            continue
        if key not in expected:
            raise ValueError(f'Unexpected weight: {key}; check checkpoint architecture/version')
        if value.shape != expected[key].shape:
            raise ValueError(f'Shape mismatch for {key}: {tuple(value.shape)} != {tuple(expected[key].shape)}')
        filtered[key] = value
    missing = [k for k in expected if k not in filtered and not k.startswith('feat_rope.')
               and not (allow_new_modules and ('.router.' in k or new_module_key(k)))]
    if missing:
        raise ValueError(f'Missing backbone weights: {missing[:12]}')
    model.load_state_dict(filtered, strict=False)
    return {'loaded': len(filtered), 'discarded': discarded,
            'new_router_parameters': [k for k in expected if '.router.' in k and k not in filtered],
            'new_token_merge_parameters': [k for k in expected if new_module_key(k) and k not in filtered]}


def save_export(model, directory, pipeline, strip_conditioning=False):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config, state = model.export_state(strip_conditioning)
    save_file(state, str(directory / 'model.safetensors'))
    (directory / 'model.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (directory / 'pipeline.json').write_text(json.dumps(pipeline, indent=2), encoding='utf-8')


def load_export(directory, device='cpu'):
    from models.lightningdit_ablation import AblationLightningDiT
    directory = Path(directory)
    config = json.loads((directory / 'model.json').read_text(encoding='utf-8'))
    model = AblationLightningDiT(**config)
    model.load_state_dict(read_weights(directory / 'model.safetensors'), strict=True)
    return model.to(device).eval()


def student_dino_config(config):
    """Top-level DINO settings (teacher) with the optional student.dino size/layer override."""
    override = (config.get('student') or {}).get('dino') or {}
    unknown = set(override) - {'size', 'layer', 'tile'}
    if unknown:
        raise ValueError(f'student.dino may only override size/layer/tile, not {sorted(unknown)}')
    return {**config['dino'], **override}


def load_dino(config, device):
    torch.hub.set_dir(config.get('cache_dir', 'preset/ckpts/torch_cache'))
    repo = config.get('local_repo') or 'facebookresearch/dinov2'
    kwargs = {'source': 'local'} if config.get('local_repo') else {}
    return torch.hub.load(repo, 'dinov2_vitl14', **kwargs).to(device).eval().requires_grad_(False)


@torch.no_grad()
def dino_features(model, lq, config):
    """Match upstream raw intermediate tokens, not normalized intermediate layers.

    With ``tile`` (HQ pixels), an LQ crop larger than the tile is split into tiles that each get their
    own ``size`` input, as when the crop is tiled at that size; the token grids are laid side by side
    into one row-major grid (CA has no positions, token pooling sees the right neighbours).
    """
    tile = config.get('tile')
    b, _, h, w = lq.shape
    if tile and (h > tile or w > tile):
        if h % tile or w % tile:
            raise ValueError(f'dino.tile {tile} must divide the {h}x{w} crop')
        rows, cols = h // tile, w // tile
        tiles = lq.reshape(b, -1, rows, tile, cols, tile).permute(0, 2, 4, 1, 3, 5).reshape(b * rows * cols, -1, tile, tile)
        tokens = dino_features(model, tiles, dict(config, tile=None))[0]
        side = math.isqrt(tokens.shape[1])
        grid = tokens.reshape(b, rows, cols, side, side, -1).permute(0, 1, 3, 2, 4, 5)
        return [grid.reshape(b, rows * side * cols * side, -1)]
    import torch.nn.functional as F
    layer = int(config['layer'])
    if not 0 <= layer < len(model.blocks):
        raise ValueError('Invalid DINOv2 layer')
    x = F.interpolate(lq.float() * 0.5 + 0.5, size=int(config['size']),
                      mode='bicubic', align_corners=False).clamp(0, 1)
    mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    x = model.prepare_tokens_with_masks((x - mean) / std)
    for i, block in enumerate(model.blocks):
        x = block(x)
        if i == layer:
            return [(model.norm(x) if i == len(model.blocks) - 1 else x)[:, 1:]]
    raise RuntimeError('Requested DINOv2 layer was not produced')


def teacher_velocity(teacher, inp, t, r, lq, features, venc, config):
    """Frozen-teacher velocity. With training.teacher_tile, a larger crop is run like
    inference_vosr_ablation.restore: overlapping square tiles, each with the DINO of its own LQ crop
    (top-level dino settings), velocities Gaussian-blended. Otherwise one forward on ``features``."""
    tile = (config.get('training') or {}).get('teacher_tile')
    h, w = inp.shape[-2:]
    scale = lq.shape[-1] // w  # HQ pixels per latent (the VAE's 8)
    if not tile or (h * scale <= tile and w * scale <= tile):
        return teacher(inp, t, r, features)
    from tiled_vae import _make_tile_grid, _gaussian_weights
    size, overlap = tile // scale, int(config['training'].get('teacher_tile_overlap', 64)) // scale
    gaussian = _gaussian_weights(size, size, 1, inp.device)
    velocity = weights = None
    for y in _make_tile_grid(h, size, overlap):
        for x in _make_tile_grid(w, size, overlap):
            crop = lq[:, :, y * scale:(y + size) * scale, x * scale:(x + size) * scale]
            tile_features = dino_features(venc, crop, config['dino'])
            v = teacher(inp[:, :, y:y + size, x:x + size], t, r, tile_features).float()
            if velocity is None:
                velocity = v.new_zeros(v.shape[0], v.shape[1], h, w)
                weights = v.new_zeros(1, 1, h, w)
            velocity[:, :, y:y + size, x:x + size] += v * gaussian
            weights[:, :, y:y + size, x:x + size] += gaussian
    return velocity / weights
