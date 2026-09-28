"""Checkpoint, paired-data and DINO helpers for VOSR2 ablation fine-tuning."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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


class PairedManifestDataset(Dataset):
    """JSONL rows contain lq and optional gt paths, relative to the manifest.

    Bicubic-upsample LQ and take aligned crops. No flips (preserve text).
    This is NOT a reproduction of the unpublished VOSR2 training data pipeline.
    """
    def __init__(self, manifest, resolution=512, upscale=4):
        self.path = Path(manifest).resolve()
        self.resolution, self.upscale = int(resolution), int(upscale)
        if self.resolution <= 0 or self.upscale <= 0:
            raise ValueError('resolution and upscale must be positive')
        self.rows = []
        for number, line in enumerate(self.path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get('lq'):
                raise ValueError(f'{self.path}:{number}: missing lq')
            self.rows.append(row)
        if not self.rows:
            raise ValueError('Empty training manifest')
        self.has_gt = all(bool(row.get('gt')) for row in self.rows)
        if any(bool(row.get('gt')) for row in self.rows) != self.has_gt:
            raise ValueError('GT must be supplied for every row or for no rows')

    def __len__(self):
        return len(self.rows)

    def _open(self, path):
        with Image.open(self.path.parent / path) as image:
            return image.convert('RGB')

    @staticmethod
    def _tensor(image):
        return torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1

    def __getitem__(self, index):
        row = self.rows[index]
        lq = self._open(row['lq'])
        size = (lq.width * self.upscale, lq.height * self.upscale)
        gt = self._open(row['gt']) if self.has_gt else None
        if gt is not None and gt.size != size:
            raise ValueError(f'Pair {index}: GT {gt.size} != LQ x upscale {size}')
        if min(size) < self.resolution:
            raise ValueError(f'Pair {index}: upscaled LQ is smaller than crop size')
        lq = lq.resize(size, Image.Resampling.BICUBIC)
        left, top = random.randint(0, size[0] - self.resolution), random.randint(0, size[1] - self.resolution)
        box = (left, top, left + self.resolution, top + self.resolution)
        batch = {'lq': self._tensor(lq.crop(box))}
        if gt is not None:
            batch['gt'] = self._tensor(gt.crop(box))
        return batch


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


def load_backbone_state(model, state, allow_new_router=False):
    from models.sdt_router import conditioning_key
    expected, filtered, discarded = model.state_dict(), {}, []
    for key, value in state.items():
        if key.startswith('feat_rope.') or (not model.use_cross_attention and conditioning_key(key)):
            discarded.append(key)
            continue
        if key not in expected:
            raise ValueError(f'Unexpected weight: {key}; check checkpoint architecture/version')
        if value.shape != expected[key].shape:
            raise ValueError(f'Shape mismatch for {key}: {tuple(value.shape)} != {tuple(expected[key].shape)}')
        filtered[key] = value
    missing = [k for k in expected if k not in filtered and not k.startswith('feat_rope.')
               and not (allow_new_router and '.router.' in k)]
    if missing:
        raise ValueError(f'Missing backbone weights: {missing[:12]}')
    model.load_state_dict(filtered, strict=False)
    return {'loaded': len(filtered), 'discarded': discarded,
            'new_router_parameters': [k for k in expected if '.router.' in k and k not in filtered]}


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


def load_dino(config, device):
    torch.hub.set_dir(config.get('cache_dir', 'preset/ckpts/torch_cache'))
    repo = config.get('local_repo') or 'facebookresearch/dinov2'
    kwargs = {'source': 'local'} if config.get('local_repo') else {}
    return torch.hub.load(repo, 'dinov2_vitl14', **kwargs).to(device).eval().requires_grad_(False)


@torch.no_grad()
def dino_features(model, lq, config):
    """Match upstream raw intermediate tokens, not normalized intermediate layers."""
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
