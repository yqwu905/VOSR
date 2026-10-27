"""Aligned LQ/HQ crops from same-size image pairs (``data.dataset_type: paired``).

The LQ images are already upsampled to the HQ size (e.g. bicubic x4 of a real low-resolution capture, as the
inference scripts do), so both crops come from the same coordinates. Optional text boxes bias the crops towards
text: with probability ``text_crop_prob`` a crop is centred on a random box of the image, jittered by up to a
quarter crop in x and y, else it is uniform. ``tools/prepare_realce.py`` writes Real-CE in this layout.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')


def _images(directory):
    files = {}
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in files:
                raise ValueError(f'{directory}: two images named {path.stem}')
            files[path.stem] = path
    return files


class PairedCropDataset(Dataset):
    """``lq`` and ``hq`` crops of ``size`` x ``size`` as [0, 1] float tensors (C, H, W).

    One item per HQ image (sorted by name); the crop is drawn with the ``random`` module, which the DataLoader
    reseeds in every worker and epoch. ``text_boxes``: JSON mapping an image name (stem) to a list of quads
    ``[x1, y1, ..., x4, y4]`` in HQ pixels; images without boxes always get uniform crops. Images smaller than the
    crop are edge-padded. ``text_mask``: items also carry ``text_mask`` (1, size, size), 1 inside the axis-aligned
    bounding boxes of the image's quads (all zero without boxes).
    """
    def __init__(self, lq_dir, hq_dir, size, text_boxes=None, text_crop_prob=0.0, text_mask=False):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError('crop size must be a positive integer')
        if isinstance(text_crop_prob, bool) or not isinstance(text_crop_prob, (int, float)) \
                or not 0 <= text_crop_prob <= 1:
            raise ValueError('data.text_crop_prob must be in [0, 1]')
        if text_crop_prob and not text_boxes:
            raise ValueError('data.text_crop_prob needs data.text_boxes')
        if text_mask and not text_boxes:
            raise ValueError('Text-box masks need data.text_boxes')
        for name, directory in (('lq_dir', lq_dir), ('hq_dir', hq_dir)):
            if not directory or not Path(directory).is_dir():
                raise FileNotFoundError(f'data.{name} is not a directory: {directory}')
        hq, lq = _images(hq_dir), _images(lq_dir)
        if not hq:
            raise ValueError(f'{hq_dir}: empty training dataset')
        missing = sorted(set(hq) - set(lq))
        if missing:
            raise FileNotFoundError(f'{lq_dir}: no LQ image for {missing[:5]} ({len(missing)} missing)')
        self.names = sorted(hq)
        self.hq, self.lq = hq, lq
        self.size, self.text_crop_prob, self.text_mask = size, float(text_crop_prob), bool(text_mask)
        self.boxes = {}
        if text_boxes:
            boxes = json.loads(Path(text_boxes).read_text(encoding='utf-8'))
            for name, quads in boxes.items():
                if not isinstance(quads, list) or any(len(q) != 8 for q in quads):
                    raise ValueError(f'{text_boxes}: {name} must map to a list of [x1, y1, ..., x4, y4] quads')
                self.boxes[name] = quads

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _load(path):
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB'))

    def sample(self, name, rng):
        """Crops of image ``name`` drawn with ``rng`` (the ``random`` module or a ``random.Random``)."""
        lq, hq = self._load(self.lq[name]), self._load(self.hq[name])
        if lq.shape != hq.shape:
            raise ValueError(f'{name}: LQ {lq.shape[:2]} and HQ {hq.shape[:2]} sizes differ; '
                             'upsample the LQ to the HQ size first')
        s = self.size
        h, w = hq.shape[:2]
        if h < s or w < s:
            pad = ((0, max(0, s - h)), (0, max(0, s - w)), (0, 0))
            lq, hq = np.pad(lq, pad, mode='edge'), np.pad(hq, pad, mode='edge')
            h, w = hq.shape[:2]
        boxes = self.boxes.get(name)
        if boxes and rng.random() < self.text_crop_prob:
            quad = np.asarray(rng.choice(boxes), dtype=np.float64).reshape(4, 2)
            cx, cy = quad[:, 0].mean(), quad[:, 1].mean()
            x0 = int(np.clip(cx - s / 2 + rng.uniform(-s / 4, s / 4), 0, w - s))
            y0 = int(np.clip(cy - s / 2 + rng.uniform(-s / 4, s / 4), 0, h - s))
        else:
            x0, y0 = rng.randint(0, w - s), rng.randint(0, h - s)
        crop = lambda image: torch.from_numpy(np.ascontiguousarray(image[y0:y0 + s, x0:x0 + s])).permute(2, 0, 1)
        item = {'hq': crop(hq).float() / 255, 'lq': crop(lq).float() / 255}
        if self.text_mask:
            mask = torch.zeros(1, s, s)
            for quad in boxes or ():
                xs, ys = np.asarray(quad[0::2], dtype=np.float64) - x0, np.asarray(quad[1::2], dtype=np.float64) - y0
                left, right = int(np.clip(np.floor(xs.min()), 0, s)), int(np.clip(np.ceil(xs.max()), 0, s))
                top, bottom = int(np.clip(np.floor(ys.min()), 0, s)), int(np.clip(np.ceil(ys.max()), 0, s))
                mask[:, top:bottom, left:right] = 1
            item['text_mask'] = mask
        return item

    def __getitem__(self, index):
        return self.sample(self.names[index], random)
