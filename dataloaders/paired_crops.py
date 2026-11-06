"""Aligned, native-scale LQ/HQ crops, optionally carrying text-line annotations.

Legacy calls keep their original tensor-only contract. data.text_metadata enables
polygon masks and variable-length text_lines (use collate_paired_samples).
Annotations are in the already upsampled HQ/output pixel coordinate system.
"""
import bisect
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from PIL import Image

IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
DATASET_OPTION_KEYS = ('text_metadata', 'small_text_height', 'text_sampling', 'alignment_valid',
                       'allow_missing_text', 'teacher_error_weight', 'deterministic_crops', 'seed', 'image_names')


def _images(directory):
    files = {}
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in files:
                raise ValueError(f'{directory}: two images named {path.stem}')
            files[path.stem] = path
    return files


def _image_name(value, known=()):
    """Accept stems or filenames without truncating a dotted stem such as page.02."""
    name = Path(value.replace('\\', '/')).name
    return name if name in known or Path(name).suffix.lower() not in IMAGE_SUFFIXES else Path(name).stem


def _selected_names(value, available):
    if value is None:
        return sorted(available)
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f'data.image_names file not found: {path}')
        value = [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
                 if line.strip() and not line.lstrip().startswith('#')]
    if not isinstance(value, (list, tuple)) or not value or any(not isinstance(x, str) or not x for x in value):
        raise ValueError('data.image_names must be a nonempty list of names or a nonempty text file')
    names = [_image_name(x, available) for x in value]
    if len(names) != len(set(names)):
        raise ValueError('data.image_names contains duplicate image stems')
    missing = sorted(set(names) - set(available))
    if missing:
        raise FileNotFoundError(f'data.image_names: HQ images not found: {missing[:5]}')
    return sorted(names)


def _bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be a boolean')
    return value


def _number(value, name, lo=0., hi=float('inf'), inclusive_lo=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value > hi or (value < lo if inclusive_lo else value <= lo):
        relation = '>=' if inclusive_lo else '>'
        raise ValueError(f'{name} must be finite, {relation} {lo} and <= {hi}')
    return float(value)


def _area(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.
    return abs(float(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                            - points[:, 1] * np.roll(points[:, 0], -1)))) * .5


def canonical_quad(value, name='text line'):
    """Validate a cyclic TL,TR,BR,BL quad, or an xyxy AABB; never silently repair crossed boxes."""
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name}: expected finite xyxy AABB or four cyclic (x,y) corners') from error
    if points.shape == (4,):
        left, top, right, bottom = points
        if right <= left or bottom <= top:
            raise ValueError(f'{name}: AABB must have x1 > x0 and y1 > y0')
        points = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
    elif points.shape == (8,):
        points = points.reshape(4, 2)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f'{name}: expected finite xyxy AABB or four cyclic (x,y) corners')
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(cross > 1e-8) or np.all(cross < -1e-8)) or _area(points) < 1e-6:
        raise ValueError(f'{name}: quad must be convex, nondegenerate and cyclic (not crossed)')
    if np.all(cross < 0):
        points = points[[0, 3, 2, 1]]
    return points


def quad_line_size(points):
    """Recognition-line width/height before any OCR resize, including the usual vertical-line rotation."""
    points = np.asarray(points, dtype=np.float64)
    width = max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
    height = max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
    return (float(height), float(width)) if height >= 1.5 * width else (float(width), float(height))


def clip_polygon(points, left, top, right, bottom):
    """Sutherland-Hodgman clipping against a crop; preserves fractional geometry."""
    polygon = [np.asarray(point, dtype=np.float64) for point in points]
    for axis, bound, lower in ((0, left, True), (0, right, False), (1, top, True), (1, bottom, False)):
        if not polygon:
            break
        result = []
        previous = polygon[-1]
        previous_inside = previous[axis] >= bound if lower else previous[axis] <= bound
        for current in polygon:
            current_inside = current[axis] >= bound if lower else current[axis] <= bound
            if current_inside != previous_inside:
                fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
                result.append(previous + fraction * (current - previous))
            if current_inside:
                result.append(current)
            previous, previous_inside = current, current_inside
        polygon = result
    return np.asarray(polygon, dtype=np.float64).reshape(-1, 2)


def polygon_mask(points, height, width):
    """Rasterize polygon coverage at pixel centers (xyxy [2,2,6,6] covers exactly 4x4 pixels)."""
    result = np.zeros((height, width), dtype=np.float32)
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3 or _area(points) < 1e-8:
        return result
    left, top = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
    right, bottom = np.minimum(np.ceil(points.max(axis=0)).astype(int), [width, height])
    if right <= left or bottom <= top:
        return result
    xs, ys = np.meshgrid(np.arange(left, right) + .5, np.arange(top, bottom) + .5)
    inside = np.zeros(xs.shape, dtype=bool)
    previous = points[-1]
    for current in points:
        # Horizontal edges never cross the ray; avoid division by zero for them.
        if abs(current[1] - previous[1]) > 1e-12:
            crossing = (current[1] > ys) != (previous[1] > ys)
            edge_x = (previous[0] - current[0]) * (ys - current[1]) / (previous[1] - current[1]) + current[0]
            inside ^= crossing & (xs < edge_x)
        previous = current
    result[top:bottom, left:right] = inside.astype(np.float32)
    return result


def _line_record(raw, name, alignment_valid):
    if isinstance(raw, dict):
        keys = [key for key in ('quad', 'points', 'bbox', 'box') if key in raw]
        if len(keys) != 1:
            raise ValueError(f'{name}: a line record needs exactly one of quad, points, bbox or box')
        quad = canonical_quad(raw[keys[0]], name)
        text = raw.get('transcript', raw.get('transcription', raw.get('text', '')))
        if not isinstance(text, str):
            raise ValueError(f'{name}: transcript must be a string')
        valid = _bool(raw.get('valid', True), f'{name}.valid')
        for flag in ('difficult', 'ignore'):
            if _bool(raw.get(flag, False), f'{name}.{flag}'):
                valid = False
        valid = valid and text not in ('###', '*')
        aligned = alignment_valid and _bool(raw.get('alignment_valid', True), f'{name}.alignment_valid')
        teacher_correct = raw.get('teacher_correct')
        if teacher_correct is not None:
            _bool(teacher_correct, f'{name}.teacher_correct')
    else:
        quad = canonical_quad(raw, name)
        text, valid, aligned, teacher_correct = '', True, alignment_valid, None
    return {'quad': quad.tolist(), 'transcript': text, 'native_height': quad_line_size(quad)[1],
            'valid': valid, 'alignment_valid': aligned, 'teacher_correct': teacher_correct}


def _sampling_spec(value):
    spec = {'mode': 'image', 'height_edges': [24., 32., 48., 64., 96.],
            'bucket_weights': None, 'jitter_fraction': .25, 'min_visible_fraction': .5}
    if value is not None and not isinstance(value, dict):
        raise ValueError('data.text_sampling must be a mapping')
    if set(value or {}) - set(spec):
        raise ValueError(f'Unknown data.text_sampling keys: {sorted(set(value) - set(spec))}')
    spec.update(value or {})
    if spec['mode'] not in ('image', 'line_balanced', 'height_buckets'):
        raise ValueError('data.text_sampling.mode must be image, line_balanced or height_buckets')
    edges = spec['height_edges']
    if not isinstance(edges, (list, tuple)) or any(isinstance(x, bool) or not isinstance(x, (int, float))
            or not math.isfinite(x) or x <= 0 for x in edges) or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError('data.text_sampling.height_edges must be strictly increasing positive heights')
    weights = spec['bucket_weights']
    if weights is None:
        weights = [1.] * (len(edges) + 1)
    if not isinstance(weights, (list, tuple)) or len(weights) != len(edges) + 1:
        raise ValueError('data.text_sampling.bucket_weights needs len(height_edges) + 1 entries')
    spec['bucket_weights'] = [_number(x, 'data.text_sampling.bucket_weights') for x in weights]
    if not any(spec['bucket_weights']):
        raise ValueError('data.text_sampling.bucket_weights cannot all be zero')
    spec['height_edges'] = list(map(float, edges))
    spec['jitter_fraction'] = _number(spec['jitter_fraction'], 'data.text_sampling.jitter_fraction', hi=.5)
    spec['min_visible_fraction'] = _number(spec['min_visible_fraction'],
                                           'data.text_sampling.min_visible_fraction', hi=1.)
    return spec


def dataset_options_from_config(config):
    """Extra keyword-only PairedCropDataset options, separate from the legacy six positional fields."""
    return {key: config[key] for key in DATASET_OPTION_KEYS if key in config}


class PairedCropDataset(Dataset):
    """Same-size [0,1] LQ/HQ crops; no rescaling, flipping, or online degradation.

    text_boxes is a JSON mapping image stems (or filenames) to lists of flattened
    quads, xyxy AABBs, or dictionaries with geometry plus optional transcript,
    valid, alignment_valid and teacher_correct. Dictionary quads must be cyclic,
    starting at the reading-order top-left corner. Crop-relative quads retain
    their unclipped corners; polygon/bbox describe visible pixels. native_height
    is measured on the original quad, never the clipped crop or OCR-resized line.

    With metadata disabled the old hq/lq/(optional AABB text_mask) contract holds.
    """
    def __init__(self, lq_dir, hq_dir, size, text_boxes=None, text_crop_prob=0.0, text_mask=False, *,
                 text_metadata=False, small_text_height=64., text_sampling=None, alignment_valid=True,
                 allow_missing_text=False, teacher_error_weight=.1, deterministic_crops=False, seed=42,
                 source_id='paired', image_names=None):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError('crop size must be a positive integer')
        if isinstance(text_crop_prob, bool) or not isinstance(text_crop_prob, (int, float)) \
                or not math.isfinite(text_crop_prob) or not 0 <= text_crop_prob <= 1:
            raise ValueError('data.text_crop_prob must be in [0, 1]')
        self.text_metadata = _bool(text_metadata, 'data.text_metadata')
        allow_missing_text = _bool(allow_missing_text, 'data.allow_missing_text')
        alignment_valid = _bool(alignment_valid, 'data.alignment_valid')
        if text_crop_prob and not text_boxes:
            raise ValueError('data.text_crop_prob needs data.text_boxes')
        if (text_mask or self.text_metadata) and not text_boxes and not allow_missing_text:
            raise ValueError('Text-box masks/metadata need data.text_boxes (or explicit allow_missing_text: true)')
        for name, directory in (('lq_dir', lq_dir), ('hq_dir', hq_dir)):
            if not directory or not Path(directory).is_dir():
                raise FileNotFoundError(f'data.{name} is not a directory: {directory}')
        hq, lq = _images(hq_dir), _images(lq_dir)
        if not hq:
            raise ValueError(f'{hq_dir}: empty training dataset')
        selected_names = _selected_names(image_names, hq)
        hq = {name: hq[name] for name in selected_names}
        missing = sorted(set(hq) - set(lq))
        if missing:
            raise FileNotFoundError(f'{lq_dir}: no LQ image for {missing[:5]} ({len(missing)} missing)')
        self.names, self.hq, self.lq = sorted(hq), hq, lq
        self.size, self.text_crop_prob, self.text_mask = size, float(text_crop_prob), bool(text_mask)
        self.small_text_height = _number(small_text_height, 'data.small_text_height', inclusive_lo=False)
        self.teacher_error_weight = _number(teacher_error_weight, 'data.teacher_error_weight', hi=1.)
        self.sampling = _sampling_spec(text_sampling)
        self.source_id = str(source_id)
        self.deterministic_crops = _bool(deterministic_crops, 'data.deterministic_crops')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('data.seed must be an integer')
        self.seed, self.epoch = seed, 0
        self.lines, self.boxes = {}, {}
        if text_boxes:
            annotations = json.loads(Path(text_boxes).read_text(encoding='utf-8'))
            if not isinstance(annotations, dict):
                raise ValueError(f'{text_boxes}: expected a mapping from image name to text lines')
            for raw_name, records in annotations.items():
                if not isinstance(raw_name, str) or not raw_name:
                    raise ValueError(f'{text_boxes}: image names must be nonempty strings')
                name = _image_name(raw_name, hq)
                if name in self.lines:
                    raise ValueError(f'{text_boxes}: duplicate image stem {name!r}')
                aligned = alignment_valid
                if isinstance(records, dict):
                    aligned = aligned and _bool(records.get('alignment_valid', True), f'{raw_name}.alignment_valid')
                    records = records.get('lines')
                if not isinstance(records, list):
                    raise ValueError(f'{text_boxes}: {raw_name} must map to a list of text quads/records')
                self.lines[name] = [_line_record(record, f'{text_boxes}:{raw_name}[{i}]', aligned)
                                    for i, record in enumerate(records)]
                self.boxes[name] = [np.asarray(line['quad']).reshape(-1).tolist() for line in self.lines[name]]
        # Global per-line sampling prevents images with few lines from dominating.
        self._all_lines = [(name, i) for name in self.names for i, line in enumerate(self.lines.get(name, ()))
                           if line['valid']]
        self._buckets = [[] for _ in self.sampling['bucket_weights']]
        for entry in self._all_lines:
            height = self.lines[entry[0]][entry[1]]['native_height']
            self._buckets[bisect.bisect_right(self.sampling['height_edges'], height)].append(entry)
        if self.text_crop_prob and self.sampling['mode'] != 'image' and not self._all_lines:
            raise ValueError('Line-balanced text sampling needs at least one valid annotated line')
        if self.text_crop_prob and self.sampling['mode'] == 'height_buckets' and not any(
                bucket and weight > 0 for bucket, weight in zip(self._buckets, self.sampling['bucket_weights'])):
            raise ValueError('No annotated lines fall in a positive-weight text height bucket')

    def __len__(self):
        return len(self.names)

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError('dataset epoch must be a nonnegative integer')
        self.epoch = epoch

    def _rng(self, index):
        if not self.deterministic_crops:
            return random
        key = f'{self.seed}:{self.epoch}:{int(index)}'.encode()
        return random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little'))

    @staticmethod
    def _load(path):
        with Image.open(path) as image:
            return np.array(image.convert('RGB'), copy=True)

    def _choose_line(self, entries, rng):
        if self.sampling['mode'] != 'height_buckets':
            return rng.choice(entries)
        buckets = [[] for _ in self.sampling['bucket_weights']]
        for entry in entries:
            height = self.lines[entry[0]][entry[1]]['native_height']
            buckets[bisect.bisect_right(self.sampling['height_edges'], height)].append(entry)
        available = [i for i, bucket in enumerate(buckets)
                     if bucket and self.sampling['bucket_weights'][i] > 0]
        if not available:
            return None
        chosen = rng.choices(available, weights=[self.sampling['bucket_weights'][i] for i in available], k=1)[0]
        return rng.choice(buckets[chosen])

    def sample_index(self, index, rng=None):
        """Reproducible entry point for calibration, including source/line/height sampling."""
        rng = self._rng(index) if rng is None else rng
        name = self.names[index % len(self.names)]
        if self.sampling['mode'] == 'image':
            return self.sample(name, rng)
        if self._all_lines and rng.random() < self.text_crop_prob:
            if self.sampling['mode'] == 'height_buckets':
                available = [i for i, bucket in enumerate(self._buckets)
                             if bucket and self.sampling['bucket_weights'][i] > 0]
                which = rng.choices(available,
                                    weights=[self.sampling['bucket_weights'][i] for i in available], k=1)[0]
                name, line_index = rng.choice(self._buckets[which])
            else:
                name, line_index = rng.choice(self._all_lines)
            return self.sample(name, rng, _line_index=line_index, _text_focus=True)
        return self.sample(name, rng, _text_focus=False)

    def sample(self, name, rng, *, _line_index=None, _text_focus=None):
        """Named-image preview; sample_index additionally honors global line balancing."""
        lq, hq = self._load(self.lq[name]), self._load(self.hq[name])
        if lq.shape != hq.shape:
            raise ValueError(f'{name}: LQ {lq.shape[:2]} and HQ {hq.shape[:2]} sizes differ; '
                             'upsample the LQ to the HQ size first')
        s = self.size
        original_h, original_w = hq.shape[:2]
        h, w = original_h, original_w
        lines = self.lines.get(name, [])
        for i, line in enumerate(lines):
            quad = np.asarray(line['quad'])
            tolerance = max(2., .02 * max(original_h, original_w))
            if (quad < -tolerance).any() or (quad[:, 0] > original_w + tolerance).any() \
                    or (quad[:, 1] > original_h + tolerance).any():
                raise ValueError(f'{name} line {i}: text coordinates exceed HQ size '
                                 f'{original_w}x{original_h}; annotations must use HQ/output pixels')
        if h < s or w < s:
            pad = ((0, max(0, s - h)), (0, max(0, s - w)), (0, 0))
            lq, hq = np.pad(lq, pad, mode='edge'), np.pad(hq, pad, mode='edge')
            h, w = hq.shape[:2]
        candidates = [(name, i) for i, line in enumerate(lines) if line['valid']]
        if _text_focus is None:
            _text_focus = bool(candidates) and rng.random() < self.text_crop_prob
        if _text_focus and _line_index is None:
            selected = self._choose_line(candidates, rng)
            _line_index = selected[1] if selected is not None else None
        if _text_focus and _line_index is not None:
            quad = np.asarray(lines[_line_index]['quad'], dtype=np.float64)
            cx, cy = quad.mean(axis=0)
            jitter = s * self.sampling['jitter_fraction']
            x0 = int(np.clip(cx - s / 2 + rng.uniform(-jitter, jitter), 0, w - s))
            y0 = int(np.clip(cy - s / 2 + rng.uniform(-jitter, jitter), 0, h - s))
        else:
            x0, y0 = rng.randint(0, w - s), rng.randint(0, h - s)
        def crop(image):
            return torch.from_numpy(np.ascontiguousarray(image[y0:y0 + s, x0:x0 + s])).permute(2, 0, 1)
        item = {'hq': crop(hq).float() / 255, 'lq': crop(lq).float() / 255}
        if not self.text_metadata:
            if self.text_mask:
                # Deliberately retain the legacy AABB mask, including its rounding.
                mask = torch.zeros(1, s, s)
                for quad in self.boxes.get(name, ()):
                    xs, ys = np.asarray(quad[0::2]) - x0, np.asarray(quad[1::2]) - y0
                    left, right = int(np.clip(np.floor(xs.min()), 0, s)), int(np.clip(np.ceil(xs.max()), 0, s))
                    top, bottom = int(np.clip(np.floor(ys.min()), 0, s)), int(np.clip(np.ceil(ys.max()), 0, s))
                    mask[:, top:bottom, left:right] = 1
                item['text_mask'] = mask
            return item
        visible_lines = []
        text_mask = np.zeros((s, s), dtype=np.float32)
        small_mask = np.zeros_like(text_mask)
        trust = np.ones_like(text_mask)
        valid_mask = np.zeros_like(text_mask)
        valid_mask[:max(0, min(s, original_h - y0)), :max(0, min(s, original_w - x0))] = 1
        for i, line in enumerate(lines):
            quad = np.asarray(line['quad']) - [x0, y0]
            polygon = clip_polygon(quad, 0, 0, min(s, original_w - x0), min(s, original_h - y0))
            if _area(polygon) <= 1e-8:
                continue
            fraction = min(1., _area(polygon) / _area(quad))
            record = {**line, 'quad': quad.tolist(), 'polygon': polygon.tolist(), 'line_index': i,
                      'bbox': [*polygon.min(axis=0).tolist(), *polygon.max(axis=0).tolist()],
                      'visible_fraction': fraction, 'fully_visible': fraction >= 1. - 1e-6,
                      'valid': line['valid'] and fraction >= self.sampling['min_visible_fraction']}
            visible_lines.append(record)
            if not record['valid']:
                continue
            mask = polygon_mask(polygon, s, s)
            text_mask = np.maximum(text_mask, mask)
            if record['native_height'] < self.small_text_height:
                small_mask = np.maximum(small_mask, mask)
            if record['teacher_correct'] is False and record['alignment_valid']:
                trust[mask > 0] = np.minimum(trust[mask > 0], self.teacher_error_weight)
        item.update(text_lines=visible_lines, text_mask=torch.from_numpy(text_mask)[None],
                    small_text_mask=torch.from_numpy(small_mask)[None],
                    kd_trust_mask=torch.from_numpy(trust)[None], valid_mask=torch.from_numpy(valid_mask)[None],
                    source_id=self.source_id, sample_name=name, crop_origin=torch.tensor([x0, y0]),
                    original_size=torch.tensor([original_h, original_w]))
        return item

    def __getitem__(self, index):
        return self.sample_index(index)


class PairedMixtureDataset(Dataset):
    """Paired sources sampled with replacement; source overrides inherit global crop options."""
    def __init__(self, config, text_mask=False):
        sources = config['sources']
        if not isinstance(sources, list) or not sources:
            raise ValueError('data.sources must be a nonempty list of paired sources')
        if config.get('lq_dir') or config.get('hq_dir'):
            raise ValueError('Use data.sources OR top-level lq_dir/hq_dir, not both')
        self.datasets, self.weights, names = [], [], set()
        allowed = {'name', 'weight', 'lq_dir', 'hq_dir', 'text_boxes', 'text_crop_prob', *DATASET_OPTION_KEYS}
        global_options = dataset_options_from_config(config)
        self.deterministic_crops = _bool(config.get('deterministic_crops', False), 'data.deterministic_crops')
        self.seed, self.epoch = config.get('seed', 42), 0
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError('data.seed must be an integer')
        for source in sources:
            if not isinstance(source, dict) or set(source) - allowed:
                raise ValueError(f'Each data.sources entry accepts only {sorted(allowed)}')
            name, weight = source.get('name'), source.get('weight')
            if not isinstance(name, str) or not name or name in names:
                raise ValueError('Paired source names must be nonempty and unique')
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                raise ValueError('Paired source weight must be finite and positive')
            names.add(name)
            self.weights.append(float(weight))
            options = {**global_options, **dataset_options_from_config(source)}
            for key, expected in (('deterministic_crops', self.deterministic_crops), ('seed', self.seed)):
                if key in source and source[key] != expected:
                    raise ValueError(f'data.sources must share the mixture-level {key}; '
                                     'one RNG controls source selection and crop sampling')
            if 'text_sampling' in source and isinstance(source['text_sampling'], dict):
                options['text_sampling'] = {**(config.get('text_sampling') or {}), **source['text_sampling']}
            if options.get('text_metadata', False) != config.get('text_metadata', False):
                raise ValueError('All paired sources must share data.text_metadata for compatible collation')
            self.datasets.append(PairedCropDataset(
                source.get('lq_dir'), source.get('hq_dir'), int(config['resolution']),
                source.get('text_boxes', config.get('text_boxes')),
                source.get('text_crop_prob', config.get('text_crop_prob', 0.0)), text_mask,
                **options, source_id=name))
        self.length = config.get('samples_per_epoch', sum(len(dataset) for dataset in self.datasets))
        if isinstance(self.length, bool) or not isinstance(self.length, int) or self.length <= 0:
            raise ValueError('data.samples_per_epoch must be a positive integer')

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError('dataset epoch must be a nonnegative integer')
        self.epoch = epoch
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def sample_index(self, index, rng=None):
        if rng is None:
            if self.deterministic_crops:
                key = f'{self.seed}:{self.epoch}:{int(index)}'.encode()
                rng = random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little'))
            else:
                rng = random
        dataset = rng.choices(self.datasets, weights=self.weights, k=1)[0]
        return dataset.sample_index(rng.randrange(len(dataset)), rng)

    def __getitem__(self, index):
        return self.sample_index(index)


def collate_paired_samples(items):
    """Stack tensor fields while preserving one variable-length annotation list per image."""
    if not items:
        raise ValueError('Cannot collate an empty paired batch')
    keys = set(items[0])
    if any(set(item) != keys for item in items):
        raise ValueError('Paired batch items have different fields; enable text_metadata for every source')
    result = default_collate([{key: value for key, value in item.items() if key != 'text_lines'} for item in items])
    if 'text_lines' in keys:
        result['text_lines'] = [item['text_lines'] for item in items]
    return result


paired_collate = collate_paired_samples
