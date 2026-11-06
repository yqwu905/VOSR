"""Native-scale, line-balanced text supervision for a decoded SR prediction.

Targets are RGB tensors in [-1,1]; pixel/edge predictions may be unclamped
decoder values. Pixel/edge losses use the original output grid and polygon
masks. Only OCR rectifies/resizes a line. The existing frozen
KLOCRLoss recognizer can be reused, but its full-width strip helper is not used.
No teacher/recognizer module is required at inference.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dataloaders.paired_crops import canonical_quad, clip_polygon, polygon_mask, quad_line_size


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')
    return float(value)


def _quad(record):
    keys = [key for key in ('quad', 'points', 'bbox', 'box') if key in record]
    if not keys:
        raise ValueError('A text line needs quad/points/bbox geometry')
    # Dataset records intentionally contain both the original quad and its clipped bbox.
    preferred = next(key for key in ('quad', 'points', 'bbox', 'box') if key in record)
    return canonical_quad(record[preferred], 'text ROI')


def rectify_text_roi(images, quad, output_size):
    """Differentiable perspective crop from cyclic corner geometry.

    Pixel-boundary coordinates are used: [x0,y0,x1,y1] samples the corresponding
    image slice without an extra half-pixel shift. A single quad is applied to
    all N images. Solving the constant 8x8 geometry on CPU also supports devices
    that do not implement float64 linear algebra; gradients remain on the image.
    """
    if images.ndim != 4:
        raise ValueError('rectify_text_roi expects NCHW images')
    quad = canonical_quad(quad, 'OCR ROI')
    out_h, out_w = output_size
    if isinstance(out_h, bool) or isinstance(out_w, bool) or not isinstance(out_h, int) \
            or not isinstance(out_w, int) or min(out_h, out_w) <= 0:
        raise ValueError('OCR output_size must contain positive integer height and width')
    matrix, rhs = [], []
    for (u, v), (x, y) in zip(((0., 0.), (1., 0.), (1., 1.), (0., 1.)), quad):
        matrix.extend(([u, v, 1., 0., 0., 0., -u * x, -v * x],
                       [0., 0., 0., u, v, 1., -u * y, -v * y]))
        rhs.extend((x, y))
    try:
        h = torch.linalg.solve(torch.tensor(matrix, dtype=torch.float64), torch.tensor(rhs, dtype=torch.float64))
    except RuntimeError as error:
        raise ValueError('OCR quad does not define a stable perspective transform') from error
    transform = torch.cat((h, torch.ones(1, dtype=torch.float64))).reshape(3, 3)
    if not torch.isfinite(transform).all():
        raise ValueError('OCR quad produced a nonfinite perspective transform')
    transform = transform.to(device=images.device, dtype=torch.float32)
    ys = (torch.arange(out_h, device=images.device, dtype=torch.float32) + .5) / out_h
    xs = (torch.arange(out_w, device=images.device, dtype=torch.float32) + .5) / out_w
    vv, uu = torch.meshgrid(ys, xs, indexing='ij')
    source = torch.stack((uu, vv, torch.ones_like(uu)), -1) @ transform.T
    denominator = source[..., 2:]
    if (denominator.abs() < 1e-7).any():
        raise ValueError('OCR quad crosses the perspective horizon')
    source = source[..., :2] / denominator
    grid = source / source.new_tensor([images.shape[-1], images.shape[-2]]) * 2 - 1
    return F.grid_sample(images.float(), grid[None].expand(images.shape[0], -1, -1, -1),
                         mode='bilinear', padding_mode='border', align_corners=False)


def _native_sobel(images):
    """Luminance x/y gradients computed before cropping, preserving the native stroke scale."""
    gray = (images * images.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    x = images.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.
    filters = torch.stack((x, x.T))[:, None]
    return F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='replicate'), filters)


class TextROILoss(nn.Module):
    """Per-line Charbonnier, Sobel and optional KL(OCR(GT) || OCR(SR)).

    Pixel losses ignore invalid or unaligned annotations. OCR additionally
    requires the entire line and valid image area; overlong lines that exceed
    ocr_max_width are explicitly skipped (counts are exposed in last_counts),
    rather than geometrically squashed to a fixed width. Every eligible line
    contributes its own normalized loss before averaging across lines.
    """
    def __init__(self, config=None, ocr_kl=None):
        super().__init__()
        defaults = dict(eps=1e-3, small_text_height=64., small_text_weight=1.,
                        min_visible_fraction=.5, ocr_min_visible_fraction=.999,
                        ocr_max_width=1024, ocr_width_multiple=8)
        if config is not None and not isinstance(config, dict):
            raise ValueError('text_roi must be a mapping')
        if set(config or {}) - set(defaults):
            raise ValueError(f'Unknown text_roi keys: {sorted(set(config) - set(defaults))}')
        self.config = {**defaults, **(config or {})}
        for name in ('eps', 'small_text_height', 'small_text_weight'):
            self.config[name] = _positive(self.config[name], f'text_roi.{name}')
        for name in ('min_visible_fraction', 'ocr_min_visible_fraction'):
            self.config[name] = _positive(self.config[name], f'text_roi.{name}', allow_zero=True)
            if self.config[name] > 1:
                raise ValueError(f'text_roi.{name} must be in [0,1]')
        for name in ('ocr_max_width', 'ocr_width_multiple'):
            value = self.config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'text_roi.{name} must be a positive integer')
        self.ocr_kl = ocr_kl
        if ocr_kl is not None:
            if not all(hasattr(ocr_kl, key) for key in ('recognizer', 'input_height', 'temperature')):
                raise ValueError('ocr_kl must expose recognizer, input_height and temperature')
            if not isinstance(ocr_kl.input_height, int) or ocr_kl.input_height <= 0:
                raise ValueError('OCR input_height must be a positive integer')
            _positive(ocr_kl.temperature, 'OCR temperature')
            ocr_kl.eval().requires_grad_(False)
        self.last_counts = {}

    def train(self, mode=True):
        super().train(mode)
        if self.ocr_kl is not None:
            self.ocr_kl.eval()
        return self

    def forward(self, prediction, target, text_lines, valid_mask=None, *, ocr_prediction=None):
        """Compute native losses; optionally use bounded pixels for OCR alone.

        ``ocr_prediction`` lets the trainer supervise raw decoder values with
        Charbonnier/Sobel while retaining the OCR recognizer's expected range.
        It remains connected to the student graph and defaults to prediction.
        """
        if prediction.ndim != 4 or prediction.shape[1] != 3 or target.shape != prediction.shape:
            raise ValueError('TextROILoss expects matching RGB NCHW prediction and target')
        if ocr_prediction is not None and (ocr_prediction.shape != prediction.shape
                                            or ocr_prediction.device != prediction.device):
            raise ValueError('Text ROI ocr_prediction must match prediction shape and device')
        if not isinstance(text_lines, (list, tuple)) or len(text_lines) != prediction.shape[0] \
                or any(not isinstance(lines, (list, tuple)) for lines in text_lines):
            raise ValueError('text_lines must contain one annotation list per batch image; use paired_collate')
        batch, _, height, width = prediction.shape
        valid_pixels = None
        if valid_mask is not None:
            if valid_mask.shape != (batch, 1, height, width):
                raise ValueError('Text ROI valid_mask must have shape (B,1,H,W)')
            # Dataset masks normally arrive on CPU. Validate/multiply constant
            # ROI geometry there instead of synchronizing the GPU once per line.
            mask_cpu = valid_mask.detach().to(device='cpu', dtype=torch.float32)
            if not torch.isfinite(mask_cpu).all() or (mask_cpu < 0).any() or (mask_cpu > 1).any():
                raise ValueError('Text ROI valid_mask must contain finite weights in [0,1]')
            valid_pixels = mask_cpu.numpy()
            valid_mask = mask_cpu.to(device=prediction.device) if self.ocr_kl is not None else None
        with torch.autocast(prediction.device.type, enabled=False):
            prediction = prediction.float()
            ocr_prediction = prediction if ocr_prediction is None else ocr_prediction.float()
            target = target.detach().to(device=prediction.device, dtype=torch.float32)
            # Keep empty batches connected to the student graph (including under DDP).
            zero = prediction.sum() * 0.
            charb_terms, edge_terms, pixel_weights, ocr_terms, ocr_weights = [], [], [], [], []
            eps = self.config['eps']
            pixel_difference = torch.sqrt((prediction - target).square() + eps ** 2) - eps
            edge_difference = torch.sqrt((_native_sobel(prediction) - _native_sobel(target)).square()
                                         + eps ** 2) - eps
            counts = dict(pixel_lines=0, ocr_lines=0, ocr_skipped_width=0)
            for image_index, records in enumerate(text_lines):
                for record in records:
                    if not isinstance(record, dict):
                        raise ValueError('Each text_lines entry must be a metadata dictionary')
                    for flag in ('valid', 'alignment_valid', 'fully_visible'):
                        if flag in record and not isinstance(record[flag], bool):
                            raise ValueError(f'Text ROI {flag} must be boolean')
                    if not record.get('valid', True) or not record.get('alignment_valid', True):
                        continue
                    visible_fraction = _positive(record.get('visible_fraction', 1.),
                                                 'text line visible_fraction', allow_zero=True)
                    if visible_fraction > 1:
                        raise ValueError('text line visible_fraction must be in [0,1]')
                    quad = _quad(record)
                    visible = clip_polygon(quad, 0, 0, width, height)
                    if len(visible) < 3:
                        continue
                    if visible_fraction < self.config['min_visible_fraction']:
                        continue
                    left, top = np.maximum(np.floor(visible.min(axis=0)).astype(int), 0)
                    right, bottom = np.minimum(np.ceil(visible.max(axis=0)).astype(int), [width, height])
                    if right <= left or bottom <= top:
                        continue
                    mask_array = polygon_mask(visible - [left, top], bottom - top, right - left)
                    if valid_pixels is not None:
                        mask_array *= valid_pixels[image_index, 0, top:bottom, left:right]
                    mass = float(mask_array.sum())
                    if mass <= 0:
                        continue
                    mask = torch.from_numpy(mask_array).to(device=prediction.device, dtype=torch.float32)
                    line_height = record.get('native_height', quad_line_size(quad)[1])
                    _positive(line_height, 'text line native_height')
                    weight = self.config['small_text_weight'] if line_height < self.config['small_text_height'] else 1.
                    charb_terms.append((pixel_difference[image_index, :, top:bottom, left:right] * mask).sum()
                                       / (3. * mass) * weight)
                    edge_terms.append((edge_difference[image_index, :, top:bottom, left:right] * mask).sum()
                                      / (2. * mass) * weight)
                    pixel_weights.append(weight)
                    counts['pixel_lines'] += 1
                    if self.ocr_kl is None or not record.get('fully_visible', True) \
                            or visible_fraction < self.config['ocr_min_visible_fraction'] \
                            or (quad < 0).any() or (quad[:, 0] > width).any() or (quad[:, 1] > height).any():
                        continue
                    line_w, line_h = quad_line_size(quad)
                    multiple = self.config['ocr_width_multiple']
                    ocr_width = max(multiple, round(line_w * self.ocr_kl.input_height / line_h / multiple) * multiple)
                    if ocr_width > self.config['ocr_max_width']:
                        counts['ocr_skipped_width'] += 1
                        continue
                    original_w = max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3]))
                    original_h = max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]))
                    if original_h >= 1.5 * original_w:
                        quad = quad[[1, 2, 3, 0]]  # Match the evaluator's 90-degree counterclockwise rotation.
                    output_size = (self.ocr_kl.input_height, ocr_width)
                    if valid_mask is not None:
                        coverage = rectify_text_roi(valid_mask[image_index:image_index + 1], quad, output_size)
                        if coverage.min().item() < self.config['ocr_min_visible_fraction']:
                            continue
                    temperature = float(self.ocr_kl.temperature)
                    self.ocr_kl.eval()
                    with torch.no_grad():
                        gt_line = rectify_text_roi(target[image_index:image_index + 1], quad, output_size)
                        target_log = F.log_softmax(self.ocr_kl.recognizer(gt_line).float() / temperature, -1)
                    # Do not detach or enter no_grad: gradients must reach the SR decoder and FFN.
                    sr_line = rectify_text_roi(ocr_prediction[image_index:image_index + 1], quad, output_size)
                    prediction_log = F.log_softmax(self.ocr_kl.recognizer(sr_line).float() / temperature, -1)
                    if prediction_log.shape != target_log.shape or prediction_log.ndim != 3:
                        raise ValueError('OCR recognizer must return matching (N,frames,classes) logits')
                    kl = F.kl_div(prediction_log, target_log, reduction='none', log_target=True).sum(-1).mean()
                    ocr_terms.append(kl * temperature ** 2 * weight)
                    ocr_weights.append(weight)
                    counts['ocr_lines'] += 1
            self.last_counts = counts
            return {
                'charbonnier': torch.stack(charb_terms).sum() / sum(pixel_weights) if charb_terms else zero,
                'edge': torch.stack(edge_terms).sum() / sum(pixel_weights) if edge_terms else zero,
                'ocr': torch.stack(ocr_terms).sum() / sum(ocr_weights) if ocr_terms else zero,
            }


def text_roi_losses(prediction, target, text_lines, valid_mask=None, config=None, ocr_kl=None,
                    *, ocr_prediction=None):
    """Functional form for small callers; reuse TextROILoss in the training loop."""
    return TextROILoss(config, ocr_kl)(prediction, target, text_lines, valid_mask,
                                    ocr_prediction=ocr_prediction)
