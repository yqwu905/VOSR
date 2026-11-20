"""Differentiable TextVerifier Stage-3 error on annotated SR glyphs."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.text_losses import rectify_text_roi


class TextVerifierGlyphLoss(nn.Module):
    """Mean distance to the closest target-character prototype. Inputs are RGB in [-1, 1]."""

    def __init__(self, encoder, prototypes, char_to_index, batch_size=8):
        super().__init__()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('glyph_loss.batch_size must be a positive integer')
        self.encoder = encoder.eval().requires_grad_(False)
        self.register_buffer('prototypes', F.normalize(prototypes.detach().float(), dim=-1), persistent=False)
        self.char_to_index = char_to_index
        self.batch_size = batch_size
        self.last_counts = (0, 0)  # scored, unsupported

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @classmethod
    def from_config(cls, config):
        import timm

        if not isinstance(config, dict) or set(config) - {'model_dir', 'batch_size'}:
            raise ValueError('glyph_loss accepts model_dir and batch_size')
        root = Path(config.get('model_dir', 'preset/ckpts/TextVerifier'))
        state = torch.load(root / 'textverifier_stage3.pth', map_location='cpu', weights_only=True, mmap=True)
        state = state.get('model_state_dict', state)
        # The released prototype readout uses only the vision tower and image projector.
        backbone = timm.create_model('vit_large_patch16_224', pretrained=False, img_size=128,
                                     patch_size=16, num_classes=0, global_pool='avg')
        dim = backbone.embed_dim
        projector = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(dim, dim, bias=False)), ('ln', nn.LayerNorm(dim)), ('act', nn.GELU()),
            ('fc2', nn.Linear(dim, dim, bias=False)),
        ]))
        for module, prefix in ((backbone, 'image_encoder.backbone.'), (projector, 'image_projector.')):
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        pack = torch.load(root / 'textverifier_k10_q90.pt', map_location='cpu', weights_only=True)
        bank, mapping = pack['prototypes'], pack['char_to_index']
        if bank.shape != (len(mapping), pack['K'], dim) or pack['K'] < 1 or len(pack['chars']) != len(mapping) \
                or any(mapping.get(char) != i for i, char in enumerate(pack['chars'])):
            raise ValueError('TextVerifier prototype dimensions or character indices are inconsistent')
        return cls(nn.Sequential(backbone, projector), bank, mapping, config.get('batch_size', 8))

    def _score(self, images, entries):
        crops, rows = [], []
        for index, glyph, row in entries:
            quad = np.asarray(glyph['quad'])
            # Rectify at native resolution before antialiased bilinear resizing, as in the release.
            top, right, bottom, left = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            size = (max(1, round(max(left, right))), max(1, round(max(top, bottom))))
            crop = rectify_text_roi(images[index:index + 1], quad, size)
            crops.append(F.interpolate(crop, (128, 128), mode='bilinear', align_corners=False, antialias=True))
            rows.append(row)
        pixels = torch.cat(crops)
        # Recompute frozen ViT activations during backward so long pages can be scored in chunks.
        features = (checkpoint(self.encoder, pixels, use_reentrant=False, preserve_rng_state=False)
                    if pixels.requires_grad and torch.is_grad_enabled() else self.encoder(pixels))
        features = F.normalize(features.float(), dim=-1)
        similarities = torch.einsum('nd,nkd->nk', features, self.prototypes[rows])
        return 1.0 - similarities.max(dim=1).values

    def forward(self, prediction, text_lines):
        """Score the dataset's fully visible, aligned, crop-relative character quads."""
        if prediction.ndim != 4 or prediction.shape[1] != 3 or len(text_lines) != prediction.shape[0]:
            raise ValueError('glyph_loss needs an RGB NCHW prediction and one annotation list per image')
        entries = []
        unsupported = 0
        for index, lines in enumerate(text_lines):
            for line in lines:
                for glyph in line.get('glyphs', ()):
                    row = self.char_to_index.get(glyph['char'])
                    if row is None:
                        unsupported += 1
                    else:
                        entries.append((index, glyph, row))
        self.last_counts = len(entries), unsupported
        if not entries:
            return prediction.flatten()[:0].float().sum()
        with torch.autocast(prediction.device.type, enabled=False):
            images = prediction.float().clamp(-1, 1)
            scores = [self._score(images, entries[i:i + self.batch_size])
                      for i in range(0, len(entries), self.batch_size)]
            return torch.cat(scores).mean()
