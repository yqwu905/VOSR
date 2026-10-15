"""Pixel-space VOSR2 student after L2P (arXiv 2605.12013, TencentYoutuResearch/T2I-L2P): no VAE.

A pixel patch embedding replaces the latent one, and a convolutional U-Net decoder replaces the final
layer together with the VAE decoder; as in L2P it reads the last block's hidden state. The input is still
``cat(LQ, x_t)``, now RGB in [-1, 1]. One VOSR2 latent token covers 16 x 16 pixels (Qwen-Image VAE 8x times patch 2), so
``patch_size`` 16 keeps the pretrained token grid and RoPE positions; 32 quarters the tokens and turns
on the decoder's 2x upsampling of the DiT features.

The decoder predicts the clean image (x-prediction, as JiT finds necessary for large pixel patches);
``forward`` returns it as the flow velocity (x_t - x0) / t, so the trainer and inference keep computing
the image as ``noise - velocity`` at the one-step t = 1.
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .lightningdit_ablation import AblationLightningDiT

PIXEL_ENCODERS = ('patch', 'unet')
DECODER_STRIDE = 16  # the decoder's four stride-2 stages
RGB = 3


def gradient_checkpoint_forward(model, use_gradient_checkpointing, use_gradient_checkpointing_offload, *args, **kwargs):
    """DiffSynth-Studio's helper of that name (diffsynth/core/gradient/gradient_checkpoint.py) without its
    DeepSpeed branch: non-reentrant checkpointing, with activations kept on the CPU when offloading."""
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            return checkpoint(model, *args, **kwargs, use_reentrant=False)
    if use_gradient_checkpointing:
        return checkpoint(model, *args, **kwargs, use_reentrant=False)
    return model(*args, **kwargs)


class MicroDiffusionModelTop3SCWide(nn.Module):
    """Pixel-space U-Net decoder -- top-3 skip variant (no full-resolution skip).

    The decoder supplied for this project, layers unchanged: encoder 48/96/192/384 channels down to 1/16,
    a 1x1 bottleneck over ``cat(p4, c)`` with the DiT feature map ``c``, transposed-conv upsampling with
    additive skips at 1/8, 1/4 and 1/2, and no skip at full resolution. Two changes:

    * ``c_scale`` is the DiT token size over 16 pixels. The supplied ``c_upsample`` (a 2x transposed conv
      initialized to nearest 2x) is for 32-pixel tokens; it is built for ``c_scale`` > 1 and omitted for
      16-pixel tokens, whose grid already is the bottleneck's.
    * ``encode`` / ``decode`` split ``forward`` so the down path can also embed the DiT tokens
      (``student.pixel.encoder: unet``); ``forward`` runs both, as the supplied forward did.
    """

    def __init__(self, in_channels, si_t_hidden_size, k_convT=2, p_convT=0, c_scale=2):
        super().__init__()

        # ---------- Encoder ----------
        self.ds1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.ds2 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)

        self.enc3 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.ds3 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)

        self.enc4 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.ds4 = nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1)

        # ---------- Bottleneck ----------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(384 + si_t_hidden_size, 384, kernel_size=1),
            nn.ReLU(),
        )

        # ---------- Decoder (inner: ConvTranspose2d + additive skip) ----------
        self.up4 = nn.ConvTranspose2d(384, 384, kernel_size=k_convT, stride=2, padding=p_convT)
        self.dec4 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.up3 = nn.ConvTranspose2d(192, 192, kernel_size=k_convT, stride=2, padding=p_convT)
        self.dec3 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(96, 96, kernel_size=k_convT, stride=2, padding=p_convT)
        self.dec2 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # ---------- Decoder (outer: ConvTranspose2d, NO skip from enc1) ----------
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(48, 48, kernel_size=k_convT, stride=2, padding=p_convT),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(48, 3, kernel_size=1)

        # DiT feature upsampling: a ConvTranspose2d(k=s=c_scale, p=0) whose weight is the identity channel
        # map times an all-ones kernel reproduces F.interpolate(mode='nearest', scale_factor=c_scale).
        if c_scale > 1:
            self.c_upsample = nn.ConvTranspose2d(
                si_t_hidden_size, si_t_hidden_size, kernel_size=c_scale, stride=c_scale, padding=0, bias=False
            )
            with torch.no_grad():
                _w = torch.eye(si_t_hidden_size).reshape(
                    si_t_hidden_size, si_t_hidden_size, 1, 1) * torch.ones(1, 1, c_scale, c_scale)
                self.c_upsample.weight.copy_(_w)
        else:
            self.c_upsample = nn.Identity()

    # ---------- stage 1: H -> H/2 (ds1); enc1_out is not a skip ----------
    def _enc_stage_1(self, x_in):
        return self.ds1(x_in)                                   # [B, 48,  H/2,  W/2 ]

    # ---------- stage 2: H/2 -> H/2 (enc2) -> H/4 (ds2) ----------
    def _enc_stage_2(self, x_in):
        enc2_out = self.enc2(x_in)                              # [B, 96,  H/2,  W/2 ]
        return enc2_out, self.ds2(enc2_out)                     # [B, 96,  H/4,  W/4 ]

    # ---------- stage 3: H/4 -> H/4 (enc3) -> H/8 (ds3) ----------
    def _enc_stage_3(self, x_in):
        enc3_out = self.enc3(x_in)                              # [B, 192, H/4,  W/4 ]
        return enc3_out, self.ds3(enc3_out)                     # [B, 192, H/8,  W/8 ]

    # ---------- stage 4: H/8 -> H/8 (enc4) -> H/16 (ds4) ----------
    def _enc_stage_4(self, x_in):
        enc4_out = self.enc4(x_in)                              # [B, 384, H/8,  W/8 ]
        return enc4_out, self.ds4(enc4_out)                     # [B, 384, H/16, W/16]

    # ---------- bottleneck: cat(p4, upsampled c) -> 1x1 conv ----------
    def _bottleneck(self, p4_in, c_in):
        c_in = self.c_upsample(c_in)
        return self.bottleneck(torch.cat([p4_in, c_in], dim=1))  # [B, 384, H/16, W/16]

    # ---------- stages 5-7 (decoder): up + additive skip + conv ----------
    def _dec_stage_4(self, bottleneck_in, enc4_skip):
        return self.dec4(self.up4(bottleneck_in) + enc4_skip)   # [B, 192, H/8, W/8]

    def _dec_stage_3(self, dec_in, enc3_skip):
        return self.dec3(self.up3(dec_in) + enc3_skip)          # [B, 96,  H/4, W/4]

    def _dec_stage_2(self, dec_in, enc2_skip):
        return self.dec2(self.up2(dec_in) + enc2_skip)          # [B, 48,  H/2, W/2]

    # ---------- stage 8 (decoder): up1 + out_conv (NO skip) ----------
    def _dec_stage_1_and_out(self, dec_in):
        return self.out_conv(self.up1(dec_in))                  # [B, 3,   H,   W]

    def encode(self, x, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
        """Down path: the skips (enc2, enc3, enc4 outputs) and the 1/16 features p4."""
        _gc = lambda fn, *a: gradient_checkpoint_forward(
            fn, use_gradient_checkpointing, use_gradient_checkpointing_offload, *a,
        )
        p1_out = _gc(self._enc_stage_1, x)
        enc2_feat, p2_out = _gc(self._enc_stage_2, p1_out)
        enc3_feat, p3_out = _gc(self._enc_stage_3, p2_out)
        enc4_feat, p4_out = _gc(self._enc_stage_4, p3_out)
        return (enc2_feat, enc3_feat, enc4_feat), p4_out

    def decode(self, skips, p4_out, c, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
        _gc = lambda fn, *a: gradient_checkpoint_forward(
            fn, use_gradient_checkpointing, use_gradient_checkpointing_offload, *a,
        )
        enc2_feat, enc3_feat, enc4_feat = skips
        bottleneck_out = _gc(self._bottleneck, p4_out, c)
        dec4_out = _gc(self._dec_stage_4, bottleneck_out, enc4_feat)
        dec3_out = _gc(self._dec_stage_3, dec4_out, enc3_feat)
        dec2_out = _gc(self._dec_stage_2, dec3_out, enc2_feat)
        return _gc(self._dec_stage_1_and_out, dec2_out)

    def forward(self, x, c, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
        skips, p4_out = self.encode(x, use_gradient_checkpointing, use_gradient_checkpointing_offload)
        return self.decode(skips, p4_out, c, use_gradient_checkpointing, use_gradient_checkpointing_offload)


def pixel_spec(config):
    """Validate a ``student.pixel`` mapping and fill in its defaults."""
    if not isinstance(config, dict):
        raise ValueError('student.pixel must be a mapping')
    unknown = set(config) - {'patch_size', 'encoder'}
    if unknown:
        raise ValueError(f'Unknown student.pixel keys: {sorted(unknown)}')
    spec = {'patch_size': config.get('patch_size', 16), 'encoder': config.get('encoder', 'patch')}
    patch = spec['patch_size']
    if isinstance(patch, bool) or not isinstance(patch, int) or patch <= 0 or patch % DECODER_STRIDE:
        raise ValueError(f'student.pixel.patch_size must be a positive multiple of {DECODER_STRIDE} '
                         '(the decoder downsamples 16x)')
    if spec['encoder'] not in PIXEL_ENCODERS:
        raise ValueError(f'student.pixel.encoder must be one of {PIXEL_ENCODERS}')
    if spec['encoder'] == 'unet' and patch != DECODER_STRIDE:
        raise ValueError(f'student.pixel.encoder unet embeds the decoder\'s 1/16 features: needs patch_size {DECODER_STRIDE}')
    return spec


def pixel_model_config(model, resolution, pixel):
    """Top-level ``model`` settings (latent VOSR2) -> PixelLightningDiT settings.

    RGB LQ and RGB x_t in, RGB out; ``resolution`` (HQ pixels) is the reference grid that the latent
    model sees as resolution // 8 with patch 2, i.e. the same RoPE positions at patch 16."""
    spec = pixel_spec(pixel)
    return dict(model, input_size=resolution, patch_size=spec['patch_size'], in_channels=2 * RGB,
                out_channels=RGB, pixel=spec)


class PixelLightningDiT(AblationLightningDiT):
    """AblationLightningDiT on RGB pixels (``student.pixel``); see the module docstring."""
    pixel_space = True
    NEW_MODULE_PREFIXES = ('x_embedder.', 'decoder.')

    def __init__(self, *, pixel, **kwargs):
        spec = pixel_spec(pixel)
        if kwargs.get('router_config') is not None or kwargs.get('token_compression') is not None:
            raise ValueError('student.pixel supports neither router_config nor token_compression')
        super().__init__(**kwargs)
        if self.patch_size != spec['patch_size'] or self.in_channels != 2 * RGB or self.out_channels != RGB:
            raise ValueError('Build the pixel student with pixel_model_config')
        self.pixel = spec
        self.export_config['pixel'] = spec
        # As L2P: no final layer; the decoder reads the last block's hidden state (VOSR2's is already about
        # unit RMS per token, see docs/ablation_training.md).
        del self.final_layer
        self.decoder = MicroDiffusionModelTop3SCWide(self.in_channels, self.hidden_size,
                                                     c_scale=self.patch_size // DECODER_STRIDE)
        if spec['encoder'] == 'unet':
            # Tokens embed the decoder's own 1/16 features (384 channels) instead of a linear pixel patchify;
            # initialized like the patch embedding (as a Linear).
            self.x_embedder.proj = nn.Conv2d(384, self.hidden_size, kernel_size=1)
            nn.init.xavier_uniform_(self.x_embedder.proj.weight.data.view(self.hidden_size, -1))
            nn.init.zeros_(self.x_embedder.proj.bias)

    def is_new_weight(self, key):
        """Patch embedding and decoder: fresh unless a checkpoint has them with the same shape (L2P loading)."""
        return key.startswith(self.NEW_MODULE_PREFIXES)

    def is_removed_weight(self, key):
        return key.startswith('final_layer.') or super().is_removed_weight(key)

    def _decoder_checkpointing(self):
        return self.use_checkpoint and self.training and torch.is_grad_enabled(), False

    def _embed(self, x):
        if self.pixel['encoder'] == 'patch':
            return self.x_embedder(x), None
        skips, p4 = self.decoder.encode(x, *self._decoder_checkpointing())
        return self.x_embedder.proj(p4).flatten(2).transpose(1, 2), (skips, p4)

    def _head(self, x, c, t, inputs, encoded):
        b, n, d = x.shape
        side = math.isqrt(n)
        features = x.transpose(1, 2).reshape(b, d, side, side)
        if encoded is None:
            image = self.decoder(inputs, features, *self._decoder_checkpointing())
        else:
            image = self.decoder.decode(*encoded, features, *self._decoder_checkpointing())
        # FP32, so that noise - velocity gives back the decoder's image without bf16 rounding.
        noisy = inputs[:, RGB:].float()
        return (noisy - image.float()) / t.float().view(-1, 1, 1, 1)
