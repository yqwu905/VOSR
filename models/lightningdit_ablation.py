"""Opt-in VOSR2 ablations; the original LightningDiT is left unchanged."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .lightningdit import LightningDiT
from .sdt_router import AblationBlock, conditioning_key
from .token_compression import (validate_compression_config, pool_tokens,
                                restore_tokens, pooled_rope)


class AblationLightningDiT(LightningDiT):
    def __init__(self, *, use_cross_attention=True, router_config=None,
                 sparse_eval=True, ca_scale=None, compression_config=None, **kwargs):
        kwargs = dict(kwargs)
        if not use_cross_attention:
            kwargs['z_dims'] = None
        elif kwargs.get('z_dims') is None:
            raise ValueError('Cross-attention requires z_dims')
        super().__init__(**kwargs)
        self.blocks = nn.ModuleList(AblationBlock(b, router_config) for b in self.blocks)
        self.use_cross_attention = use_cross_attention
        self.ca_scale = 1.0 if use_cross_attention else 0.0
        if ca_scale is not None:
            self.set_ca_scale(ca_scale)
        self.sparse_eval = sparse_eval
        self.compression_config = validate_compression_config(compression_config, len(self.blocks))
        self.export_config = dict(kwargs, use_cross_attention=use_cross_attention,
                                  router_config=router_config, sparse_eval=sparse_eval,
                                  compression_config=self.compression_config)

    def set_ca_scale(self, scale):
        if not 0 <= scale <= 1 or (not self.use_cross_attention and scale != 0):
            raise ValueError('Invalid CA scale for this architecture')
        self.ca_scale = float(scale)

    def freeze_conditioning(self):
        for name, parameter in self.named_parameters():
            if conditioning_key(name):
                parameter.requires_grad_(False)

    def enable_fused_attn(self):
        self._set_fused_attn(True)

    def disable_fused_attn(self):
        self._set_fused_attn(False)

    def _set_fused_attn(self, enabled):
        for block in self.blocks:
            block.attn.fused_attn = enabled
            if hasattr(block, 'cross_attn'):
                block.cross_attn.fused_attn = enabled

    def _forward_once(self, x, t, r, z, force_dense, disable_compression=False, feature_layers=()):
        _, _, h, w = x.shape
        if h != w or h % self.patch_size:
            raise ValueError('Use square latent crops divisible by patch_size; tile non-square images')
        feature_layers = tuple(feature_layers)
        if len(set(feature_layers)) != len(feature_layers) or any(
                type(i) is not int or not 0 <= i < len(self.blocks) for i in feature_layers):
            raise ValueError('feature_layers must contain unique valid zero-based block indices')
        cfg = self.compression_config
        compress = bool(cfg and cfg['enabled'] and cfg['factor'] > 1 and
                        not force_dense and not disable_compression)
        if compress and (h // self.patch_size) % cfg['factor']:
            raise ValueError('Latent patch grid must be divisible by compression factor; adjust tile/crop size')
        x = self.x_embedder(x)
        c = self.t_embedder(t)
        if self.r_embedder is not None:
            if r is None:
                raise ValueError('r is required for a one-step checkpoint')
            c = c + self.r_embedder(r) * (t - r).unsqueeze(-1)
        c0 = self.t_block(c)
        if self.use_cross_attention and self.ca_scale != 0:
            if not isinstance(z, (list, tuple)) or len(z) != 1:
                raise ValueError('Expected one DINOv2 feature tensor in a list')
            z = self.mlp_ca(self.layer_norm(z[0]))
        else:
            z = None
        rope = self.feat_rope
        if self.use_rope and h != self.x_embedder.img_size[0]:
            rope = self._get_dynamic_rope(h // self.patch_size, x.device, x.dtype)
        statistics, token_counts, features = [], [], {}
        full_tokens = x.shape[1]
        full_rope = rope
        entry = pooled_entry = None
        for index, block in enumerate(self.blocks):
            if compress and index == cfg['start_block']:
                entry = x
                pooled_entry = pool_tokens(x, cfg['factor'])
                x = pooled_entry
                rope = pooled_rope(h // self.patch_size, cfg['factor'],
                                   self.x_embedder.img_size[0] // self.patch_size,
                                   self.hidden_size // self.num_heads, x.device) if self.use_rope else None
            token_counts.append(x.shape[1])
            inputs = (x, c0, z, rope, self.ca_scale, force_dense, self.sparse_eval)
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x, stats = checkpoint(block, *inputs, use_reentrant=False, preserve_rng_state=True)
            else:
                x, stats = block(*inputs)
            statistics.append(stats)
            if compress and index + 1 == cfg['end_block']:
                x = restore_tokens(x, entry, pooled_entry, cfg['factor'])
                entry = pooled_entry = None
                rope = full_rope
            if index in feature_layers:
                # A canonical dense grid makes teacher/student alignment explicit.
                features[index] = (restore_tokens(x, entry, pooled_entry, cfg['factor'])
                                   if entry is not None else x)
        stats = torch.stack(statistics)
        counts = x.new_tensor(token_counts, dtype=torch.float32)
        result_stats = {
            'keep_probabilities': stats[:, 0],
            'keep_fractions': stats[:, 1],
            'keep_fraction': stats[:, 1].mean(),
            'attention_tokens': counts,
            'attention_token_ratio': counts.mean() / full_tokens,
            'attention_pair_ratio': counts.square().mean() / full_tokens**2,
        }
        if feature_layers:
            result_stats['features'] = features
        return self.unpatchify(self.final_layer(x, c)), result_stats

    def forward(self, x, t, r=None, z=None, *, return_stats=False,
                force_dense=False, include_dense=False, disable_compression=False, feature_layers=()):
        output, stats = self._forward_once(x, t, r, z, force_dense, disable_compression, feature_layers)
        if include_dense:
            # Both passes belong to one DDP forward; statistics have no mutable cache.
            dense, _ = self._forward_once(x, t, r, z, True)
            return output, stats, dense
        return (output, stats) if return_stats else output

    def forward_flexible(self, x, t, r=None, z=None):
        return self.forward(x, t, r, z)

    def export_state(self, strip_conditioning=False):
        if strip_conditioning and self.ca_scale != 0:
            raise ValueError('Finish the CA fade before stripping conditioning')
        config = dict(self.export_config)
        state = self.state_dict()
        if strip_conditioning:
            config.update(use_cross_attention=False, z_dims=None)
            state = {k: v for k, v in state.items() if not conditioning_key(k)}
        config['ca_scale'] = 0.0 if strip_conditioning else self.ca_scale
        config['use_checkpoint'] = False
        return config, {k: v.detach().cpu().contiguous() for k, v in state.items()}
