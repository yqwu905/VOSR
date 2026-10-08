"""Opt-in VOSR2 ablations; the original LightningDiT is left unchanged."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .lightningdit import LightningDiT
from .sdt_router import AblationBlock, conditioning_key
from .token_hourglass import hourglass_spec, pool_tokens, rope_at, TokenMerge, TokenUnmerge


class AblationLightningDiT(LightningDiT):
    def __init__(self, *, use_cross_attention=True, router_config=None,
                 sparse_eval=True, ca_scale=None, token_compression=None,
                 coarse_depth=None, **kwargs):
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
        self.active_blocks = list(range(len(self.blocks)))
        self.token_compression, self.coarse_depth, self.max_coarse_depth = None, None, 0
        if token_compression is not None:
            self._build_hourglass(token_compression, router_config, coarse_depth)
        elif coarse_depth is not None:
            raise ValueError('coarse_depth requires token_compression')
        self.export_config = dict(kwargs, use_cross_attention=use_cross_attention,
                                  router_config=router_config, sparse_eval=sparse_eval,
                                  token_compression=self.token_compression)

    def _build_hourglass(self, config, router_config, coarse_depth):
        if router_config is not None:
            raise ValueError('SDT routing and token merging need a joint compute budget; enable only one')
        spec = hourglass_spec(config, len(self.blocks))
        if (self.x_embedder.img_size[0] // self.patch_size) % spec['factor']:
            raise ValueError('input_size / patch_size must be divisible by token_compression.factor')
        for index in spec['drop_blocks']:
            self.blocks[index] = nn.Identity()
        active = self.active_blocks = [i for i in range(len(self.blocks)) if i not in spec['drop_blocks']]
        fine = active[:spec['fine_in']] + active[len(active) - spec['fine_out']:]
        # Permanent full-grid blocks may drop CA; their pretrained CA weights are discarded on load.
        self.fine_blocks_without_ca = [] if spec['fine_cross_attention'] or not self.use_cross_attention else fine
        for index in self.fine_blocks_without_ca:
            del self.blocks[index].cross_attn
            self.blocks[index].z_dims = None
        self.token_merge = TokenMerge(self.hidden_size, spec['factor'])
        self.token_unmerge = TokenUnmerge(self.hidden_size, spec['factor'])
        self.token_compression = spec
        self.max_coarse_depth = len(active) - spec['fine_in'] - spec['fine_out']
        self.set_coarse_depth(self.max_coarse_depth if coarse_depth is None else coarse_depth)

    def set_coarse_depth(self, depth):
        """Number of merged blocks, centered between the fine entry and exit (merge curriculum)."""
        if self.token_compression is None or isinstance(depth, bool) or not isinstance(depth, int) \
                or not 0 <= depth <= self.max_coarse_depth:
            raise ValueError(f'coarse_depth must be an integer in [0, {self.max_coarse_depth}] '
                             'for a token-merging model')
        self.coarse_depth = depth

    @property
    def latent_multiple(self):
        """Latent sides must be multiples of patch_size times the merge factor."""
        return self.patch_size * (self.token_compression['factor'] if self.token_compression else 1)

    def is_removed_weight(self, key):
        """Pretrained weights this student deliberately lacks: dropped blocks and fine-block CA."""
        parts = key.split('.')
        if self.token_compression is None or len(parts) < 3 or parts[0] != 'blocks' or not parts[1].isdigit():
            return False
        index = int(parts[1])
        return index in self.token_compression['drop_blocks'] or \
            (index in self.fine_blocks_without_ca and parts[2] == 'cross_attn')

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
        for index in self.active_blocks:
            block = self.blocks[index]
            block.attn.fused_attn = enabled
            if hasattr(block, 'cross_attn'):
                block.cross_attn.fused_attn = enabled

    def _coarse_span(self):
        if not self.coarse_depth:
            return None
        start = self.token_compression['fine_in'] + (self.max_coarse_depth - self.coarse_depth) // 2
        return start, start + self.coarse_depth

    def _coarse_rope(self, grid, device):
        if not self.use_rope:
            return None
        factor = self.token_compression['factor']
        offset = (factor - 1) / 2 if self.token_compression['rope'] == 'centroid' else 0.
        # Same spacing as the full-grid RoPE; centroid = mean of the merged sub-token positions.
        scale = (self.x_embedder.img_size[0] // self.patch_size) / grid
        positions = (torch.arange(grid // factor, device=device, dtype=torch.float32) * factor + offset) * scale
        return rope_at(positions, self.hidden_size // self.num_heads // 2)

    def _forward_once(self, x, t, r, z, force_dense, feature_layers=None):
        _, _, h, w = x.shape
        if h != w or h % self.latent_multiple:
            raise ValueError('Use square latent crops divisible by patch_size (times the token merge '
                             'factor); tile non-square images')
        if feature_layers is not None and (len(set(feature_layers)) != len(feature_layers)
                                           or any(i not in self.active_blocks for i in feature_layers)):
            raise ValueError('feature_layers must be unique indices of blocks present in this model')
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
            z = z[0]
            if self.token_compression is not None:
                z = pool_tokens(z, self.token_compression['cond_pool'])
            z = self.mlp_ca(self.layer_norm(z))
        else:
            z = None
        rope = self.feat_rope
        if self.use_rope and h != self.x_embedder.img_size[0]:
            rope = self._get_dynamic_rope(h // self.patch_size, x.device, x.dtype)
        span = self._coarse_span()
        coarse_rope = self._coarse_rope(h // self.patch_size, x.device) if span else None
        wanted = {layer: i for i, layer in enumerate(feature_layers or ())}
        statistics, features, current = [], [None] * len(wanted), rope
        if span and span[0] == 0:
            fine, x, current = x, self.token_merge(x), coarse_rope
            merged = x
        for position, index in enumerate(self.active_blocks):
            inputs = (x, c0, z, current, self.ca_scale, force_dense, self.sparse_eval)
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x, stats = checkpoint(self.blocks[index], *inputs, use_reentrant=False, preserve_rng_state=True)
            else:
                x, stats = self.blocks[index](*inputs)
            statistics.append(stats)
            if span and position + 1 == span[1]:
                # Bypass: each full-grid token keeps its own detail and receives its group's update.
                if self.token_compression['bypass']:
                    x = fine + self.token_unmerge(x - merged)
                else:
                    x = self.token_unmerge(x)
                current = rope
            elif span and position + 1 == span[0]:
                fine, x, current = x, self.token_merge(x), coarse_rope
                merged = x
            if index in wanted:
                # The state handed to the next block, i.e. after any merge/unmerge at this boundary.
                features[wanted[index]] = x
        stats = torch.stack(statistics)
        result = {
            'keep_probabilities': stats[:, 0],
            'keep_fractions': stats[:, 1],
            'keep_fraction': stats[:, 1].mean(),
        }
        if feature_layers is not None:
            result['features'] = features
        return self.unpatchify(self.final_layer(x, c)), result

    def forward(self, x, t, r=None, z=None, *, return_stats=False,
                force_dense=False, include_dense=False, feature_layers=None):
        output, stats = self._forward_once(x, t, r, z, force_dense, feature_layers)
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
        if self.token_compression is not None:
            config['coarse_depth'] = self.coarse_depth
        config['use_checkpoint'] = False
        return config, {k: v.detach().cpu().contiguous() for k, v in state.items()}
