"""Opt-in VOSR2 ablations; the original LightningDiT is left unchanged."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .lightningdit import LightningDiT
from .sdt_router import AblationBlock, conditioning_key
from .token_hourglass import (depth_to_space_tokens, detail_positions, DetailStream, GroupRouter, hourglass_spec,
                             LQDetailScore, pool_tokens, rope_at, rope_tables, space_to_depth_tokens, table_rope,
                             TokenMerge, TokenUnmerge)


ROPE_MODES = ('interpolate', 'extrapolate')


class AblationLightningDiT(LightningDiT):
    def __init__(self, *, use_cross_attention=True, router_config=None,
                 sparse_eval=True, ca_scale=None, token_compression=None,
                 coarse_depth=None, rope_mode='interpolate', **kwargs):
        kwargs = dict(kwargs)
        if rope_mode not in ROPE_MODES:
            raise ValueError(f'rope_mode must be one of {ROPE_MODES}')
        # Token grids other than input_size / patch_size: interpolate squeezes the positions into the
        # trained range (upstream _get_dynamic_rope); extrapolate keeps integer positions, i.e. spacing 1.
        self.rope_mode = rope_mode
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
        self.group_router = self.detail_stream = None
        if token_compression is not None:
            self._build_hourglass(token_compression, router_config, coarse_depth)
        elif coarse_depth is not None:
            raise ValueError('coarse_depth requires token_compression')
        if self.group_router is not None and kwargs.get('attn_type', 'full') != 'full':
            raise ValueError('token_compression.routing mixes full-grid and merged tokens; use attn_type full')
        self.export_config = dict(kwargs, use_cross_attention=use_cross_attention,
                                  router_config=router_config, sparse_eval=sparse_eval,
                                  token_compression=self.token_compression, rope_mode=rope_mode)

    def _build_hourglass(self, config, router_config, coarse_depth):
        if router_config is not None:
            raise ValueError('SDT routing and token merging need a joint compute budget; enable only one')
        spec = hourglass_spec(config, len(self.blocks))
        if (self.x_embedder.img_size[0] // self.patch_size) % spec['factor'] ** (2 if spec['inner_depth'] else 1):
            raise ValueError('input_size / patch_size must be divisible by token_compression.factor '
                             '(its square with inner_depth)')
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
        if spec['inner_depth']:
            self.token_merge_inner = TokenMerge(self.hidden_size, spec['factor'])
            self.token_unmerge_inner = TokenUnmerge(self.hidden_size, spec['factor'])
        self.token_compression = spec
        self.max_coarse_depth = len(active) - spec['fine_in'] - spec['fine_out']
        self.set_coarse_depth(self.max_coarse_depth if coarse_depth is None else coarse_depth)
        self.fine_blocks = list(spec['fine_blocks'])
        if spec.get('routing'):
            # Mixed resolution: the router picks which groups stay on the full grid in the merged span.
            self.group_router = (LQDetailScore(self.patch_size, spec['factor'])
                                 if spec['routing'].get('score') == 'lq_detail'
                                 else GroupRouter(self.hidden_size, spec['factor']))
            self.keep_ratio = self.train_keep_ratio = spec['routing']['eval_keep_ratio']
        if spec.get('detail'):
            # Dual resolution: a narrow full-grid stream next to the merged span (no merge curriculum).
            if self.coarse_depth != self.max_coarse_depth:
                raise ValueError('token_compression.detail needs the full merged span (no merge curriculum)')
            self.detail_stream = DetailStream(self.hidden_size, spec['factor'], **spec['detail'])
            merged = [p for p in range(spec['fine_in'], spec['fine_in'] + self.max_coarse_depth)
                      if active[p] not in spec['fine_blocks']]
            self.detail_after = detail_positions(merged, spec['detail']['depth'])

    def set_keep_ratio(self, ratio):
        """Share of groups kept on the full grid in eval mode (previews, exports, inference)."""
        self.keep_ratio = self._check_keep_ratio(ratio)

    def set_train_keep_ratio(self, ratio):
        """Share of groups kept on the full grid in training mode; the trainer samples it every step."""
        self.train_keep_ratio = self._check_keep_ratio(ratio)

    def _check_keep_ratio(self, ratio):
        if self.group_router is None:
            raise ValueError('keep ratios need token_compression.routing')
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
            raise ValueError('keep ratio must be in [0, 1]')
        return float(ratio)

    @property
    def current_keep_ratio(self):
        return self.train_keep_ratio if self.training else self.keep_ratio

    def set_coarse_depth(self, depth):
        """Number of merged blocks, centered between the fine entry and exit (merge curriculum).

        With inner_depth, the last inner_depth blocks of growth go to the inner level: it holds
        max(0, inner_depth - (max_coarse_depth - depth)) blocks, centered as well.
        """
        if self.token_compression is None or isinstance(depth, bool) or not isinstance(depth, int) \
                or not 0 <= depth <= self.max_coarse_depth:
            raise ValueError(f'coarse_depth must be an integer in [0, {self.max_coarse_depth}] '
                             'for a token-merging model')
        if self.detail_stream is not None and depth != self.max_coarse_depth:
            raise ValueError('token_compression.detail needs the full merged span (no merge curriculum)')
        self.coarse_depth = depth

    @property
    def latent_multiple(self):
        """Latent sides must be multiples of patch_size times the merge factor (squared with inner_depth)."""
        if not self.token_compression:
            return self.patch_size
        return self.patch_size * self.token_compression['factor'] ** (2 if self.token_compression['inner_depth'] else 1)

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

    def set_attn_type(self, attn_type):
        """Self-attention of every kept block; also recorded for export (inference override)."""
        if self.group_router is not None and attn_type != 'full':
            raise ValueError('token_compression.routing mixes full-grid and merged tokens; use attn_type full')
        for index in self.active_blocks:
            self.blocks[index].attn.set_attn_type(attn_type)
        self.export_config['attn_type'] = attn_type

    def set_rope_mode(self, rope_mode):
        """RoPE positions on non-reference grids; also recorded for export (inference override)."""
        if rope_mode not in ROPE_MODES:
            raise ValueError(f'rope_mode must be one of {ROPE_MODES}')
        self.rope_mode = rope_mode
        self.export_config['rope_mode'] = rope_mode

    def _set_fused_attn(self, enabled):
        for index in self.active_blocks:
            block = self.blocks[index]
            block.attn.fused_attn = enabled
            if hasattr(block, 'cross_attn'):
                block.cross_attn.fused_attn = enabled

    def _merge_spans(self):
        """Active-block positions of the merged span and, nested in it, the inner span."""
        if not self.coarse_depth or (self.group_router is not None and self.current_keep_ratio >= 1):
            return []  # routing with every group kept is the dense model, exactly
        inner = max(0, self.token_compression['inner_depth'] - (self.max_coarse_depth - self.coarse_depth))
        spans = []
        for depth in (self.coarse_depth, inner):
            if depth:
                start = self.token_compression['fine_in'] + (self.max_coarse_depth - depth) // 2
                spans.append((start, start + depth))
        return spans

    def _change_level(self, x, stack, level, tables=None, scores=None):
        """Merge or unmerge until the tokens are at merge level ``level`` (0 = full grid)."""
        while len(stack) < level:
            if self.group_router is not None:  # routing has a single merge level
                x, state = self._route_merge(x, tables, scores)
                stack.append(state)
                continue
            merged = (self.token_merge_inner if stack else self.token_merge)(x)
            # The detail stream (single merge level) starts from the full-grid state; its slot is updated in place.
            stack.append((x, merged) if self.detail_stream is None else [x, merged, self.detail_stream.enter(x)])
            x = merged
        while len(stack) > level:
            if isinstance(stack[-1], dict):
                x = self._route_unmerge(x, stack.pop())
                continue
            fine, merged, *detail = stack.pop()
            unmerge = self.token_unmerge_inner if stack else self.token_unmerge
            # Bypass: each finer token keeps its own detail and receives its group's update.
            x = fine + unmerge(x - merged) if self.token_compression['bypass'] else unmerge(x)
            if detail:
                x = x + self.detail_stream.exit(detail[0])
        return x

    def _route_merge(self, x, tables, scores=None):
        """Mixed-resolution merge: the top-scored groups keep their tokens, the others are merged.

        Returns the sequence [fine tokens of the kept groups, one merged token per other group] and the
        state ``_route_unmerge`` needs. RoPE: kept tokens at their own grid positions, merged tokens as
        ``_coarse_rope``. In training the group scores get gradients through a straight-through gate
        that is exactly 1 in the forward pass: +sigmoid on the kept groups' updates, -sigmoid on the
        merged groups' updates. Precomputed ``scores`` (``routing.score: lq_detail``) carry no gradient.
        """
        b, n, d = x.shape
        factor = self.token_compression['factor']
        groups = space_to_depth_tokens(x, factor)  # (B, G, factor^2 D), sub-position major
        count = groups.shape[1]
        kept = min(count, int(round(self.current_keep_ratio * count)))
        if scores is None:
            scores = self.group_router(groups)  # (B, G), FP32
        order = torch.argsort(scores.detach(), dim=1, descending=True, stable=True)
        index = lambda idx, width: idx.unsqueeze(-1).expand(-1, -1, width)
        state = dict(groups=groups, fine_idx=order[:, :kept], coarse_idx=order[:, kept:], rope=None, gate=None)
        fine = groups.gather(1, index(state['fine_idx'], groups.shape[-1])).reshape(b, kept * factor * factor, d)
        coarse = self.token_merge.proj(groups.gather(1, index(state['coarse_idx'], groups.shape[-1])))
        state.update(fine=fine, coarse=coarse)
        if self.training and torch.is_grad_enabled():
            probability = scores.sigmoid()
            state['gate'] = probability - probability.detach()
        if tables is not None:
            (fine_cos, fine_sin), (coarse_cos, coarse_sin) = tables
            # Full-grid token index of every (group, sub-position), in the space_to_depth layout.
            tokens = space_to_depth_tokens(torch.arange(n, device=x.device).view(1, n, 1), factor)[0]
            fine_tokens = tokens[state['fine_idx']].reshape(b, -1)
            pick = lambda table, idx: table[idx].unsqueeze(1)  # (B, 1, L, head_dim) for (B, heads, L, head_dim)
            state['rope'] = table_rope(
                torch.cat((pick(fine_cos, fine_tokens), pick(coarse_cos, state['coarse_idx'])), 2),
                torch.cat((pick(fine_sin, fine_tokens), pick(coarse_sin, state['coarse_idx'])), 2))
        return torch.cat((fine, coarse), 1), state

    def _route_unmerge(self, x, state):
        """Back to the full grid: kept groups take their tokens' outputs, merged groups the hourglass unmerge."""
        groups, fine_idx, coarse_idx = state['groups'], state['fine_idx'], state['coarse_idx']
        b, count, width = groups.shape
        fine_out, coarse_out = x[:, :state['fine'].shape[1]], x[:, state['fine'].shape[1]:]
        coarse_groups = groups.gather(1, coarse_idx.unsqueeze(-1).expand(-1, -1, width))
        fine_delta = (fine_out - state['fine']).reshape(b, -1, width)
        if self.token_compression['bypass']:
            coarse_delta = self.token_unmerge.proj(coarse_out - state['coarse'])
        else:
            coarse_delta = self.token_unmerge.proj(coarse_out) - coarse_groups
        if state['gate'] is not None:
            gate = state['gate'].to(fine_delta.dtype).unsqueeze(-1)
            fine_delta = fine_delta * (1 + gate.gather(1, fine_idx.unsqueeze(-1)))
            coarse_delta = coarse_delta * (1 - gate.gather(1, coarse_idx.unsqueeze(-1)))
        fine_groups = groups.gather(1, fine_idx.unsqueeze(-1).expand(-1, -1, width))
        out = torch.zeros_like(groups)
        out = out.scatter(1, fine_idx.unsqueeze(-1).expand(-1, -1, width), fine_groups + fine_delta)
        out = out.scatter(1, coarse_idx.unsqueeze(-1).expand(-1, -1, width), coarse_groups + coarse_delta)
        return depth_to_space_tokens(out, self.token_compression['factor'])

    def _rope_tables(self, grid, device, dtype, factor=1):
        """cos/sin rows of the RoPE that the dense blocks (factor 1) or ``_coarse_rope`` use on this grid."""
        half = self.hidden_size // self.num_heads // 2
        reference = self.x_embedder.img_size[0] // self.patch_size
        if factor == 1 and grid == reference:
            return self.feat_rope.freqs_cos, self.feat_rope.freqs_sin
        if factor == 1 and self.rope_mode == 'interpolate':  # as _get_dynamic_rope, including its dtype
            cos, sin = rope_tables(torch.arange(grid, device=device).float() / grid * reference, half)
            return cos.to(dtype), sin.to(dtype)
        offset = (factor - 1) / 2 if factor > 1 and self.token_compression['rope'] == 'centroid' else 0.
        scale = reference / grid if self.rope_mode == 'interpolate' else 1.
        return rope_tables((torch.arange(grid // factor, device=device, dtype=torch.float32) * factor + offset) * scale,
                           half)

    def _coarse_rope(self, grid, device, factor=None):
        if not self.use_rope:
            return None
        factor = factor or self.token_compression['factor']
        offset = (factor - 1) / 2 if self.token_compression['rope'] == 'centroid' else 0.
        # Same spacing as the full-grid RoPE; centroid = mean of the merged sub-token positions.
        scale = (self.x_embedder.img_size[0] // self.patch_size) / grid if self.rope_mode == 'interpolate' else 1.
        positions = (torch.arange(grid // factor, device=device, dtype=torch.float32) * factor + offset) * scale
        return rope_at(positions, self.hidden_size // self.num_heads // 2)

    def _forward_once(self, x, t, r, z, force_dense, feature_layers=None):
        _, _, h, w = x.shape
        if h != w or h % self.latent_multiple:
            raise ValueError('Use square latent crops divisible by patch_size (times the token merge factor, '
                             'squared with inner_depth); tile non-square images')
        if feature_layers is not None and (len(set(feature_layers)) != len(feature_layers)
                                           or any(i not in self.active_blocks for i in feature_layers)):
            raise ValueError('feature_layers must be unique indices of blocks present in this model')
        if feature_layers and self.group_router is not None:
            raise ValueError('Hidden-state distillation is not supported with token_compression.routing')
        latent = x
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
            rope = (self._get_dynamic_rope(h // self.patch_size, x.device, x.dtype) if self.rope_mode == 'interpolate'
                    else rope_at(torch.arange(h // self.patch_size, device=x.device), self.hidden_size // self.num_heads // 2))
        spans = self._merge_spans()
        tables = scores = None
        if self.group_router is not None and spans and self.use_rope:
            grid = h // self.patch_size
            tables = (self._rope_tables(grid, x.device, x.dtype),
                      self._rope_tables(grid, x.device, x.dtype, self.token_compression['factor']))
        if self.group_router is not None and spans and self.group_router.uses_lq:
            scores = self.group_router(latent[:, :latent.shape[1] // 2])  # the LQ half of cat(lq_latent, noise)
        # Per merge level (0 = full grid): RoPE positions and DINO tokens.
        ropes = [rope] + [self._coarse_rope(h // self.patch_size, x.device, self.token_compression['factor'] ** level)
                          for level in range(1, len(spans) + 1)]
        # coarse_cond_pool: merged levels attend to DINO tokens pooled once more (after mlp_ca); full-grid blocks keep all.
        coarse_z = z if z is None else pool_tokens(z, (self.token_compression or {}).get('coarse_cond_pool', 1))
        conds = (z, coarse_z, None if z is None or len(spans) < 2
                 else pool_tokens(coarse_z, self.token_compression['inner_cond_pool']))
        wanted = {layer: i for i, layer in enumerate(feature_layers or ())}
        statistics, features, stack = [], [None] * len(wanted), []
        fine = {self.active_blocks.index(i) for i in (self.fine_blocks if spans else ())}
        level_at = lambda position: 0 if position in fine else sum(start <= position < end for start, end in spans)
        x = self._change_level(x, stack, level_at(0), tables, scores)
        for position, index in enumerate(self.active_blocks):
            # A routed span carries its own RoPE: kept tokens at their grid positions, merged ones at centroids.
            rope_now = stack[-1]['rope'] if stack and isinstance(stack[-1], dict) else ropes[len(stack)]
            inputs = (x, c0, conds[len(stack)], rope_now, self.ca_scale, force_dense, self.sparse_eval)
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x, stats = checkpoint(self.blocks[index], *inputs, use_reentrant=False, preserve_rng_state=True)
            else:
                x, stats = self.blocks[index](*inputs)
            statistics.append(stats)
            if self.detail_stream is not None and position in self.detail_after:
                # Detail step after this merged block: read the merged tokens, mix on the full grid, write back.
                entry = stack[-1]
                entry[2], x = self.detail_stream.blocks[self.detail_after[position]](entry[2], x)
            x = self._change_level(x, stack, level_at(position + 1), tables, scores)
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
