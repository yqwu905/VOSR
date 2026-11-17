"""Opt-in VOSR2 ablations; the original LightningDiT is left unchanged."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .attn_compression import (attn_compression_spec, head_macs, HeadSearchAttention, HeadSplitAttention,
                               pooled_rope, search_price, search_spec)
from .lightningdit import LightningDiT
from .pos_embed import rope_at
from .mixed_resolution_ffn import MixedResolutionSwiGLU, mixed_resolution_spec, split_mlp_state
from .sdt_router import AblationBlock, conditioning_key
from .swiglu_ffn import SwiGLUFFN


ROPE_MODES = ('interpolate', 'extrapolate')


class AblationLightningDiT(LightningDiT):
    def __init__(self, *, use_cross_attention=True, router_config=None,
                 sparse_eval=True, ca_scale=None, rope_mode='interpolate',
                 attn_compression=None, mlp_compression=None,
                 **kwargs):
        kwargs = dict(kwargs)
        # Older supported exports may include null fields for retired token merging.
        for key in ('token_compression', 'coarse_depth'):
            if kwargs.pop(key, None) is not None:
                raise ValueError(f'{key} is no longer supported')
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
        self.attn_compression = None
        if attn_compression is not None:
            self._build_head_split(attn_compression)
        self.mlp_compression = None
        if mlp_compression is not None:
            self._build_mlp_compression(mlp_compression, router_config)
        self.export_config = dict(kwargs, use_cross_attention=use_cross_attention,
                                  router_config=router_config, sparse_eval=sparse_eval,
                                  rope_mode=rope_mode)
        if self.attn_compression is not None:  # absent keeps the model.json of other exports unchanged
            self.export_config['attn_compression'] = self.attn_compression
        if self.mlp_compression is not None:
            self.export_config['mlp_compression'] = self.mlp_compression

    def _build_head_split(self, config):
        """Head-split self-attention (models/attn_compression.py) in the chosen blocks; CA unchanged, the MLP
        unchanged or, with router_config, routed per token (its budget counts MLP tokens only)."""
        spec, kernels = attn_compression_spec(config, len(self.blocks), self.num_heads)
        if (self.x_embedder.img_size[0] // self.patch_size) % spec['factor']:
            raise ValueError('input_size / patch_size must be divisible by attn_compression.factor')
        search, none = spec.get('search'), [[] for _ in self.blocks]
        for index in spec['blocks']:
            block = self.blocks[index]
            local = spec['local_heads'][index]
            if search is not None:
                block.attn = HeadSearchAttention(block.attn, local, spec['factor'], spec['kernel'], spec['upsample'],
                                                 search['candidates'], self._search_costs(spec, local))
            else:
                block.attn = HeadSplitAttention(block.attn, local, spec['factor'], spec['kernel'], spec['upsample'],
                                                spec.get('kv_pool_heads', none)[index],
                                                spec.get('dense_heads', none)[index])
            if kernels is not None and index in kernels:
                block.attn.set_local_kernels(kernels[index])
        self.attn_compression = spec
        if search is not None:
            self._check_search_budget(search['attn_budget'])

    def _search_costs(self, spec, local_heads):
        """(heads, 1 + candidates) MACs of [base, *candidates] per head, in dense heads at the reference grid."""
        tokens, head_dim = (self.x_embedder.img_size[0] // self.patch_size) ** 2, self.hidden_size // self.num_heads
        unit = lambda kind: (head_macs(kind, tokens, self.hidden_size, head_dim, spec['factor'], spec['kernel'])
                             / head_macs('dense', tokens, self.hidden_size, head_dim, spec['factor'], spec['kernel']))
        return torch.tensor([[unit('local' if h in local_heads else 'pooled')]
                             + [unit(c) for c in spec['search']['candidates']] for h in range(self.num_heads)],
                            device='cpu')

    @property
    def head_search(self):
        """The resolved ``attn_compression.search``, or None."""
        return (self.attn_compression or {}).get('search')

    def _search_modules(self):
        return [self.blocks[index].attn for index in self.attn_compression['blocks']]

    def _search_target(self, budget=None):
        """Budget of the searched heads in dense heads: the blocks outside attn_compression.blocks stay dense."""
        budget = self.head_search['attn_budget'] if budget is None else budget
        total, fixed = len(self.blocks) * self.num_heads, (len(self.blocks) - len(self.attn_compression['blocks']))
        return budget * total - fixed * self.num_heads

    def _check_search_budget(self, budget):
        cheapest = sum(module.costs.min(1).values.sum().item() for module in self._search_modules())
        if cheapest >= self._search_target(budget):
            fixed = len(self.blocks) - len(self.attn_compression['blocks'])
            base = (cheapest + fixed * self.num_heads) / (len(self.blocks) * self.num_heads)
            raise ValueError(f'attn_compression.search.attn_budget {budget} must exceed the head-split base '
                             f'structure ({base:.4f} of the dense self-attention)')

    def set_search_budget(self, budget):
        """Another self-attention budget for the learned head logits (tools/head_search_select.py)."""
        if self.head_search is None:
            raise ValueError('The model has no attn_compression.search')
        search = search_spec(dict(self.head_search, attn_budget=budget))
        self._check_search_budget(search['attn_budget'])
        self.attn_compression['search'] = search

    def set_search_temperature(self, temperature):
        """Gumbel-softmax temperature of the straight-through gradients (the trainer anneals it)."""
        for module in self._search_modules():
            module.temperature = float(temperature)

    def _search_tables(self):
        modules = self._search_modules()
        logits = torch.cat([module.option_logits().detach() for module in modules])
        return modules, logits, torch.cat([module.costs for module in modules])

    def _set_search_price(self):
        """Training: the price at which the expected cost meets the budget; eval: the price of the fixed choice."""
        modules, logits, costs = self._search_tables()
        price = search_price(logits, costs, self._search_target(), hard=not self.training)
        for module in modules:
            module.price = price

    def head_search_report(self, probabilities=False):
        """Fixed choice at the budget (the price at which the argmax options fit), its self-attention MACs and the
        expected MACs of training's sampling, as shares of the dense self-attention; optionally per block the
        option probabilities and the chosen option (0 = base) of every head."""
        modules, logits, costs = self._search_tables()
        logits, costs = logits.double().cpu(), costs.double().cpu()
        target, total = self._search_target(), len(self.blocks) * self.num_heads
        fixed = total - len(modules) * self.num_heads
        expected_price, price = search_price(logits, costs, target), search_price(logits, costs, target, hard=True)
        expected = (logits - expected_price * costs).softmax(1)
        chosen = (logits - price * costs).argmax(1)
        report = {'attn_budget': self.head_search['attn_budget'], 'price': price, 'expected_price': expected_price,
                  'expected_attn_ratio': ((expected * costs).sum().item() + fixed) / total,
                  'selected_attn_ratio': (costs.gather(1, chosen[:, None]).sum().item() + fixed) / total}
        for option, candidate in enumerate(self.head_search['candidates'], 1):
            report[f'{candidate}_heads'] = int((chosen == option).sum())
        if probabilities:
            report.update(options=['base'] + self.head_search['candidates'], blocks=self.attn_compression['blocks'],
                          probabilities=expected.view(len(modules), self.num_heads, -1).tolist(),
                          choice=chosen.view(len(modules), self.num_heads).tolist())
        return report

    def selected_attn_compression(self):
        """The fixed head-split structure of the current head choice (no search): local, kv_pool, dense lists."""
        modules, logits, costs = self._search_tables()
        price = search_price(logits, costs, self._search_target(), hard=True)
        chosen = (logits.double().cpu() - price * costs.double().cpu()).argmax(1).view(len(modules), self.num_heads)
        lists = {key: [[] for _ in self.blocks] for key in ('local_heads', 'kv_pool_heads', 'dense_heads')}
        for index, module, row in zip(self.attn_compression['blocks'], modules, chosen.tolist()):
            for head, option in enumerate(row):  # option 0 keeps the base: local or (unlisted) pooled
                if option:
                    lists[f'{module.candidates[option - 1]}_heads'][index].append(head)
                elif head in module.local_heads:
                    lists['local_heads'][index].append(head)
        config = {key: value for key, value in self.attn_compression.items() if key != 'search'}
        return attn_compression_spec(dict(config, **lists), len(self.blocks), self.num_heads)[0]

    def _build_mlp_compression(self, config, router_config):
        """Split SwiGLU hidden units between full and pooled token grids."""
        if router_config is not None:
            raise ValueError('mlp_compression cannot be combined with router_config')
        if not all(isinstance(block.mlp, SwiGLUFFN) for block in self.blocks):
            raise ValueError('mlp_compression needs SwiGLU MLPs; set use_swiglu: true')
        self.mlp_full_hidden = self.blocks[0].mlp.w3.in_features
        spec, self.mlp_split_init = mixed_resolution_spec(config, self.mlp_full_hidden)
        if spec['coarse_hidden'] and (self.x_embedder.img_size[0] // self.patch_size) % spec['factor']:
            raise ValueError('input_size / patch_size must be divisible by mixed_resolution.factor')
        for block in self.blocks:
            block.mlp = MixedResolutionSwiGLU(self.hidden_size, spec['full_hidden'], spec['coarse_hidden'],
                                             spec['factor'], spec['upsample'])
        self.mlp_compression = spec

    def prepare_pretrained_state(self, state):
        """Convert a full-width source to spatial branches; converted checkpoints pass through."""
        if self.mlp_compression is None:
            return state
        return split_mlp_state(state, self.mlp_compression, self.mlp_split_init,
                               len(self.blocks), self.mlp_full_hidden)

    @property
    def latent_multiple(self):
        """Latent sides must align with patches and both attention and FFN pooling factors."""
        factor = 1
        if self.mlp_compression is not None and self.mlp_compression['coarse_hidden']:
            factor = self.mlp_compression['factor']
        if self.attn_compression is not None:
            return self.patch_size * math.lcm(factor, self.attn_compression['factor'])
        return self.patch_size * factor

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

    def _pooled_rope(self, grid, device, dtype):
        """RoPE of the head-split blocks' pooled grid: the centroids of the factor x factor groups, spaced like
        the full-grid RoPE (``rope_mode``). Tables in ``dtype``, the residual dtype at the block."""
        factor = self.attn_compression['factor']
        scale = (self.x_embedder.img_size[0] // self.patch_size) / grid if self.rope_mode == 'interpolate' else 1.
        positions = (torch.arange(grid // factor, device=device, dtype=torch.float32) * factor + (factor - 1) / 2) * scale
        return pooled_rope(positions, self.hidden_size // self.num_heads // 2, dtype)

    def _forward_once(self, x, t, r, z, force_dense, feature_layers=None, mlp_update_layers=None):
        _, _, h, w = x.shape
        if h != w or h % self.latent_multiple:
            raise ValueError('Use square latent crops divisible by patch_size times the attention/FFN '
                             'pooling factors; tile non-square images')
        if feature_layers is not None and (len(set(feature_layers)) != len(feature_layers)
                                           or any(i not in self.active_blocks for i in feature_layers)):
            raise ValueError('feature_layers must be unique indices of blocks present in this model')
        if mlp_update_layers is not None and (len(set(mlp_update_layers)) != len(mlp_update_layers)
                                              or any(i not in self.active_blocks for i in mlp_update_layers)):
            raise ValueError('mlp_update_layers must be unique indices of blocks present in this model')
        model_input, (x, encoded) = x, self._embed(x)
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
            z = self.mlp_ca(self.layer_norm(z))
        else:
            z = None
        rope = self.feat_rope
        if self.use_rope and h != self.x_embedder.img_size[0]:
            rope = (self._get_dynamic_rope(h // self.patch_size, x.device, x.dtype) if self.rope_mode == 'interpolate'
                    else rope_at(torch.arange(h // self.patch_size, device=x.device), self.hidden_size // self.num_heads // 2))
        wanted = {layer: i for i, layer in enumerate(feature_layers or ())}
        update_wanted = {layer: i for i, layer in enumerate(mlp_update_layers or ())}
        updates = [None] * len(update_wanted)
        statistics, features = [], [None] * len(wanted)
        split = set(self.attn_compression['blocks']) if self.attn_compression is not None else set()
        if self.head_search is not None:
            self._set_search_price()
        pooled_ropes = {}  # per residual dtype: the first block may still see the embedding's autocast dtype
        for index in self.active_blocks:
            rope_now = rope
            if index in split and self.use_rope:  # head-split blocks attend on the pooled grid
                if x.dtype not in pooled_ropes:
                    pooled_ropes[x.dtype] = self._pooled_rope(h // self.patch_size, x.device, x.dtype)
                # Full-grid queries (dense, key/value-pooled and searched heads) also need the full-grid RoPE.
                rope_now = ((pooled_ropes[x.dtype], rope_now) if self.blocks[index].attn.needs_full_rope
                            else pooled_ropes[x.dtype])
            inputs = (x, c0, z, rope_now, self.ca_scale, force_dense, self.sparse_eval)
            if index in update_wanted:
                inputs = (*inputs, True)
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                block_result = checkpoint(self.blocks[index], *inputs, use_reentrant=False, preserve_rng_state=True)
            else:
                block_result = self.blocks[index](*inputs)
            x, stats = block_result[:2]
            if index in update_wanted:
                updates[update_wanted[index]] = block_result[2]
            statistics.append(stats)
            if index in wanted:
                # The hidden state handed to the next block.
                features[wanted[index]] = x
        stats = torch.stack(statistics)
        result = {
            'keep_probabilities': stats[:, 0],
            'keep_fractions': stats[:, 1],
            'keep_fraction': stats[:, 1].mean(),
        }
        if feature_layers is not None:
            result['features'] = features
        if mlp_update_layers is not None:
            result['mlp_updates'] = updates
        return self._head(x, c, t, model_input, encoded), result

    def _embed(self, x):
        """Model input -> tokens, and what ``_head`` needs besides them (models/pixel_dit.py)."""
        return self.x_embedder(x), None

    def _head(self, x, c, t, inputs, encoded):
        """Last hidden state -> model output; the pixel student swaps in its decoder."""
        return self.unpatchify(self.final_layer(x, c))

    def forward(self, x, t, r=None, z=None, *, return_stats=False,
                force_dense=False, include_dense=False, feature_layers=None, mlp_update_layers=None):
        output, stats = self._forward_once(x, t, r, z, force_dense, feature_layers, mlp_update_layers)
        if include_dense:
            # Both passes belong to one DDP forward; statistics have no mutable cache.
            dense, _ = self._forward_once(x, t, r, z, True)
            return output, stats, dense
        return (output, stats) if return_stats else output

    def forward_flexible(self, x, t, r=None, z=None):
        return self.forward(x, t, r, z)

    def export_state(self, strip_conditioning=False, selected_heads=False):
        """Config and weights, optionally stripping conditioning or fixing searched heads."""
        if strip_conditioning and self.ca_scale != 0:
            raise ValueError('Finish the CA fade before stripping conditioning')
        config = dict(self.export_config)
        state = self.state_dict()
        if selected_heads:
            if self.head_search is None:
                raise ValueError('selected_heads needs attn_compression.search')
            config['attn_compression'] = self.selected_attn_compression()
            state = {k: v for k, v in state.items() if not k.endswith('.attn.head_logits')}
            hd = self.hidden_size // self.num_heads
            for index in self.attn_compression['blocks']:
                key, local = f'blocks.{index}.attn.local_conv.weight', self.blocks[index].attn.local_heads
                kept = [i for i, head in enumerate(local) if head in config['attn_compression']['local_heads'][index]]
                if kept and key in state:
                    state[key] = state[key][[i * hd + c for i in kept for c in range(hd)]]
                else:
                    state.pop(key, None)
        if strip_conditioning:
            config.update(use_cross_attention=False, z_dims=None)
            state = {k: v for k, v in state.items() if not conditioning_key(k)}
        config['ca_scale'] = 0.0 if strip_conditioning else self.ca_scale
        config['use_checkpoint'] = False
        return config, {k: v.detach().cpu().contiguous() for k, v in state.items()}
