"""Full-backbone smoke tests; require the upstream timm/triton environment."""
import os
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import pytest
import torch
pytest.importorskip('timm', reason='Full backbone requires the upstream timm environment')
from models.lightningdit import LightningDiT
from models.lightningdit_ablation import AblationLightningDiT
from ablation_utils import load_backbone_state, save_export, load_export


def kwargs():
    return dict(input_size=8, patch_size=2, in_channels=4, out_channels=2,
                hidden_size=64, depth=2, num_heads=4, mlp_ratio=2,
                z_dims=16, auxiliary_time_cond=True, use_rope=True)


def test_dense_backbone_matches_original():
    base = LightningDiT(**kwargs()).eval()
    torch.nn.init.normal_(base.final_layer.linear.weight, std=.02)
    model = AblationLightningDiT(**kwargs()).eval()
    load_backbone_state(model, base.state_dict())
    x, t, r, z = torch.randn(2, 4, 8, 8), torch.ones(2), torch.zeros(2), [torch.randn(2, 3, 16)]
    with torch.no_grad():
        torch.testing.assert_close(base(x, t, r, z), model(x, t, r, z))


def test_stripped_export_roundtrip(tmp_path):
    model = AblationLightningDiT(**kwargs()).eval()
    torch.nn.init.normal_(model.final_layer.linear.weight, std=.02)
    model.set_ca_scale(0)
    save_export(model, tmp_path, {}, strip_conditioning=True)
    restored = load_export(tmp_path)
    assert not restored.use_cross_attention
    assert not any('cross_attn' in k or k.startswith(('mlp_ca.', 'layer_norm.')) for k in restored.state_dict())
    x, t, r = torch.randn(2, 4, 8, 8), torch.ones(2), torch.zeros(2)
    with torch.no_grad():
        torch.testing.assert_close(model(x, t, r), restored(x, t, r))


def test_routed_backbone_backward_checkpointed():
    model = AblationLightningDiT(**kwargs(), router_config={'init_keep_prob': .5}, use_checkpoint=True)
    torch.nn.init.normal_(model.final_layer.linear.weight, std=.02)
    y, stats = model(torch.randn(2, 4, 8, 8), torch.ones(2), torch.zeros(2),
                     [torch.randn(2, 3, 16)], return_stats=True)
    (y.square().mean() + stats['keep_probabilities'].mean()).backward()
    assert all(b.router.net[-1].weight.grad is not None for b in model.blocks)
