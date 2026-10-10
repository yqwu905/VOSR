"""attn_type full | sparse | local: the copied block-sparse attention inside VOSR2's self-attention."""
import json
import os
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import pytest
import torch
import torch.nn.functional as F
pytest.importorskip('timm', reason='Full backbone requires the upstream timm environment')
from models.lightningdit import LightningDiT
from models.lightningdit_ablation import AblationLightningDiT
from models.sparse_attention import build_sparse_attn, windowed_attention
from ablation_utils import load_backbone_state, load_export, save_export


def reference(q, k, v, attn_type):
    """Row-major reference: the same window masks built directly on the (side, side) grid."""
    b, h, n, d = q.shape
    side = int(n ** 0.5)
    rows, cols = torch.arange(n) // side, torch.arange(n) % side
    window = (rows // 8) * (side // 8) + cols // 8                       # window id of each token
    windows = (side // 8) ** 2
    if attn_type == 'local':
        wr, wc = torch.arange(windows) // (side // 8), torch.arange(windows) % (side // 8)
        keep = ((wr[:, None] - wr[None]).abs() <= 1) & ((wc[:, None] - wc[None]).abs() <= 1)
        keep = keep.expand(b, h, windows, windows)
    else:
        pool = lambda t: torch.zeros(b, h, windows, d).index_add_(2, window, t) / 64
        top = (pool(q) @ pool(k).transpose(-1, -2)).topk(8, dim=-1).indices
        keep = torch.zeros(b, h, windows, windows, dtype=torch.bool).scatter_(3, top, True)
    mask = keep[:, :, window][:, :, :, window]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@pytest.mark.parametrize('attn_type', ['sparse', 'local'])
@pytest.mark.parametrize('side', [32, 40])
def test_windowed_attention_matches_row_major_reference(attn_type, side):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, side * side, 16) for _ in range(3))
    out = windowed_attention(build_sparse_attn(attn_type, 16, 3), q, k, v)
    torch.testing.assert_close(out, reference(q, k, v, attn_type))
    assert not torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-3)


@pytest.mark.parametrize('side', [8, 16])
def test_eight_or_fewer_windows_is_full_attention(side):
    q, k, v = (torch.randn(1, 2, side * side, 16) for _ in range(3))
    full = F.scaled_dot_product_attention(q, k, v)
    for attn_type in ('sparse', 'local'):
        torch.testing.assert_close(windowed_attention(build_sparse_attn(attn_type, 16, 2), q, k, v), full)


def test_rejects_unknown_type_and_grids_without_8x8_windows():
    with pytest.raises(ValueError, match='attn_type must be one of'):
        build_sparse_attn('topk', 16, 2)
    q = torch.randn(1, 2, 12 * 12, 16)
    with pytest.raises(ValueError, match='sides divisible by 8'):
        windowed_attention(build_sparse_attn('local', 16, 2), q, q, q)


def kwargs(**extra):
    return dict(dict(input_size=64, patch_size=2, in_channels=4, out_channels=2, hidden_size=32, depth=2,
                     num_heads=2, mlp_ratio=2, z_dims=16, auxiliary_time_cond=True, use_rope=True,
                     use_qknorm=True, use_swiglu=True, use_rmsnorm=True), **extra)


def inputs():
    torch.manual_seed(1)
    return torch.randn(1, 4, 64, 64), torch.ones(1), torch.zeros(1), [torch.randn(1, 16, 16)]


def test_model_attn_type_and_override_share_weights():
    torch.manual_seed(0)
    full = LightningDiT(**kwargs()).eval()
    torch.nn.init.normal_(full.final_layer.linear.weight, std=.02)
    model = LightningDiT(**kwargs(attn_type='sparse')).eval()
    model.load_state_dict(full.state_dict())
    with torch.no_grad():
        dense = full(*inputs())
        assert not torch.allclose(model(*inputs()), dense, atol=1e-6)
        model.set_attn_type('local')
        assert all(b.attn.attn_type == 'local' for b in model.blocks)
        assert not torch.allclose(model(*inputs()), dense, atol=1e-6)
        model.set_attn_type('full')
        torch.testing.assert_close(model(*inputs()), dense)


def test_export_records_attn_type_and_inference_override(tmp_path):
    torch.manual_seed(0)
    teacher = AblationLightningDiT(**kwargs()).eval()
    torch.nn.init.normal_(teacher.final_layer.linear.weight, std=.02)
    student = AblationLightningDiT(**kwargs(), attn_type='local').eval()
    load_backbone_state(student, teacher.state_dict())
    save_export(student, tmp_path, {})
    assert json.loads((tmp_path / 'model.json').read_text())['attn_type'] == 'local'
    restored = load_export(tmp_path)
    assert all(b.attn.attn_type == 'local' for b in restored.blocks)
    with torch.no_grad():
        torch.testing.assert_close(restored(*inputs()), student(*inputs()))
        restored.set_attn_type('full')  # what inference_vosr_ablation.py --attn-type full does
        torch.testing.assert_close(restored(*inputs()), teacher(*inputs()))
    assert restored.export_config['attn_type'] == 'full'


def test_hourglass_student_runs_sparse_on_full_grid_and_full_when_merged():
    torch.manual_seed(0)
    student = AblationLightningDiT(**kwargs(depth=3), token_compression={'type': 'hourglass'},
                                   attn_type='sparse').eval()
    seen = []
    for block in student.blocks:
        block.attn.register_forward_hook(lambda m, i, o: seen.append(i[0].shape[1]))
    y = student(*inputs())
    assert seen == [1024, 256, 1024] and torch.isfinite(y).all()


def test_backward_through_sparse_attention():
    model = AblationLightningDiT(**kwargs(), attn_type='sparse', use_checkpoint=True)
    torch.nn.init.normal_(model.final_layer.linear.weight, std=.02)
    model(*inputs()).square().mean().backward()
    assert all(b.attn.qkv.weight.grad is not None and torch.isfinite(b.attn.qkv.weight.grad).all()
               for b in model.blocks)


def test_trainer_builds_student_attn_type_and_keeps_teacher_full(tmp_path):
    import yaml
    from test_ablation_hourglass import run_trainer, trainer_config
    student = {'use_cross_attention': True, 'router_config': None, 'attn_type': 'local'}
    path, _ = trainer_config(tmp_path, student)
    run_trainer(path)
    assert json.loads((tmp_path / 'run/export/model.json').read_text())['attn_type'] == 'local'
    cfg = yaml.safe_load(path.read_text())
    cfg['model']['attn_type'] = 'local'
    cfg['training']['output_dir'] = str(tmp_path / 'run_model_key')
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(AssertionError, match='Set student.attn_type'):
        run_trainer(path)
