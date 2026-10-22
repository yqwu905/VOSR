"""CPU-only numerical, intervention, budget, migration and experiment-entry contracts."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import numpy as np
import pytest
import torch
from torch import nn

from ablation_utils import load_backbone_state, load_config, load_export, save_export
from models.attn_compression import HeadSplitAttention, attn_compression_spec, pooled_rope
from models.head_sensitivity import (assign_heads, attention_macs, attention_values, conv_values,
    head_modes, intervention, measure_heads, reconfigure, weighted_mse)
from models.lightningdit import Attention
from models.lightningdit_ablation import AblationLightningDiT
from tools.attn_head_sensitivity import (aggregate, make_groups, profile, ablate, roi_masks, signature,
                                        sha256, translated_boxes, write_json)
from tools.head_ocr_compare import compare
from tools.plan_head_budget import protect, random_control, reclassify

ROOT = Path(__file__).resolve().parents[1]


def attn(bias=True):
    torch.manual_seed(7)
    return Attention(32, 2, qkv_bias=bias, qk_norm=True, use_rmsnorm=True).eval()


def rope(side, factor=1):
    return pooled_rope(torch.arange(side//factor).float()*factor+(factor-1)/2, 8, torch.float32)


def model():
    torch.manual_seed(3)
    net = AblationLightningDiT(input_size=8, patch_size=1, in_channels=4, out_channels=2,
            hidden_size=32, depth=2, num_heads=2, mlp_ratio=2, z_dims=8,
            auxiliary_time_cond=True, use_rope=True, use_qknorm=True, use_rmsnorm=True).eval()
    nn.init.normal_(net.final_layer.linear.weight, std=.05)  # default zero would hide perturbations
    return net


def inputs():
    torch.manual_seed(5)
    return torch.randn(1, 4, 8, 8), torch.ones(1), torch.zeros(1), [torch.randn(1, 4, 8)]


@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('dense,local', [([0], []), ([0], [1]), ([0, 1], [])])
def test_mixed_heads_match_independent_contributions_and_bias_once(bias, dense, local):
    base = attn(bias)
    x = torch.randn(2, 64, 32)
    split = HeadSplitAttention(base, local, 2, 3, 'bilinear', dense).eval()
    kernels = torch.zeros(2, 3, 3)
    kernels[:, 1, 1] = 1
    expected = base.proj.bias.clone()[None, None].expand(2, 64, -1)
    for h in range(2):
        if h in local:
            value = conv_values(base, x, [h], kernels)[:, 0]
        else:
            f = 1 if h in dense else 2
            value = attention_values(base, x, [h], rope(8, f), f)[:, 0]
        expected = expected + value @ base.proj.weight[:, h*16:(h+1)*16].T
    with torch.no_grad():
        actual = split(x, (rope(8, 2), rope(8)))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_dense_head_validation_and_legacy_schema():
    legacy = dict(type='head_split', local_heads=[[], []])
    assert 'dense_heads' not in attn_compression_spec(legacy, 2, 2)[0]
    for dense in ([[2], []], [[0, 0], []], [[True], []], [[0]]):
        with pytest.raises(ValueError):
            attn_compression_spec(dict(legacy, dense_heads=dense), 2, 2)
    with pytest.raises(ValueError, match='overlap'):
        attn_compression_spec(dict(legacy, local_heads=[[0], []], dense_heads=[[0], []]), 2, 2)
    with pytest.raises(ValueError, match='full_rope'):
        HeadSplitAttention(attn(), [], 2, 3, 'bilinear', [0])(torch.randn(1,64,32), rope(8,2))


def test_intervention_restores_state_and_matches_production_export(tmp_path):
    net = model()
    original = [b.attn for b in net.blocks]
    with torch.no_grad():
        dense = net(*inputs())
        with intervention(net, [(0, 1, 'pool')]):
            changed = net(*inputs())
            assert net.blocks[1].attn is original[1]
            assert net.blocks[0].attn.dense_heads == [0]
            assert not torch.equal(dense, changed)
        torch.testing.assert_close(net(*inputs()), dense, atol=0, rtol=0)
    assert net.attn_compression is None and 'attn_compression' not in net.export_config
    with pytest.raises(RuntimeError):
        with intervention(net, [(0, 1, 'pool')]):
            raise RuntimeError('test cleanup')
    assert net.blocks[0].attn is original[0]
    spec = assign_heads(None, [(0, 1, 'pool')], 2, 2)
    reconfigure(net, spec)
    with torch.no_grad():
        torch.testing.assert_close(net(*inputs()), changed)
    save_export(net, tmp_path/'export', {})
    loaded = load_export(tmp_path/'export')
    with torch.no_grad():
        torch.testing.assert_close(loaded(*inputs()), net(*inputs()))


def test_reconfigure_migrates_conv_by_head_id_and_supports_gradients():
    net = model()
    spec = dict(type='head_split', factor=2, kernel=3, local_heads=[[0,1], []])
    reconfigure(net, spec)
    with torch.no_grad():
        net.blocks[0].attn.local_conv.weight[:16].fill_(2)
        net.blocks[0].attn.local_conv.weight[16:].fill_(7)
    new = assign_heads(net.attn_compression, [(0,0,'dense')], 2, 2)
    reconfigure(net, new)
    assert torch.all(net.blocks[0].attn.local_conv.weight == 7)
    net.use_checkpoint = True
    net.train()
    net(*inputs()).square().mean().backward()
    a = net.blocks[0].attn
    assert a.local_conv.weight.grad is not None and a.qkv.weight.grad.abs().sum() > 0


def test_locality_is_not_sensitivity_and_empty_roi_is_not_zero():
    base = attn()
    with torch.no_grad():
        base.proj.weight[:, :16].zero_()  # arbitrary locality, exactly zero causal contribution
    x, gate = torch.randn(1,64,32), torch.ones(1,1,32)
    with torch.no_grad():
        rows = list(measure_heads(base, x, gate, rope(8), rope(8,2),
                                 masks={'small': torch.zeros(1,64,1)}, head_chunk=1, query_chunk=7))
    assert rows[0]['mass3x3'] > 0 and rows[0]['head_mse'] > 0
    assert rows[0]['residual_mse'] == 0 and rows[0]['small_mse'] is None
    assert rows[1]['residual_mse'] > 0
    with torch.no_grad():
        identity = list(measure_heads(base, x, gate, rope(8), rope(8), factor=1))
    assert all(row['head_mse'] == 0 for row in identity)


def test_analytic_macs_match_real_shape_flop_counter_and_budget():
    cfg = load_config(ROOT/'configs/ablations/head_split_f2.yml')
    spec, _ = attn_compression_spec(cfg['student']['attn_compression'],36,24)
    baseline = attention_macs(spec,36,24,1536,32)
    assert baseline['macs'] == 107875336192
    assert baseline['ratio'] == pytest.approx(.2325618885)
    scores = {(l,h): float(864-l*24-h) for l in range(36) for h in range(24)}
    candidate, selected = protect(spec,scores,36,24,1536,32,.27)
    random = random_control(spec, selected,36,24,11)
    assert attention_macs(candidate,36,24,1536,32)['ratio'] <= .27
    assert attention_macs(candidate,36,24,1536,32)['macs'] == attention_macs(random,36,24,1536,32)['macs']
    from torch.utils.flop_counter import FlopCounterMode
    with torch.device('meta'):
        net = AblationLightningDiT(**dict(cfg['model'],auxiliary_time_cond=True,input_size=64),
                                  attn_compression=candidate).eval()
        inp = (torch.randn(1,32,64,64),torch.ones(1),torch.zeros(1),[torch.randn(1,1024,1024)])
    with torch.no_grad(), FlopCounterMode(display=False) as count:
        net(*inp)
    sa = sum(sum(ops.values()) for name,ops in count.get_flop_counts().items()
             if name.count('.') == 3 and name.endswith('.attn'))/2
    assert sa == attention_macs(candidate,36,24,1536,32)['macs']


def test_roi_translation_aggregation_and_group_validation():
    q = [[10,10,80,10,80,50,10,50], [0,0,10,0,10,80,0,80]]
    masks = roi_masks(translated_boxes(q,(0,0),128),128)
    assert masks['small'].sum() == masks['text'].sum()  # vertical line uses rotated height
    assert translated_boxes(q,(20,0),128,complete=True) == []
    rows = [dict(layer=0,head=0,mode='pool',sample='a',image='a',noise_seed=42,small_mse=None),
            dict(layer=0,head=0,mode='pool',sample='b',image='b',noise_seed=42,small_mse=2.)]
    assert aggregate(rows,('layer','head','mode'))[0]['small_mse_count'] == 1
    args = SimpleNamespace(groups=None,layers=[0],heads=[0,0],unit='layer',mode='pool')
    with pytest.raises(ValueError):
        make_groups(args,2,2)


def test_paired_ocr_bootstrap_and_strict_alignment(tmp_path):
    ref, arm = [], []
    for i in range(12):
        base = dict(image=f'{i}.png',index=0,label='abcd',norm_label='abcd',size=40.,
                    pred=dict(text='abcd',edit_distance=0))
        ref.append(base)
        arm.append(dict(base,pred=dict(text='abxd',edit_distance=1)))
    paths = [tmp_path/'ref.jsonl',tmp_path/'arm.jsonl']
    for path,rows in zip(paths,[ref,arm]):
        path.write_text('\n'.join(json.dumps(r) for r in rows))
    result = compare(*paths,bootstrap=50)
    assert result['bins']['<64']['delta']['CER'] == .25
    assert result['bins']['<64']['delta']['OCR-A'] == -1
    assert result['bins']['<64']['ci95']['CER'] == [.25,.25]
    assert result['bins']['>=96']['delta'] is None
    grouped = compare(*paths,bootstrap=50,source_images={f'{i}.png':f'source{i//2}' for i in range(12)})
    assert grouped['bins']['<64']['images'] == 6 and grouped['bins']['<64']['ci95'] is None
    paths[1].write_text('\n'.join(json.dumps(r) for r in arm[:-1]))
    with pytest.raises(ValueError, match='Unpaired'):
        compare(*paths)


def test_cpu_profile_and_causal_entrypoints(tmp_path,monkeypatch):
    import tools.attn_head_sensitivity as tool
    net = model()
    cfg = dict(teacher_checkpoint='fixture',vae_path='fixture',dino={'size':56,'layer':0},
               model=dict(depth=2,num_heads=2,hidden_size=32,patch_size=1),data={'resolution':64},
               student={'attn_compression':dict(type='head_split',factor=2,kernel=3,local_heads=[[],[]])})
    cache = tmp_path/'cache'
    cache.mkdir()
    inp,t,r,z = inputs()
    payload = dict(inp=inp,noise=inp[:,2:],z=z,masks={'small':torch.ones(1,1,64,64),
                                                   'text':torch.ones(1,1,64,64)},mean=None,std=None)
    torch.save(payload,cache/'a.pt')
    entry = dict(id='a',image='source',noise_seed=42,file='a.pt',sha256=sha256(cache/'a.pt'))
    meta = dict(version=1,config=cfg,precision='fp32',crop=64,entries=[entry],annotations=False)
    meta['signature'] = signature(meta)
    write_json(cache/'manifest.json',meta)
    monkeypatch.setattr(tool,'load_teacher',lambda *a: copy.deepcopy(net))
    args = SimpleNamespace(cache=str(cache),precision='fp32',layers=None,head_chunk=1,query_chunk=16,
                          shortlist=2,output=str(tmp_path/'profile'))
    profile(args,cfg,torch.device('cpu'))
    ranking = json.loads((tmp_path/'profile/ranking.json').read_text())
    assert len(ranking['rows']) == 4 and all(r['small_mse'] > 0 for r in ranking['rows'])
    args = SimpleNamespace(cache=str(cache),precision='fp32',export=None,layers=[0],heads=None,groups=None,
        unit='head',mode='pool',decode=False,ocr=False,output=str(tmp_path/'ablate'))
    ablate(args,cfg,torch.device('cpu'))
    ranking = json.loads((tmp_path/'ablate/ranking.json').read_text())
    assert ranking['kind'] == 'dense_drop' and len(ranking['rows']) == 2
    assert all(r['kd_mse'] > 0 and r['delta_kd_mse'] == r['kd_mse'] for r in ranking['rows'])


def test_prepare_decode_ocr_and_budget_export_roundtrip(tmp_path, monkeypatch):
    """Tiny real DiT math; synthetic VAE/DINO/OCR fixtures explicitly replace external weights."""
    from PIL import Image
    from safetensors.torch import save_file
    import sys
    import yaml
    import tools.attn_head_sensitivity as tool
    import evaluate
    from tools.plan_head_budget import build, migrate

    class TinyVAE(nn.Module):
        config = SimpleNamespace(latents_mean=[0.,0.],latents_std=[1.,1.])
        def encode(self,x):
            z = torch.nn.functional.avg_pool2d(x[:,:2],8)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))
        def decode(self,z,return_dict=True):
            value = torch.nn.functional.interpolate(torch.cat((z,z[:,:1]),1),scale_factor=8,mode='nearest')
            return SimpleNamespace(sample=value) if return_dict else (value,)
        @classmethod
        def from_pretrained(cls,*args):
            return cls()
    monkeypatch.setitem(sys.modules,'models.qwenimage_vae2d',SimpleNamespace(AutoencoderKLQwenImage2D=TinyVAE))
    monkeypatch.setattr(tool,'load_dino',lambda *a: nn.Identity())
    monkeypatch.setattr(tool,'dino_features',lambda m,x,c: [torch.ones(x.shape[0],4,8,device=x.device)])
    monkeypatch.setattr(evaluate,'PaddleTextRecognizer',lambda *a: lambda crops: [('abcd',1.) for c in crops])
    for name in ('LQ','HQ'):
        (tmp_path/name).mkdir()
        Image.fromarray(np.random.default_rng(0).integers(0,255,(64,64,3),dtype=np.uint8)).save(tmp_path/name/'a.png')
    write_json(tmp_path/'boxes.json',{'a':[[10,10,50,10,50,40,10,40]]})
    (tmp_path/'Label.txt').write_text('a.png\t'+json.dumps([dict(transcription='abcd',points=[[10,10],[50,10],[50,40],[10,40]])]))
    net = model()
    save_file(net.state_dict(),str(tmp_path/'teacher.safetensors'))
    cfg = dict(teacher_checkpoint=str(tmp_path/'teacher.safetensors'),vae_path='fixture',
        model={k:v for k,v in net.export_config.items() if k not in ('input_size','use_cross_attention','router_config',
                                               'sparse_eval','token_compression','rope_mode')},
        dino={'size':56,'layer':0},data=dict(dataset_type='paired',resolution=64,
             lq_dir=str(tmp_path/'LQ'),hq_dir=str(tmp_path/'HQ'),text_boxes=str(tmp_path/'boxes.json'),upscale=4),
        student=dict(attn_compression=dict(type='head_split',factor=2,kernel=3,local_heads=[[],[]])),
        training={'vae_tile_size':0})
    path = tmp_path/'config.yml'
    path.write_text(yaml.safe_dump(cfg))
    cache = tmp_path/'cache'
    args = SimpleNamespace(output=str(cache),samples=1,seed=11,noise_seeds=[42,43],names=None,
                           ann=str(tmp_path/'Label.txt'),precision='fp32')
    tool.prepare(args,cfg,torch.device('cpu'))
    meta = json.loads((cache/'manifest.json').read_text())
    assert len(meta['entries']) == 2 and meta['annotations']
    profile_args = SimpleNamespace(output=str(tmp_path/'profile'),cache=str(cache),precision='fp32',layers=None,
                                   head_chunk=1,query_chunk=16,shortlist=2)
    tool.profile(profile_args,cfg,torch.device('cpu'))
    reconfigure(net,cfg['student']['attn_compression'])
    pipeline = dict(dino=cfg['dino'],noise_input='gaussian',precision='fp32',vae_path='fixture',resolution=64,upscale=4)
    save_export(net,tmp_path/'release',pipeline)
    ablate_args = SimpleNamespace(output=str(tmp_path/'rescue'),cache=str(cache),precision='fp32',
        export=str(tmp_path/'release'),decode=True,ocr=True,groups=None,unit='head',mode='dense',layers=[0],heads=[0],
        ocr_model='fixture',ocr_device='cpu',bootstrap=10,seed=0)
    tool.ablate(ablate_args,cfg,torch.device('cpu'))
    ocr = json.loads((tmp_path/'rescue/ocr_deltas.json').read_text())
    assert ocr[0]['bins']['<64']['images'] == 1  # same original image across two noise seeds
    assert ocr[0]['bins']['<64']['observations'] == 2
    assert ocr[0]['bins']['<64']['delta']['CER'] == 0
    budget = tmp_path/'budget'
    build(SimpleNamespace(config=str(path),ranking=str(tmp_path/'profile/ranking.json'),source_export=str(tmp_path/'release'),
        metric='small_mse',ratios=[.8],random_seeds=[11],steps=10,reclassify=False,output=str(budget)))
    experiments = json.loads((budget/'experiments.json').read_text())
    selected = experiments['experiments'][0]
    dest = budget/selected['name']
    migrate(SimpleNamespace(config=str(path),source=str(tmp_path/'release'),spec=str(budget/selected['layout']),output=str(dest)))
    migrated = load_export(dest)
    assert migrated.attn_compression['dense_heads']
    restored_cfg = load_config(budget/selected['config'])
    assert restored_cfg['student_checkpoint'] == str(dest/'model.safetensors')
    with torch.no_grad():
        assert torch.isfinite(migrated(*inputs())).all()


@pytest.mark.parametrize('bf16',[False,True])
def test_all_three_branches_numeric_reference(bf16):
    torch.manual_seed(11)
    base = Attention(64,4,qkv_bias=True,qk_norm=True,use_rmsnorm=True).eval()
    split = HeadSplitAttention(base,[1],2,3,'bilinear',[0]).eval()
    kernels = torch.zeros(4,3,3)
    kernels[:,1,1] = 1
    x = torch.randn(1,64,64)
    with torch.no_grad(), torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
        expected = base.proj.bias[None,None].expand(1,64,-1)
        for h in range(4):
            if h == 1:
                values = conv_values(base,x,[h],kernels)[:,0]
            else:
                factor = 1 if h == 0 else 2
                values = attention_values(base,x,[h],rope(8,factor),factor)[:,0]
            expected = expected + torch.nn.functional.linear(values,base.proj.weight[:,h*16:(h+1)*16])
        output = split(x,(rope(8,2),rope(8)))
    torch.testing.assert_close(output.float(),expected.float(),atol=.01 if bf16 else 2e-6,
                               rtol=.03 if bf16 else 2e-5)


def test_reclassification_uses_output_error():
    spec = dict(type='head_split',factor=2,kernel=3,local_heads=[[0],[]])
    rows = [dict(layer=l,head=h,mode=m,small_mse=value,mass3x3=.99 if h else .01)
            for l in range(2) for h in range(2) for m,value in
            (('pool',.1 if h == 0 else .4),('conv',.3 if h == 0 else .2))]
    revised = reclassify(spec,dict(kind='same_input_feature_screen',rows=rows),'small_mse',2,2)
    assert revised['local_heads'] == [[1],[1]]
    with pytest.raises(ValueError,match='Budget'):
        protect(revised,{(0,0):1.},2,2,32,8,.01)
