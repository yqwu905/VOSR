import json
import pytest

from token_compression_eval import attention_cost, edit_distance, read_manifest, score_predictions
from benchmark_token_compression import AttentionAudit, percentile


def manifest():
    return [dict(id='page', lq='lq.png', size=[100, 100], tags=['dense'], regions=[
        dict(id='r1', bbox=[0, 0, 50, 20], text='中文小字', tags=['small', 'chinese']),
        dict(id='r2', bbox=[0, 30, 50, 50], text='AB', tags=[])])]


def test_cer_corpus_weighting_strata_empty_prediction_and_insertion():
    predictions = [dict(id='page', regions=[dict(id='r1', text='中文错字'), dict(id='r2', text='')])]
    groups = score_predictions(manifest(), predictions)['groups']
    assert groups['all']['cer'] == .5  # (1 substitution + 2 deletions) / 6
    assert groups['small']['cer'] == .25
    assert groups['chinese']['regions'] == 1
    assert groups['dense']['regions'] == 2
    predictions[0]['regions'][1]['text'] = 'ABCD'
    assert score_predictions(manifest(), predictions)['groups']['all']['cer'] == .5
    assert edit_distance('中文', '中午文') == 1


@pytest.mark.parametrize('predictions', [[], [dict(id='page', regions=[])],
    [dict(id='page', regions=[dict(id='r1', text='中'), dict(id='r1', text='中')])]])
def test_missing_or_duplicate_predictions_fail(predictions):
    with pytest.raises(ValueError):
        score_predictions(manifest(), predictions)


def test_manifest_coordinate_and_id_validation(tmp_path):
    path = tmp_path/'eval.jsonl'
    row = manifest()[0]
    path.write_text(json.dumps(row), encoding='utf-8')
    assert read_manifest(path)[0]['lq'] == str(tmp_path/'lq.png')
    row['regions'][0]['bbox'][2] = 101
    path.write_text(json.dumps(row), encoding='utf-8')
    with pytest.raises(ValueError, match='bbox'):
        read_manifest(path)


def test_flop_convention_and_whole_model_ratio():
    original = attention_cost([1024]*36, 1536, condition_tokens=1024, dense_tokens=1024)
    compressed = attention_cost([1024]*4+[256]*28+[1024]*4, 1536, condition_tokens=1024, dense_tokens=1024)
    assert compressed['self_pair_ratio'] == pytest.approx(13/48)
    assert compressed['self_projection_ratio'] == pytest.approx(5/12)
    assert compressed['self_qk_av_flops']/original['self_qk_av_flops'] == pytest.approx(13/48)
    # Cross K,V still process every condition token, so CA projections do not shrink by 4x.
    assert compressed['cross_projection_flops']/original['cross_projection_flops'] > 5/12
    assert original['self_qk_av_flops'] == 4*36*1024**2*1536


def test_latency_percentile():
    assert percentile([10, 20, 30], .5) == 20
    assert percentile([10, 20, 30], .95) == 29


def test_actual_shape_hook_costs_and_cleanup():
    import torch
    from models.lightningdit_ablation import AblationLightningDiT
    model = AblationLightningDiT(input_size=8, patch_size=2, in_channels=4, out_channels=2,
        hidden_size=32, depth=2, num_heads=4, z_dims=8,
        compression_config=dict(factor=2, start_block=0, end_block=2)).eval()
    audit = AttentionAudit(model)
    with torch.no_grad():
        model(torch.randn(1, 4, 8, 8), torch.ones(1), z=[torch.randn(1, 3, 8)])
    audit.close()
    summary = audit.summary()
    assert [r['n'] for r in summary['observed_shapes']] == [4]*4
    assert summary['analytic_flops'] == {k:v for k,v in attention_cost([4,4],32,condition_tokens=3).items()
                                         if k.endswith('flops')}
    assert all(not block.attn._forward_pre_hooks for block in model.blocks)
