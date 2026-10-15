"""calib_am.py: loss weights from the lr-0 calibration runs configs/ablations/calib/calib_am_*.yml (run from the repo root).
grad_norm in metrics.jsonl is the pre-clipping norm of the batch-8 gradient of one objective; all runs see the same
batches, so ratios are taken per step and the median over the 20 steps is used, rounded to one significant digit."""
import glob, json, math, statistics
def norms(name):
    runs = sorted(glob.glob(f'exp_vosr/calib/{name}/*/metrics.jsonl'))
    if not runs:
        raise SystemExit(f'missing exp_vosr/calib/{name}')
    return {r['step']: r['grad_norm'] for r in map(json.loads, open(runs[-1], encoding='utf-8')) if 'grad_norm' in r}
def ratio(a, b):
    v = [a[s] / b[s] for s in a if s in b and b[s] > 0]
    return statistics.median(v), min(v), max(v)
r1 = lambda x: float(f'{x:.1g}')
sft = norms('am_sft')
for label, num, den, scale in (('kd_weight (KD_AM)', sft, norms('am_kd'), 1.0),
                               ('feature_distill_weight (FKD_AM)', sft, norms('am_fkd'), 1.0),
                               ('ocr_kl_weight for 48/16 strips', norms('am_ocr64'), norms('am_ocr48'), 0.0005)):
    m, lo, hi = ratio(num, den)
    print(f'{label:34s} {r1(scale * m):g}   (per-step range {scale * lo:.3g}-{scale * hi:.3g})')
