"""calib_am.py: loss weights from the lr-0 calibration runs configs/ablations/calib/calib_am_*.yml (run from the repo root).
grad_norm in metrics.jsonl is the pre-clipping norm of the batch-8 gradient of one objective; all runs see the same
batches, so ratios are taken per step and the median over the 20 steps is used, rounded to one significant digit.
For each objective the newest run that logged all 20 steps is used; an interrupted relaunch is skipped."""
import glob, json, statistics
STEPS = range(1, 21)
def norms(name):
    for run in sorted(glob.glob(f'exp_vosr/calib/{name}/*/metrics.jsonl'), reverse=True):
        values = {r['step']: r['grad_norm'] for r in map(json.loads, open(run, encoding='utf-8')) if 'grad_norm' in r}
        if all(s in values for s in STEPS):
            print(f'{name}: {run}')
            return values
    raise SystemExit(f'no complete run (steps 1-20) under exp_vosr/calib/{name}')
def ratio(a, b):
    v = [a[s] / b[s] for s in STEPS if b[s] > 0]
    return statistics.median(v), min(v), max(v)
r1 = lambda x: float(f'{x:.1g}')
sft, kd, fkd, ocr64, ocr48 = (norms(n) for n in ('am_sft', 'am_kd', 'am_fkd', 'am_ocr64', 'am_ocr48'))
for label, num, den, scale in (('kd_weight (KD_AM)', sft, kd, 1.0),
                               ('feature_distill_weight (FKD_AM)', sft, fkd, 1.0),
                               ('ocr_kl_weight for 48/16 strips', ocr64, ocr48, 0.0005)):
    m, lo, hi = ratio(num, den)
    print(f'{label:34s} {r1(scale * m):g}   (per-step range {scale * lo:.3g}-{scale * hi:.3g})')
