"""cer_gap.py REF_EVAL_DIR [REF_EVAL_DIR ...] -- ARM_EVAL_DIR [ARM_EVAL_DIR ...]
Pooled CER (sum of edits / normalized label chars, as evaluate.py) of evaluate.py ocr_details.jsonl by text size,
averaged over the given seed directories, and a paired image bootstrap of ARM - REF (2000 resamples; 95% interval).
Each size bin is resampled over the images that have a box in it; with fewer than 10 such images the interval is
not reported."""
import collections, json, random, sys, unicodedata
args = sys.argv[1:]; cut = args.index('--'); ref_dirs, arm_dirs = args[:cut], args[cut + 1:]
BINS = {'<16': (0, 16), '16-24': (16, 24), '24-32': (24, 32), 'S<32': (0, 32), 'L>=32': (32, 1e9), 'all': (0, 1e9)}
# evaluate.py writes the normalized label it scores against as norm_label; older files only have the transcription.
chars = lambda x: len(x.get('norm_label') or ''.join(unicodedata.normalize('NFKC', x['label']).split()))
def load(dirs):  # image -> list of (size, edits averaged over seeds, label chars)
    runs = []
    for d in dirs:
        r = collections.defaultdict(dict)
        for line in open(f'{d}/ocr_details.jsonl', encoding='utf-8'):
            x = json.loads(line)
            if chars(x): r[x['image']][x['index']] = (x['size'], x['pred']['edit_distance'], chars(x))
        runs.append(r)
    out = {}
    for img in runs[0]:
        out[img] = [(s, sum(run[img][i][1] for run in runs) / len(runs), c) for i, (s, _, c) in runs[0][img].items()]
    return out
ref, arm = load(ref_dirs), load(arm_dirs)
images = sorted(set(ref) & set(arm)); rng = random.Random(0)
def cer(R, imgs, lo, hi):
    e = n = 0
    for i in imgs:
        for s, d, c in R[i]:
            if lo <= s < hi: e += d; n += c
    return e / n if n else float('nan')
for name, (lo, hi) in BINS.items():
    holders = [i for i in images if any(lo <= s < hi for s, _, _ in ref[i])]
    boxes = sum(lo <= s < hi for i in holders for s, _, _ in ref[i])
    gap = cer(arm, holders, lo, hi) - cer(ref, holders, lo, hi)
    interval = 'n/a (fewer than 10 images)'
    if len(holders) >= 10:
        boot = sorted(cer(arm, s, lo, hi) - cer(ref, s, lo, hi)
                      for s in ([rng.choice(holders) for _ in holders] for _ in range(2000)))
        interval = f'[{boot[50]:+.4f}, {boot[1949]:+.4f}]'
    print(f'{name:6s} images={len(holders):4d} boxes={boxes:5d} ref={cer(ref, holders, lo, hi):.4f} '
          f'arm={cer(arm, holders, lo, hi):.4f} gap={gap:+.4f} 95%={interval}')
