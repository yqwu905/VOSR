"""cer_gap.py REF_EVAL_DIR [REF_EVAL_DIR ...] -- ARM_EVAL_DIR [ARM_EVAL_DIR ...]
Pooled CER (sum of edits / label chars) of evaluate.py ocr_details.jsonl by text size, averaged over the given seed
directories, and a paired image bootstrap of ARM - REF (2000 resamples; 95% interval)."""
import collections, json, random, sys
args = sys.argv[1:]; cut = args.index('--'); ref_dirs, arm_dirs = args[:cut], args[cut + 1:]
BINS = {'<16': (0, 16), '16-24': (16, 24), '24-32': (24, 32), 'S<32': (0, 32), 'L>=32': (32, 1e9), 'all': (0, 1e9)}
def load(dirs):  # image -> list of (size, edits averaged over seeds, label chars)
    runs = []
    for d in dirs:
        r = collections.defaultdict(dict)
        for line in open(f'{d}/ocr_details.jsonl', encoding='utf-8'):
            x = json.loads(line)
            if x['label']: r[x['image']][x['index']] = (x['size'], x['pred']['edit_distance'], len(x['label']))
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
samples = [[rng.choice(images) for _ in images] for _ in range(2000)]
for name, (lo, hi) in BINS.items():
    boxes = sum(lo <= s < hi for i in images for s, _, _ in ref[i])
    gap = cer(arm, images, lo, hi) - cer(ref, images, lo, hi)
    boot = sorted(cer(arm, s, lo, hi) - cer(ref, s, lo, hi) for s in samples)
    print(f'{name:6s} boxes={boxes:5d} ref={cer(ref, images, lo, hi):.4f} arm={cer(arm, images, lo, hi):.4f} '
          f'gap={gap:+.4f} 95%=[{boot[50]:+.4f}, {boot[1949]:+.4f}]')
