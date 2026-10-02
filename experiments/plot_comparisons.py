"""Plot measured paired differences; positive x always means improvement."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_pipeline.pipeline import atomic_json, file_digest
from experiments.compare import LOWER, METRICS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--metrics', nargs='+', choices=METRICS, default=['PSNR', 'LPIPS', 'CER', '1-NED'])
    p.add_argument('--title', default='Held-out validation: paired change from baseline')
    p.add_argument('--labels', nargs='*', default=[], help='name=display label')
    args = p.parse_args()
    data = json.loads(args.comparison.read_text())['comparisons']
    if not data:
        raise ValueError('No completed comparisons to plot')
    labels = dict(item.split('=', 1) for item in args.labels)
    names = list(data)
    groups = {row['groups'] for row in data.values()}
    baselines = {row['baseline_directory'] for row in data.values()}
    if len(groups) != 1 or len(baselines) != 1:
        raise ValueError('A figure must share one baseline and grouping protocol')
    nrows = math.ceil(len(args.metrics)/2)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, nrows*max(2.2, 0.4*len(names)+1.3)),
                             squeeze=False)
    colors = plt.get_cmap('tab10')(np.arange(len(names)) % 10)
    for ax, metric in zip(axes.ravel(), args.metrics):
        scale = 100 if metric in ['OCR-A', 'CER', '1-NED'] else 1
        direction = -1 if metric in LOWER else 1
        for index, name in enumerate(names):
            row = data[name]['metrics'][metric]
            value = row['delta']*direction*scale
            low, high = sorted(x*direction*scale for x in row['delta_ci95'])
            ax.hlines(index, low, high, color=colors[index], linewidth=1.2)
            ax.vlines([low, high], index-0.06, index+0.06, color=colors[index], linewidth=1.2)
            ax.plot(value, index, 'o', color=colors[index], markersize=5)
        unit = ' (dB)' if metric == 'PSNR' else ' (percentage points)' if scale == 100 else ''
        ax.set_title(('Reduction in ' if direction < 0 else 'Increase in ')+metric+unit, fontsize=10)
        ax.set_yticks(range(len(names)), [labels.get(n, n) for n in names], fontsize=9)
        ax.set_ylim(len(names)-0.5, -0.5)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.2)
        ax.set_xlabel('Better →', fontsize=9)
        ax.spines[['right', 'top']].set_visible(False)
    for ax in axes.ravel()[len(args.metrics):]:
        ax.set_visible(False)
    fig.suptitle(args.title, fontsize=13)
    fig.text(0.5, 0.015, f'95% paired bootstrap intervals over {next(iter(groups))} source groups. '
             'Fixed trained seed; intervals do not account for model selection.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=180)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)
    atomic_json(args.output.with_suffix('.json'), dict(comparison_sha256=file_digest(args.comparison),
                code_sha256=file_digest(__file__),
                input=str(args.comparison), metrics=args.metrics, labels=labels, title=args.title,
                sign_convention='Positive is improvement: lower-is-better metrics have negated deltas.'))


if __name__ == '__main__':
    main()
