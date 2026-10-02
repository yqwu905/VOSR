"""Plot the completed source-ratio screen without selecting on final test data."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_pipeline.pipeline import atomic_json, file_digest
from experiments.compare import LOWER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--title', default='AnyWord validation: source ratios at 500 steps, seed 42')
    args = parser.parse_args()
    entries = json.loads(args.selection.read_text())['ranking']
    comparisons = json.loads(args.comparison.read_text())['comparisons']
    by_recipe = {}
    for row in entries:
        if row['steps'] != 500 or row['seed'] != 42:
            raise ValueError('Ratio figure requires the matched 500-step seed-42 screen')
        by_recipe.setdefault(row['recipe'], []).append(row)
    if len(by_recipe) != 2 or any(sorted(r['anyword_probability'] for r in rows) !=
                                  [0, 0.25, 0.5, 0.75, 1] for rows in by_recipe.values()):
        raise ValueError('Expected two recipes, each with all five source ratios')
    if {row['name'] for row in entries} != set(comparisons):
        raise ValueError('Ratio selection and comparison coverage differ')
    groups = {r['groups'] for r in comparisons.values()}
    baselines = {r['baseline_directory'] for r in comparisons.values()}
    if len(groups) != 1 or len(baselines) != 1:
        raise ValueError('All ratios must share a baseline and source grouping')
    metrics = ['PSNR', 'LPIPS', 'CER', '1-NED']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.ravel(), metrics):
        factor = (-1 if metric in LOWER else 1) * (100 if metric in ['CER', '1-NED'] else 1)
        for index, (recipe, rows) in enumerate(sorted(by_recipe.items())):
            rows.sort(key=lambda r: r['anyword_probability'])
            x = [r['anyword_probability']*100 for r in rows]
            values, intervals = [], []
            color = f'C{index}'
            for row in rows:
                result = comparisons[row['name']]['metrics'][metric]
                if abs(result['candidate'] - row['metrics'][metric]) > 1e-9:
                    raise ValueError('Compared metric differs from frozen selection')
                values.append(result['delta']*factor)
                intervals.append(sorted(v*factor for v in result['delta_ci95']))
            ax.plot(x, values, marker='o', color=color, label=recipe.replace('_', ' '))
            for pos, (low, high) in zip(x, intervals):
                ax.vlines(pos, low, high, color=color, linewidth=1, alpha=0.6)
                ax.hlines([low, high], pos-1, pos+1, color=color, linewidth=1, alpha=0.6)
        unit = ' (dB)' if metric == 'PSNR' else ' (percentage points)' if metric in ['CER', '1-NED'] else ''
        ax.set_title(('Reduction in ' if metric in LOWER else 'Increase in ')+metric+unit)
        ax.axhline(0, color='#555555', linestyle='--', linewidth=0.8)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('AnyWord sampling probability (%)')
        ax.set_ylabel('Improvement vs released VOSR2')
        ax.grid(alpha=0.2)
        ax.spines[['right', 'top']].set_visible(False)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(args.title)
    fig.text(0.5, 0.02, f'EasyText probability = 100% - AnyWord. 95% paired intervals over '
             f'{next(iter(groups))} source groups.\nLines connect measured ratios only; '
             'intervals omit training-seed and selection uncertainty.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=180)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)
    atomic_json(args.output.with_suffix('.json'), dict(
        selection=str(args.selection), selection_sha256=file_digest(args.selection),
        comparison=str(args.comparison), comparison_sha256=file_digest(args.comparison),
        code_sha256=file_digest(__file__), metrics=metrics, title=args.title,
        sign_convention='Positive changes indicate improvement; lower-is-better deltas are negated.'))


if __name__ == '__main__':
    main()
