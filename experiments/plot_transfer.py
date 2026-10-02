"""Plot the same fixed models against their released-model baseline in two domains."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_pipeline.pipeline import atomic_json, file_digest
from experiments.compare import LOWER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())['comparisons']
    test = json.loads(args.test.read_text())['comparisons']
    if not validation or set(validation) != set(test):
        raise ValueError('Both domains must contain the same complete set of fixed models')
    names = list(test)
    for name in names:
        a, b = validation[name], test[name]
        if Path(a['candidate_directory']).parent.name != Path(b['candidate_directory']).parent.name:
            raise ValueError(f'Domain results do not identify the same model: {name}')
    labels = {
        'original_text_s42': 'Original: complete text',
        'original_random_s42': 'Original: random crop',
        'original_strong_s42': 'Original: strong degradation',
        'original_ocr_s42': 'Original: OCR filter',
        'original_ocr_iqa_s42': 'Original: OCR + IQA',
        'teacher_text_s42': 'Teacher: unfiltered',
        'teacher_ocr_s42': 'Teacher: OCR filter',
        'teacher_fidelity_s42': 'Teacher: OCR + fidelity',
        'teacher_ocr_iqa_s42': 'Teacher: OCR + IQA',
        'teacher_fidelity_iqa_s42': 'Teacher: OCR + fidelity + IQA'}
    names = [name for name in labels if name in test] + [name for name in names if name not in labels]
    metrics = ['1-NED', 'LPIPS']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, metric in zip(axes, metrics):
        factor = (-1 if metric in LOWER else 1) * (100 if metric == '1-NED' else 1)
        for dataset, label, color, offset in [
            (validation, 'AnyWord validation', '#2878B5', -0.13),
            (test, 'RealCE official test', '#D55E00', 0.13)]:
            for i, name in enumerate(names):
                values = dataset[name]['metrics'][metric]
                x = factor * values['delta']
                low, high = sorted(factor * value for value in values['delta_ci95'])
                y = i + offset
                ax.hlines(y, low, high, color=color, linewidth=1.2)
                ax.vlines([low, high], y - 0.04, y + 0.04, color=color, linewidth=1.1)
                ax.plot(x, y, 'o', color=color, markersize=4, label=label if i == 0 else None)
        ax.axvline(0, linestyle='--', linewidth=0.8, color='#555555')
        ax.set_title('Increase in 1-NED (percentage points)' if metric == '1-NED' else 'Reduction in LPIPS')
        ax.set_xlabel('Improvement vs released VOSR2 →')
        ax.grid(axis='x', alpha=0.2)
        ax.spines[['right', 'top']].set_visible(False)
    axes[0].set_yticks(np.arange(len(names)), [labels.get(n, n) for n in names], fontsize=9)
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Processing ablations: synthetic validation and real test domains\n500 steps, seed 42, AnyWord:EasyText = 50:50')
    fig.text(0.5, 0.02,
        'Each interval independently resamples source groups within its own dataset. '
        'Fixed models; no adjustment for selection or multiple comparisons.\n'
        'These are within-domain baseline differences, not a confidence interval for a difference between domains.',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=180)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)
    atomic_json(args.output.with_suffix('.json'), dict(
        validation=str(args.validation), validation_sha256=file_digest(args.validation),
        test=str(args.test), test_sha256=file_digest(args.test), code_sha256=file_digest(__file__),
        models=names, metrics=metrics, official_test_used_for_selection=False,
        sign_convention='Positive means improvement relative to the released model within each domain.'))


if __name__ == '__main__':
    main()
