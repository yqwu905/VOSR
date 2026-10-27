"""Fix the head choice of a head search (``student.attn_compression.search``) as a head-split export.

A search run's export/ already holds its choice at the configured budget; its checkpoints keep the search state
(head logits). This tool fixes a checkpoint's choice, optionally at another self-attention budget (the learned
logits rank the heads, the budget sets how many get upgraded), into an export that ``attn_compression.heads_from``
and ``student_checkpoint`` of the fine-tuning config read. head_selection.json records the choice, its MACs and
every head's option probabilities.

usage: python tools/head_search_select.py --checkpoint exp_vosr/head_split_f2_search/<run>/checkpoint-00002000 \
    --output exp_vosr/head_split_f2_search/<run>/select_0.30 [--attn-budget 0.30]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_utils import load_export, save_export  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint directory of a head search run (model.json with attn_compression.search)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--attn-budget', type=float,
                        help="Self-attention MACs as a share of the dense model's (default: the run's attn_budget)")
    args = parser.parse_args()
    source, out = Path(args.checkpoint), Path(args.output)
    model = load_export(source)
    if model.head_search is None:
        raise ValueError(f'{source} has no attn_compression.search (an export/ of a search run is already fixed)')
    if args.attn_budget is not None:
        model.set_search_budget(args.attn_budget)
    pipeline = json.loads((source / 'pipeline.json').read_text(encoding='utf-8'))
    save_export(model, out, pipeline, selected_heads=True)
    report = model.head_search_report(probabilities=True)
    (out / 'head_selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}, indent=2))


if __name__ == '__main__':
    main()
