"""Join the same-run performance and optional OCR files into a comparison table."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    reports = {}
    for name in ('original', 'sdt', 'compressed'):
        directory = Path(root) / name
        performance = json.loads((directory / 'performance.json').read_text(encoding='utf-8'))
        ocr_path = directory / 'ocr.json'
        ocr = json.loads(ocr_path.read_text(encoding='utf-8')) if ocr_path.exists() else None
        if ocr and ocr['manifest_sha256'] != performance['manifest_sha256']:
            raise ValueError(f'{name}: OCR/performance manifest mismatch')
        if ocr and 'image_sha256' in ocr and ocr['image_sha256'] != {r['id']: r['output_sha256'] for r in performance['samples']}:
            raise ValueError(f'{name}: OCR images do not match the measured outputs')
        reports[name] = dict(performance=performance, ocr=ocr)
    baseline = reports['original']['performance']
    ids = [(r['id'], r['lq_sha256']) for r in baseline['samples']]
    ocr_protocols = set()
    rows = []
    for name, report in reports.items():
        perf, ocr = report['performance'], report['ocr']
        if (perf['manifest_sha256'] != baseline['manifest_sha256'] or perf['pipeline'] != baseline['pipeline'] or
                [(r['id'], r['lq_sha256']) for r in perf['samples']] != ids):
            raise ValueError('Cannot compare different inputs/pipelines')
        for key in ('device', 'device_name', 'torch', 'torchdynamo_disable'):
            if perf[key] != baseline[key]:
                raise ValueError(f'Cannot compare different {key}')
        for key in ('seed', 'warmup', 'repeats', 'tile_size', 'tile_overlap', 'vae_tile_size'):
            if perf['config'][key] != baseline['config'][key]:
                raise ValueError(f'Cannot compare different {key}')
        if ocr:
            ocr_protocols.add((ocr['engine_id'], ocr['normalization'], ocr['protocol'],
                               ocr.get('language'), ocr.get('psm')))
        samples = perf['samples']
        medians = [r['latency_median_ms'] for r in samples]
        memories = [m['peak_allocated_bytes'] for r in samples for m in r['memory'] if m['peak_allocated_bytes'] is not None]
        flops = sum(r['attention']['analytic_flops']['attention_matmul_flops'] for r in samples)
        rows.append(dict(variant=name, mean_image_median_ms=statistics.mean(medians),
            speedup=sum(r['latency_median_ms'] for r in baseline['samples'])/sum(medians),
            max_peak_allocated_bytes=max(memories) if memories else None,
            attention_matmul_flops_dataset=flops,
            mean_self_pair_ratio=statistics.mean(r['attention_pair_ratio'] for r in samples),
            ocr_status='measured' if ocr else 'not_measured', ocr=ocr['groups'] if ocr else None))
    if len(ocr_protocols) > 1:
        raise ValueError('OCR engine/normalization/protocol differ across variants')
    baseline_ocr = reports['original']['ocr']
    for row in rows:
        if row['ocr'] and baseline_ocr:
            for name, group in row['ocr'].items():
                reference = baseline_ocr['groups'].get(name)
                if reference is None or (group['regions'], group['characters']) != (reference['regions'], reference['characters']):
                    raise ValueError('OCR strata/coverage differ across variants')
                group['delta_cer_vs_original'] = group['cer'] - reference['cer']
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    rows = summarize(args.run)
    root = Path(args.run)
    (root/'comparison.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['| Variant | Latency ms* | Speedup | Peak MiB | SA pair ratio | CER all | CER small | CER Chinese | CER dense |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        memory = f"{row['max_peak_allocated_bytes']/2**20:.1f}" if row['max_peak_allocated_bytes'] is not None else '未实测'
        cers = [f"{row['ocr'][tag]['cer']:.4f}" if row['ocr'] and tag in row['ocr'] else '未实测'
                for tag in ('all', 'small', 'chinese', 'dense')]
        lines.append(f"| {row['variant']} | {row['mean_image_median_ms']:.2f} | {row['speedup']:.3f} | {memory} | "
                     f"{row['mean_self_pair_ratio']:.4f} | " + ' | '.join(cers) + ' |')
    lines += ['', '*Latency: 每张图 warmed end-to-end 延迟中位数的算术平均；原始重复观测与 p95 在 performance.json。',
              'SA pair ratio 与 Attention FLOPs 是按实际输入形状计算的解析量，不是计时或显存预测。',
              'OCR 使用固定 GT 框识别；CER 越低越好，不包含文字检测能力。精确匹配率、字符准确率和逐框错误见 ocr.json。']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
