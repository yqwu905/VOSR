"""Sequential full-model training and held-out evaluation; never selects on test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from data_pipeline.pipeline import atomic_json


def run(command, logfile):
    print(json.dumps(dict(event='start', command=command, logfile=str(logfile), time=time.time())), flush=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', USE_TF='0', TORCH_COMPILE_DISABLE='1', TORCHDYNAMO_DISABLE='1')
    with logfile.open('a') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); inspect {logfile}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs', nargs='+', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    validation, out = Path(args.validation).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((validation / 'protocol.json').read_text())
    if protocol.get('purpose') == 'final_test' or protocol.get('archive_split') == 'val':
        raise ValueError('Do not run selection against the official test set')
    expected_images = len((validation / 'pairs.jsonl').read_text().splitlines())
    summaries = {}
    for path in args.configs:
        cfg = json.loads(Path(path).read_text())
        training = Path(cfg['training']['output_dir']).resolve()
        name = training.name
        if (training / 'complete.json').exists():
            complete = json.loads((training / 'complete.json').read_text())
            provenance = json.loads((training / 'provenance.json').read_text())
            if complete['steps'] != cfg['training']['max_steps'] or provenance['config'] != cfg:
                raise ValueError('Completed training does not match the requested experiment')
        else:
            command = [sys.executable, '-u', 'train_vosr2_sft.py', '--config', path]
            if (training / 'latest/training_state.pt').exists():
                command.append('--resume')
            run(command, out / 'logs' / f'{name}_train.log')
        prediction = out / name / 'validation_sr'
        run([sys.executable, '-u', '-m', 'experiments.infer', '--checkpoint', str(training / 'export'),
             '--pairs', str(validation / 'pairs.jsonl'), '--output', str(prediction)],
            out / 'logs' / f'{name}_infer.log')
        metrics = out / name / 'validation_metrics'
        if not (metrics / 'summary.json').exists():
            run([sys.executable, '-u', 'evaluate.py', '--pred', str(prediction),
                 '--gt', str(validation / 'gt'), '--ann', str(validation / 'Label.txt'), '--ann-ref', 'gt',
                 '--ocr-python', '.venv-ocr/bin/python', '--ocr-device', 'cpu', '--ocr-pred-only',
                 '--strict', '--rgb', '--crop-border', '2', '--output', str(metrics)],
                out / 'logs' / f'{name}_evaluate.log')
        summary = json.loads((metrics / 'summary.json').read_text())
        if summary['num_images'] != expected_images or not all(k in summary for k in ['fr_iqa', 'ocr']):
            raise ValueError('Incomplete held-out evaluation')
        summaries[name] = summary
        atomic_json(out / 'validation_results.json', summaries)
        print(json.dumps(dict(event='experiment_complete', name=name, fr=summary['fr_iqa'],
                              ocr=summary['ocr']['pred'])), flush=True)
    atomic_json(out / 'queue_complete.json', dict(experiments=list(summaries), validation=str(validation)))


if __name__ == '__main__':
    main()
