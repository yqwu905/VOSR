"""Export the unmodified pretrained backbone for the SAME tiled evaluation path."""
import argparse
import json
from pathlib import Path

from ablation_utils import load_config, read_weights, load_backbone_state, save_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ablations/base_vosr2.yml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--upscale', type=int, default=4)
    args = parser.parse_args()
    if args.upscale <= 0:
        parser.error('--upscale must be positive')
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        parser.error('--output must be empty; refusing to replace an export')
    from models.lightningdit_ablation import AblationLightningDiT
    cfg = load_config(args.config)
    state = read_weights(cfg['teacher_checkpoint'])
    model_cfg = dict(cfg['model'])
    aux = model_cfg.pop('auxiliary_time_cond', 'auto')
    model_cfg['auxiliary_time_cond'] = any(k.startswith('r_embedder.') for k in state) if aux == 'auto' else bool(aux)
    model_cfg['input_size'] = int(cfg['data']['resolution']) // 8
    model = AblationLightningDiT(**model_cfg, use_cross_attention=True)
    report = load_backbone_state(model, state)
    pipeline = dict(vae_path=cfg['vae_path'], dino=cfg['dino'], resolution=cfg['data']['resolution'],
                    upscale=args.upscale, precision=cfg['training']['precision'])
    save_export(model.eval(), args.output, pipeline)
    (Path(args.output) / 'provenance.json').write_text(json.dumps(dict(
        kind='original_pretrained_backbone_export', source_checkpoint=str(cfg['teacher_checkpoint']),
        source_config=str(args.config), load_report=report), indent=2), encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
