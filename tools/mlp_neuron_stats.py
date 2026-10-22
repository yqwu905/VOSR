"""SwiGLU neuron statistics of a student's full-width MLPs, for ``student.mlp_compression.neuron_stats``.

The network measured is the one the pruned student is initialized from: the config's model and student structure
(e.g. ``attn_compression``) without ``mlp_compression``, loaded from ``student_checkpoint`` (the teacher's weights when
it is null) exactly as train_vosr_ablation.py loads the student. Per block and hidden neuron j of
``a = silu(x1) * x2`` (the input of w3), over every token of the samples:

- ``mean`` and ``variance`` of a_j;
- ``output_energy`` = |g2 * w3[:, j]|^2, what the neuron writes into the residual stream per unit of activation
  (g2 = the block's MLP gate, a constant of the one-step model at t=1, r=0);
- ``w3_sq_norm`` = |w3|_F^2 per block, so the trainer can check that the statistics belong to the checkpoint it
  prunes.

``score: fluctuation`` keeps the neurons with the largest variance * output_energy; ``compensation: mean`` moves the
pruned neurons' mean output into the w3 bias (models/mlp_compression.py). Measured at the one-step input (t=1, r=0,
Gaussian noise) on text-centred crops of the config's paired training data (``data.dataset_type: paired``) in FP32,
with the student's DINO input. Crop i is drawn by ``random.Random(seed * 1000003 + i)`` as in tools/attn_head_stats.py.

    python tools/mlp_neuron_stats.py --config configs/ablations/ffn_prune_k75.yml \\
        --output configs/ablations/ffn_prune/head_split_f2_neuron_stats.safetensors --device cuda
"""
import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')  # forward hooks on the eager SwiGLU modules
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_utils import (dino_features, load_backbone_state, load_config, load_dino,  # noqa: E402
                            read_weights, student_dino_config, vae_tiling)


@torch.no_grad()
def neuron_statistics(model, samples):
    """``samples``: iterable of (model input, DINO features). Returns the STATS tensors of models/mlp_compression.py
    on the CPU (FP32), from FP64 sums over all tokens."""
    from models.swiglu_ffn import SwiGLUFFN
    if not all(isinstance(block.mlp, SwiGLUFFN) for block in model.blocks):
        raise ValueError('Neuron statistics need the full-width SwiGLU MLPs (a model without mlp_compression)')
    depth, hidden = len(model.blocks), model.blocks[0].mlp.w3.in_features
    device = next(model.parameters()).device
    sums = torch.zeros(depth, hidden, dtype=torch.float64, device=device)
    squares, counts = torch.zeros_like(sums), [0] * depth

    def hook(index):
        def gather(module, args):
            a = args[0].detach().double().reshape(-1, hidden)
            sums[index] += a.sum(0)
            squares[index] += a.square().sum(0)
            counts[index] += a.shape[0]
        return gather
    handles = [block.mlp.w3.register_forward_pre_hook(hook(i)) for i, block in enumerate(model.blocks)]
    t = r = None
    try:
        for inp, features in samples:
            t, r = inp.new_ones(inp.shape[0]), inp.new_zeros(inp.shape[0])
            model(inp, t, r, features)
    finally:
        for handle in handles:
            handle.remove()
    if t is None or min(counts) == 0:
        raise ValueError('No samples reached the MLPs')
    count = sums.new_tensor(counts).unsqueeze(1)
    mean = sums / count
    variance = (squares / count - mean.square()).clamp_min(0)
    c = model.t_embedder(t[:1])
    if model.r_embedder is not None:
        c = c + model.r_embedder(r[:1]) * (t[:1] - r[:1]).unsqueeze(-1)
    c0 = model.t_block(c).reshape(1, 6, -1)
    energy, norms = [], []
    for block in model.blocks:
        g2 = (block.scale_shift_table[None] + c0)[0, 5].double()  # s1, a1, g1, s2, a2, g2 (AblationBlock)
        w3 = block.mlp.w3.weight.double()
        energy.append((g2[:, None] * w3).square().sum(0))
        norms.append(w3.square().sum())
    stats = dict(mean=mean, variance=variance, output_energy=torch.stack(energy), w3_sq_norm=torch.stack(norms))
    return {key: value.float().cpu().contiguous() for key, value in stats.items()}


def source_model(cfg):
    """The student before pruning, initialized as train_vosr_ablation.py initializes it."""
    from models.lightningdit_ablation import AblationLightningDiT
    state = read_weights(cfg['teacher_checkpoint'])
    model_cfg = dict(cfg['model'])
    aux = model_cfg.pop('auxiliary_time_cond', 'auto')
    model_cfg['auxiliary_time_cond'] = any(k.startswith('r_embedder.') for k in state) if aux == 'auto' else bool(aux)
    model_cfg['input_size'] = int(cfg['data']['resolution']) // 8
    student = cfg['student']
    model = AblationLightningDiT(**model_cfg, use_cross_attention=student['use_cross_attention'],
                                 attn_compression=student.get('attn_compression'),
                                 attn_type=student.get('attn_type', 'full'))
    initial = state if not cfg.get('student_checkpoint') else read_weights(cfg['student_checkpoint'])
    load_backbone_state(model, initial, allow_new_modules=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='Training config with student.mlp_compression')
    parser.add_argument('--output', required=True, help='.safetensors file for mlp_compression.neuron_stats')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg['data']
    if data.get('dataset_type') != 'paired' or not data.get('text_boxes'):
        raise ValueError('The config needs data.dataset_type: paired with data.text_boxes')
    if cfg['student'].get('token_compression') is not None or cfg['student'].get('router_config') is not None:
        raise ValueError('mlp_compression is not combined with token_compression or router_config')
    if args.samples <= 0:
        raise ValueError('--samples must be positive')
    device = torch.device(args.device)
    from dataloaders.paired_crops import PairedCropDataset
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    from tiled_vae import encode_dispatch
    model = source_model(cfg).to(device).eval()
    vae = AutoencoderKLQwenImage2D.from_pretrained(cfg['vae_path']).to(device).eval()
    venc = load_dino(cfg['dino'], device)
    dino = student_dino_config(cfg)
    crop = int(data.get('crop_size') or data['resolution'])
    dataset = PairedCropDataset(data['lq_dir'], data['hq_dir'], crop, data['text_boxes'], 1.0)

    def samples():
        for index in range(args.samples):
            rng = random.Random(args.seed * 1_000_003 + index)
            lq = dataset.sample(rng.choice(dataset.names), rng)['lq'][None].to(device) * 2 - 1
            latent = encode_dispatch(vae, lq, vae_tiling(cfg)[1], device)[0]
            torch.manual_seed(index)
            yield torch.cat((latent, torch.randn_like(latent)), 1), dino_features(venc, lq, dino)
            print(f'sample {index + 1}/{args.samples}', flush=True)

    stats = neuron_statistics(model, samples())
    student_checkpoint = cfg.get('student_checkpoint')
    metadata = {'description': 'SwiGLU neuron statistics (tools/mlp_neuron_stats.py): per block and hidden neuron of '
                               'a = silu(x1) * x2 the mean and variance over all tokens, output_energy = '
                               '|g2 * w3[:, j]|^2, and w3_sq_norm = |w3|_F^2 per block',
                'source': str(student_checkpoint or cfg['teacher_checkpoint']),
                'attn_compression': str(cfg['student'].get('attn_compression') is not None),
                'data': str(data['hq_dir']), 'samples': str(args.samples), 'seed': str(args.seed), 'crop': str(crop)}
    from safetensors.torch import save_file
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_file(stats, args.output, metadata=metadata)
    score = stats['variance'] * stats['output_energy']
    top = score.sort(1, descending=True).values
    share = (top[:, :top.shape[1] // 4].sum(1) / top.sum(1)).mean().item()
    print(f'wrote {args.output}: {score.shape[0]} blocks x {score.shape[1]} neurons; the top 25% of each block hold '
          f'{share:.1%} of its fluctuation on average')


if __name__ == '__main__':
    main()
