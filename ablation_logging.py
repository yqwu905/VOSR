"""Accelerate trackers and reproducible training previews for ablations."""
from contextlib import contextmanager
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

from ablation_utils import prepare_training_batch, dino_features, student_dino_config

logger = get_logger(__name__)


TRACKING_CONFIG_KEYS = frozenset({
    'report_to', 'tracker_project_name', 'run_name', 'wandb_mode', 'log_every',
    'preview_every', 'preview_num_images', 'preview_seed',
})


def resume_configs_match(saved, current):
    """Allow instrumentation changes on resume, but preserve training semantics."""
    def without_tracking(config):
        result = dict(config)
        result['training'] = {key: value for key, value in config['training'].items()
                              if key not in TRACKING_CONFIG_KEYS}
        return result
    return without_tracking(saved) == without_tracking(current)


class TrainingLogger:
    """Use Accelerate for tracking only; the trainer owns DDP and optimization."""
    def __init__(self, config, device, start_step=0):
        tc = config['training']
        targets = tc.get('report_to', ['tensorboard', 'wandb'])
        if targets is None or targets == 'none':
            targets = []
        elif isinstance(targets, str):
            targets = [targets]
        if not isinstance(targets, list) or any(t not in ('tensorboard', 'wandb') for t in targets):
            raise ValueError('report_to must be none, tensorboard, wandb, or a list of these backends')
        for target in targets:
            if importlib.util.find_spec(target) is None:
                raise ImportError(f'Install {target} to enable training.report_to={targets}')
        self.out = Path(tc['output_dir'])
        log_dir = self.out / 'logs'
        self.accelerator = Accelerator(cpu=device.type == 'cpu', mixed_precision='no',
                                       log_with=targets or None, project_dir=str(log_dir))
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        kwargs = {}
        if 'wandb' in targets:
            kwargs['wandb'] = dict(name=tc.get('run_name') or self.out.name,
                                  dir=str(log_dir),
                                  mode=os.environ.get('WANDB_MODE', tc.get('wandb_mode', 'offline')))
        if 'tensorboard' in targets and start_step:
            kwargs['tensorboard'] = {'purge_step': start_step + 1}
        if self.accelerator.is_main_process:
            log_dir.mkdir(parents=True, exist_ok=True)
        # TensorBoard's hparams API requires scalar values; keep the resolved config as JSON.
        self.accelerator.init_trackers(tc.get('tracker_project_name', 'vosr-ablation'),
                                       config={'resolved_config': json.dumps(config),
                                               'start_step': start_step}, init_kwargs=kwargs)

    def log(self, record, images=None):
        if not self.accelerator.is_main_process:
            return
        step = record['step']
        if images:
            folder = self.out / 'previews' / f'step-{step:08d}'
            folder.mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(images):
                image.save(folder / f'sample-{i:02d}.png')
            for tracker in self.accelerator.trackers:
                if tracker.name == 'tensorboard':
                    tracker.log_images({'preview/LQ_student_teacher_HQ': np.stack([np.asarray(im) for im in images])},
                                       step=step, dataformats='NHWC')
                elif tracker.name == 'wandb':
                    # Commit images together with the scalars below, at the SAME W&B step.
                    tracker.log_images({'preview/LQ_student_teacher_HQ': images}, step=step, commit=False)
        logger.info(json.dumps(record))
        with (self.out / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        self.accelerator.log({key: value for key, value in record.items() if key != 'step'}, step=step)

    def close(self):
        # Tracker.finish only: Accelerator.end_training also destroys the process group,
        # which is owned and closed by the trainer after its final barrier.
        if self.accelerator.is_main_process:
            for tracker in self.accelerator.trackers:
                tracker.finish()


@contextmanager
def isolated_rng(seed, device):
    """Preserve Python, NumPy, CPU and the current CUDA device RNG states."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@torch.no_grad()
def build_preview_samples(dataset, degradation, device, count, seed):
    samples = []
    with isolated_rng(seed, device):
        for i in range(min(count, len(dataset))):
            batch = {'hq': dataset[i]['hq'].unsqueeze(0)}
            hq, lq = prepare_training_batch(batch, degradation, device)
            samples.append((hq.cpu(), lq.cpu()))
    return samples


def comparison_image(pixels, labels=('LQ', 'Student SR', 'Teacher SR', 'HQ')):
    panels = []
    for tensor in pixels:
        array = ((tensor[0].detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        panels.append(Image.fromarray(array))
    width, height = panels[0].size
    canvas = Image.new('RGB', (width * len(panels), height + 24), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, panel) in enumerate(zip(labels, panels)):
        canvas.paste(panel, (i * width, 24))
        draw.text((i * width + 4, 5), label, fill='black')
    return canvas


@torch.no_grad()
def render_previews(student, teacher, vae, venc, samples, config, device):
    from tiled_vae import encode_latent, decode_latent
    seed = int(config['training'].get('preview_seed', 1234))
    was_training, was_sparse = student.training, student.sparse_eval
    args = SimpleNamespace(ae_type='qwen')
    student_dino = student_dino_config(config)
    images, keeps, expected_keeps, layer_keeps = [], [], [], []
    try:
        student.eval()
        # Deterministic routing with dense masked MLPs avoids backend-specific sparse ops.
        student.sparse_eval = False
        with isolated_rng(seed, device):
            for hq, lq in samples:
                lq = lq.to(device)
                latent, mean, std = encode_latent(vae, lq, args, device, posterior_mode=True)
                noise = torch.randn_like(latent)
                with torch.autocast(device.type, dtype=torch.bfloat16,
                                    enabled=config['training'].get('precision', 'bf16') == 'bf16'):
                    features = dino_features(venc, lq, config['dino'])
                    student_features = (features if student_dino == config['dino'] else
                                        dino_features(venc, lq, student_dino))
                    inp = torch.cat((latent, noise), 1)
                    t, r = latent.new_ones(1), latent.new_zeros(1)
                    teacher_v = teacher(inp, t, r, features) if teacher is not None else None
                    student_v, stats = student(inp, t, r, student_features, return_stats=True)
                sr = decode_latent(vae, noise - student_v.float(), args, mean, std)
                if teacher is None:  # plain SFT runs without a teacher
                    images.append(comparison_image((lq, sr, hq), ('LQ', 'Student SR', 'HQ')))
                else:
                    teacher_sr = decode_latent(vae, noise - teacher_v.float(), args, mean, std)
                    images.append(comparison_image((lq, sr, teacher_sr, hq)))
                keeps.append(stats['keep_fraction'].item())
                if 'keep_probabilities' in stats:
                    expected_keeps.append(stats['keep_probabilities'].mean().item())
                if 'keep_fractions' in stats:
                    layer_keeps.append(stats['keep_fractions'].detach().float().cpu())
    finally:
        student.train(was_training)
        student.sparse_eval = was_sparse
    metrics = {'preview/deterministic_mlp_keep': float(np.mean(keeps))}
    if expected_keeps:
        metrics['preview/expected_mlp_keep'] = float(np.mean(expected_keeps))
        metrics['preview/keep_gap'] = metrics['preview/deterministic_mlp_keep'] - metrics['preview/expected_mlp_keep']
    if layer_keeps:
        for layer, keep in enumerate(torch.stack(layer_keeps).mean(0).tolist()):
            metrics[f'preview/layer_{layer:02d}/actual_keep'] = keep
    return images, metrics
