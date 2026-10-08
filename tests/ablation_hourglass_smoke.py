"""CPU integration fixture: real trainer and real small AblationLightningDiT student/teacher.

Only the VAE (8x average pooling), DINO (size-dependent pooled tokens) and the
degradation are substituted, so token merging, curriculum, feature distillation,
student-side DINO, previews, checkpoints and exports run through the real code.
"""
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch import nn


class TinyVAE(nn.Module):
    """f8 stand-in: three latent channels, average pooling down and nearest upsampling."""
    config = SimpleNamespace(latents_mean=[0, 0, 0], latents_std=[1, 1, 1])

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def encode(self, x):
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: F.avg_pool2d(x, 8)))

    def decode(self, x, return_dict=False):
        return (F.interpolate(x, scale_factor=8, mode='nearest'),)


class Degradation:
    opt = {'scale': 4}

    def __init__(self, *args, **kwargs):
        pass

    def degrade_process(self, hq, resize_bak=False):
        assert resize_bak
        return hq, hq * .8


def fake_dino_features(model, lq, config):
    """(size // 14)^2 tokens with z_dims=6, so teacher/student DINO sizes give different token counts."""
    side = int(config['size']) // 14
    pooled = F.adaptive_avg_pool2d(lq.float(), side).flatten(2).transpose(1, 2)
    return [torch.cat((pooled, pooled.square()), -1)]


def inject(name, key, value):
    module = ModuleType(name)
    setattr(module, key, value)
    sys.modules[name] = module


if __name__ == '__main__':
    inject('models.qwenimage_vae2d', 'AutoencoderKLQwenImage2D', TinyVAE)
    inject('dataloaders.realesrgan_gpu', 'RealESRGAN_degradation', Degradation)
    import ablation_logging
    import train_vosr_ablation as trainer
    trainer.load_dino = lambda *args: nn.Identity()
    trainer.dino_features = ablation_logging.dino_features = fake_dino_features
    trainer.main()
