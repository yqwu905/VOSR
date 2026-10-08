"""Exercise the real trainer + real tiny DiT; replace data/VAE/DINO/degradation.

Synthetic fixture only. This does not test pretrained VOSR2 image quality.
"""
import os
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import torch.nn.functional as F


class SyntheticData(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {'hq': torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(index))}


class TinyVAE(nn.Module):
    config = SimpleNamespace(latents_mean=[0, 0], latents_std=[1, 1])

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def encode(self, x):
        latent = F.avg_pool2d(x[:, :2], 8)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))


class Degradation:
    opt = {'scale': 4}

    def __init__(self, *args, **kwargs):
        pass

    def degrade_process(self, x, resize_bak=True):
        return x, x * .8


for name, key, value in [('models.qwenimage_vae2d', 'AutoencoderKLQwenImage2D', TinyVAE),
                         ('dataloaders.realesrgan_gpu', 'RealESRGAN_degradation', Degradation)]:
    module = ModuleType(name)
    setattr(module, key, value)
    sys.modules[name] = module

import train_vosr_ablation as trainer
trainer.build_txt_dataset = lambda cfg: SyntheticData()
trainer.load_dino = lambda *args: nn.Identity()
trainer.dino_features = lambda net, lq, cfg: [lq.new_zeros(lq.shape[0], 3, 16)]
trainer.main()
