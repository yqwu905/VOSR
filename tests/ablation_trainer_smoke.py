"""CPU integration fixture: real trainer/DDP/data/trackers, tiny model dependencies."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from models.sdt_router import SDTRouter


class TinyModel(nn.Module):
    def __init__(self, use_cross_attention=True, router_config=None, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))
        self.use_cross_attention = use_cross_attention
        self.ca_scale = 1. if use_cross_attention else 0.
        self.sparse_eval = True
        self.blocks = nn.ModuleList()
        self.router_config = router_config
        if router_config is not None:
            for _ in range(2):
                block = nn.Module()
                block.router = SDTRouter(3, **router_config)
                self.blocks.append(block)

    def set_ca_scale(self, value):
        self.ca_scale = float(value)

    def forward(self, x, t, r, features, return_stats=False, force_dense=False, include_dense=False):
        output = x[:, 3:] * self.weight
        stats = {'keep_fraction': x.new_ones(()), 'keep_probabilities': x.new_ones(1)}
        dense = output
        if self.blocks:
            tokens = output.flatten(2).transpose(1, 2)
            probabilities, keeps = [], []
            for block in self.blocks:
                mask, p = block.router(tokens)
                probabilities.append(p.mean())
                keeps.append(mask.detach().float().mean() if not force_dense else p.new_ones(()))
                tokens = tokens + p.mean() * 0 if force_dense else tokens * mask
            output = tokens.transpose(1, 2).reshape_as(output)
            stats = {'keep_fraction': torch.stack(keeps).mean(),
                     'keep_fractions': torch.stack(keeps), 'keep_probabilities': torch.stack(probabilities)}
        if include_dense:
            return output, stats, dense
        return (output, stats) if return_stats else output

    def export_state(self, strip_conditioning=False):
        return {'use_cross_attention': self.use_cross_attention, 'router_config': self.router_config}, self.state_dict()


class TinyVAE(nn.Module):
    config = SimpleNamespace(latents_mean=[0, 0, 0], latents_std=[1, 1, 1])

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def encode(self, x):
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: x))

    def decode(self, x, return_dict=False):
        return (x,)


class Degradation:
    opt = {'scale': 4}

    def __init__(self, *args, **kwargs):
        pass

    def degrade_process(self, hq, resize_bak=False):
        assert resize_bak
        return hq, hq * .8


def inject(name, key, value):
    module = ModuleType(name)
    setattr(module, key, value)
    sys.modules[name] = module


inject('models.lightningdit_ablation', 'AblationLightningDiT', TinyModel)
inject('models.qwenimage_vae2d', 'AutoencoderKLQwenImage2D', TinyVAE)
inject('dataloaders.realesrgan_gpu', 'RealESRGAN_degradation', Degradation)
import ablation_logging
import train_vosr_ablation as trainer
trainer.load_dino = lambda *args: nn.Identity()
trainer.dino_features = lambda *args: [torch.zeros(1, 1, 1)]
ablation_logging.dino_features = trainer.dino_features
trainer.main()
