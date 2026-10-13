"""Pixel-space adversarial loss for one-step SFT: Real-ESRGAN's U-Net discriminator.

``UNetDiscriminatorSN`` reproduces BasicSR 1.4.2 ``basicsr/archs/discriminator_arch.py``
(Apache-2.0) layer for layer, so the network matches Real-ESRGAN's ``network_d``; it is
copied rather than imported because importing basicsr pulls in its data package. The
losses are BasicSR's ``vanilla`` GANLoss (BCE with logits), averaged over the per-pixel
logits.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class UNetDiscriminatorSN(nn.Module):
    """U-Net discriminator with spectral normalization; (N, C, H, W) -> per-pixel logits (N, 1, H, W).
    H and W must be multiples of 8."""
    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        act = lambda y: F.leaky_relu(y, negative_slope=0.2, inplace=True)
        up = lambda y: F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        x0 = act(self.conv0(x))
        x1 = act(self.conv1(x0))
        x2 = act(self.conv2(x1))
        x3 = act(self.conv3(x2))
        x4 = act(self.conv4(up(x3)))
        if self.skip_connection:
            x4 = x4 + x2
        x5 = act(self.conv5(up(x4)))
        if self.skip_connection:
            x5 = x5 + x1
        x6 = act(self.conv6(up(x5)))
        if self.skip_connection:
            x6 = x6 + x0
        out = act(self.conv7(x6))
        out = act(self.conv8(out))
        return self.conv9(out)


def gan_spec(config):
    """Validate ``training.gan`` (a mapping, or null/absent = off) and fill in defaults."""
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError('training.gan must be a mapping or null')
    defaults = dict(weight=0.1, start_step=0, max_scale=1.0e4, num_feat=64, learning_rate=1.0e-4,
                    betas=[0.9, 0.99])
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown training.gan options: {sorted(unknown)}')
    spec = {**defaults, **config}
    if not float(spec['weight']) > 0 or not float(spec['max_scale']) > 0 or not float(spec['learning_rate']) > 0:
        raise ValueError('training.gan weight, max_scale and learning_rate must be positive')
    for key, low in (('start_step', 0), ('num_feat', 1)):
        if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] < low:
            raise ValueError(f'training.gan.{key} must be an integer >= {low}')
    if not (isinstance(spec['betas'], (list, tuple)) and len(spec['betas']) == 2
            and all(0 <= float(b) < 1 for b in spec['betas'])):
        raise ValueError('training.gan.betas must be two numbers in [0, 1)')
    return spec


def generator_loss(logits):
    """Non-saturating generator loss: BCE of the fake logits against the real label."""
    return F.softplus(-logits.float()).mean()


def discriminator_loss(real_logits, fake_logits):
    """BCE real -> 1 plus BCE fake -> 0 (BasicSR backpropagates the two terms separately; same gradients)."""
    return F.softplus(-real_logits.float()).mean() + F.softplus(fake_logits.float()).mean()


def adaptive_scale(reference_loss, adversarial_loss, image, weight, max_scale):
    """Weight that gives the adversarial gradient on ``image`` ``weight`` times the norm of
    ``reference_loss``'s gradient there (VQGAN-style adaptive weight, measured on the decoded
    image instead of the last decoder layer). Detached; graphs are kept for the main backward."""
    reference, = torch.autograd.grad(reference_loss, image, retain_graph=True)
    adversarial, = torch.autograd.grad(adversarial_loss, image, retain_graph=True)
    scale = weight * reference.float().norm() / (adversarial.float().norm() + 1e-12)
    return scale.clamp(max=max_scale).detach()
