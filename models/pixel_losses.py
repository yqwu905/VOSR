"""Optional pixel-space losses on the VAE-decoded student prediction: LPIPS and KL-OCR.

Both compare the decoded one-step prediction with the HQ crop, RGB in [-1, 1].
The networks are frozen and run in fp32 outside autocast; only the prediction
receives gradients.
"""
import torch
import torch.nn.functional as F
from torch import nn

from models.ocr_recognizers import build_ocr_recognizer, ocr_spec


def build_lpips(net='vgg', model_path=None):
    """pyiqa's LPIPS v0.1 network (pyiqa's own ``lpips``/``lpips-vgg`` metrics) as a frozen loss.

    The architecture is built directly: ``pyiqa.create_metric`` would also switch
    cuDNN to deterministic, non-benchmark mode for the whole training process.
    """
    if net not in ('vgg', 'alex'):
        raise ValueError('training.lpips_net must be vgg or alex')
    from pyiqa.archs.lpips_arch import LPIPS
    return LPIPS(net=net, version='0.1', pretrained_model_path=model_path).eval().requires_grad_(False)


def lpips_loss(network, prediction, target):
    with torch.autocast(prediction.device.type, enabled=False):
        return network(prediction.float(), target.float(), normalize=False).mean()


def text_strips(images, height, stride):
    """Full-width horizontal strips with overlap, so any text line up to ``height - stride`` px fits one strip."""
    full = images.shape[-2]
    height = min(height, full)
    starts = list(range(0, full - height + 1, stride))
    if starts[-1] + height < full:
        starts.append(full - height)
    return torch.cat([images[..., start:start + height, :] for start in starts])


class KLOCRLoss(nn.Module):
    """KL(OCR(HQ) || OCR(prediction)) over the recognizer's per-frame CTC distributions.

    The training crops carry no text boxes, so both images are cut into the same
    overlapping full-width strips and each strip is resized to the recognizer's
    line height. The KL is summed over classes and averaged over all frames,
    scaled by temperature ** 2.
    """
    def __init__(self, recognizer, input_height, strip_height=64, strip_stride=32, temperature=1.0):
        super().__init__()
        self.recognizer = recognizer
        self.input_height = int(input_height)
        self.strip_height = int(strip_height)
        self.strip_stride = int(strip_stride)
        self.temperature = float(temperature)

    def logits(self, images):
        strips = text_strips(images.float(), self.strip_height, self.strip_stride)
        width = max(8, round(strips.shape[-1] * self.input_height / strips.shape[-2] / 8) * 8)
        strips = F.interpolate(strips, size=(self.input_height, width), mode='bilinear', align_corners=False)
        return self.recognizer(strips).float() / self.temperature

    def forward(self, prediction, target):
        with torch.autocast(prediction.device.type, enabled=False):
            with torch.no_grad():
                target_log = F.log_softmax(self.logits(target), -1)
            prediction_log = F.log_softmax(self.logits(prediction), -1)
            frames = prediction_log.shape[0] * prediction_log.shape[1]
            kl = F.kl_div(prediction_log, target_log, reduction='sum', log_target=True) / frames
        return kl * self.temperature ** 2


def build_ocr_kl(config):
    """``config`` is the top-level ``ocr`` mapping; see models/ocr_recognizers.py."""
    spec = ocr_spec(config)
    recognizer, height = build_ocr_recognizer(spec)
    return KLOCRLoss(recognizer, height, spec['strip_height'], spec['strip_stride'], spec['temperature'])
