"""Frozen, differentiable text recognizer for the KL-OCR training loss.

The recognizer maps RGB text-line images in [-1, 1] to per-frame CTC logits
``(N, T, classes)``; gradients flow to the input, never into the weights.

``ppocr`` (the only ``ocr.type``): a PyTorch port of PP-OCRv5_server_rec (PPHGNetV2-B4 backbone, SVTR
neck and CTC head), the recognizer evaluate.py runs through PaddleOCR. It loads
the official ``PaddlePaddle/PP-OCRv5_server_rec_safetensors`` weights; module
names follow the Hugging Face transformers port (Apache-2.0) so the file loads
strictly. PaddleOCR trains it on BGR lines and PaddleX's Paddle-inference path
(the one evaluate.py uses) feeds BGR, 48 px high, (x / 255 - 0.5) / 0.5, so the
port flips its RGB input to BGR.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

PPOCR_CHECKPOINT = 'PaddlePaddle/PP-OCRv5_server_rec_safetensors'
OCR_DEFAULTS = {'ppocr': dict(checkpoint=PPOCR_CHECKPOINT)}


class ConvBN(nn.Module):
    """Conv (no bias) + BatchNorm + optional activation; HF ``HGNetV2ConvLayer`` naming."""
    def __init__(self, cin, cout, kernel, stride=1, groups=1, act=None):
        super().__init__()
        kernel = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
        padding = ((kernel[0] - 1) // 2, (kernel[1] - 1) // 2)
        self.convolution = nn.Conv2d(cin, cout, kernel, stride, padding, groups=groups, bias=False)
        self.normalization = nn.BatchNorm2d(cout)
        self.act = act

    def forward(self, x):
        x = self.normalization(self.convolution(x))
        return self.act(x) if self.act is not None else x


class LightConvBN(nn.Module):
    def __init__(self, cin, cout, kernel):
        super().__init__()
        self.conv1 = ConvBN(cin, cout, 1)
        self.conv2 = ConvBN(cout, cout, kernel, groups=cout, act=F.relu)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class HGBlock(nn.Module):
    def __init__(self, cin, mid, cout, layers, kernel, residual, light):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList(LightConvBN(cin if i == 0 else mid, mid, kernel) if light else
                                    ConvBN(cin if i == 0 else mid, mid, kernel, act=F.relu) for i in range(layers))
        self.aggregation = nn.Sequential(ConvBN(cin + layers * mid, cout // 2, 1, act=F.relu),
                                         ConvBN(cout // 2, cout, 1, act=F.relu))

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        y = self.aggregation(torch.cat(outputs, 1))
        return y + x if self.residual else y


class HGStage(nn.Module):
    def __init__(self, cin, mid, cout, blocks, light, kernel, layers, stride):
        super().__init__()
        self.downsample = ConvBN(cin, cin, 3, stride, groups=cin)
        self.blocks = nn.ModuleList(HGBlock(cin if i == 0 else cout, mid, cout, layers, kernel, i > 0, light)
                                    for i in range(blocks))

    def forward(self, x):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class HGStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = ConvBN(3, 32, 3, 2, act=F.relu)
        self.stem2a = ConvBN(32, 16, 2, act=F.relu)
        self.stem2b = ConvBN(16, 32, 2, act=F.relu)
        self.stem3 = ConvBN(64, 32, 3, act=F.relu)
        self.stem4 = ConvBN(32, 48, 1, act=F.relu)
        self.pool = nn.MaxPool2d(2, 1, ceil_mode=True)

    def forward(self, x):
        x = F.pad(self.stem1(x), (0, 1, 0, 1))
        branch = self.stem2b(F.pad(self.stem2a(x), (0, 1, 0, 1)))
        return self.stem4(self.stem3(torch.cat((self.pool(x), branch), 1)))


class HGNetV2B4Rec(nn.Module):
    """PPHGNetV2-B4 with the text-recognition strides of PP-OCRv5_server_rec."""
    STAGES = ((48, 48, 128, 1, False, 3, 6, (2, 1)), (128, 96, 512, 1, False, 3, 6, (1, 2)),
              (512, 192, 1024, 3, True, 5, 6, (2, 1)), (1024, 384, 2048, 1, True, 5, 6, (2, 1)))

    def __init__(self):
        super().__init__()
        self.embedder = HGStem()
        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList(HGStage(*stage) for stage in self.STAGES)

    def forward(self, x):
        x = self.embedder(x)
        for stage in self.encoder.stages:
            x = stage(x)
        return x


class SVTRBlock(nn.Module):
    def __init__(self, dim=120, heads=8, mlp_ratio=2.0):
        super().__init__()
        self.heads = heads
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = nn.Module()
        self.self_attn.qkv = nn.Linear(dim, 3 * dim)
        self.self_attn.projection = nn.Linear(dim, dim)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.mlp.fc2 = nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.self_attn.qkv(self.layer_norm1(x)).reshape(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attention = (q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5).softmax(-1)
        x = x + self.self_attn.projection((attention @ v).transpose(1, 2).reshape(b, n, c))
        return x + self.mlp.fc2(F.silu(self.mlp.fc1(self.layer_norm2(x))))


class SVTRNeck(nn.Module):
    def __init__(self, cin=2048, dim=120, depth=2):
        super().__init__()
        self.conv_block = nn.ModuleList([ConvBN(cin, cin // 8, (1, 3), act=F.silu),
                                         ConvBN(cin // 8, dim, 1, act=F.silu),
                                         ConvBN(dim, cin, 1, act=F.silu),
                                         ConvBN(2 * cin, cin // 8, (1, 3), act=F.silu),
                                         ConvBN(cin // 8, dim, 1, act=F.silu)])
        self.svtr_block = nn.ModuleList(SVTRBlock(dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.conv_block[1](self.conv_block[0](x))
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        for block in self.svtr_block:
            x = block(x)
        x = self.norm(x).transpose(1, 2).reshape(b, c, h, w)
        x = self.conv_block[3](torch.cat((residual, self.conv_block[2](x)), 1))
        return self.conv_block[4](x).squeeze(2).transpose(1, 2)


class PPOCRv5ServerRec(nn.Module):
    """PP-OCRv5_server_rec CTC branch. Input (N, 3, 48, W) RGB in [-1, 1] (fed to the network as BGR);
    logits (N, W // 8, classes)."""
    input_height = 48

    def __init__(self, num_classes=18385):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = HGNetV2B4Rec()
        self.head = nn.Module()
        self.head.encoder = SVTRNeck()
        self.head.head = nn.Linear(120, num_classes)

    def forward(self, x):
        if x.shape[-2] != self.input_height:
            raise ValueError(f'PP-OCRv5_server_rec expects {self.input_height}-pixel-high text lines')
        x = F.avg_pool2d(self.model.backbone(x.flip(1)), (3, 2))
        return self.head.head(self.head.encoder(x))


def load_ppocr(checkpoint=PPOCR_CHECKPOINT):
    """Build the port from a local file/directory or a Hugging Face repo id; loading is strict."""
    from safetensors.torch import load_file
    path = Path(checkpoint)
    if not path.exists():
        if Path(checkpoint).is_absolute() or str(checkpoint).count('/') != 1:
            raise FileNotFoundError(f'PP-OCR checkpoint not found: {checkpoint}')
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(checkpoint, 'model.safetensors')).parent
        hf_hub_download(checkpoint, 'config.json')
    if path.is_dir():
        config = path / 'config.json'
        if config.is_file():
            model_type = json.loads(config.read_text(encoding='utf-8')).get('model_type')
            if model_type != 'pp_ocrv5_server_rec':
                raise ValueError(f'ocr.type ppocr ports PP-OCRv5_server_rec, not {model_type}')
        path = path / 'model.safetensors'
    state = load_file(str(path), device='cpu')
    if 'head.head.weight' not in state:
        raise ValueError(f'{path} is not a PP-OCRv5_server_rec safetensors file')
    model = PPOCRv5ServerRec(state['head.head.weight'].shape[0])
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith('num_batches_tracked')]
    if missing or unexpected:
        raise ValueError(f'PP-OCR weights do not match the port: missing {missing[:8]}, unexpected {unexpected[:8]}')
    return model


def ocr_spec(config):
    """Validate the top-level ``ocr`` mapping and fill in its type's defaults."""
    config = dict(config or {})
    kind = config.pop('type', 'ppocr')
    if kind not in OCR_DEFAULTS:
        raise ValueError(f'ocr.type must be one of {sorted(OCR_DEFAULTS)}, not {kind!r}')
    common = {'strip_height', 'strip_stride', 'temperature'}
    unknown = set(config) - set(OCR_DEFAULTS[kind]) - common
    if unknown:
        raise ValueError(f'Unknown ocr options for type {kind}: {sorted(unknown)}')
    spec = {**OCR_DEFAULTS[kind], 'strip_height': 64, 'strip_stride': 32, 'temperature': 1.0, **config}
    spec['type'] = kind
    for key in ('strip_height', 'strip_stride'):
        if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] <= 0:
            raise ValueError(f'ocr.{key} must be a positive integer')
    if not float(spec['temperature']) > 0:
        raise ValueError('ocr.temperature must be positive')
    return spec


def build_ocr_recognizer(spec):
    """Return a frozen, eval-mode recognizer and its input line height."""
    return load_ppocr(spec['checkpoint']).eval().requires_grad_(False), PPOCRv5ServerRec.input_height
