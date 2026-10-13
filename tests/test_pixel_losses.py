import json
import pytest
import torch
from torch import nn
from models.ocr_recognizers import PPOCRv5ServerRec, load_ppocr, ocr_spec
from models.pixel_losses import KLOCRLoss, text_strips


def test_text_strips_cover_every_row_with_overlap():
    rows = torch.arange(100.).view(1, 1, 100, 1).expand(2, 3, 100, 5)
    strips = text_strips(rows, 64, 32)
    assert strips.shape == (6, 3, 64, 5)  # starts 0, 32 and the bottom-aligned 36, for both images
    assert strips[:, 0, :, 0].unique().tolist() == list(range(100))
    assert text_strips(rows, 128, 32).shape == (2, 3, 100, 5)


def test_ppocr_port_layout_matches_official_safetensors():
    model = PPOCRv5ServerRec(num_classes=11).eval()
    keys = [k for k in model.state_dict() if not k.endswith('num_batches_tracked')]
    # PaddlePaddle/PP-OCRv5_server_rec_safetensors holds exactly these 458 tensors.
    assert len(keys) == 458
    for key in ('model.backbone.embedder.stem2a.convolution.weight',
                'model.backbone.encoder.stages.2.blocks.1.layers.5.conv2.normalization.running_var',
                'model.backbone.encoder.stages.3.blocks.0.aggregation.0.convolution.weight',
                'head.encoder.conv_block.3.convolution.weight', 'head.encoder.svtr_block.1.self_attn.qkv.bias',
                'head.head.weight'):
        assert key in keys
    assert model.state_dict()['model.backbone.encoder.stages.3.blocks.0.aggregation.0.convolution.weight'].shape == (1024, 3328, 1, 1)
    model.requires_grad_(False)
    x = torch.rand(2, 3, 48, 64, requires_grad=True)
    logits = model(x * 2 - 1)
    assert logits.shape == (2, 8, 11)
    logits.sum().backward()
    assert x.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='48-pixel-high'):
        model(torch.zeros(1, 3, 32, 64))


def test_load_ppocr_is_strict(tmp_path):
    from safetensors.torch import save_file
    torch.manual_seed(0)
    source = PPOCRv5ServerRec(num_classes=7).eval()
    state = {k: v for k, v in source.state_dict().items() if not k.endswith('num_batches_tracked')}
    save_file(state, str(tmp_path / 'model.safetensors'))
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'pp_ocrv5_server_rec'}))
    loaded = load_ppocr(str(tmp_path)).eval()
    x = torch.rand(1, 3, 48, 32) * 2 - 1
    with torch.no_grad():
        assert torch.equal(loaded(x), source(x))
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'pp_ocrv5_mobile_rec'}))
    with pytest.raises(ValueError, match='PP-OCRv5_server_rec'):
        load_ppocr(str(tmp_path))
    del state['head.encoder.norm.weight']
    save_file(state, str(tmp_path / 'model.safetensors'))
    with pytest.raises(ValueError, match='missing'):
        load_ppocr(str(tmp_path / 'model.safetensors'))
    with pytest.raises(FileNotFoundError):
        load_ppocr(str(tmp_path / 'absent' / 'model.safetensors'))


def test_ocr_spec_validation():
    spec = ocr_spec(None)
    assert spec['type'] == 'ppocr' and spec['strip_height'] == 64 and spec['strip_stride'] == 32
    with pytest.raises(ValueError, match='ocr.type'):
        ocr_spec({'type': 'basicsr'})
    with pytest.raises(ValueError, match='input_height'):
        ocr_spec({'type': 'ppocr', 'input_height': 32})  # PP-OCRv5 lines are always 48 px
    with pytest.raises(ValueError, match='strip_stride'):
        ocr_spec({'strip_stride': 0})
    with pytest.raises(ValueError, match='temperature'):
        ocr_spec({'temperature': 0})


class TinyRecognizer(nn.Module):
    """(N, 3, 16, W) -> (N, W // 4, 5) logits; a stand-in for the 84 MB PP-OCR weights."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, (16, 4), stride=(16, 4))

    def forward(self, x):
        return self.conv(x).squeeze(2).transpose(1, 2) * 10


def test_kl_ocr_loss_zero_positive_and_gradients():
    torch.manual_seed(0)
    loss = KLOCRLoss(TinyRecognizer().eval().requires_grad_(False), 16, strip_height=16, strip_stride=8,
                     temperature=2.0)
    hq = torch.rand(2, 3, 40, 48) * 2 - 1
    assert loss(hq, hq).item() == pytest.approx(0, abs=1e-6)
    prediction = (hq + 0.5 * torch.randn_like(hq)).requires_grad_(True)
    value = loss(prediction, hq)
    assert value.item() > 1e-4
    value.backward()
    assert prediction.grad.abs().sum() > 0
    assert all(p.grad is None for p in loss.parameters())


class FixedRecognizer(nn.Module):
    """Returns preset logits for each strip, ignoring the pixels."""
    def __init__(self, logits):
        super().__init__()
        self.preset = logits

    def forward(self, x):
        return self.preset[:x.shape[0]]


def test_text_mask_maps_frames_to_columns_and_strips_to_rows():
    # A 40x48 image cut into 16-row strips with stride 8: starts 0, 8, 16, 24 (bottom-aligned).
    # Each strip is resized to 16 x 48 and read as 12 frames of 4 columns each.
    strips, frames = 4, 12
    logits = torch.zeros(strips, frames, 3)
    logits[..., 0] = 10  # blank everywhere
    logits[1, 3, 1] = 12  # strip 1 (rows 8-23) reads class 1 at frame 3 (columns 12-15), p = 0.88
    logits[2, 6, 1:] = 11  # strip 2 is unsure between two classes at frame 6: p = 0.42 each
    loss = KLOCRLoss(FixedRecognizer(logits), 16, strip_height=16, strip_stride=8)
    target_logits = loss.logits(torch.zeros(1, 3, 40, 48))
    mask = loss.text_mask(target_logits, (1, 3, 40, 48), threshold=0.5, smooth_frames=1)
    assert mask.shape == (1, 1, 40, 48)
    rows = mask[0, 0].amax(1).nonzero().flatten().tolist()
    cols = mask[0, 0].amax(0).nonzero().flatten().tolist()
    assert rows == list(range(8, 24))  # all rows of the strip that fired
    assert cols[0] >= 11 and cols[-1] <= 16 and len(cols) >= 4
    # Max-pooling over 3 frames widens the text by one frame on each side.
    wide = loss.text_mask(target_logits, (1, 3, 40, 48), threshold=0.5, smooth_frames=3)
    assert wide.sum() > mask.sum() and torch.all(wide >= mask)
    assert loss.text_mask(target_logits, (1, 3, 40, 48), threshold=0.95, smooth_frames=1).sum() == 0
    # Lowering the threshold below 0.42 also takes in the unsure frame of strip 2 (rows 16-31).
    low = loss.text_mask(target_logits, (1, 3, 40, 48), threshold=0.4, smooth_frames=1)
    assert low[0, 0].amax(1).nonzero().flatten().tolist() == list(range(8, 32))


def test_kl_ocr_reuses_precomputed_target_logits():
    torch.manual_seed(0)
    loss = KLOCRLoss(TinyRecognizer().eval().requires_grad_(False), 16, strip_height=16, strip_stride=8)
    hq = torch.rand(2, 3, 40, 48) * 2 - 1
    prediction = hq + 0.3 * torch.randn_like(hq)
    with torch.no_grad():
        logits = loss.logits(hq)
    assert torch.equal(loss(prediction, hq), loss(prediction, hq, logits))


def test_spatial_lpips_map_averages_to_lpips_and_weight_removes_pixels(monkeypatch):
    pytest.importorskip('pyiqa')
    import torchvision
    from pyiqa.archs import lpips_arch
    from models.pixel_losses import lpips_loss
    # pyiqa's alexnet trunk always asks torchvision for ImageNet weights; random weights suffice here.
    alexnet = torchvision.models.alexnet
    monkeypatch.setattr(lpips_arch.models, 'alexnet', lambda weights=None: alexnet())
    torch.manual_seed(0)
    plain = lpips_arch.LPIPS(pretrained=False, net='alex', version='0.1').eval()
    for module in plain.modules():  # nonnegative linear heads, like the calibrated LPIPS weights
        if isinstance(module, nn.Conv2d) and module.out_channels == 1:
            module.weight.data.abs_()
    spatial = lpips_arch.LPIPS(pretrained=False, net='alex', version='0.1', spatial=True).eval()
    spatial.load_state_dict(plain.state_dict())
    a, b = torch.rand(2, 3, 64, 64) * 2 - 1, torch.rand(2, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        value = lpips_loss(plain, a, b)
        assert spatial(a, b, normalize=False).shape == (2, 1, 64, 64)
        torch.testing.assert_close(lpips_loss(spatial, a, b, torch.ones(2, 1, 64, 64)), value, rtol=0.05, atol=0)
        half = torch.ones(2, 1, 64, 64)
        half[..., :32, :] = 0
        assert lpips_loss(spatial, a, b, half) < value
        assert lpips_loss(spatial, a, b, torch.zeros(2, 1, 64, 64)) == 0
