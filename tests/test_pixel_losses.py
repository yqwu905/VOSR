import json
import pytest
import torch
from models.ocr_recognizers import CRNN, PPOCRv5ServerRec, load_ppocr, ocr_spec
from models.pixel_losses import KLOCRLoss, build_ocr_kl, text_strips


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


def test_crnn_keeps_crnn_pytorch_parameter_names():
    keys = set(CRNN().state_dict())
    assert {'cnn.conv0.weight', 'cnn.conv6.bias', 'cnn.batchnorm4.running_mean', 'rnn.0.rnn.weight_ih_l0_reverse',
            'rnn.1.embedding.weight'} <= keys
    assert {k.split('.')[1] for k in keys if 'batchnorm' in k} == {'batchnorm2', 'batchnorm4', 'batchnorm6'}
    assert CRNN().eval()(torch.zeros(2, 1, 32, 100)).shape == (2, 26, 37)  # crnn.pytorch: 26 frames at width 100


def test_ocr_spec_validation():
    spec = ocr_spec(None)
    assert spec['type'] == 'ppocr' and spec['strip_height'] == 64 and spec['strip_stride'] == 32
    with pytest.raises(ValueError, match='ocr.type'):
        ocr_spec({'type': 'tesseract'})
    with pytest.raises(ValueError, match='input_height'):
        ocr_spec({'type': 'ppocr', 'input_height': 32})  # PP-OCRv5 lines are always 48 px
    with pytest.raises(ValueError, match='strip_stride'):
        ocr_spec({'strip_stride': 0})
    with pytest.raises(ValueError, match='temperature'):
        ocr_spec({'temperature': 0})


def test_kl_ocr_with_basicsr_crnn(tmp_path):
    pytest.importorskip('basicsr')
    torch.manual_seed(0)
    crnn = CRNN(in_channels=1, num_classes=5, hidden_size=8)
    with torch.no_grad():  # random tiny CRNNs are almost input-independent; amplify features and logits
        for module in crnn.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_var.fill_(1e-3)
        crnn.rnn[1].embedding.weight.mul_(100)
    torch.save({'params': crnn.state_dict()}, tmp_path / 'crnn.pth')
    with pytest.raises(ValueError, match='checkpoint'):
        build_ocr_kl({'type': 'basicsr'})
    loss = build_ocr_kl({'type': 'basicsr', 'checkpoint': str(tmp_path / 'crnn.pth'),
                         'network': {'type': 'CRNN', 'in_channels': 1, 'num_classes': 5, 'hidden_size': 8},
                         'strip_height': 16, 'strip_stride': 8, 'temperature': 2.0})
    assert isinstance(loss, KLOCRLoss) and loss.input_height == 32 and loss.recognizer.channels == 1
    assert not loss.recognizer.training and not any(p.requires_grad for p in loss.parameters())
    hq = torch.rand(2, 3, 40, 48) * 2 - 1
    assert loss(hq, hq).item() == pytest.approx(0, abs=1e-6)
    prediction = (hq + 0.5 * torch.randn_like(hq)).requires_grad_(True)
    value = loss(prediction, hq)
    assert value.item() > 1e-4
    value.backward()
    assert prediction.grad.abs().sum() > 0
