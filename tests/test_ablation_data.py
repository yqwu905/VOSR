"""TXT data contract tests using the actual upstream HQ dataset."""
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ablation_utils import build_txt_dataset, prepare_training_batch
from dataloaders.realsr_dataset import TxtPairDataset


def write_config(tmp_path, content):
    path = tmp_path / 'datasets.txt'
    path.write_text(content, encoding='utf-8')
    return dict(train_dataset_config=str(path), dataset_type='txt', resolution=16)


def test_upstream_txt_lists_repeats_and_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'lists').mkdir()
    (tmp_path / 'images').mkdir()
    Image.new('RGB', (24, 20), (255, 0, 0)).save('images/red image.png')
    Image.new('L', (8, 10), 255).save('images/white.png')
    (tmp_path / 'lists/a.txt').write_text('images/red image.png\n', encoding='utf-8')
    (tmp_path / 'lists/b.txt').write_text('images/white.png\n', encoding='utf-8')
    config = write_config(tmp_path / 'lists', '\n# datasets\nlists/a.txt, 2\nlists/b.txt\nlists/a.txt, 0\n')
    dataset = build_txt_dataset(config)
    assert isinstance(dataset, TxtPairDataset)
    assert len(dataset) == 3
    # Image paths resolve from cwd, not from the lists/ directory.
    batch = next(iter(DataLoader(dataset, batch_size=3)))
    assert set(batch) == {'hq'}
    assert batch['hq'].shape == (3, 3, 16, 16)
    assert batch['hq'].dtype == torch.float32
    red = torch.tensor([1., 0., 0.])[:, None, None].expand(3, 16, 16)
    torch.testing.assert_close(batch['hq'][0], red)
    torch.testing.assert_close(batch['hq'][1], red)
    torch.testing.assert_close(batch['hq'][2], torch.ones(3, 16, 16))


@pytest.mark.parametrize('row', ['images.txt, -1', 'images.txt, 0.5',
                                'images.txt, nope', 'images.txt, 1, 2', ', 1'])
def test_invalid_dataset_rows_report_line(tmp_path, row):
    config = write_config(tmp_path, '# header\n' + row + '\n')
    with pytest.raises(ValueError, match=r'datasets.txt:2: expected'):
        build_txt_dataset(config)


def test_missing_image_list_reports_path(tmp_path):
    config = write_config(tmp_path, str(tmp_path / 'missing.txt') + ', 1\n')
    with pytest.raises(FileNotFoundError, match='HQ image-list TXT not found'):
        build_txt_dataset(config)


@pytest.mark.parametrize('kind', ['no_lists', 'empty_list', 'zero_repeats'])
def test_empty_training_dataset_rejected(tmp_path, kind):
    image_list = tmp_path / 'images.txt'
    image_list.write_text('', encoding='utf-8')
    content = '# no lists\n' if kind == 'no_lists' else f'{image_list}, {0 if kind == "zero_repeats" else 1}\n'
    with pytest.raises(ValueError, match='empty training dataset'):
        build_txt_dataset(write_config(tmp_path, content))


@pytest.mark.parametrize('config, message', [
    ({'manifest': 'pairs.jsonl'}, 'data.train_dataset_config'),
    ({'dataset_type': 'webdataset'}, 'data.dataset_type: txt'),
    ({'resolution': 0}, 'resolution must be positive'),
])
def test_unsupported_data_config_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        build_txt_dataset(config)


def test_online_degradation_receives_hq_before_normalization():
    hq = torch.tensor([0., 0.25, 0.75, 1.]).reshape(1, 1, 2, 2).repeat(2, 3, 1, 1)

    class DeterministicDegradation:
        def degrade_process(self, pixels, resize_bak=False):
            assert not torch.is_grad_enabled()
            assert resize_bak is True
            torch.testing.assert_close(pixels, hq)
            return pixels, torch.full_like(pixels, 0.25)

    gt, lq = prepare_training_batch({'hq': hq}, DeterministicDegradation(), torch.device('cpu'))
    torch.testing.assert_close(gt, hq * 2 - 1)
    torch.testing.assert_close(lq, torch.full_like(hq, -0.5))
    assert gt.shape == lq.shape == (2, 3, 2, 2)
    assert not gt.requires_grad and not lq.requires_grad


def paired_folders(tmp_path, sizes=(('a', 40, 48), ('b', 20, 24))):
    """HQ images and LQ = 255 - HQ (so a crop pair is aligned when lq + hq == 1)."""
    import numpy as np
    rng = np.random.default_rng(0)
    for folder in ('LQ', 'HQ'):
        (tmp_path / folder).mkdir()
    for name, h, w in sizes:
        hq = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(hq).save(tmp_path / 'HQ' / f'{name}.png')
        Image.fromarray(255 - hq).save(tmp_path / 'LQ' / f'{name}.png')
    return dict(dataset_type='paired', lq_dir=str(tmp_path / 'LQ'), hq_dir=str(tmp_path / 'HQ'), upscale=4,
                resolution=32)


def test_paired_crops_are_aligned_and_small_images_edge_padded(tmp_path):
    from ablation_utils import build_train_dataset
    from dataloaders.paired_crops import PairedCropDataset
    dataset = build_train_dataset(paired_folders(tmp_path))
    assert isinstance(dataset, PairedCropDataset) and dataset.names == ['a', 'b'] and len(dataset) == 2
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert set(batch) == {'hq', 'lq'} and batch['hq'].shape == (2, 3, 32, 32) and batch['hq'].dtype == torch.float32
    torch.testing.assert_close(batch['lq'] + batch['hq'], torch.ones_like(batch['hq']))
    small = dataset[1]  # 20x24 image, edge-padded to the crop
    torch.testing.assert_close(small['hq'][:, 20:], small['hq'][:, 19:20].expand(-1, 12, -1))
    torch.testing.assert_close(small['hq'][:, :, 24:], small['hq'][:, :, 23:24].expand(-1, -1, 8))


def test_paired_text_crops_centre_on_a_box(tmp_path):
    import json
    import random
    from ablation_utils import build_train_dataset
    config = paired_folders(tmp_path, sizes=(('a', 200, 300),))
    box = [200, 140, 240, 140, 240, 160, 200, 160]  # centre (220, 150)
    (tmp_path / 'boxes.json').write_text(json.dumps({'a': [box]}))
    dataset = build_train_dataset(dict(config, text_boxes=str(tmp_path / 'boxes.json'), text_crop_prob=1.0,
                                       resolution=64))
    full = torch.from_numpy(np.asarray(Image.open(tmp_path / 'HQ/a.png'))).permute(2, 0, 1).float() / 255
    for seed in range(20):
        crop = dataset.sample('a', random.Random(seed))['hq']
        matches = [(y, x) for y in range(200 - 63) for x in range(300 - 63)
                   if torch.equal(full[:, y:y + 64, x:x + 64], crop)]
        (y, x), = matches
        # Centred on the box up to a quarter-crop jitter (16 px), clipped to the image.
        assert abs(x + 32 - 220) <= 16 and abs(y + 32 - 150) <= 16


@pytest.mark.parametrize('change, error, message', [
    ({'dataset_type': 'lmdb'}, ValueError, 'data.dataset_type must be one of'),
    ({'upscale': None}, ValueError, 'data.upscale must be a positive integer'),
    ({'lq_dir': 'missing'}, FileNotFoundError, 'data.lq_dir is not a directory'),
    ({'text_crop_prob': .5}, ValueError, 'text_crop_prob needs data.text_boxes'),
    ({'text_crop_prob': 2.}, ValueError, r'text_crop_prob must be in \[0, 1\]'),
])
def test_paired_config_errors(tmp_path, change, error, message):
    from ablation_utils import build_train_dataset
    config = paired_folders(tmp_path)
    with pytest.raises(error, match=message):
        build_train_dataset(dict(config, **change))


def test_paired_pairs_need_an_lq_of_the_same_size(tmp_path):
    from ablation_utils import build_train_dataset
    config = paired_folders(tmp_path)
    (tmp_path / 'LQ/b.png').unlink()
    with pytest.raises(FileNotFoundError, match=r"no LQ image for \['b'\]"):
        build_train_dataset(config)
    Image.new('RGB', (10, 10)).save(tmp_path / 'LQ/b.png')
    with pytest.raises(ValueError, match='sizes differ'):
        build_train_dataset(config)[1]


def test_paired_batches_and_previews_use_the_dataset_lq(tmp_path):
    from ablation_logging import build_preview_samples
    from ablation_utils import build_train_dataset

    class NoDegradation:
        def degrade_process(self, *args, **kwargs):
            raise AssertionError('paired data must not be degraded online')
    hq = torch.full((1, 3, 2, 2), .75)
    gt, lq = prepare_training_batch({'hq': hq, 'lq': torch.full_like(hq, .25)}, NoDegradation(), torch.device('cpu'))
    torch.testing.assert_close(gt, torch.full_like(hq, .5))
    torch.testing.assert_close(lq, torch.full_like(hq, -.5))
    dataset = build_train_dataset(paired_folders(tmp_path))
    (gt, lq), = build_preview_samples(dataset, NoDegradation(), torch.device('cpu'), 1, 3)
    torch.testing.assert_close(gt + lq, torch.zeros_like(gt), rtol=0, atol=1e-6)  # (2 hq - 1) + (2 (1 - hq) - 1)
