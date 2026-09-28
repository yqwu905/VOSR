"""TXT data contract tests using the actual upstream HQ dataset."""
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
