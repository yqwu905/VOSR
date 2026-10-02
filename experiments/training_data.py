"""Source-balanced, deterministic sampling of explicitly declared research data."""
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from data_pipeline.augmentation import degrade, sample_seed


class ResearchDataset(Dataset):
    def __init__(self, config, length, seed, start=0):
        self.config, self.length, self.seed, self.start = config, length, seed, start
        if config.get('use_policy') != 'research_ablation_unreviewed':
            raise ValueError('Research candidates require an explicit research use policy')
        if config['split'] != 'train':
            raise ValueError('Training must use the train split')
        groups = defaultdict(lambda: defaultdict(list))
        mode = config.get('quality_filter', 'none')
        if mode not in ('none', 'confidence', 'fidelity'):
            raise ValueError('Unknown quality filter')
        for line in Path(config['manifest']).read_text().splitlines():
            row = json.loads(line)
            if row['split'] != config['split'] or row['crop_mode'] != config['crop_mode']:
                continue
            if row['target'] != config['target'] or row['use_policy'] != config['use_policy']:
                raise ValueError('Data target/use policy does not match the experiment')
            if mode != 'none':
                evidence = row.get('quality')
                if evidence is None:
                    raise ValueError('Quality filtering requested but evidence is absent')
                if evidence['min_confidence'] < config.get('ocr_min_confidence', 0.9):
                    continue
                if mode == 'fidelity' and not evidence['reference_match']:
                    continue
            if not Path(row['hr_path']).is_file():
                raise FileNotFoundError(row['hr_path'])
            groups[row['source']][row['group_id']].append(row)
        self.sources = [name for name, weight in config['source_weights'].items() if weight > 0]
        self.weights = [float(config['source_weights'][name]) for name in self.sources]
        if not self.sources or any(weight < 0 for weight in config['source_weights'].values()):
            raise ValueError('Source weights must be nonnegative and not all zero')
        keep = float(config.get('iqa_keep_fraction', 1.0))
        if not 0 < keep <= 1:
            raise ValueError('iqa_keep_fraction must be in (0, 1]')
        thresholds = {}
        if keep < 1:
            # Fit source-specific quantiles on training candidates after OCR
            # selection; held-out examples never influence these thresholds.
            metric = config.get('iqa_metric', 'musiq')
            for source in self.sources:
                records = [r for group in groups[source].values() for r in group]
                try:
                    values = [r['quality']['iqa'][metric] for r in records]
                except KeyError as error:
                    raise ValueError('IQA filtering requested but scores are absent') from error
                if not values:
                    raise ValueError('No candidates for IQA quantile selection')
                thresholds[source] = float(np.quantile(values, 1-keep))
                selected = defaultdict(list)
                for row, value in zip(records, values):
                    if value >= thresholds[source]:
                        selected[row['group_id']].append(row)
                groups[source] = selected
        self.groups = {name: list(groups[name].values()) for name in self.sources}
        if any(not group for group in self.groups.values()):
            raise ValueError('A requested training source is empty after filtering')
        self.summary = {name: dict(probability=weight/sum(self.weights), groups=len(self.groups[name]),
                                  crops=sum(len(g) for g in self.groups[name]),
                                  iqa_threshold=thresholds.get(name))
                        for name, weight in zip(self.sources, self.weights)}

    def __len__(self):
        return self.length - self.start

    def __getitem__(self, index):
        index += self.start
        rng = random.Random(sample_seed(self.seed, str(index), 'choose_sample'))
        source = rng.choices(self.sources, weights=self.weights)[0]
        group = rng.choice(self.groups[source])
        record = rng.choice(group)
        with Image.open(record['hr_path']) as image:
            hq = image.convert('RGB')
        if hq.size != (self.config['resolution'],) * 2:
            raise ValueError('Preprocessed training crop size changed')
        lq, _, _ = degrade(hq, [], self.config['degradation'],
                           sample_seed(self.seed, str(index), 'degradation'))
        lq = lq.resize(hq.size, Image.Resampling.BICUBIC)
        def tensor(image):
            return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 127.5 - 1
        return dict(hq=tensor(hq), lq=tensor(lq), source=source, id=record['id'], sample_index=index)
