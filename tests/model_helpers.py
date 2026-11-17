"""Shared meta-device compute measurements for the fixed shipped VOSR2 architecture."""
from functools import lru_cache
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from ablation_utils import load_config
from models.attn_compression import attn_compression_spec
from models.lightningdit_ablation import AblationLightningDiT

ROOT = Path(__file__).resolve().parents[1]


def vosr2_macs(attn_compression=None):
    # Resolve head lists so equivalent raw/resolved configs share the measurement.
    spec = (attn_compression_spec(attn_compression, 36, 24)[0]
            if attn_compression is not None else None)
    return _vosr2_macs(json.dumps(spec, sort_keys=True))


@lru_cache(maxsize=8)
def _vosr2_macs(spec_json):
    cfg = load_config(ROOT / 'configs/ablations/base/base_vosr2.yml')
    model = dict(cfg['model'], auxiliary_time_cond=False, input_size=64)
    with torch.device('meta'):
        network = AblationLightningDiT(**model, attn_compression=json.loads(spec_json)).eval()
        x, t, r, z = torch.randn(1, 32, 64, 64), torch.ones(1), torch.zeros(1), [torch.randn(1, 1024, 1024)]
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        network(x, t, r, z)
    attention = sum(sum(ops.values()) for name, ops in counter.get_flop_counts().items()
                    if name.count('.') == 3 and name.endswith('.attn'))
    return counter.get_total_flops() / 2, attention / 2
