"""Compare real two-rank DDP budget gradients with a global-batch reference."""
import copy
from contextlib import nullcontext
from datetime import timedelta
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.sdt_router import mlp_budget_loss


def check_distributed_budget():
    dist.init_process_group('gloo', timeout=timedelta(seconds=30))
    try:
        rank = dist.get_rank()
        linear = nn.Linear(1, 2)  # two equal-cost layers, one image per rank
        with torch.no_grad():
            linear.weight.fill_(1.)
            linear.bias.zero_()
        model = DDP(linear)
        balanced = torch.tensor([[math.log(.55 / .45)], [math.log(.95 / .05)]])
        unbalanced = torch.tensor([[-1.], [.5]])
        # The first case must have zero budget and gradient, even though each
        # rank separately misses .75. The others detect missing autograd or
        # incorrect world-size scaling, also alongside task loss / no_sync().
        cases = [
            ('global', [balanced], 0.),
            ('global', [unbalanced], 0.),
            ('global', [balanced, unbalanced], .2),
            ('layer', [balanced], 0.),
        ]
        for scope, batches, task_weight in cases:
            reference = copy.deepcopy(model.module)
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            for micro, batch in enumerate(batches):
                sync = model.no_sync() if micro < len(batches) - 1 else nullcontext()
                with sync:
                    p = model(batch[rank:rank + 1]).sigmoid()
                    budget = mlp_budget_loss(p.mean(0), .75, scope)
                    loss = budget + task_weight * p.square().mean()
                    (loss / len(batches)).backward()
                global_p = reference(batch).sigmoid()
                expected_budget = ((global_p.mean() - .75).square() if scope == 'global'
                                   else (global_p - .75).square().mean())
                expected_loss = expected_budget + task_weight * global_p.square().mean()
                (expected_loss / len(batches)).backward()
                if scope == 'global':
                    torch.testing.assert_close(budget, expected_budget)
                    if batch is balanced:
                        assert budget.item() < 1e-12
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('VOSR_TEST_DDP') != '1',
                    reason='Set VOSR_TEST_DDP=1 on a host that permits Gloo sockets')
def test_global_budget_matches_concatenated_batch_loss_and_gradients():
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=2', str(Path(__file__).resolve())]
    result = subprocess.run(command, text=True, capture_output=True, timeout=90,
                            env=dict(os.environ, OMP_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == '__main__':
    check_distributed_budget()
