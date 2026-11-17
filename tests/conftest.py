"""Keep tiny CPU regressions and their subprocesses cheap and deterministic."""
import os

import pytest


def pytest_sessionstart(session):
    threads = int(os.environ.get('VOSR_TEST_THREADS', '1'))
    if threads < 1:
        raise pytest.UsageError('VOSR_TEST_THREADS must be a positive integer')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = str(threads)
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
    import torch
    torch.set_num_threads(threads)
