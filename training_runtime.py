"""Small backend and checkpoint helpers shared by FFN experiments.

No global CUDA-to-NPU monkeypatching is needed by these helpers. RNG payloads
contain only tensors and Python primitives and support torch.load(weights_only=True).
"""
import importlib.util
import os
from pathlib import Path
import random

import numpy as np
import torch


def select_device(requested='auto', local_rank=0):
    if requested is None:
        requested = 'auto'
    if requested == 'auto' or str(requested).startswith('npu'):
        if not hasattr(torch, 'npu') and importlib.util.find_spec('torch_npu') is not None:
            import torch_npu  # noqa: F401
    if requested == 'auto':
        requested = ('cuda' if torch.cuda.is_available() else
                     'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu')
    device = torch.device(requested)
    if int(os.environ.get('WORLD_SIZE', '1')) > 1 and device.index is not None:
        raise ValueError('Distributed launches require auto/cuda/npu without an index; LOCAL_RANK selects the device')
    if device.type not in ('cpu', 'cuda', 'npu'):
        raise ValueError('Training device must be cpu, cuda, npu or auto')
    if device.type != 'cpu':
        device = torch.device(device.type, local_rank if device.index is None else device.index)
        backend = getattr(torch, device.type)
        if not backend.is_available():
            raise RuntimeError(f'Requested {device.type} backend is not available')
        backend.set_device(device)
    if device.type == 'npu':
        import torch._dynamo as dynamo
        dynamo.config.disable = True
    return device


def distributed_backend(device):
    return {'cpu': 'gloo', 'cuda': 'nccl', 'npu': 'hccl'}[device.type]


def capture_rng(device):
    numpy_state = np.random.get_state()
    state = dict(python=random.getstate(), torch=torch.get_rng_state(),
                 numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]])
    if device.type != 'cpu':
        state['accelerator'] = getattr(torch, device.type).get_rng_state(device).cpu()
    return state


def restore_rng(state, device):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if 'accelerator' in state and device.type != 'cpu':
        getattr(torch, device.type).set_rng_state(state['accelerator'].cpu(), device)


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_checkpoint_complete(state, directory):
    """New trainer checkpoints require their last-written completion marker.

    Historical states without per-rank RNG remain loadable; they predate this
    marker protocol and do not promise bitwise data/noise continuation.
    """
    if 'rng_by_rank' not in state:
        return
    marker = Path(directory) / 'checkpoint.complete'
    if not marker.is_file() or marker.read_text(encoding='utf-8').strip() != str(int(state['step'])):
        raise ValueError('Incomplete checkpoint: checkpoint.complete is missing or disagrees with training_state.pt')


def rank_zero_action(action, rank=0, world=1, description='Checkpoint write'):
    """Propagate rank-zero I/O failures before other ranks enter their next collective."""
    error = [None]
    if rank == 0:
        try:
            action()
        except Exception as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    if world > 1:
        import torch.distributed as dist
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f'{description} failed: {error[0]}')
