"""Seeding + deterministic ops (GOAL Phase 4). Seeds python/numpy/torch/cuda and captures /
restores RNG state so a resumed checkpoint continues the same stream.

Determinism is opt-in (`deterministic=True`) because fully-deterministic CUDA kernels can be
slower; `warn_only` keeps ops that lack a deterministic implementation from crashing training.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs. With ``deterministic`` also request deterministic torch algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for det. cuBLAS
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


@dataclass
class RngState:
    python: Any
    numpy: Any
    torch: Any
    cuda: Any


def capture_rng_state() -> RngState:
    """Snapshot RNG state (for checkpointing mid-training)."""
    import torch

    try:
        import numpy as np

        np_state = np.random.get_state()
    except ImportError:
        np_state = None
    return RngState(
        python=random.getstate(),
        numpy=np_state,
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng_state(state: RngState) -> None:
    """Restore a snapshot from :func:`capture_rng_state` (on resume)."""
    import torch

    random.setstate(state.python)
    if state.numpy is not None:
        import numpy as np

        np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch)
    if state.cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.cuda)
