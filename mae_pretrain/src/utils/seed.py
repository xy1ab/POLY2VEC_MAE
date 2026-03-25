"""Random seed helpers for reproducible training.

This module provides one entrypoint to initialize Python, NumPy, and PyTorch
random states in a consistent way across scripts and training processes.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set global random seed for Python, NumPy, and PyTorch.

    Args:
        seed: Global integer seed.
        deterministic: Whether to enable deterministic CUDA backend flags.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
