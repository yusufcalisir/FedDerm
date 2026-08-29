"""
fedderm/utils/seed.py
---------------------
Global reproducibility seeding for PyTorch, NumPy, and Python random.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU) for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic convolutions (slight speed penalty, but ensures reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
