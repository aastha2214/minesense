"""Reproducibility helpers."""
import os
import random

import numpy as np
import torch

from .config import GLOBAL_SEED


def set_seed(seed: int = GLOBAL_SEED) -> np.random.Generator:
    """Seed Python, NumPy and PyTorch (CPU and MPS) and return a NumPy RNG."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the MPS generator
    torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)
