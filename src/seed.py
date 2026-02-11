"""Utilities to enforce reproducibility."""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
