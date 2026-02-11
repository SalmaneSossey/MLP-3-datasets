"""Evaluation metrics."""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy from label arrays."""
    return float(np.mean(y_true == y_pred))
