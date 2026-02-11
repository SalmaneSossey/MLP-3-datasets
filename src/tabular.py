"""Helpers for tabular datasets."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TabularDataset:
    X: np.ndarray
    y: np.ndarray


def train_test_split(dataset: TabularDataset, test_ratio: float = 0.2):
    """Simple NumPy-based split."""
    n = len(dataset.X)
    split = int(n * (1 - test_ratio))
    return (
        TabularDataset(dataset.X[:split], dataset.y[:split]),
        TabularDataset(dataset.X[split:], dataset.y[split:]),
    )
