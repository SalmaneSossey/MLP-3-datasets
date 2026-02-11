"""Bridges image arrays to tabular-like features."""

import numpy as np


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten (N, H, W[, C]) images to (N, D)."""
    return images.reshape(images.shape[0], -1)
