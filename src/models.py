"""Model definitions for MLP experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TinyMLP:
    input_dim: int
    hidden_dim: int
    output_dim: int

    def __post_init__(self):
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.01
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.01
        self.b2 = np.zeros((1, self.output_dim))

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ self.W1 + self.b1)
        return h @ self.W2 + self.b2
