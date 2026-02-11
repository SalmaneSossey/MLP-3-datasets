"""Training entrypoint placeholder."""

import numpy as np

from models import TinyMLP
from seed import set_seed


def main() -> None:
    set_seed(42)
    X = np.random.randn(64, 10)
    model = TinyMLP(input_dim=10, hidden_dim=16, output_dim=3)
    logits = model.forward(X)
    print("Logits shape:", logits.shape)


if __name__ == "__main__":
    main()
