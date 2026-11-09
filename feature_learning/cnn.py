"""CNN feature learner: a 1D CNN maps a signal (2, 256) to n_features numbers.
As the old Simple_CNN_1D (src/architectures/features_extractors/cnn.py). Trained by an ML approach (ml_approach/)."""

import torch
from torch import nn


def block(c_in, c_out):
    """conv -> ReLU -> BatchNorm -> halve the length -> conv -> ReLU -> BatchNorm (the old order)."""
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=3), nn.ReLU(), nn.BatchNorm1d(c_out),
        nn.MaxPool1d(2),
        nn.Conv1d(c_out, c_out, kernel_size=3), nn.ReLU(), nn.BatchNorm1d(c_out),
    )


class CNNEncoder(nn.Module):
    """4 blocks (16, 32, 64, 128 channels), then one linear layer to n_features.
    Per signal: (2, 256) -> (16, 125) -> (32, 59) -> (64, 26) -> (128, 10) -> 1280 -> n_features."""

    def __init__(self, n_features, channels=(16, 32, 64, 128), length=256):
        super().__init__()
        layers, c_in = [], 2                                  # 2 input channels: I and Q
        for c in channels:
            layers.append(block(c_in, c))
            c_in = c
            length = (length - 2) // 2 - 2                    # conv: -2, pool: /2, conv: -2
        self.blocks = nn.Sequential(*layers)
        self.linear = nn.Linear(c_in * length, n_features)    # 128 x 10 = 1280 -> n_features

    def forward(self, x):                                     # torch (B, 2, 256) -> (B, n_features)
        return self.linear(self.blocks(x).flatten(1))

    @torch.no_grad()
    def encode(self, x, batch_size=1024):
        """Same as KANEncoder.encode: numpy (N, 2, 256) -> numpy (N, n_features). Here eval() matters:
        BatchNorm then uses its running averages, so each signal's features don't depend on the others."""
        self.eval()
        device = next(self.parameters()).device
        out = [self(torch.from_numpy(x[i:i + batch_size]).to(device)).cpu()
               for i in range(0, len(x), batch_size)]
        return torch.cat(out).numpy()


if __name__ == "__main__":
    # Quick check on 8 fake signals: shape and number of learnable numbers.
    import numpy as np
    x = np.random.default_rng(0).uniform(-1, 1, size=(8, 2, 256)).astype(np.float32)
    encoder = CNNEncoder(n_features=20)
    print("features:", encoder.encode(x).shape)
    print("parameters:", sum(p.numel() for p in encoder.parameters()))