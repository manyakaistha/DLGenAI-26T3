import torch.nn as nn


def build_model():
    """Replace this body with YOUR architecture (same layers as in training).
    Output must be 4 channels: (batch, 4, 128, 800) raw logits."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 4, 1),
    )
