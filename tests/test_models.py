"""
tests/test_models.py
--------------------
Unit tests for neural network architectures.
"""

import torch
from fedderm.models import MiniCNN, build_model


def test_minicnn_forward() -> None:
    model = build_model(num_classes=7, dropout=0.2)
    x = torch.randn(4, 3, 28, 28)
    out = model(x)
    assert out.shape == (4, 7)
