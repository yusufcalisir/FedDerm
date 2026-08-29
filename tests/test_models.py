"""
tests/test_models.py
--------------------
Unit tests for neural network architectures.
"""

import torch
from fedderm.models import MiniCNN, build_model, build_vit_lora_model


def test_minicnn_forward() -> None:
    model = build_model(num_classes=7, dropout=0.2)
    x = torch.randn(4, 3, 28, 28)
    out = model(x)
    assert out.shape == (4, 7)


def test_vit_lora_forward() -> None:
    # Use pretrained=False for unit testing to avoid network calls
    model = build_vit_lora_model(
        model_name="vit_base_patch16_224",
        num_classes=7,
        rank=4,
        lora_alpha=8,
        pretrained=False,
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)

    # Check trainable parameter count is small (< 500k)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params < 500_000
