"""
tests/test_data.py
------------------
Unit tests for data loading and transform pipelines.
"""

import torch
from fedderm.data import get_dataloaders, get_class_names


def test_class_names() -> None:
    names = get_class_names()
    assert len(names) == 7
    assert "melanoma" in names


def test_dataloaders_shapes() -> None:
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root="data",
        image_size=28,
        batch_size=16,
        num_workers=0,
    )
    images, targets = next(iter(train_loader))
    assert images.shape == (16, 3, 28, 28)
    assert targets.shape[0] == 16
