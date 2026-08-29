"""
fedderm/data/dermamnist.py
--------------------------
DermaMNIST dataset loading via the medmnist package.

Wraps the official medmnist.DermaMNIST class and returns standard
torch DataLoaders using the official train/val/test splits.
No custom splitting logic -- medmnist provides the split natively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import medmnist
from medmnist import DermaMNIST


# Per-channel mean and std computed from the DermaMNIST training set.
# Source: MedMNIST v2 paper (Yang et al. 2023).
_MEAN = (0.7635, 0.5461, 0.5705)
_STD  = (0.1409, 0.1526, 0.1692)


def get_transforms(image_size: int, augment: bool) -> transforms.Compose:
    """Return torchvision transform pipeline.

    Training pipeline applies light augmentation (random horizontal flip,
    colour jitter) that is appropriate for dermoscopy images.
    Validation/test pipeline is deterministic.
    """
    resize = [transforms.Resize((image_size, image_size))] if image_size != 28 else []
    normalize = transforms.Normalize(mean=_MEAN, std=_STD)

    if augment:
        return transforms.Compose([
            *resize,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        *resize,
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(
    data_root: str | Path,
    image_size: int = 28,
    batch_size: int = 128,
    num_workers: int = 0,
    balanced_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for DermaMNIST.

    Args:
        data_root:        Directory where medmnist will store / look for the npz file.
        image_size:       Spatial size to resize to (28 keeps native resolution).
        batch_size:       Mini-batch size for all loaders.
        num_workers:      DataLoader worker processes (0 = main process, safe on Windows).
        balanced_sampler: If True, use WeightedRandomSampler on the training loader so
                          minority classes appear more frequently per batch. Important
                          for GroupNorm models (which lack BatchNorm's implicit
                          cross-sample normalization that helps minority class learning).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    train_ds = DermaMNIST(
        split="train",
        transform=get_transforms(image_size, augment=True),
        download=True,
        root=str(root),
        size=28,
    )
    val_ds = DermaMNIST(
        split="val",
        transform=get_transforms(image_size, augment=False),
        download=True,
        root=str(root),
        size=28,
    )
    test_ds = DermaMNIST(
        split="test",
        transform=get_transforms(image_size, augment=False),
        download=True,
        root=str(root),
        size=28,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,  # CPU-only machine
    )

    if balanced_sampler:
        sampler = get_weighted_sampler(train_ds)
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

    val_loader  = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def get_class_names() -> list[str]:
    """Return the 7 DermaMNIST class label strings."""
    from medmnist import INFO  # noqa: PLC0415
    return list(INFO["dermamnist"]["label"].values())


def get_class_weights(
    data_root: str | Path,
    image_size: int = 28,
    exponent: float = 0.3,
) -> torch.Tensor:
    """Compute tempered inverse-frequency class weights from training set.

    Using exponent=1.0 is full inverse frequency (can over-penalize majority class).
    Using exponent=0.3 provides a mild rebalancing that boosts minority classes
    without collapsing the majority class (nv).
    """
    root = Path(data_root)
    ds = DermaMNIST(
        split="train",
        transform=transforms.ToTensor(),
        download=True,
        root=str(root),
        size=28,
    )
    label_loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    all_labels: list[int] = []
    for _, targets in label_loader:
        all_labels.extend(targets.view(-1).tolist())
    labels = torch.tensor(all_labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=7).float()
    weights = counts.pow(-exponent)
    weights = weights / weights.sum() * 7  # normalise so weights sum to num_classes
    return weights


def get_weighted_sampler(
    dataset: torch.utils.data.Dataset,
    beta: float = 0.5,
) -> torch.utils.data.WeightedRandomSampler:
    """Build a WeightedRandomSampler that up-samples minority classes.

    Each sample weight is inversely proportional to class_count^beta.
    beta=1.0: fully inverse frequency (exact class balance in each epoch --
             df gets sampled 59x as often as nv, causing memorization).
    beta=0.5: sqrt inverse frequency (moderate balance -- gentler oversampling,
             df gets sampled ~7.7x more than nv instead of 59x).
    beta=0.0: uniform sampling (no correction).

    Use beta=0.5 (default) to avoid memorization of minority class examples
    while still providing meaningful minority class exposure.

    Args:
        dataset: The training dataset (must return (image, label) pairs).
        beta:    Exponent controlling oversampling strength (0=none, 1=full inverse freq).

    Returns:
        A WeightedRandomSampler with len(dataset) draws.
    """
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    all_labels: list[int] = []
    for _, targets in loader:
        all_labels.extend(targets.view(-1).tolist())

    labels = torch.tensor(all_labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=7).float()
    # Proportional weight per class: 1 / count^beta
    class_weights = counts.pow(-beta)
    sample_weights = class_weights[labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
