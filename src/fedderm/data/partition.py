"""
fedderm/data/partition.py
--------------------------
Non-IID dataset partitioning utilities using Dirichlet sampling.

Splits a dataset into K client shards where each client's class
distribution is drawn from a Dirichlet(alpha) distribution.
Lower alpha -> more heterogeneous (each client holds fewer classes).
Alpha=inf -> perfectly IID uniform split.

Reference:
    Hsieh et al. (2020), "Quagmire in the Stone Age" / standard FL non-IID
    setup following Yurochkin et al. (2019) and Lin et al. (2020).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from medmnist import DermaMNIST


def _collect_labels_fast(root: str, image_size: int) -> np.ndarray:
    """Return all training labels as a numpy array using batched DataLoader."""
    ds = DermaMNIST(
        split="train",
        transform=transforms.ToTensor(),
        download=True,
        root=root,
        size=28,
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    labels: list[int] = []
    for _, targets in loader:
        labels.extend(targets.squeeze().tolist())
    return np.array(labels, dtype=np.int64)


def dirichlet_partition(
    dataset: torch.utils.data.Dataset,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_samples_per_client: int = 5,
    seed: int = 42,
) -> list[list[int]]:
    """Partition dataset indices into num_clients shards via Dirichlet sampling.

    Each class's indices are distributed across clients by drawing proportions
    from a Dirichlet(alpha) distribution. Lower alpha = more skewed = more non-IID.

    Args:
        dataset:                 PyTorch Dataset (used only for len()).
        labels:                  Integer class label for each sample (same order as dataset).
        num_clients:             Number of clients (simulated hospitals).
        alpha:                   Dirichlet concentration parameter.
        min_samples_per_client:  Discard any client shard with fewer samples.
        seed:                    Random seed for reproducibility.

    Returns:
        List of length num_clients, where each element is a list of sample indices
        belonging to that client.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    n = len(labels)

    # Group indices by class
    class_indices: list[np.ndarray] = [
        np.where(labels == c)[0] for c in range(num_classes)
    ]

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c_indices in class_indices:
        rng.shuffle(c_indices)
        # Draw proportions for this class across clients
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        # Convert proportions to integer counts (must sum to len(c_indices))
        counts = (proportions * len(c_indices)).astype(int)
        # Fix rounding so counts sum exactly to len(c_indices)
        diff = len(c_indices) - counts.sum()
        counts[: int(abs(diff))] += int(np.sign(diff))

        # Assign slices of this class's indices to each client
        start = 0
        for k, count in enumerate(counts):
            client_indices[k].extend(c_indices[start : start + count].tolist())
            start += count

    # Shuffle each client's indices for random batch ordering
    for k in range(num_clients):
        rng.shuffle(np.array(client_indices[k]))

    # Warn if any client is below minimum threshold
    for k, idx in enumerate(client_indices):
        if len(idx) < min_samples_per_client:
            print(
                f"[partition] WARNING: client {k} has only {len(idx)} samples "
                f"(< min {min_samples_per_client}). Consider raising alpha or "
                f"reducing num_clients."
            )

    return client_indices


def report_partition(
    client_indices: list[list[int]],
    labels: np.ndarray,
    class_names: Sequence[str],
    out_path: str | Path | None = None,
) -> dict:
    """Compute and optionally save a per-client class distribution report.

    Args:
        client_indices: Output of dirichlet_partition().
        labels:         Label array for the full training dataset.
        class_names:    Human-readable class name list.
        out_path:       Optional JSON path to save the report.

    Returns:
        dict with keys 'per_client' (list of dicts) and 'summary'.
    """
    num_classes = len(class_names)
    report: dict = {"per_client": [], "summary": {}}

    total_counts = np.zeros(num_classes, dtype=int)

    for k, idx in enumerate(client_indices):
        client_labels = labels[np.array(idx)]
        counts = np.bincount(client_labels, minlength=num_classes)
        total_counts += counts
        report["per_client"].append(
            {
                "client_id": k,
                "total_samples": int(len(idx)),
                "class_counts": {
                    class_names[c]: int(counts[c]) for c in range(num_classes)
                },
            }
        )

    report["summary"] = {
        "num_clients": len(client_indices),
        "total_samples": int(total_counts.sum()),
        "global_class_counts": {
            class_names[c]: int(total_counts[c]) for c in range(num_classes)
        },
    }

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

    # Print a compact table to stdout
    short = [n[:20] for n in class_names]
    header = f"{'Client':>7} | {'N':>5} | " + " | ".join(f"{s:>5}" for s in short)
    print("\n" + "=" * len(header))
    print("Non-IID Dirichlet partition: per-client class counts")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in report["per_client"]:
        counts_vals = list(row["class_counts"].values())
        row_str = (
            f"{row['client_id']:>7} | {row['total_samples']:>5} | "
            + " | ".join(f"{v:>5}" for v in counts_vals)
        )
        print(row_str)
    print("=" * len(header))

    return report


def make_client_loaders(
    root: str,
    image_size: int,
    client_indices: list[list[int]],
    batch_size: int,
    augment: bool = True,
) -> list[DataLoader]:
    """Build one DataLoader per client using the given index partitions.

    Args:
        root:            Data download root.
        image_size:      Image spatial size.
        client_indices:  Per-client index lists from dirichlet_partition().
        batch_size:      Mini-batch size.
        augment:         Whether to apply training augmentation.

    Returns:
        List of DataLoaders, one per client.
    """
    from fedderm.data.dermamnist import get_transforms  # avoid circular import

    full_ds = DermaMNIST(
        split="train",
        transform=get_transforms(image_size, augment=augment),
        download=True,
        root=root,
        size=28,
    )

    loaders: list[DataLoader] = []
    for idx_list in client_indices:
        subset = Subset(full_ds, idx_list)
        loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
        )
    return loaders
