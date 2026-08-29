"""
tests/test_federated.py
-----------------------
Unit tests for non-IID Dirichlet partitioning and federated client components.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedderm.data.partition import dirichlet_partition, report_partition
from fedderm.federated.client import DermClient, get_parameters, set_parameters
from fedderm.models import build_model


def test_dirichlet_partition() -> None:
    # 100 samples across 7 classes
    labels = np.array([i % 7 for i in range(100)])
    dummy_ds = type("_DS", (), {"__len__": lambda self: len(labels)})()
    num_clients = 5
    client_indices = dirichlet_partition(
        dummy_ds,
        labels,
        num_clients=num_clients,
        alpha=0.3,
        seed=42,
    )
    assert len(client_indices) == num_clients
    total_assigned = sum(len(idx) for idx in client_indices)
    assert total_assigned == 100

    report = report_partition(
        client_indices,
        labels,
        [f"class_{i}" for i in range(7)],
    )
    assert len(report["per_client"]) == num_clients
    assert report["summary"]["total_samples"] == 100


def test_derm_client_fit() -> None:
    model = build_model(num_classes=7, dropout=0.1)
    x = torch.randn(8, 3, 28, 28)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)

    client = DermClient(
        client_id=0,
        train_loader=loader,
        val_loader=loader,
        model=model,
        local_epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        class_weights=torch.ones(7),
        device=torch.device("cpu"),
    )

    params = client.get_parameters({})
    new_params, num_samples, metrics = client.fit(params, {})
    assert num_samples == 8
    assert "train_loss" in metrics
    assert "train_acc" in metrics
    assert len(new_params) == len(params)
