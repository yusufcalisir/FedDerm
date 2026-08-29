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


def test_derm_client_fedprox_proximal_penalty() -> None:
    """Verify that FedProx (mu > 0) constrains local parameter drift."""
    torch.manual_seed(42)
    model_unreg = build_model(num_classes=7, dropout=0.0)
    model_prox = build_model(num_classes=7, dropout=0.0)

    # Initialize with identical weights (deep copied numpy arrays)
    init_params = [p.copy() for p in get_parameters(model_unreg)]
    set_parameters(model_prox, init_params)

    # Synthetic data
    torch.manual_seed(123)
    x = torch.randn(16, 3, 28, 28)
    y = torch.tensor([i % 7 for i in range(16)])
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    # Client without proximal term (FedAvg: mu = 0.0)
    client_unreg = DermClient(
        client_id=0,
        train_loader=loader,
        val_loader=loader,
        model=model_unreg,
        local_epochs=3,
        lr=1e-2,
        weight_decay=0.0,
        class_weights=torch.ones(7),
        device=torch.device("cpu"),
        mu=0.0,
    )

    # Client with strong proximal penalty (FedProx: mu = 1.0)
    client_prox = DermClient(
        client_id=1,
        train_loader=loader,
        val_loader=loader,
        model=model_prox,
        local_epochs=3,
        lr=1e-2,
        weight_decay=0.0,
        class_weights=torch.ones(7),
        device=torch.device("cpu"),
        mu=1.0,
    )

    params_unreg, _, _ = client_unreg.fit(init_params, {})
    params_prox, _, _ = client_prox.fit(init_params, {})

    # Compute Euclidean distance from initial parameters
    drift_unreg = sum(
        np.linalg.norm(p_new - p_init) ** 2
        for p_new, p_init in zip(params_unreg, init_params)
    )
    drift_prox = sum(
        np.linalg.norm(p_new - p_init) ** 2
        for p_new, p_init in zip(params_prox, init_params)
    )

    # Proximal regularization must reduce parameter drift
    assert drift_prox < drift_unreg, (
        f"FedProx drift ({drift_prox:.4f}) should be strictly smaller than FedAvg drift ({drift_unreg:.4f})"
    )


def test_scaffold_control_variates_persistence() -> None:
    """Verify that SCAFFOLD client control variates persist across simulated rounds."""
    from fedderm.federated.scaffold import PersistentControlVariates, ScaffoldClient

    torch.manual_seed(42)
    model = build_model(num_classes=7, dropout=0.0)
    params = list(model.parameters())

    # Initialize manager for 3 clients
    manager = PersistentControlVariates(num_clients=3, param_templates=params)

    # Initially all control variates are zero
    for c_i in manager.client_controls[0]:
        assert torch.all(c_i == 0.0)
    for c_g in manager.c_global:
        assert torch.all(c_g == 0.0)

    # Synthetic data for client 0
    torch.manual_seed(10)
    x = torch.randn(8, 3, 28, 28)
    y = torch.tensor([i % 7 for i in range(8)])
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    client0 = ScaffoldClient(
        client_id=0,
        train_loader=loader,
        val_loader=loader,
        model=model,
        control_manager=manager,
        local_epochs=2,
        lr=1e-2,
        weight_decay=0.0,
        class_weights=torch.ones(7),
        device=torch.device("cpu"),
    )

    init_weights = [p.copy() for p in get_parameters(model)]
    client0.fit(init_weights, {})

    # After fit, client 0's control variate must be updated to non-zero values
    c0_updated = manager.client_controls[0]
    total_norm_c0 = sum(c.norm().item() for c in c0_updated)
    assert total_norm_c0 > 0.0, "Client 0 control variate should be non-zero after fit"

    # Client 1's control variate should still be exactly zero
    for c1 in manager.client_controls[1]:
        assert torch.all(c1 == 0.0)

    # Server aggregates global control variate
    manager.aggregate_global_control()
    total_norm_cg = sum(c.norm().item() for c in manager.c_global)
    assert total_norm_cg > 0.0, "Global control variate should be updated after aggregation"

    # Simulate Round 2: A newly instantiated client 0 must retrieve the non-zero c_0
    fresh_model = build_model(num_classes=7, dropout=0.0)
    client0_round2 = ScaffoldClient(
        client_id=0,
        train_loader=loader,
        val_loader=loader,
        model=fresh_model,
        control_manager=manager,
        local_epochs=1,
        lr=1e-2,
        weight_decay=0.0,
        class_weights=torch.ones(7),
        device=torch.device("cpu"),
    )

    retrieved_c0 = manager.get_client_control(0, torch.device("cpu"))
    retrieved_norm = sum(c.norm().item() for c in retrieved_c0)
    assert abs(retrieved_norm - total_norm_c0) < 1e-6, (
        "Client 0 control variate must persist across simulated rounds"
    )
