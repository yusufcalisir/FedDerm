"""
tests/test_privacy.py
---------------------
Unit tests for Opacus DP-SGD integration, PrivacyEngine wrapping,
and Federated Privacy Accounting.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus.validators import ModuleValidator

from fedderm.models import build_model
from fedderm.privacy.accountant import FederatedPrivacyAccountant
from fedderm.privacy.engine import (
    check_opacus_compatibility,
    make_private_client,
    train_one_epoch_dp,
)
from fedderm.federated.client import get_parameters
from fedderm.federated.dp_client import DPDermClient


def test_opacus_module_validator_passes() -> None:
    """Verify that GroupNorm MiniCNN is 100% compatible with Opacus DP-SGD."""
    model = build_model(num_classes=7, dropout=0.4)
    errors = check_opacus_compatibility(model)
    assert len(errors) == 0, f"MiniCNN has Opacus compatibility errors: {errors}"


def test_make_private_client() -> None:
    """Verify PrivacyEngine wrapping of model, optimizer, and DataLoader."""
    model = build_model(num_classes=7, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(16, 3, 28, 28)
    y = torch.tensor([i % 7 for i in range(16)])
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

    dp_model, dp_optimizer, dp_loader, privacy_engine = make_private_client(
        model=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
    )

    assert privacy_engine is not None
    assert hasattr(dp_model, "autograd_grad_sample_hooks") or hasattr(dp_model, "_module")
    assert hasattr(dp_optimizer, "noise_multiplier") or hasattr(dp_optimizer, "original_optimizer")


def test_dp_derm_client_fit() -> None:
    """Verify DPDermClient executes local DP-SGD training with proximal term."""
    torch.manual_seed(42)
    model = build_model(num_classes=7, dropout=0.0)

    x = torch.randn(16, 3, 28, 28)
    y = torch.tensor([i % 7 for i in range(16)])
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

    client = DPDermClient(
        client_id=0,
        train_loader=loader,
        val_loader=loader,
        model=model,
        local_epochs=2,
        lr=1e-2,
        weight_decay=0.0,
        class_weights=torch.ones(7),
        device=torch.device("cpu"),
        mu=0.01,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
    )

    init_params = get_parameters(model)
    new_params, num_samples, metrics = client.fit(init_params, {})

    assert num_samples == 16
    assert "train_loss" in metrics
    assert "train_acc" in metrics
    assert "steps" in metrics
    assert metrics["steps"] == 4  # 2 epochs * (16 / 8) batches = 4 steps
    assert len(new_params) == len(init_params)


def test_federated_privacy_accountant() -> None:
    """Verify RDP privacy accountant correctly accumulates epsilon across rounds."""
    accountant = FederatedPrivacyAccountant(
        target_delta=1e-5,
        total_samples=7007,
        num_clients=10,
        clients_per_round=5,
        batch_size=64,
    )

    # Initial state (no steps) -> epsilon is inf
    assert accountant.get_epsilon() == float("inf")

    # Step 5 rounds with noise_multiplier = 1.0
    for _ in range(5):
        accountant.step_round(
            noise_multiplier=1.0,
            num_local_steps_per_client=33,
            client_dataset_size=700,
        )

    eps_server = accountant.get_epsilon(delta=1e-5)
    eps_client = accountant.get_client_epsilon(delta=1e-5)

    assert 0.0 < eps_server < 100.0, f"Server epsilon ({eps_server}) should be reasonable"
    assert 0.0 < eps_client < 500.0, f"Client epsilon ({eps_client}) should be reasonable"
    assert eps_server < eps_client, "Server-level subsampled epsilon must be smaller than local client epsilon"
