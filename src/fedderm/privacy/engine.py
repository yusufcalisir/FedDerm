"""
fedderm/privacy/engine.py
-------------------------
Opacus PrivacyEngine wrappers and DP-SGD local training loop for FedDerm.
"""

from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def check_opacus_compatibility(model: nn.Module) -> list[str]:
    """Validate that the model architecture is compatible with Opacus DP-SGD."""
    errors = ModuleValidator.validate(model, strict=False)
    return [str(e) for e in errors]


def make_private_client(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    noise_multiplier: float,
    max_grad_norm: float = 1.0,
) -> tuple[nn.Module, torch.optim.Optimizer, DataLoader, PrivacyEngine | None]:
    """Wrap model, optimizer, and data_loader with Opacus PrivacyEngine for DP-SGD.

    If noise_multiplier <= 0.0, DP-SGD is effectively bypassed (clipping/noise disabled).

    Args:
        model: PyTorch model (MiniCNN with GroupNorm).
        optimizer: PyTorch optimizer (e.g. Adam).
        data_loader: PyTorch DataLoader.
        noise_multiplier: Standard deviation multiplier for Gaussian noise (sigma).
        max_grad_norm: Maximum L2 gradient norm for per-sample clipping (C).

    Returns:
        (dp_model, dp_optimizer, dp_loader, privacy_engine)
    """
    if noise_multiplier <= 0.0:
        return model, optimizer, data_loader, None

    privacy_engine = PrivacyEngine(secure_mode=False)
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return dp_model, dp_optimizer, dp_loader, privacy_engine


def train_one_epoch_dp(
    dp_model: nn.Module,
    dp_loader: DataLoader,
    dp_optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    proximal_term_ref: Sequence[torch.Tensor] | None = None,
    mu: float = 0.0,
) -> tuple[float, float, int]:
    """Execute one local training epoch with DP-SGD and FedProx proximal regularization.

    Args:
        dp_model: Model (wrapped with GradSampleModule if DP is active).
        dp_loader: DataLoader (wrapped with DPDataLoader if DP is active).
        dp_optimizer: Optimizer (wrapped with DPOptimizer if DP is active).
        criterion: Loss function (CrossEntropyLoss).
        device: Torch compute device.
        proximal_term_ref: Reference global parameters w^t for proximal term.
        mu: FedProx proximal penalty coefficient (default 0.01).

    Returns:
        (epoch_loss, epoch_accuracy, step_count)
    """
    dp_model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    for images, targets in dp_loader:
        images = images.to(device)
        targets = targets.view(-1).long().to(device)

        dp_optimizer.zero_grad()
        logits = dp_model(images)
        loss = criterion(logits, targets)

        # FedProx proximal penalty: (mu / 2) * sum_l ||w_l - w_l^t||^2
        if proximal_term_ref is not None and mu > 0.0:
            prox_loss = torch.tensor(0.0, device=device)
            # dp_model.parameters() yields the underlying trainable parameters
            for p, p_ref in zip(dp_model.parameters(), proximal_term_ref):
                prox_loss = prox_loss + 0.5 * mu * torch.sum((p - p_ref) ** 2)
            loss = loss + prox_loss

        loss.backward()
        dp_optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += images.size(0)
        steps += 1

    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc, steps
