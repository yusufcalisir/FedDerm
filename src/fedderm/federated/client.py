"""
fedderm/federated/client.py
----------------------------
Flower NumPyClient for local training on a simulated hospital client.

Each client holds a fixed Dirichlet-partitioned shard of DermaMNIST and
trains a MiniCNN locally for a configurable number of epochs per round.
Model parameters are exchanged with the server as flat NumPy arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import NDArrays, Scalar

from fedderm.trainer import train_one_epoch, eval_one_epoch


def get_parameters(model: nn.Module) -> NDArrays:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    """Load a list of NumPy arrays into the model's state dict (in-place)."""
    state_dict = model.state_dict()
    for key, param in zip(state_dict.keys(), parameters):
        state_dict[key] = torch.tensor(param)
    model.load_state_dict(state_dict, strict=True)


class DermClient(fl.client.NumPyClient):
    """Flower NumPyClient for one simulated hospital/client.

    Args:
        client_id:        Integer client index (for logging).
        train_loader:     DataLoader for this client's local training shard.
        val_loader:       DataLoader for the global validation set (for fit metrics).
        model:            Freshly instantiated MiniCNN (shared architecture).
        local_epochs:     Number of training epochs per communication round.
        lr:               Local optimizer learning rate.
        weight_decay:     Optimizer weight decay.
        class_weights:    Inverse-frequency class weights for CrossEntropyLoss.
        device:           Torch device.
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        local_epochs: int,
        lr: float,
        weight_decay: float,
        class_weights: torch.Tensor,
        device: torch.device,
        mu: float = 0.0,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.device = device
        self.mu = mu

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Receive global parameters, train locally, return updated parameters."""
        set_parameters(self.model, parameters)

        ref_params = (
            [p.detach().clone() for p in self.model.parameters()]
            if self.mu > 0.0
            else None
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        train_loss = 0.0
        train_acc = 0.0
        for _ in range(self.local_epochs):
            train_loss, train_acc = train_one_epoch(
                self.model,
                self.train_loader,
                optimizer,
                self.criterion,
                self.device,
                proximal_term_ref=ref_params,
                mu=self.mu,
            )

        num_train = len(self.train_loader.dataset)  # type: ignore[arg-type]
        return (
            get_parameters(self.model),
            num_train,
            {"train_loss": train_loss, "train_acc": train_acc},
        )

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Receive global parameters, evaluate on local validation set."""
        set_parameters(self.model, parameters)
        val_loss, val_acc, val_macro_f1 = eval_one_epoch(
            self.model, self.val_loader, self.criterion, self.device
        )
        num_val = len(self.val_loader.dataset)  # type: ignore[arg-type]
        return val_loss, num_val, {
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
        }
