"""
fedderm/federated/dp_client.py
------------------------------
Flower NumPyClient for DP-FedProx local training with Opacus DP-SGD.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import NDArrays, Scalar

from fedderm.federated.client import get_parameters, set_parameters
from fedderm.privacy.engine import make_private_client, train_one_epoch_dp
from fedderm.trainer import eval_one_epoch


class DPDermClient(fl.client.NumPyClient):
    """Flower client executing local DP-SGD training with FedProx regularization.

    Args:
        client_id:        Client integer ID.
        train_loader:     DataLoader for local data shard.
        val_loader:       DataLoader for validation set.
        model:            MiniCNN instance.
        local_epochs:     Number of local epochs per round.
        lr:               Learning rate.
        weight_decay:     Weight decay.
        class_weights:    CrossEntropyLoss class weights.
        device:           Compute device.
        mu:               FedProx proximal penalty weight.
        noise_multiplier: Opacus DP noise multiplier (sigma).
        max_grad_norm:    Opacus DP max gradient clipping norm (C).
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
        mu: float = 0.01,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
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
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
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

        dp_model, dp_optimizer, dp_loader, _ = make_private_client(
            model=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        train_loss = 0.0
        train_acc = 0.0
        total_steps = 0

        for _ in range(self.local_epochs):
            train_loss, train_acc, steps = train_one_epoch_dp(
                dp_model=dp_model,
                dp_loader=dp_loader,
                dp_optimizer=dp_optimizer,
                criterion=self.criterion,
                device=self.device,
                proximal_term_ref=ref_params,
                mu=self.mu,
            )
            total_steps += steps

        # If wrapped by GradSampleModule, self.model holds the updated parameters
        num_train = len(self.train_loader.dataset)  # type: ignore[arg-type]
        return (
            get_parameters(self.model),
            num_train,
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "steps": total_steps,
            },
        )

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        set_parameters(self.model, parameters)
        val_loss, val_acc, val_macro_f1 = eval_one_epoch(
            self.model, self.val_loader, self.criterion, self.device
        )
        num_val = len(self.val_loader.dataset)  # type: ignore[arg-type]
        return val_loss, num_val, {
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
        }
