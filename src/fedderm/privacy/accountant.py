"""
fedderm/privacy/accountant.py
-----------------------------
Differential privacy accountant for federated learning simulations.

Uses Opacus RDP (Rényi Differential Privacy) accountant to compute the cumulative
privacy spending (epsilon, delta) across federated communication rounds.

Supports two complementary privacy accounting perspectives:
  1. Client-Level Sample DP (Local DP): Tracks privacy spent on a single client's
     local dataset across the rounds that client participated in.
  2. Server-Level Subsampled Record DP: Tracks privacy of any individual data sample
     across the entire federation using Poisson/uniform client subsampling and
     mini-batch subsampling (q = (B / N_total) * (C / N_clients)).
"""

from __future__ import annotations

from typing import Any
import numpy as np
from opacus.accountants import RDPAccountant


class FederatedPrivacyAccountant:
    """Federated RDP accountant tracking cumulative privacy spending.

    Args:
        target_delta: Target delta parameter for (epsilon, delta)-DP (default 1e-5).
        total_samples: Total number of training samples across all clients (7,007).
        num_clients: Total number of clients (10).
        clients_per_round: Number of clients sampled per round (5).
        batch_size: Local training mini-batch size (64).
    """

    def __init__(
        self,
        target_delta: float = 1.0e-5,
        total_samples: int = 7007,
        num_clients: int = 10,
        clients_per_round: int = 5,
        batch_size: int = 64,
    ) -> None:
        self.target_delta = target_delta
        self.total_samples = total_samples
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.batch_size = batch_size

        self.accountant_server = RDPAccountant()
        self.accountant_client = RDPAccountant()
        self.total_steps = 0
        self.total_rounds = 0

    def step_round(
        self,
        noise_multiplier: float,
        num_local_steps_per_client: int,
        client_dataset_size: int = 700,
    ) -> None:
        """Record privacy consumption for one federated communication round.

        Args:
            noise_multiplier: Noise multiplier sigma used during local DP-SGD.
            num_local_steps_per_client: Number of mini-batch steps taken locally (K).
            client_dataset_size: Average dataset size of sampled clients.
        """
        if noise_multiplier <= 0.0:
            self.total_rounds += 1
            self.total_steps += num_local_steps_per_client
            return

        # Server-level record subsampling rate: q_server = (batch_size / total_samples) * (clients_per_round / num_clients)
        client_subsample_ratio = float(self.clients_per_round) / float(self.num_clients)
        q_server = (float(self.batch_size) / float(self.total_samples)) * client_subsample_ratio

        # Client-level local subsampling rate: q_client = batch_size / client_dataset_size
        q_client = float(self.batch_size) / max(1.0, float(client_dataset_size))

        for _ in range(num_local_steps_per_client):
            self.accountant_server.step(noise_multiplier=noise_multiplier, sample_rate=q_server)
            self.accountant_client.step(noise_multiplier=noise_multiplier, sample_rate=q_client)
            self.total_steps += 1

        self.total_rounds += 1

    def get_epsilon(self, delta: float | None = None) -> float:
        """Return server-level subsampled record DP epsilon at target delta."""
        target_d = delta if delta is not None else self.target_delta
        if self.total_steps == 0:
            return float("inf")
        try:
            return float(self.accountant_server.get_epsilon(delta=target_d))
        except Exception:
            return float("inf")

    def get_client_epsilon(self, delta: float | None = None) -> float:
        """Return client-level local DP epsilon at target delta."""
        target_d = delta if delta is not None else self.target_delta
        if self.total_steps == 0:
            return float("inf")
        try:
            return float(self.accountant_client.get_epsilon(delta=target_d))
        except Exception:
            return float("inf")

    def get_summary(self) -> dict[str, Any]:
        """Return a complete privacy accounting report."""
        return {
            "target_delta": self.target_delta,
            "server_epsilon": self.get_epsilon(),
            "client_local_epsilon": self.get_client_epsilon(),
            "total_rounds": self.total_rounds,
            "total_steps": self.total_steps,
        }
