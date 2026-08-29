"""
scripts/train_federated.py
---------------------------
Entry point for FedAvg simulation with non-IID Dirichlet hospital splits.

Usage:
    python scripts/train_federated.py
    python scripts/train_federated.py federation.rounds=50
    python scripts/train_federated.py federation.dirichlet_alpha=0.1

Driven by configs/experiment/fedavg_dirichlet.yaml via Hydra.
Results are written to results/federated_fedavg/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from fedderm.federated.simulation import run_federated


@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    run_federated(cfg)


if __name__ == "__main__":
    # Default override so this script uses the federated config without
    # requiring explicit CLI flag every time.
    sys.argv.append("experiment=fedavg_dirichlet")
    main()
