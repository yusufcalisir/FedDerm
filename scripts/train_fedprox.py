"""
scripts/train_fedprox.py
------------------------
Entry point for FedProx simulation with non-IID Dirichlet hospital splits.

Usage:
    python scripts/train_fedprox.py
    python scripts/train_fedprox.py federation.mu=0.01
    python scripts/train_fedprox.py federation.rounds=20 federation.mu=0.1

Driven by configs/experiment/fedprox_dirichlet.yaml via Hydra.
Results are written to results/federated_fedprox/.
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
    sys.argv.append("experiment=fedprox_dirichlet")
    main()
