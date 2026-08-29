"""
scripts/train_dp.py
-------------------
Entry point for Federated DP-SGD + FedProx simulation with non-IID Dirichlet splits.

Usage:
    python scripts/train_dp.py
    python scripts/train_dp.py privacy.noise_multiplier=0.5
    python scripts/train_dp.py privacy.max_grad_norm=1.0

Driven by configs/experiment/dp_fedprox_dirichlet.yaml via Hydra.
Results are written to results/federated_dp_fedprox/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from fedderm.federated.dp_simulation import run_federated_dp


@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    run_federated_dp(cfg)


if __name__ == "__main__":
    sys.argv.append("experiment=dp_fedprox_dirichlet")
    main()
