"""
scripts/train_scaffold.py
-------------------------
Entry point for SCAFFOLD simulation with non-IID Dirichlet hospital splits.

Usage:
    python scripts/train_scaffold.py
    python scripts/train_scaffold.py federation.rounds=20
    python scripts/train_scaffold.py federation.dirichlet_alpha=0.3

Driven by configs/experiment/scaffold_dirichlet.yaml via Hydra.
Results are written to results/federated_scaffold/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from fedderm.federated.scaffold import run_scaffold


@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    run_scaffold(cfg)


if __name__ == "__main__":
    sys.argv.append("experiment=scaffold_dirichlet")
    main()
