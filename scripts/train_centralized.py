"""
scripts/train_centralized.py
-----------------------------
Entry point for centralized baseline training on DermaMNIST.

Usage:
    python scripts/train_centralized.py
    python scripts/train_centralized.py training.epochs=50
    python scripts/train_centralized.py training.lr=5e-4 training.batch_size=32

Driven by configs/experiments/centralized_baseline.yaml via Hydra.
Results are written to results/centralized_baseline/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script (editable install handles
# this automatically, but explicit is safer for direct python invocation)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from fedderm.trainer import run_centralized


@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    run_centralized(cfg)


if __name__ == "__main__":
    main()
