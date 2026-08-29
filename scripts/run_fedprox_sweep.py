"""
scripts/run_fedprox_sweep.py
----------------------------
Automated hyperparameter sweep over proximal term penalty (mu) for FedProx.

Sweeps mu over: [0.001, 0.01, 0.1]
  - 0.001: Weak regularization (closer to FedAvg)
  - 0.01:  Standard literature baseline
  - 0.1:   Strong regularization (strictly constrained client drift)

Saves:
  - Individual run outputs: results/federated_fedprox/mu_<val>/
  - Consolidated sweep artifact: results/federated_fedprox/mu_sweep_results.json
  - Best run artifact: results/federated_fedprox/ (top-level)
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from fedderm.data import get_class_names
from fedderm.federated.simulation import run_federated


SWEEP_MU_VALUES = [0.001, 0.01, 0.1]


def run_sweep() -> dict[str, dict]:
    """Execute the FedProx mu parameter sweep."""
    base_dir = Path("results/federated_fedprox")
    base_dir.mkdir(parents=True, exist_ok=True)
    class_names = get_class_names()

    sweep_results: dict[str, dict] = {}
    best_mu = None
    best_macro_f1 = -1.0

    print("=" * 70)
    print("FEDPROX PROXIMAL REGULARIZATION (MU) SWEEP")
    print(f"Candidate mu values: {SWEEP_MU_VALUES}")
    print("=" * 70)

    t_start = time.time()

    with initialize(version_base="1.3", config_path="../configs"):
        for i, mu in enumerate(SWEEP_MU_VALUES, start=1):
            mu_str = str(mu)
            sub_dir = base_dir / f"mu_{mu_str}"
            sub_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{i}/{len(SWEEP_MU_VALUES)}] Starting FedProx simulation for mu = {mu}...")
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=fedprox_dirichlet",
                    f"federation.mu={mu}",
                    f"output_dir={sub_dir}",
                ],
            )

            metrics = run_federated(cfg)
            sweep_results[mu_str] = metrics

            m_f1 = metrics["macro_f1"]
            acc = metrics["accuracy"]
            b_acc = metrics["balanced_accuracy"]
            print(f"\n>>> Completed mu = {mu}: Test Acc = {acc*100:.2f}%, Bal Acc = {b_acc*100:.2f}%, Macro F1 = {m_f1*100:.2f}%")

            if m_f1 > best_macro_f1:
                best_macro_f1 = m_f1
                best_mu = mu_str

    total_time = time.time() - t_start
    print(f"\n[sweep] complete in {total_time / 60:.1f} minutes")
    print(f"[sweep] winning mu = {best_mu} with Test Macro F1 = {best_macro_f1 * 100:.2f}%\n")

    # Save aggregated sweep results artifact
    sweep_summary = {
        "sweep_mu_values": SWEEP_MU_VALUES,
        "best_mu": float(best_mu) if best_mu is not None else None,
        "total_sweep_time_seconds": total_time,
        "results_by_mu": sweep_results,
    }
    sweep_path = base_dir / "mu_sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump(sweep_summary, f, indent=2)
    print(f"[sweep] sweep results saved to {sweep_path}")

    # Copy winning run artifacts to top-level results/federated_fedprox/
    if best_mu is not None:
        best_dir = base_dir / f"mu_{best_mu}"
        for f in best_dir.glob("*"):
            if f.is_file():
                shutil.copy(f, base_dir / f.name)
        print(f"[sweep] copied winning run (mu={best_mu}) artifacts to {base_dir}/")

    # Print summary table
    print("\n" + "=" * 75)
    print("FEDPROX MU SWEEP SUMMARY TABLE")
    print("=" * 75)
    header = f"{'mu':<10} | {'Test Acc':<10} | {'Bal Acc':<10} | {'Macro F1':<10} | {'Weighted F1':<12} | {'Time (min)':<10}"
    print(header)
    print("-" * 75)
    for mu_str, res in sweep_results.items():
        row = (
            f"{mu_str:<10} | "
            f"{res['accuracy']*100:>8.2f}% | "
            f"{res['balanced_accuracy']*100:>8.2f}% | "
            f"{res['macro_f1']*100:>8.2f}% | "
            f"{res['weighted_f1']*100:>10.2f}% | "
            f"{res['training_time_seconds']/60:>9.1f}"
        )
        print(row)
    print("=" * 75)

    print("\nPER-CLASS F1 BREAKDOWN ACROSS MU VALUES:")
    print("-" * 75)
    col_names = " | ".join([f"mu={mu}" for mu in SWEEP_MU_VALUES])
    print(f"{'Class':<35} | {col_names}")
    print("-" * 75)
    for c_idx, c_name in enumerate(class_names):
        f1_vals = [f"{sweep_results[str(mu)]['per_class_f1'][c_idx]*100:>6.1f}%" for mu in SWEEP_MU_VALUES]
        print(f"{c_name[:35]:<35} | " + " | ".join(f1_vals))
    print("=" * 75 + "\n")

    return sweep_results


if __name__ == "__main__":
    run_sweep()
