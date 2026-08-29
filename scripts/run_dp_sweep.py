"""
scripts/run_dp_sweep.py
-----------------------
Automated DP-SGD noise multiplier sweep for FedDerm.

Evaluates noise_multiplier in [0.3, 0.5, 1.0, 2.0] on the exact same
Dirichlet non-IID partition (alpha=0.3, seed 42, 10 clients, 5 sampled/round,
20 rounds, 3 local epochs, FedProx mu=0.01, max_grad_norm=1.0).

Outputs:
  - results/federated_dp_fedprox/sigma_0.3/
  - results/federated_dp_fedprox/sigma_0.5/
  - results/federated_dp_fedprox/sigma_1.0/
  - results/federated_dp_fedprox/sigma_2.0/
  - results/federated_dp_fedprox/dp_sweep_summary.json
  - results/federated_dp_fedprox/privacy_utility_tradeoff.png
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, open_dict

from fedderm.federated.dp_simulation import run_federated_dp

NOISE_MULTIPLIERS = [0.3, 0.5, 1.0, 2.0]


def plot_privacy_utility_curve(
    summary: list[dict],
    out_path: Path,
) -> None:
    """Generate privacy-utility tradeoff plots (Epsilon vs Accuracy / Macro F1)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sigmas = [s["noise_multiplier"] for s in summary]
    epsilons = [s["server_epsilon"] for s in summary]
    accs = [s["accuracy"] * 100 for s in summary]
    macro_f1s = [s["macro_f1"] * 100 for s in summary]
    weighted_f1s = [s["weighted_f1"] * 100 for s in summary]

    # Plot 1: Metrics vs Noise Multiplier (sigma)
    ax1.plot(sigmas, accs, "o-", label="Accuracy (%)", color="#1f77b4", linewidth=2)
    ax1.plot(sigmas, weighted_f1s, "s--", label="Weighted F1 (%)", color="#2ca02c", linewidth=2)
    ax1.plot(sigmas, macro_f1s, "^-.", label="Macro F1 (%)", color="#d62728", linewidth=2)
    ax1.set_xlabel("Noise Multiplier ($\\sigma$)", fontsize=11)
    ax1.set_ylabel("Metric Score (%)", fontsize=11)
    ax1.set_title("Performance vs. Noise Multiplier", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Plot 2: Metrics vs Server Epsilon (log scale x)
    valid_eps_idx = [i for i, e in enumerate(epsilons) if e > 0 and e != float("inf")]
    if valid_eps_idx:
        v_eps = [epsilons[i] for i in valid_eps_idx]
        v_acc = [accs[i] for i in valid_eps_idx]
        v_mf1 = [macro_f1s[i] for i in valid_eps_idx]
        ax2.plot(v_eps, v_acc, "o-", label="Accuracy (%)", color="#1f77b4", linewidth=2)
        ax2.plot(v_eps, v_mf1, "^-.", label="Macro F1 (%)", color="#d62728", linewidth=2)
        ax2.set_xscale("log")
        ax2.set_xlabel("Privacy Budget $\\epsilon$ (Server-Level, $\\delta=10^{-5}$)", fontsize=11)
        ax2.set_ylabel("Metric Score (%)", fontsize=11)
        ax2.set_title("Privacy-Utility Tradeoff", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    base_out_dir = Path("results/federated_dp_fedprox")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    summary_results: list[dict] = []
    t_start_sweep = time.time()

    print("\n" + "=" * 65)
    print("STARTING DP-SGD NOISE MULTIPLIER SWEEP")
    print(f"Noise multipliers: {NOISE_MULTIPLIERS}")
    print("=" * 65 + "\n")

    for i, nm in enumerate(NOISE_MULTIPLIERS, 1):
        run_dir = base_out_dir / f"sigma_{nm}"
        run_dir.mkdir(parents=True, exist_ok=True)

        run_cfg = copy.deepcopy(cfg)
        with open_dict(run_cfg):
            run_cfg.privacy.noise_multiplier = float(nm)
            run_cfg.output_dir = str(run_dir)

        print(f"\n[{i}/{len(NOISE_MULTIPLIERS)}] Running DP-FedProx with noise_multiplier={nm}...")
        t_run_start = time.time()
        metrics = run_federated_dp(run_cfg)
        run_duration = time.time() - t_run_start

        record = {
            "noise_multiplier": nm,
            "max_grad_norm": float(run_cfg.privacy.max_grad_norm),
            "target_delta": float(run_cfg.privacy.target_delta),
            "server_epsilon": metrics.get("server_epsilon", float("inf")),
            "client_epsilon": metrics.get("client_epsilon", float("inf")),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "best_val_acc": metrics.get("best_val_acc", 0.0),
            "best_val_macro_f1": metrics.get("best_val_macro_f1", 0.0),
            "per_class_f1": metrics.get("per_class_f1", []),
            "runtime_seconds": run_duration,
        }
        summary_results.append(record)

        # Save summary incrementally after each run
        summary_path = base_out_dir / "dp_sweep_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_results, f, indent=2)

    total_sweep_time = time.time() - t_start_sweep

    # Plot summary
    plot_path = base_out_dir / "privacy_utility_tradeoff.png"
    plot_privacy_utility_curve(summary_results, plot_path)

    # Print final summary table
    print("\n" + "=" * 75)
    print("DP-SGD NOISE MULTIPLIER SWEEP SUMMARY (FedProx mu=0.01, C=1.0, delta=1e-5)")
    print("=" * 75)
    print(f"{'sigma':<8} {'Server eps':<12} {'Client eps':<12} {'Test Acc':<10} {'Bal Acc':<10} {'Macro F1':<10} {'Weighted F1':<12} {'Runtime':<8}")
    print("-" * 75)
    for r in summary_results:
        s_eps = f"{r['server_epsilon']:.2f}" if r['server_epsilon'] != float("inf") else "inf"
        c_eps = f"{r['client_epsilon']:.2f}" if r['client_epsilon'] != float("inf") else "inf"
        print(
            f"{r['noise_multiplier']:<8.2f} "
            f"{s_eps:<12} "
            f"{c_eps:<12} "
            f"{r['accuracy']*100:<9.2f}% "
            f"{r['balanced_accuracy']*100:<9.2f}% "
            f"{r['macro_f1']*100:<9.2f}% "
            f"{r['weighted_f1']*100:<11.2f}% "
            f"{r['runtime_seconds']/60:<7.1f}m"
        )
    print("=" * 75)
    print(f"Total sweep time: {total_sweep_time / 60:.1f} minutes\n")


if __name__ == "__main__":
    sys.argv.append("experiment=dp_fedprox_dirichlet")
    main()
