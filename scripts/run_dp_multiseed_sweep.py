"""
scripts/run_dp_multiseed_sweep.py
-----------------------------------
Automated multi-seed DP-SGD noise multiplier sweep for FedDerm.

Runs 3 random seeds (42, 43, 44) for each noise_multiplier in [0.3, 0.5, 1.0, 2.0]
on the exact same Dirichlet non-IID partition (alpha=0.3, partition_seed=42,
10 clients, 5 sampled/round, 20 rounds, 3 local epochs, FedProx mu=0.01, max_grad_norm=1.0).

Seeds control:
  - Global model parameter initialization
  - Opacus DP-SGD Gaussian noise sampling
  - Flower client selection and DataLoader shuffling

Outputs:
  - results/federated_dp_fedprox_multiseed/sigma_{nm}/seed_{s}/
  - results/federated_dp_fedprox_multiseed/multiseed_summary.json
  - results/federated_dp_fedprox_multiseed/multiseed_summary.csv
  - results/federated_dp_fedprox_multiseed/privacy_utility_tradeoff_multiseed.png
"""

from __future__ import annotations

import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, open_dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from fedderm.federated.dp_simulation import run_federated_dp

NOISE_MULTIPLIERS = [0.3, 0.5, 1.0, 2.0]
SEEDS = [42, 43, 44]

# Non-DP FedProx reference from Phase 4 (mu=0.01, Dirichlet alpha=0.3, 20 rounds)
NON_DP_REFERENCE = {
    "noise_multiplier": 0.0,
    "server_epsilon": float("inf"),
    "client_epsilon": float("inf"),
    "accuracy_mean": 67.73,
    "accuracy_std": 0.0,
    "balanced_accuracy_mean": 22.03,
    "balanced_accuracy_std": 0.0,
    "macro_f1_mean": 20.36,
    "macro_f1_std": 0.0,
    "weighted_f1_mean": 63.58,
    "weighted_f1_std": 0.0,
    "per_class_f1_mean": [0.0, 0.0, 35.51, 0.0, 21.32, 85.68, 0.0],
}


def compute_statistics(seed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean and sample standard deviation across seeds."""
    n_seeds = len(seed_metrics)
    accs = [m["accuracy"] * 100 for m in seed_metrics]
    bal_accs = [m["balanced_accuracy"] * 100 for m in seed_metrics]
    macro_f1s = [m["macro_f1"] * 100 for m in seed_metrics]
    weighted_f1s = [m["weighted_f1"] * 100 for m in seed_metrics]
    best_val_accs = [m.get("best_val_acc", 0.0) * 100 for m in seed_metrics]
    best_val_mf1s = [m.get("best_val_macro_f1", 0.0) * 100 for m in seed_metrics]

    server_eps_list = [m.get("server_epsilon", float("inf")) for m in seed_metrics]
    client_eps_list = [m.get("client_epsilon", float("inf")) for m in seed_metrics]

    # Verify epsilons are identical
    if len(set(server_eps_list)) > 1:
        print(f"[WARNING] Server epsilon differs across seeds: {server_eps_list}")
    if len(set(client_eps_list)) > 1:
        print(f"[WARNING] Client epsilon differs across seeds: {client_eps_list}")

    ddof = 1 if n_seeds > 1 else 0

    # Per-class F1 statistics
    num_classes = len(seed_metrics[0].get("per_class_f1", []))
    per_class_means = []
    per_class_stds = []
    for c in range(num_classes):
        c_f1s = [m["per_class_f1"][c] * 100 for m in seed_metrics if len(m.get("per_class_f1", [])) > c]
        per_class_means.append(float(np.mean(c_f1s)))
        per_class_stds.append(float(np.std(c_f1s, ddof=ddof)) if n_seeds > 1 else 0.0)

    return {
        "n_seeds": n_seeds,
        "seeds": [m["seed"] for m in seed_metrics],
        "server_epsilon": server_eps_list[0],
        "client_epsilon": client_eps_list[0],
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=ddof)) if n_seeds > 1 else 0.0,
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "balanced_accuracy_std": float(np.std(bal_accs, ddof=ddof)) if n_seeds > 1 else 0.0,
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s, ddof=ddof)) if n_seeds > 1 else 0.0,
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s, ddof=ddof)) if n_seeds > 1 else 0.0,
        "best_val_acc_mean": float(np.mean(best_val_accs)),
        "best_val_acc_std": float(np.std(best_val_accs, ddof=ddof)) if n_seeds > 1 else 0.0,
        "best_val_macro_f1_mean": float(np.mean(best_val_mf1s)),
        "best_val_macro_f1_std": float(np.std(best_val_mf1s, ddof=ddof)) if n_seeds > 1 else 0.0,
        "per_class_f1_mean": per_class_means,
        "per_class_f1_std": per_class_stds,
        "individual_runs": seed_metrics,
    }


def plot_privacy_utility_curve_with_errorbars(
    summary_data: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Generate privacy-utility tradeoff plots with error bars (mean +- std)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sigmas = [s["noise_multiplier"] for s in summary_data]
    epsilons = [s["server_epsilon"] for s in summary_data]
    acc_means = [s["accuracy_mean"] for s in summary_data]
    acc_stds = [s["accuracy_std"] for s in summary_data]
    mf1_means = [s["macro_f1_mean"] for s in summary_data]
    mf1_stds = [s["macro_f1_std"] for s in summary_data]
    wf1_means = [s["weighted_f1_mean"] for s in summary_data]
    wf1_stds = [s["weighted_f1_std"] for s in summary_data]

    # Panel 1: Performance vs Noise Multiplier (sigma)
    ax1.errorbar(
        sigmas, acc_means, yerr=acc_stds,
        fmt="o-", color="#1f77b4", linewidth=2, capsize=5, capthick=1.5,
        label="Test Accuracy (%)"
    )
    ax1.errorbar(
        sigmas, wf1_means, yerr=wf1_stds,
        fmt="s--", color="#2ca02c", linewidth=2, capsize=5, capthick=1.5,
        label="Weighted F1 (%)"
    )
    ax1.errorbar(
        sigmas, mf1_means, yerr=mf1_stds,
        fmt="^-.", color="#d62728", linewidth=2, capsize=5, capthick=1.5,
        label="Macro F1 (%)"
    )

    # Reference line for Non-DP FedProx
    ax1.axhline(NON_DP_REFERENCE["accuracy_mean"], color="#1f77b4", linestyle=":", alpha=0.5, label="Non-DP Acc (67.7%)")
    ax1.axhline(NON_DP_REFERENCE["macro_f1_mean"], color="#d62728", linestyle=":", alpha=0.5, label="Non-DP Macro F1 (20.4%)")

    ax1.set_xlabel(r"Noise Multiplier ($\sigma$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Performance vs. DP Noise Multiplier (3-Seed Mean $\\pm$ Std)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="best", fontsize=9)

    # Panel 2: Performance vs Server Privacy Budget Epsilon (log scale)
    valid_idx = [i for i, e in enumerate(epsilons) if e > 0 and e != float("inf")]
    if valid_idx:
        v_eps = [epsilons[i] for i in valid_idx]
        v_acc = [acc_means[i] for i in valid_idx]
        v_acc_std = [acc_stds[i] for i in valid_idx]
        v_mf1 = [mf1_means[i] for i in valid_idx]
        v_mf1_std = [mf1_stds[i] for i in valid_idx]
        v_wf1 = [wf1_means[i] for i in valid_idx]
        v_wf1_std = [wf1_stds[i] for i in valid_idx]

        ax2.errorbar(
            v_eps, v_acc, yerr=v_acc_std,
            fmt="o-", color="#1f77b4", linewidth=2, capsize=5, capthick=1.5,
            label="Test Accuracy (%)"
        )
        ax2.errorbar(
            v_eps, v_wf1, yerr=v_wf1_std,
            fmt="s--", color="#2ca02c", linewidth=2, capsize=5, capthick=1.5,
            label="Weighted F1 (%)"
        )
        ax2.errorbar(
            v_eps, v_mf1, yerr=v_mf1_std,
            fmt="^-.", color="#d62728", linewidth=2, capsize=5, capthick=1.5,
            label="Macro F1 (%)"
        )

        ax2.set_xscale("log")
        ax2.set_xlabel(r"Server Privacy Budget $\epsilon$ ($\delta=10^{-5}$, log scale)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
        ax2.set_title(r"Privacy-Utility Tradeoff ($\epsilon$ vs. Utility)", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="best", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved multi-seed privacy-utility curve to {out_path}")


def save_csv_summary(summary_data: list[dict[str, Any]], out_path: Path) -> None:
    """Export summary metrics to CSV."""
    headers = [
        "noise_multiplier",
        "server_epsilon",
        "client_epsilon",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "weighted_f1_mean",
        "weighted_f1_std",
        "best_val_macro_f1_mean",
        "best_val_macro_f1_std",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for s in summary_data:
            writer.writerow([
                s["noise_multiplier"],
                f"{s['server_epsilon']:.4f}" if s["server_epsilon"] != float("inf") else "inf",
                f"{s['client_epsilon']:.4f}" if s["client_epsilon"] != float("inf") else "inf",
                f"{s['accuracy_mean']:.2f}",
                f"{s['accuracy_std']:.2f}",
                f"{s['balanced_accuracy_mean']:.2f}",
                f"{s['balanced_accuracy_std']:.2f}",
                f"{s['macro_f1_mean']:.2f}",
                f"{s['macro_f1_std']:.2f}",
                f"{s['weighted_f1_mean']:.2f}",
                f"{s['weighted_f1_std']:.2f}",
                f"{s['best_val_macro_f1_mean']:.2f}",
                f"{s['best_val_macro_f1_std']:.2f}",
            ])
    print(f"[csv] Saved CSV summary to {out_path}")


@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    base_out_dir = Path("results/federated_dp_fedprox_multiseed")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STARTING MULTI-SEED DP-SGD NOISE MULTIPLIER SWEEP")
    print(f"Noise multipliers: {NOISE_MULTIPLIERS}")
    print(f"Random seeds:     {SEEDS}")
    print("Fixed Dirichlet partition: alpha=0.3, partition_seed=42")
    print("=" * 80 + "\n")

    all_summaries: list[dict[str, Any]] = []
    t_start_sweep = time.time()
    total_runs = len(NOISE_MULTIPLIERS) * len(SEEDS)
    current_run = 0

    for nm in NOISE_MULTIPLIERS:
        nm_dir = base_out_dir / f"sigma_{nm}"
        nm_dir.mkdir(parents=True, exist_ok=True)
        seed_metrics: list[dict[str, Any]] = []

        print("\n" + "-" * 70)
        print(f"RUNNING SWEEP FOR NOISE MULTIPLIER sigma={nm}")
        print("-" * 70)

        for seed in SEEDS:
            current_run += 1
            run_dir = nm_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            metrics_file = run_dir / "test_metrics.json"

            # Check if this run has already completed (e.g. from previous run)
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        cached_m = json.load(f)
                    if "accuracy" in cached_m and "macro_f1" in cached_m:
                        print(f"[{current_run}/{total_runs}] [CACHED] sigma={nm}, seed={seed} (Acc={cached_m['accuracy']*100:.2f}%, mF1={cached_m['macro_f1']*100:.2f}%)")
                        cached_m["seed"] = seed
                        seed_metrics.append(cached_m)
                        continue
                except Exception:
                    pass

            # Check if seed 42 exists in previous single-seed sweep folder
            if seed == 42:
                single_seed_file = Path(f"results/federated_dp_fedprox/sigma_{nm}/test_metrics.json")
                if single_seed_file.exists():
                    try:
                        with open(single_seed_file, "r") as f:
                            cached_m = json.load(f)
                        if "accuracy" in cached_m and "macro_f1" in cached_m:
                            print(f"[{current_run}/{total_runs}] [IMPORTED FROM SINGLE-SEED] sigma={nm}, seed={seed}")
                            cached_m["seed"] = seed
                            with open(metrics_file, "w") as f:
                                json.dump(cached_m, f, indent=2)
                            seed_metrics.append(cached_m)
                            continue
                    except Exception:
                        pass

            print(f"\n[{current_run}/{total_runs}] Running DP-FedProx with sigma={nm}, seed={seed}...")
            run_cfg = copy.deepcopy(cfg)
            with open_dict(run_cfg):
                run_cfg.seed = int(seed)
                run_cfg.privacy.noise_multiplier = float(nm)
                run_cfg.output_dir = str(run_dir)
                run_cfg.federation.partition_seed = 42

            import ray
            if ray.is_initialized():
                ray.shutdown()
            time.sleep(1.0)

            t_run_start = time.time()
            try:
                metrics = run_federated_dp(run_cfg)
            finally:
                if ray.is_initialized():
                    ray.shutdown()
            metrics["seed"] = seed
            metrics["runtime_seconds"] = time.time() - t_run_start
            seed_metrics.append(metrics)

        # Aggregate statistics for this noise multiplier
        stat = compute_statistics(seed_metrics)
        stat["noise_multiplier"] = nm
        all_summaries.append(stat)

        # Save summary incrementally
        summary_path = base_out_dir / "multiseed_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)

    total_sweep_time = time.time() - t_start_sweep

    # Export CSV summary
    csv_path = base_out_dir / "multiseed_summary.csv"
    save_csv_summary(all_summaries, csv_path)

    # Plot multi-seed errorbar curve
    plot_path = base_out_dir / "privacy_utility_tradeoff_multiseed.png"
    plot_privacy_utility_curve_with_errorbars(all_summaries, plot_path)

    # Final summary display
    print("\n" + "=" * 90)
    print("MULTI-SEED DP-SGD NOISE MULTIPLIER SWEEP SUMMARY (FedProx mu=0.01, 3 Seeds: 42, 43, 44)")
    print("=" * 90)
    print(
        f"{'sigma':<7} {'Server eps':<12} {'Client eps':<12} "
        f"{'Test Acc (%)':<16} {'Bal Acc (%)':<16} {'Macro F1 (%)':<16} {'Weighted F1 (%)':<16}"
    )
    print("-" * 90)

    # Print non-DP reference
    print(
        f"{'0.00*':<7} {'inf':<12} {'inf':<12} "
        f"{NON_DP_REFERENCE['accuracy_mean']:<5.2f}            "
        f"{NON_DP_REFERENCE['balanced_accuracy_mean']:<5.2f}            "
        f"{NON_DP_REFERENCE['macro_f1_mean']:<5.2f}            "
        f"{NON_DP_REFERENCE['weighted_f1_mean']:<5.2f}"
    )

    for s in all_summaries:
        s_eps = f"{s['server_epsilon']:.2f}" if s['server_epsilon'] != float("inf") else "inf"
        c_eps = f"{s['client_epsilon']:.2f}" if s['client_epsilon'] != float("inf") else "inf"
        acc_str = f"{s['accuracy_mean']:.2f} +- {s['accuracy_std']:.2f}"
        bal_str = f"{s['balanced_accuracy_mean']:.2f} +- {s['balanced_accuracy_std']:.2f}"
        mf1_str = f"{s['macro_f1_mean']:.2f} +- {s['macro_f1_std']:.2f}"
        wf1_str = f"{s['weighted_f1_mean']:.2f} +- {s['weighted_f1_std']:.2f}"
        print(
            f"{s['noise_multiplier']:<7.2f} {s_eps:<12} {c_eps:<12} "
            f"{acc_str:<16} {bal_str:<16} {mf1_str:<16} {wf1_str:<16}"
        )

    print("=" * 90)
    print("(* 0.00 is Non-DP FedProx mu=0.01 reference)")
    print(f"Total multi-seed sweep wall-clock time: {total_sweep_time / 60:.1f} minutes\n")


if __name__ == "__main__":
    sys.argv.append("experiment=dp_fedprox_dirichlet")
    main()
