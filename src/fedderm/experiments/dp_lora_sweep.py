"""
fedderm/experiments/dp_lora_sweep.py
------------------------------------
Automated DP-LoRA (ViT-B/16 + LoRA r=8) multi-seed privacy-utility sweep for FedDerm.
"""

from __future__ import annotations

import copy
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, open_dict

import hydra
from hydra.core.global_hydra import GlobalHydra
from fedderm.federated.dp_lora_simulation import run_federated_dp_lora

DEFAULT_NOISE_MULTIPLIERS = [0.3, 0.5, 1.0, 2.0]
DEFAULT_SEEDS = [42, 43, 44]


def _extract_metric_pct(data: dict[str, Any], key: str, fallback_key: str) -> float:
    """Safely extract a metric float percentage from dictionary."""
    val = data.get(key)
    if val is None:
        val = data.get(fallback_key, 0.0)
    val_float = float(val) if val is not None else 0.0
    return val_float * 100.0 if val_float <= 1.0 else val_float


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
    non_dp_data: dict[str, Any] | None,
    out_path: Path,
) -> None:
    """Generate privacy-utility tradeoff plots with error bars (mean +- std)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sorted_dp = sorted(
        [d for d in summary_data if d["server_epsilon"] != float("inf")],
        key=lambda x: x["server_epsilon"],
    )

    eps_vals = [d["server_epsilon"] for d in sorted_dp]
    acc_means = [d["accuracy_mean"] for d in sorted_dp]
    acc_stds = [d["accuracy_std"] for d in sorted_dp]
    f1_means = [d["macro_f1_mean"] for d in sorted_dp]
    f1_stds = [d["macro_f1_std"] for d in sorted_dp]

    # Panel 1: Test Accuracy vs Server Epsilon
    ax1.errorbar(
        eps_vals,
        acc_means,
        yerr=acc_stds,
        fmt="-o",
        color="#1f77b4",
        ecolor="#aec7e8",
        elinewidth=2,
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=7,
        label="DP-LoRA ViT-B/16 (Mean +- 1 Std)",
    )

    if non_dp_data is not None:
        non_dp_acc = _extract_metric_pct(non_dp_data, "accuracy", "accuracy_mean")
        ax1.axhline(
            y=non_dp_acc,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.8,
            label=f"Non-DP ViT-LoRA ({non_dp_acc:.2f}%)",
        )

    # MiniCNN reference line
    ax1.axhline(
        y=67.73,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.5,
        label="Non-DP MiniCNN Ceiling (67.73%)",
    )

    ax1.set_xscale("log")
    ax1.set_xlabel("Server Privacy Budget eps (log scale, lower = stronger privacy)", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax1.set_title("DP-LoRA: Privacy vs. Accuracy Tradeoff", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="lower right")

    # Panel 2: Macro F1 vs Server Epsilon
    ax2.errorbar(
        eps_vals,
        f1_means,
        yerr=f1_stds,
        fmt="-s",
        color="#d62728",
        ecolor="#ff9896",
        elinewidth=2,
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=7,
        label="DP-LoRA ViT-B/16 (Mean +- 1 Std)",
    )

    if non_dp_data is not None:
        non_dp_f1 = _extract_metric_pct(non_dp_data, "macro_f1", "macro_f1_mean")
        ax2.axhline(
            y=non_dp_f1,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.8,
            label=f"Non-DP ViT-LoRA ({non_dp_f1:.2f}%)",
        )

    # MiniCNN reference line
    ax2.axhline(
        y=20.36,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.5,
        label="Non-DP MiniCNN Ceiling (20.36%)",
    )

    ax2.set_xscale("log")
    ax2.set_xlabel("Server Privacy Budget eps (log scale, lower = stronger privacy)", fontsize=11)
    ax2.set_ylabel("Macro F1-Score (%)", fontsize=11)
    ax2.set_title("DP-LoRA: Privacy vs. Macro F1 Tradeoff", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[sweep] Privacy-utility curve with error bars saved to {out_path}")


def save_summary_table_csv(
    summary_data: list[dict[str, Any]],
    non_dp_data: dict[str, Any] | None,
    out_path: Path,
) -> None:
    """Save aggregated sweep metrics to CSV."""
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
        "n_seeds",
    ]

    rows = []
    if non_dp_data is not None:
        non_dp_acc = _extract_metric_pct(non_dp_data, "accuracy", "accuracy_mean")
        non_dp_bal_acc = _extract_metric_pct(non_dp_data, "balanced_accuracy", "balanced_accuracy_mean")
        non_dp_mf1 = _extract_metric_pct(non_dp_data, "macro_f1", "macro_f1_mean")
        non_dp_wf1 = _extract_metric_pct(non_dp_data, "weighted_f1", "weighted_f1_mean")
        rows.append({
            "noise_multiplier": 0.0,
            "server_epsilon": "inf",
            "client_epsilon": "inf",
            "accuracy_mean": f"{non_dp_acc:.2f}",
            "accuracy_std": "0.00",
            "balanced_accuracy_mean": f"{non_dp_bal_acc:.2f}",
            "balanced_accuracy_std": "0.00",
            "macro_f1_mean": f"{non_dp_mf1:.2f}",
            "macro_f1_std": "0.00",
            "weighted_f1_mean": f"{non_dp_wf1:.2f}",
            "weighted_f1_std": "0.00",
            "n_seeds": 1,
        })

    for d in summary_data:
        rows.append({
            "noise_multiplier": d.get("noise_multiplier", ""),
            "server_epsilon": f"{d['server_epsilon']:.2f}" if d["server_epsilon"] != float("inf") else "inf",
            "client_epsilon": f"{d['client_epsilon']:.2f}" if d["client_epsilon"] != float("inf") else "inf",
            "accuracy_mean": f"{d['accuracy_mean']:.2f}",
            "accuracy_std": f"{d['accuracy_std']:.2f}",
            "balanced_accuracy_mean": f"{d['balanced_accuracy_mean']:.2f}",
            "balanced_accuracy_std": f"{d['balanced_accuracy_std']:.2f}",
            "macro_f1_mean": f"{d['macro_f1_mean']:.2f}",
            "macro_f1_std": f"{d['macro_f1_std']:.2f}",
            "weighted_f1_mean": f"{d['weighted_f1_mean']:.2f}",
            "weighted_f1_std": f"{d['weighted_f1_std']:.2f}",
            "n_seeds": d["n_seeds"],
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sweep] Summary CSV saved to {out_path}")


def run_sweep(
    noise_multipliers: list[float] | None = None,
    seeds: list[int] | None = None,
    run_sanity: bool = True,
    run_dp: bool = True,
    results_base: str = "results/dp_lora_fedprox",
    rounds_override: int | None = None,
    clients_override: int | None = None,
    sampled_override: int | None = None,
    epochs_override: int | None = None,
) -> None:
    """Run full automated DP-LoRA sweep with optional hyperparameter overrides."""
    if noise_multipliers is None:
        noise_multipliers = DEFAULT_NOISE_MULTIPLIERS
    if seeds is None:
        seeds = DEFAULT_SEEDS

    base_out = Path(results_base)
    base_out.mkdir(parents=True, exist_ok=True)

    # Ensure Hydra is clean
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with hydra.initialize(version_base="1.3", config_path="../../../configs"):
        cfg_template = hydra.compose(config_name="config", overrides=["experiment=dp_lora_fedprox_dirichlet"])

    # Apply overrides if provided (useful for fast smoke-tests)
    with open_dict(cfg_template):
        if rounds_override is not None:
            cfg_template.federation.rounds = rounds_override
        if clients_override is not None:
            cfg_template.federation.num_clients = clients_override
        if sampled_override is not None:
            cfg_template.federation.clients_per_round = sampled_override
        if epochs_override is not None:
            cfg_template.local_training.epochs = epochs_override

    non_dp_metrics: dict[str, Any] | None = None
    sweep_start = time.time()

    # 1. Non-DP Sanity Check Run
    if run_sanity:
        print("\n" + "=" * 70)
        print("STAGE 1: NON-DP SANITY CHECK (ViT-B/16 + LoRA, sigma=0.0, seed=42)")
        print("=" * 70)
        sanity_dir = base_out / "non_dp_sanity"
        sanity_dir.mkdir(parents=True, exist_ok=True)

        cfg_sanity = copy.deepcopy(cfg_template)
        with open_dict(cfg_sanity):
            cfg_sanity.seed = 42
            cfg_sanity.federation.partition_seed = 42
            cfg_sanity.privacy.noise_multiplier = 0.0
            cfg_sanity.output_dir = str(sanity_dir)

        t_run0 = time.time()
        non_dp_metrics = run_federated_dp_lora(cfg_sanity)
        print(f"[sanity] Non-DP Sanity Check completed in {(time.time() - t_run0) / 60:.1f} min")

    # 2. Multi-Seed DP Sweep
    summary_results: list[dict[str, Any]] = []

    if run_dp:
        total_runs = len(noise_multipliers) * len(seeds)
        current_run = 0

        print("\n" + "=" * 70)
        print(f"STAGE 2: MULTI-SEED DP SWEEP ({len(noise_multipliers)} noise levels x {len(seeds)} seeds = {total_runs} runs)")
        print("=" * 70)

        for nm in noise_multipliers:
            seed_metrics_list: list[dict[str, Any]] = []
            print(f"\n>>> Running Noise Multiplier sigma = {nm} <<<")

            for s in seeds:
                current_run += 1
                run_out_dir = base_out / f"sigma_{nm}" / f"seed_{s}"
                run_out_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n--- Run {current_run}/{total_runs} (sigma={nm}, seed={s}) ---")
                cfg = copy.deepcopy(cfg_template)
                with open_dict(cfg):
                    cfg.seed = s
                    cfg.federation.partition_seed = 42  # Fixed partition across all seeds
                    cfg.privacy.noise_multiplier = nm
                    cfg.output_dir = str(run_out_dir)

                t_run = time.time()
                metrics = run_federated_dp_lora(cfg)
                metrics["seed"] = s
                seed_metrics_list.append(metrics)
                print(f"[run {current_run}/{total_runs}] completed in {(time.time() - t_run) / 60:.1f} min")

            # Compute aggregated statistics for this noise level
            stats = compute_statistics(seed_metrics_list)
            stats["noise_multiplier"] = nm
            summary_results.append(stats)

            print("\n" + "-" * 60)
            print(f"Aggregated Stats for sigma = {nm} ({len(seeds)} seeds):")
            print(f"  Server Epsilon:        {stats['server_epsilon']:.2f}")
            print(f"  Client Epsilon:        {stats['client_epsilon']:.2f}")
            print(f"  Accuracy:              {stats['accuracy_mean']:.2f}% +- {stats['accuracy_std']:.2f}%")
            print(f"  Balanced Accuracy:     {stats['balanced_accuracy_mean']:.2f}% +- {stats['balanced_accuracy_std']:.2f}%")
            print(f"  Macro F1:              {stats['macro_f1_mean']:.2f}% +- {stats['macro_f1_std']:.2f}%")
            print(f"  Weighted F1:           {stats['weighted_f1_mean']:.2f}% +- {stats['weighted_f1_std']:.2f}%")
            print("-" * 60)

        # Save multi-seed summary JSON
        summary_json_path = base_out / "multiseed_summary.json"
        summary_payload = {
            "model": "vit_base_patch16_224_lora_r8",
            "federation": {
                "algorithm": "dp_lora_fedprox",
                "mu": 0.01,
                "rounds": cfg_template.federation.rounds,
                "clients": cfg_template.federation.num_clients,
                "sampled_per_round": cfg_template.federation.clients_per_round,
                "local_epochs": cfg_template.local_training.epochs,
                "dirichlet_alpha": 0.3,
                "partition_seed": 42,
            },
            "noise_multipliers": noise_multipliers,
            "seeds": seeds,
            "non_dp_reference": non_dp_metrics,
            "sweep_summary": summary_results,
        }
        with open(summary_json_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\n[sweep] Aggregated summary JSON saved to {summary_json_path}")

        # Save summary CSV
        summary_csv_path = base_out / "multiseed_summary.csv"
        save_summary_table_csv(summary_results, non_dp_metrics, summary_csv_path)

        # Plot privacy-utility tradeoff curve
        plot_path = base_out / "privacy_utility_tradeoff_multiseed.png"
        plot_privacy_utility_curve_with_errorbars(summary_results, non_dp_metrics, plot_path)

    total_sweep_time = time.time() - sweep_start
    print("\n" + "=" * 70)
    print(f"DP-LORA SWEEP COMPLETE in {total_sweep_time / 60:.1f} min ({total_sweep_time / 3600:.2f} hours)")
    print(f"Outputs written to: {base_out.resolve()}")
    print("=" * 70)
