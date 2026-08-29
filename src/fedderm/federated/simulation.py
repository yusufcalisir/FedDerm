"""
fedderm/federated/simulation.py
--------------------------------
FedAvg simulation driver using Flower's simulation backend.

Orchestrates:
  1. Non-IID Dirichlet partitioning of DermaMNIST training data
  2. Per-round client sampling and local training via Flower
  3. FedAvg server-side aggregation and centralized validation tracking
  4. Post-training evaluation on the official test split
  5. Result persistence mirroring the centralized baseline structure
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

import flwr as fl
from flwr.common import (
    Context,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from fedderm.data import (
    get_dataloaders,
    get_class_names,
    get_class_weights,
    dirichlet_partition,
    report_partition,
    make_client_loaders,
)
from fedderm.data.partition import _collect_labels_fast
from fedderm.federated.client import DermClient, get_parameters, set_parameters
from fedderm.models import build_model
from fedderm.trainer import eval_one_epoch
from fedderm.utils import (
    evaluate,
    save_metrics,
    seed_everything,
    plot_training_curves,
    plot_confusion_matrix,
)


class SaveableFedAvg(FedAvg):
    """FedAvg strategy that caches the latest aggregated parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.latest_parameters: Parameters | None = None

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)
        if agg_params is not None:
            self.latest_parameters = agg_params
        return agg_params, agg_metrics


def run_federated(cfg: DictConfig) -> dict[str, Any]:
    """Full FedAvg simulation run driven by a Hydra config.

    Design choices for CPU-only execution:
        - 10 clients (hospitals), 5 sampled per round (50% participation)
        - 20 communication rounds: captures convergence behavior in ~30-45 min
        - 3 local epochs per round: standard FedAvg, avoids excessive client drift
        - Dirichlet alpha=0.3: realistic hospital heterogeneity without making
          minority classes completely absent from any single client

    Args:
        cfg: Hydra DictConfig with federation.*, local_training.*, and output_dir.

    Returns:
        dict of final test metrics.
    """
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_clients = cfg.federation.num_clients
    clients_per_round = cfg.federation.clients_per_round
    num_rounds = cfg.federation.rounds
    alpha = cfg.federation.dirichlet_alpha
    local_epochs = cfg.local_training.epochs
    batch_size = cfg.local_training.batch_size
    lr = cfg.local_training.lr
    weight_decay = cfg.local_training.weight_decay
    image_size = cfg.image_size

    print(f"\n[federated] device: {device}")
    print(f"[federated] clients: {num_clients} | sampled/round: {clients_per_round}")
    print(f"[federated] rounds: {num_rounds} | local_epochs: {local_epochs}")
    print(f"[federated] dirichlet alpha: {alpha}")
    print(f"[federated] output: {out_dir}\n")

    # -- Partition training data across clients -----------------------------------
    print("[federated] partitioning data...")
    labels = _collect_labels_fast("data", image_size)
    dummy_ds = type("_DS", (), {"__len__": lambda self: len(labels)})()
    client_indices = dirichlet_partition(
        dummy_ds,
        labels,
        num_clients=num_clients,
        alpha=alpha,
        min_samples_per_client=cfg.federation.get("min_samples_per_client", 5),
        seed=cfg.seed,
    )

    class_names = get_class_names()
    partition_report = report_partition(
        client_indices, labels, class_names, out_path=out_dir / "partition_report.json"
    )
    print(f"\n[federated] partition report saved to {out_dir}/partition_report.json")

    # -- Build per-client loaders ------------------------------------------------
    client_loaders = make_client_loaders(
        root="data",
        image_size=image_size,
        client_indices=client_indices,
        batch_size=batch_size,
        augment=True,
    )

    # -- Shared objects ----------------------------------------------------------
    _, val_loader, test_loader = get_dataloaders(
        data_root="data",
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,
    )
    class_weights = get_class_weights("data", image_size=image_size).to(device)
    val_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initial global model and parameters
    global_model = build_model(
        num_classes=cfg.num_classes,
        dropout=cfg.model.get("dropout", 0.4),
    ).to(device)
    initial_params = ndarrays_to_parameters(get_parameters(global_model))

    print(f"[federated] model: MiniCNN | params: {sum(p.numel() for p in global_model.parameters()):,}\n")

    best_val_acc = 0.0
    best_ckpt_path = out_dir / "best_model.pt"
    history_records: dict[str, list[float]] = {
        "val_loss": [],
        "val_acc": [],
    }

    # -- Centralized evaluation function for server rounds -----------------------
    eval_model = build_model(num_classes=cfg.num_classes, dropout=0.4).to(device)

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, dict[str, Scalar]] | None:
        nonlocal best_val_acc
        set_parameters(eval_model, parameters)
        val_loss, val_acc = eval_one_epoch(eval_model, val_loader, val_criterion, device)
        history_records["val_loss"].append(val_loss)
        history_records["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(eval_model.state_dict(), best_ckpt_path)

        print(
            f"Round {server_round:2d}/{num_rounds} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}% | "
            f"best_val_acc: {best_val_acc*100:.2f}%"
        )
        return float(val_loss), {"val_acc": float(val_acc)}

    strategy = SaveableFedAvg(
        fraction_fit=clients_per_round / num_clients,
        fraction_evaluate=0.0,  # evaluate centralized on val_loader instead
        min_fit_clients=clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
        evaluate_fn=evaluate_fn,
    )

    # -- Flower client factory ---------------------------------------------------
    def client_fn(context: Context) -> fl.client.Client:
        """Flower client factory: instantiate the correct client for each node."""
        client_id = int(context.node_id) % num_clients
        model = build_model(num_classes=cfg.num_classes, dropout=0.4).to(device)
        client = DermClient(
            client_id=client_id,
            train_loader=client_loaders[client_id],
            val_loader=val_loader,
            model=model,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            class_weights=class_weights,
            device=device,
        )
        return client.to_client()

    # -- Run simulation ----------------------------------------------------------
    t0 = time.time()
    print("[federated] starting Flower simulation...\n")

    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False},
    )

    total_time = time.time() - t0
    print(f"\n[federated] simulation complete in {total_time / 60:.1f} min")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history_records, f, indent=2)

    # -- Load best model for test evaluation -------------------------------------
    if best_ckpt_path.exists():
        global_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"[federated] loaded best model from {best_ckpt_path}")
    elif strategy.latest_parameters is not None:
        final_params = parameters_to_ndarrays(strategy.latest_parameters)
        set_parameters(global_model, final_params)
        torch.save(global_model.state_dict(), best_ckpt_path)

    # -- Test evaluation ---------------------------------------------------------
    test_metrics = evaluate(global_model, test_loader, device)
    test_metrics["training_time_seconds"] = total_time
    test_metrics["best_val_acc"] = best_val_acc
    test_metrics["num_rounds"] = num_rounds
    test_metrics["num_clients"] = num_clients
    test_metrics["dirichlet_alpha"] = alpha
    test_metrics["local_epochs"] = local_epochs

    save_metrics(test_metrics, out_dir / "test_metrics.json")

    # -- Print results -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEDERATED TEST RESULTS (FedAvg)")
    print("=" * 60)
    print(f"  Accuracy:          {test_metrics['accuracy']*100:.2f}%")
    print(f"  Balanced accuracy: {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro F1:          {test_metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted F1:       {test_metrics['weighted_f1']*100:.2f}%")
    print("\nPer-class F1:")
    for name, f1 in zip(class_names, test_metrics["per_class_f1"]):
        print(f"  {name[:40]:<40}  {f1*100:.1f}%")
    print("\nClassification report:")
    print(test_metrics["classification_report"])

    # -- Plots -------------------------------------------------------------------
    if history_records["val_loss"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        rounds = list(range(1, len(history_records["val_loss"]) + 1))

        ax1.plot(rounds, history_records["val_loss"], marker="o", color="#2563eb", linewidth=2)
        ax1.set_title("Global Validation Loss per Round")
        ax1.set_xlabel("Communication Round")
        ax1.set_ylabel("Loss")
        ax1.grid(alpha=0.3)

        ax2.plot(
            rounds,
            [acc * 100 for acc in history_records["val_acc"]],
            marker="o",
            color="#16a34a",
            linewidth=2,
        )
        ax2.set_title("Global Validation Accuracy per Round")
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=120)
        plt.close(fig)

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        out_dir / "confusion_matrix.png",
        normalize=True,
    )
    print(f"\n[federated] plots saved to {out_dir}/")

    return test_metrics
