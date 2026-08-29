"""
fedderm/federated/scaffold.py
-----------------------------
SCAFFOLD (Stochastic Controlled Averaging for Federated Learning)
implementation for Flower simulation backend.

Reference:
    Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated
    Learning", ICML 2020. (https://arxiv.org/abs/1910.06378)

Key components:
  1. PersistentControlVariates: In-memory manager for global c and per-client c_i.
     Persists across Flower virtual client instantiations to prevent state reset.
  2. ScaffoldClient: Flower NumPyClient that applies gradient correction
     g_tilde = g - c_i + c during local steps and computes Option II control updates.
  3. ScaffoldStrategy: Flower FedAvg extension that aggregates client parameter
     updates and updates global control variate c += (1/N) * sum(delta_c_i).
  4. run_scaffold: Complete simulation runner for SCAFFOLD driven by Hydra config.
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
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    Context,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig
from flwr.server.client_proxy import ClientProxy
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
from fedderm.federated.client import get_parameters, set_parameters
from fedderm.models import build_model
from fedderm.trainer import eval_one_epoch
from fedderm.utils import (
    evaluate,
    save_metrics,
    seed_everything,
    plot_training_curves,
    plot_confusion_matrix,
)


from collections.abc import Sequence


class PersistentControlVariates:
    """Persistent storage for global and per-client control variates.

    Flower creates and destroys virtual client instances across rounds.
    This manager holds the persistent state in simulation memory so that
    c_i is preserved when a client is re-sampled in subsequent rounds.

    Args:
        num_clients: Total number of clients (N).
        param_shapes: List of (shape, dtype) for all trainable model parameters.
    """

    def __init__(
        self,
        num_clients: int,
        param_templates: Sequence[torch.Tensor],
    ) -> None:
        self.num_clients = num_clients
        # Global control variate c
        self.c_global: list[torch.Tensor] = [
            torch.zeros_like(p, device="cpu") for p in param_templates
        ]
        # Per-client control variates {client_id: [c_i^1, c_i^2, ...]}
        self.client_controls: dict[int, list[torch.Tensor]] = {
            i: [torch.zeros_like(p, device="cpu") for p in param_templates]
            for i in range(num_clients)
        }
        # Buffer for latest delta_c received from clients during the current round
        self.round_delta_c: dict[int, list[torch.Tensor]] = {}

    def get_global_control(self, device: torch.device) -> list[torch.Tensor]:
        """Return a copy of the global control variate on the specified device."""
        return [c.clone().to(device) for c in self.c_global]

    def get_client_control(
        self, client_id: int, device: torch.device
    ) -> list[torch.Tensor]:
        """Return a copy of client i's control variate on the specified device."""
        return [c.clone().to(device) for c in self.client_controls[client_id]]

    def update_client_control(
        self, client_id: int, new_c_i: list[torch.Tensor], delta_c_i: list[torch.Tensor]
    ) -> None:
        """Persist client i's new control variate and record delta_c_i for server aggregation."""
        self.client_controls[client_id] = [c.detach().cpu().clone() for c in new_c_i]
        self.round_delta_c[client_id] = [dc.detach().cpu().clone() for dc in delta_c_i]

    def aggregate_global_control(self) -> None:
        """Update global control variate: c += (1 / N) * sum_{i in S} delta_c_i."""
        if not self.round_delta_c:
            return

        scale = 1.0 / float(self.num_clients)
        for delta_c in self.round_delta_c.values():
            for c_g, dc in zip(self.c_global, delta_c):
                c_g.add_(dc * scale)

        self.round_delta_c.clear()


class ScaffoldClient(fl.client.NumPyClient):
    """Flower NumPyClient implementing SCAFFOLD local gradient correction.

    At each local step:
        grad_corrected = grad - c_i + c
    Option II control variate update at end of local training:
        c_i^+ = c_i - c + (1 / (K * eta_l)) * (x - y_K)
        delta_c_i = c_i^+ - c_i

    Args:
        client_id:        Client index.
        train_loader:     DataLoader for local training shard.
        val_loader:       DataLoader for validation set.
        model:            MiniCNN instance.
        control_manager:  PersistentControlVariates reference.
        local_epochs:     Local epochs per round.
        lr:               Local learning rate (eta_l).
        weight_decay:     Optimizer weight decay.
        class_weights:    CrossEntropyLoss class weights.
        device:           Torch device.
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        control_manager: PersistentControlVariates,
        local_epochs: int,
        lr: float,
        weight_decay: float,
        class_weights: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.control_manager = control_manager
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.device = device

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Run SCAFFOLD local training with control variate gradient correction."""
        set_parameters(self.model, parameters)

        # Store initial global parameters x for Option II update
        init_params = [p.detach().clone() for p in self.model.parameters()]

        # Retrieve persistent c_i and global c
        c_i = self.control_manager.get_client_control(self.client_id, self.device)
        c_global = self.control_manager.get_global_control(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_steps = 0

        for _ in range(self.local_epochs):
            for images, targets in self.train_loader:
                images = images.to(self.device)
                targets = targets.view(-1).long().to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss.backward()

                # SCAFFOLD gradient correction: grad <- grad - c_i + c
                for param, c_i_p, c_g_p in zip(
                    self.model.parameters(), c_i, c_global
                ):
                    if param.grad is not None:
                        param.grad.add_(c_g_p - c_i_p)

                optimizer.step()

                total_loss += loss.item() * images.size(0)
                correct += (logits.argmax(dim=1) == targets).sum().item()
                total += images.size(0)
                num_steps += 1

        # Option II control variate update:
        # c_i^+ = c_i - c + (1 / (K * eta_l)) * (x - y_K)
        # delta_c_i = c_i^+ - c_i = (1 / (K * eta_l)) * (x - y_K) - c
        # (where K = num_steps, eta_l = self.lr)
        step_factor = 1.0 / (float(num_steps) * self.lr) if num_steps > 0 else 0.0

        new_c_i: list[torch.Tensor] = []
        delta_c_i: list[torch.Tensor] = []

        for p_init, p_final, c_i_p, c_g_p in zip(
            init_params, self.model.parameters(), c_i, c_global
        ):
            # diff = x - y_K
            diff = p_init - p_final.detach()
            # delta_c = step_factor * diff - c_g_p
            delta_c = step_factor * diff - c_g_p
            # c_i_plus = c_i_p + delta_c = c_i_p - c_g_p + step_factor * diff
            c_i_plus = c_i_p + delta_c

            new_c_i.append(c_i_plus)
            delta_c_i.append(delta_c)

        # Persist updated client control variate in manager
        self.control_manager.update_client_control(
            self.client_id, new_c_i, delta_c_i
        )

        num_train = len(self.train_loader.dataset)  # type: ignore[arg-type]
        train_acc = correct / total if total > 0 else 0.0
        train_loss = total_loss / total if total > 0 else 0.0

        return (
            get_parameters(self.model),
            num_train,
            {"train_loss": train_loss, "train_acc": train_acc},
        )

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        set_parameters(self.model, parameters)
        val_loss, val_acc, val_macro_f1 = eval_one_epoch(
            self.model, self.val_loader, self.criterion, self.device
        )
        num_val = len(self.val_loader.dataset)  # type: ignore[arg-type]
        return val_loss, num_val, {
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
        }


class ScaffoldStrategy(FedAvg):
    """Flower FedAvg strategy extension for SCAFFOLD.

    Aggregates model parameters and calls the PersistentControlVariates
    manager to update the global control variate c after each fit round.
    """

    def __init__(
        self,
        control_manager: PersistentControlVariates,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.control_manager = control_manager
        self.latest_parameters: Parameters | None = None

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        agg_params, agg_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if agg_params is not None:
            self.latest_parameters = agg_params
            # Update global control variate c from round delta_c_i
            self.control_manager.aggregate_global_control()

        return agg_params, agg_metrics


def run_scaffold(cfg: DictConfig) -> dict[str, Any]:
    """Execute SCAFFOLD simulation driven by Hydra config.

    Args:
        cfg: Hydra DictConfig containing federation.*, local_training.*, etc.

    Returns:
        dict of final test metrics on official DermaMNIST test split.
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

    print(f"\n[federated] algorithm: SCAFFOLD")
    print(f"[federated] device: {device}")
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

    # Initialize SCAFFOLD persistent control variates
    control_manager = PersistentControlVariates(
        num_clients=num_clients,
        param_templates=list(global_model.parameters()),
    )

    print(f"[federated] model: MiniCNN | params: {sum(p.numel() for p in global_model.parameters()):,}\n")

    best_val_macro_f1 = 0.0
    best_val_acc = 0.0
    best_ckpt_path = out_dir / "best_model.pt"
    history_records: dict[str, list[float]] = {
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    # -- Centralized evaluation callback -----------------------------------------
    eval_model = build_model(num_classes=cfg.num_classes, dropout=0.4).to(device)

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, dict[str, Scalar]] | None:
        nonlocal best_val_acc, best_val_macro_f1
        set_parameters(eval_model, parameters)
        val_loss, val_acc, val_macro_f1 = eval_one_epoch(
            eval_model, val_loader, val_criterion, device
        )
        history_records["val_loss"].append(val_loss)
        history_records["val_acc"].append(val_acc)
        history_records["val_macro_f1"].append(val_macro_f1)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_acc = val_acc
            torch.save(eval_model.state_dict(), best_ckpt_path)

        print(
            f"Round {server_round:2d}/{num_rounds} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}% | "
            f"val_mF1: {val_macro_f1*100:.2f}% | "
            f"best_val_mF1: {best_val_macro_f1*100:.2f}%"
        )
        return val_loss, {"val_acc": val_acc, "val_macro_f1": val_macro_f1}

    strategy = ScaffoldStrategy(
        control_manager=control_manager,
        fraction_fit=clients_per_round / num_clients,
        fraction_evaluate=0.0,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
        evaluate_fn=evaluate_fn,
    )

    # -- Flower client factory ---------------------------------------------------
    def client_fn(context: Context) -> fl.client.Client:
        client_id = context.node_id % num_clients
        model = build_model(num_classes=cfg.num_classes, dropout=0.4).to(device)
        client = ScaffoldClient(
            client_id=client_id,
            train_loader=client_loaders[client_id],
            val_loader=val_loader,
            model=model,
            control_manager=control_manager,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            class_weights=class_weights,
            device=device,
        )
        return client.to_client()

    # -- Run simulation ----------------------------------------------------------
    t0 = time.time()
    print("[federated] starting SCAFFOLD Flower simulation...\n")

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
    test_metrics["best_val_macro_f1"] = best_val_macro_f1
    test_metrics["num_rounds"] = num_rounds
    test_metrics["num_clients"] = num_clients
    test_metrics["dirichlet_alpha"] = alpha
    test_metrics["local_epochs"] = local_epochs
    test_metrics["algorithm"] = "SCAFFOLD"

    save_metrics(test_metrics, out_dir / "test_metrics.json")

    # -- Print results -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEDERATED TEST RESULTS (SCAFFOLD)")
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
    plot_training_curves(
        [0.0] * len(history_records["val_loss"]),
        history_records["val_loss"],
        [0.0] * len(history_records["val_acc"]),
        history_records["val_acc"],
        out_dir / "training_curves.png",
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        out_dir / "confusion_matrix.png",
        normalize=True,
    )
    print(f"\n[federated] plots saved to {out_dir}/")

    return test_metrics
