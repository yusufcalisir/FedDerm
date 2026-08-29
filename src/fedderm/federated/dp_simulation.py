"""
fedderm/federated/dp_simulation.py
----------------------------------
Federated DP-SGD + FedProx simulation driver using Flower and Opacus.
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
from fedderm.federated.dp_client import DPDermClient
from fedderm.federated.simulation import SaveableFedAvg
from fedderm.models import build_model
from fedderm.privacy.accountant import FederatedPrivacyAccountant
from fedderm.privacy.engine import check_opacus_compatibility
from fedderm.trainer import eval_one_epoch
from fedderm.utils import (
    evaluate,
    save_metrics,
    seed_everything,
    plot_training_curves,
    plot_confusion_matrix,
)


def run_federated_dp(cfg: DictConfig) -> dict[str, Any]:
    """Execute DP-FedProx simulation driven by Hydra config.

    Args:
        cfg: Hydra configuration.

    Returns:
        dict of test metrics including computed differential privacy epsilon.
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
    mu = float(cfg.federation.get("mu", 0.01))

    # Privacy configuration
    noise_multiplier = float(cfg.privacy.get("noise_multiplier", 1.0))
    max_grad_norm = float(cfg.privacy.get("max_grad_norm", 1.0))
    target_delta = float(cfg.privacy.get("target_delta", 1.0e-5))

    print(f"\n[federated-dp] algorithm: DP-FedProx (mu={mu})")
    print(f"[federated-dp] privacy: noise_multiplier={noise_multiplier}, max_grad_norm={max_grad_norm}, delta={target_delta}")
    print(f"[federated-dp] device: {device}")
    print(f"[federated-dp] clients: {num_clients} | sampled/round: {clients_per_round}")
    print(f"[federated-dp] rounds: {num_rounds} | local_epochs: {local_epochs}")
    print(f"[federated-dp] dirichlet alpha: {alpha}")
    print(f"[federated-dp] output: {out_dir}\n")

    # -- Validate Opacus compatibility --------------------------------------------
    test_model = build_model(num_classes=cfg.num_classes, dropout=0.4)
    compat_errors = check_opacus_compatibility(test_model)
    if compat_errors:
        print(f"[federated-dp] WARNING: Opacus compatibility errors: {compat_errors}")
    else:
        print("[federated-dp] Opacus ModuleValidator: PASSED (GroupNorm MiniCNN is 100% compatible)")

    # -- Partition training data across clients -----------------------------------
    labels = _collect_labels_fast("data", image_size)
    dummy_ds = type("_DS", (), {"__len__": lambda self: len(labels)})()
    partition_seed = int(cfg.federation.get("partition_seed", 42))
    client_indices = dirichlet_partition(
        dummy_ds,
        labels,
        num_clients=num_clients,
        alpha=alpha,
        min_samples_per_client=cfg.federation.get("min_samples_per_client", 5),
        seed=partition_seed,
    )

    class_names = get_class_names()
    partition_report = report_partition(
        client_indices, labels, class_names, out_path=out_dir / "partition_report.json"
    )

    # Build client DataLoaders
    client_loaders = make_client_loaders(
        root="data",
        image_size=image_size,
        client_indices=client_indices,
        batch_size=batch_size,
        augment=True,
    )

    # Shared loaders and loss
    _, val_loader, test_loader = get_dataloaders(
        data_root="data",
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,
    )
    class_weights = get_class_weights("data", image_size=image_size).to(device)
    val_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initial global model
    global_model = build_model(
        num_classes=cfg.num_classes,
        dropout=cfg.model.get("dropout", 0.4),
    ).to(device)
    initial_params = ndarrays_to_parameters(get_parameters(global_model))

    # Initialize Federated Privacy Accountant
    total_train_samples = len(labels)
    privacy_accountant = FederatedPrivacyAccountant(
        target_delta=target_delta,
        total_samples=total_train_samples,
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        batch_size=batch_size,
    )

    best_val_macro_f1 = 0.0
    best_val_acc = 0.0
    best_ckpt_path = out_dir / "best_model.pt"
    history_records: dict[str, list[float]] = {
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "server_epsilon": [],
        "client_epsilon": [],
    }

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

        if server_round > 0:
            # Average steps per sampled client in round: local_epochs * (avg_shard / batch_size)
            avg_steps_per_client = local_epochs * int(np.ceil((total_train_samples / num_clients) / batch_size))
            privacy_accountant.step_round(
                noise_multiplier=noise_multiplier,
                num_local_steps_per_client=avg_steps_per_client,
                client_dataset_size=total_train_samples // num_clients,
            )

        eps_server = privacy_accountant.get_epsilon()
        eps_client = privacy_accountant.get_client_epsilon()

        history_records["val_loss"].append(val_loss)
        history_records["val_acc"].append(val_acc)
        history_records["val_macro_f1"].append(val_macro_f1)
        history_records["server_epsilon"].append(eps_server)
        history_records["client_epsilon"].append(eps_client)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_acc = val_acc
            torch.save(eval_model.state_dict(), best_ckpt_path)

        eps_str = f"eps(server)={eps_server:.2f}" if eps_server != float("inf") else "eps=inf"
        print(
            f"Round {server_round:2d}/{num_rounds} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}% | "
            f"val_mF1: {val_macro_f1*100:.2f}% | {eps_str}"
        )
        return val_loss, {"val_acc": val_acc, "val_macro_f1": val_macro_f1}

    strategy = SaveableFedAvg(
        fraction_fit=clients_per_round / num_clients,
        fraction_evaluate=0.0,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
        evaluate_fn=evaluate_fn,
    )

    def client_fn(context: Context) -> fl.client.Client:
        client_id = context.node_id % num_clients
        model = build_model(num_classes=cfg.num_classes, dropout=0.4).to(device)
        client = DPDermClient(
            client_id=client_id,
            train_loader=client_loaders[client_id],
            val_loader=val_loader,
            model=model,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            class_weights=class_weights,
            device=device,
            mu=mu,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        return client.to_client()

    t0 = time.time()
    print("[federated-dp] starting Flower DP simulation...\n")

    import ray

    if ray.is_initialized():
        ray.shutdown()

    try:
        start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
            ray_init_args={"num_cpus": 4, "ignore_reinit_error": True, "include_dashboard": False},
        )
    finally:
        if ray.is_initialized():
            ray.shutdown()

    total_time = time.time() - t0
    print(f"\n[federated-dp] simulation complete in {total_time / 60:.1f} min")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history_records, f, indent=2)

    # Load best model
    if best_ckpt_path.exists():
        global_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    elif strategy.latest_parameters is not None:
        final_params = parameters_to_ndarrays(strategy.latest_parameters)
        set_parameters(global_model, final_params)
        torch.save(global_model.state_dict(), best_ckpt_path)

    # Test evaluation
    test_metrics = evaluate(global_model, test_loader, device)
    test_metrics["training_time_seconds"] = total_time
    test_metrics["best_val_acc"] = best_val_acc
    test_metrics["best_val_macro_f1"] = best_val_macro_f1
    test_metrics["num_rounds"] = num_rounds
    test_metrics["num_clients"] = num_clients
    test_metrics["dirichlet_alpha"] = alpha
    test_metrics["local_epochs"] = local_epochs
    test_metrics["algorithm"] = "DP-FedProx"
    test_metrics["mu"] = mu
    test_metrics["noise_multiplier"] = noise_multiplier
    test_metrics["max_grad_norm"] = max_grad_norm
    test_metrics["target_delta"] = target_delta
    test_metrics["server_epsilon"] = privacy_accountant.get_epsilon()
    test_metrics["client_epsilon"] = privacy_accountant.get_client_epsilon()

    save_metrics(test_metrics, out_dir / "test_metrics.json")

    # Print summary
    print("\n" + "=" * 60)
    print(f"DP-FEDPROX TEST RESULTS (sigma={noise_multiplier}, eps={test_metrics['server_epsilon']:.2f})")
    print("=" * 60)
    print(f"  Accuracy:          {test_metrics['accuracy']*100:.2f}%")
    print(f"  Balanced accuracy: {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro F1:          {test_metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted F1:       {test_metrics['weighted_f1']*100:.2f}%")
    print(f"  Server Epsilon:    {test_metrics['server_epsilon']:.2f} (delta={target_delta})")
    print(f"  Client Epsilon:    {test_metrics['client_epsilon']:.2f}")

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

    return test_metrics
