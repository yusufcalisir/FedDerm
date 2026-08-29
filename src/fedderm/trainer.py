"""
fedderm/trainer.py
------------------
Centralized training loop for DermaMNIST classification.

Designed to be called from scripts/train_centralized.py via Hydra.
Handles: optimizer setup, LR scheduling, training loop, checkpointing,
metric logging, and result persistence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from fedderm.data import get_dataloaders, get_class_names, get_class_weights
from fedderm.models import build_model
from fedderm.utils import (
    evaluate,
    save_metrics,
    seed_everything,
    plot_training_curves,
    plot_confusion_matrix,
)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (mean_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.squeeze().long().to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def eval_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Single-pass validation: returns (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.squeeze().long().to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def run_centralized(cfg: DictConfig) -> dict[str, Any]:
    """Full centralized training run driven by a Hydra config.

    Args:
        cfg: Hydra DictConfig containing all hyperparameters.

    Returns:
        dict of final test metrics.
    """
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[trainer] device: {device}")
    print(f"[trainer] output: {out_dir}\n")

    # -- Data ------------------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root="data",
        image_size=cfg.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=0,  # Windows-safe
    )

    # Inverse-frequency class weights to handle nevi dominance
    class_weights = get_class_weights("data", image_size=cfg.image_size).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # -- Model -----------------------------------------------------------------
    model = build_model(
        num_classes=cfg.num_classes,
        dropout=cfg.model.get("dropout", 0.4),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[trainer] model: MiniCNN | params: {total_params:,}")

    # -- Optimizer & scheduler -------------------------------------------------
    optimizer: torch.optim.Optimizer
    if cfg.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
    elif cfg.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.lr,
            momentum=0.9,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")

    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    if cfg.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )

    # -- Training loop ---------------------------------------------------------
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }

    best_val_acc = 0.0
    best_ckpt_path = out_dir / "best_model.pt"
    t0 = time.time()

    for epoch in range(1, cfg.training.epochs + 1):
        t_ep = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Checkpoint best model by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)

        ep_time = time.time() - t_ep
        print(
            f"Epoch {epoch:3d}/{cfg.training.epochs} | "
            f"loss {train_loss:.4f} | acc {train_acc*100:.1f}% | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc*100:.1f}% | "
            f"{ep_time:.0f}s"
        )

    total_time = time.time() - t0
    print(f"\n[trainer] training complete in {total_time/60:.1f} min")

    # -- Save training history -------------------------------------------------
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # -- Final evaluation on test set (best checkpoint) -----------------------
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    test_metrics["training_time_seconds"] = total_time
    test_metrics["best_val_acc"] = best_val_acc

    save_metrics(test_metrics, out_dir / "test_metrics.json")

    # Print final results
    class_names = get_class_names()
    print("\n" + "=" * 60)
    print("TEST RESULTS")
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

    # -- Plots -----------------------------------------------------------------
    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"], history["val_acc"],
        out_dir / "training_curves.png",
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        out_dir / "confusion_matrix.png",
        normalize=True,
    )
    print(f"\n[trainer] plots saved to {out_dir}/")

    return test_metrics



