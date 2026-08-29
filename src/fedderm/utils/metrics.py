"""
fedderm/utils/metrics.py
------------------------
Evaluation metrics for multi-class classification.
Wraps sklearn to compute accuracy, per-class F1, balanced accuracy,
and produce a confusion matrix -- all needed for the imbalanced DermaMNIST task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate(
    model: "torch.nn.Module",
    loader: "torch.utils.data.DataLoader",
    device: torch.device,
) -> dict[str, Any]:
    """Run inference on loader and return a metrics dict.

    Returns:
        dict with keys: accuracy, balanced_accuracy, macro_f1, weighted_f1,
                        per_class_f1, confusion_matrix (as list-of-lists),
                        y_true (list), y_pred (list).
    """
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            # medmnist targets are shape (N, 1) -- squeeze to (N,)
            tgts = targets.squeeze().numpy().tolist()
            if isinstance(tgts, int):
                tgts = [tgts]
            all_preds.extend(preds)
            all_targets.extend(tgts)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    cm = confusion_matrix(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0
        ),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Save metrics dict to a JSON file (excluding large arrays)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save everything except raw prediction arrays (those are large)
    saveable = {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred")}
    with open(path, "w") as f:
        json.dump(saveable, f, indent=2)
