"""
fedderm/utils/plotting.py
-------------------------
Visualisation utilities: training curves, confusion matrix.
All plots are saved to disk (no interactive display needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless backend -- no display required
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_accs: Sequence[float],
    val_accs: Sequence[float],
    out_path: str | Path,
) -> None:
    """Save a 2-panel figure: loss curves (left) and accuracy curves (right).

    Args:
        train_losses: Per-epoch training loss values.
        val_losses:   Per-epoch validation loss values.
        train_accs:   Per-epoch training accuracy values.
        val_accs:     Per-epoch validation accuracy values.
        out_path:     File path to save the figure (PNG).
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train", linewidth=1.5)
    ax1.plot(epochs, val_losses,   label="Val",   linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in train_accs], label="Train", linewidth=1.5)
    ax2.plot(epochs, [a * 100 for a in val_accs],   label="Val",   linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy curves")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    class_names: Sequence[str],
    out_path: str | Path,
    normalize: bool = True,
) -> None:
    """Save a confusion matrix heatmap.

    Args:
        cm:           Confusion matrix as list-of-lists (rows=true, cols=pred).
        class_names:  Ordered list of class label strings.
        out_path:     File path to save the figure (PNG).
        normalize:    If True, normalise rows to show recall per class.
    """
    cm_arr = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_arr = np.where(row_sums > 0, cm_arr / row_sums, 0.0)
        fmt_str = ".2f"
        title = "Confusion matrix (row-normalised recall)"
    else:
        fmt_str = "d"
        title = "Confusion matrix (counts)"

    n = len(class_names)
    # Shorten long labels for readability
    short_names = [
        n_[:18] + ".." if len(n_) > 20 else n_
        for n_ in class_names
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm_arr.max() / 2.0 if not normalize else 0.5
    for i in range(n):
        for j in range(n):
            val = cm_arr[i, j]
            text = f"{val:{fmt_str}}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7,
                    color="white" if val > thresh else "black")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
