"""Shared utilities: metrics, seeding, visualisation."""

from fedderm.utils.metrics import evaluate, save_metrics
from fedderm.utils.seed import seed_everything
from fedderm.utils.plotting import plot_training_curves, plot_confusion_matrix

__all__ = [
    "evaluate",
    "save_metrics",
    "seed_everything",
    "plot_training_curves",
    "plot_confusion_matrix",
]
