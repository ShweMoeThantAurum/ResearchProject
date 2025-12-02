"""
High-level comparison plot across datasets for AEFL thesis figures.
Shows cross-dataset accuracy/energy comparison in one figure.
"""

import os
import matplotlib.pyplot as plt
from .load_results import load_metrics
from .plot_utils import base_plot, ensure_plot_dir

DATASETS = ["sz", "los", "pems08"]


def plot_cross_dataset_accuracy(mode):
    """Compare MAE across datasets for one mode."""
    out_dir = ensure_plot_dir("all", "comparison")

    maes = []
    for d in DATASETS:
        m = load_metrics(d, mode)
        maes.append(m["MAE"])

    base_plot(f"{mode.upper()} - MAE Across Datasets", "Dataset", "MAE")
    plt.bar(DATASETS, maes)
    plt.savefig(os.path.join(out_dir, f"{mode}_mae_cross_dataset.png"))
    plt.close()

    print(f"[analysis] Saved cross-dataset accuracy comparison to {out_dir}")
