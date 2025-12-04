"""
High-level comparison plot across datasets for AEFL thesis figures.
Shows cross-dataset accuracy/energy comparison in one figure.
"""

import os
import matplotlib.pyplot as plt
from .load_results import load_metrics
from .plot_utils import base_plot, ensure_plot_dir, upload_plot

DATASETS = ["sz", "los", "pems08"]


def plot_cross_dataset_accuracy(mode):
    out_dir = ensure_plot_dir("all", "comparison")

    maes = [load_metrics(d, mode)["MAE"] for d in DATASETS]

    base_plot(f"{mode.upper()} - MAE Across Datasets", "Dataset", "MAE")
    plt.bar(DATASETS, maes)

    filename = f"{mode}_mae_cross_dataset.png"
    local_path = os.path.join(out_dir, filename)
    plt.savefig(local_path)
    plt.close()

    upload_plot(local_path, f"experiments/all/plots/comparison")

    print("[analysis] Saved + uploaded cross-dataset accuracy comparison")
