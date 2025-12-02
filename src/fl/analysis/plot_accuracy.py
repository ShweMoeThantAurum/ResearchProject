"""
Plot accuracy metrics (MAE, RMSE, MAPE) across FL modes.
Used for Experiment 1: Accuracy Comparison.
"""

import os
import matplotlib.pyplot as plt
from .load_results import load_metrics
from .plot_utils import base_plot, ensure_plot_dir


MODES = ["localonly", "fedavg", "fedprox", "aefl"]


def plot_accuracy_metrics(dataset):
    """Plot MAE, RMSE, MAPE across FL modes."""
    out_dir = ensure_plot_dir(dataset, "accuracy")

    maes = []
    rmses = []
    mapes = []

    for mode in MODES:
        m = load_metrics(dataset, mode)
        maes.append(m["MAE"])
        rmses.append(m["RMSE"])
        mapes.append(m["MAPE"])

    # MAE
    base_plot(f"{dataset.upper()} - MAE Comparison", "Mode", "MAE")
    plt.bar(MODES, maes)
    plt.savefig(os.path.join(out_dir, f"{dataset}_mae.png"))
    plt.close()

    # RMSE
    base_plot(f"{dataset.upper()} - RMSE Comparison", "Mode", "RMSE")
    plt.bar(MODES, rmses)
    plt.savefig(os.path.join(out_dir, f"{dataset}_rmse.png"))
    plt.close()

    # MAPE
    base_plot(f"{dataset.upper()} - MAPE Comparison", "Mode", "MAPE")
    plt.bar(MODES, mapes)
    plt.savefig(os.path.join(out_dir, f"{dataset}_mape.png"))
    plt.close()

    print(f"[analysis] Saved accuracy plots to {out_dir}")
