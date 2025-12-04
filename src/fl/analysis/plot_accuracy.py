"""
Plot accuracy metrics (MAE, RMSE, MAPE) across FL modes.
Used for Experiment 1: Accuracy Comparison.
"""

import os
import matplotlib.pyplot as plt
from .load_results import load_metrics
from .plot_utils import base_plot, ensure_plot_dir, upload_plot

MODES = ["localonly", "fedavg", "fedprox", "aefl"]


def plot_accuracy_metrics(dataset):
    out_dir = ensure_plot_dir(dataset, "accuracy")

    maes, rmses, mapes = [], [], []

    for mode in MODES:
        m = load_metrics(dataset, mode)
        maes.append(m["MAE"])
        rmses.append(m["RMSE"])
        mapes.append(m["MAPE"])

    def save_and_upload(name):
        local_path = os.path.join(out_dir, name)
        plt.savefig(local_path)
        plt.close()
        upload_plot(local_path, f"experiments/{dataset}/plots/accuracy")

    # MAE
    base_plot(f"{dataset.upper()} - MAE Comparison", "Mode", "MAE")
    plt.bar(MODES, maes)
    save_and_upload(f"{dataset}_mae.png")

    # RMSE
    base_plot(f"{dataset.upper()} - RMSE Comparison", "Mode", "RMSE")
    plt.bar(MODES, rmses)
    save_and_upload(f"{dataset}_rmse.png")

    # MAPE
    base_plot(f"{dataset.upper()} - MAPE Comparison", "Mode", "MAPE")
    plt.bar(MODES, mapes)
    save_and_upload(f"{dataset}_mape.png")

    print(f"[analysis] Saved + uploaded accuracy plots for {dataset}")
