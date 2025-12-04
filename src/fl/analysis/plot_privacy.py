"""
Privacy plots for DP trade-offs.
Shows sigma vs noise scale vs expected accuracy drop.
Used for Experiment 3: Privacy-Accuracy-Energy Trade-Offs.
"""

import os
import matplotlib.pyplot as plt
from .plot_utils import base_plot, ensure_plot_dir, upload_plot


def plot_privacy_tradeoff(dataset, sigmas, accuracy_losses):
    out_dir = ensure_plot_dir(dataset, "privacy")

    base_plot(
        f"{dataset.upper()} - Differential Privacy Trade-Off",
        "DP Sigma",
        "Accuracy Loss (ΔMAE)"
    )

    plt.plot(sigmas, accuracy_losses, marker="o")

    filename = f"{dataset}_privacy_tradeoff.png"
    local_path = os.path.join(out_dir, filename)
    plt.savefig(local_path)
    plt.close()

    upload_plot(local_path, f"experiments/{dataset}/plots/privacy")

    print(f"[analysis] Saved + uploaded DP privacy plot for {dataset}")
