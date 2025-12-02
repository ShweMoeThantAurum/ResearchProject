"""
Privacy plots for DP trade-offs.
Shows sigma vs noise scale vs expected accuracy drop.
Used for Experiment 3: Privacy-Accuracy-Energy Trade-Offs.
"""

import os
import matplotlib.pyplot as plt
from .plot_utils import base_plot, ensure_plot_dir


def plot_privacy_tradeoff(dataset, sigmas, accuracy_losses):
    """Plot accuracy loss as a function of DP sigma."""
    out_dir = ensure_plot_dir(dataset, "privacy")

    base_plot(
        f"{dataset.upper()} - Differential Privacy Trade-Off",
        "DP Sigma",
        "Accuracy Loss (ΔMAE)"
    )

    plt.plot(sigmas, accuracy_losses, marker="o")
    plt.savefig(os.path.join(out_dir, f"{dataset}_privacy_tradeoff.png"))
    plt.close()

    print(f"[analysis] Saved privacy plot to {out_dir}")
