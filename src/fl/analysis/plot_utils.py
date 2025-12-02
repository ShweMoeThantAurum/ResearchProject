"""
Utility helpers for all plotting modules.
"""

import os
import matplotlib.pyplot as plt


def ensure_plot_dir():
    """
    Make sure outputs/plots exists.
    """
    out = "outputs/plots"
    if not os.path.exists(out):
        os.makedirs(out)
    return out


def save_plot(name):
    """
    Save a plot as PNG inside outputs/plots.
    """
    out = ensure_plot_dir()
    path = os.path.join(out, name)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print("[PLOT] Saved:", path)
