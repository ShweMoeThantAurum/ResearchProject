"""
Utility helpers for plotting and saving figures.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(path):
    """Create the parent directory for a file path if it does not exist."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def set_plot_style():
    """Set a consistent plotting style for all figures."""
    # Use a clean, publication-friendly style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def save_figure(fig, out_path):
    """Save a Matplotlib figure to disk with tight layout."""
    ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def nice_mode_label(mode):
    """Return a nicer label for a mode string."""
    mapping = {
        "aefl": "AEFL",
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "localonly": "Local-Only",
    }
    key = str(mode).lower()
    return mapping.get(key, mode)
