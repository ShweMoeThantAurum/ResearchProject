"""
Common plotting utilities for AEFL experiment analysis.
Defines consistent style, colors, and output folders.
"""

import os
import matplotlib.pyplot as plt


def ensure_plot_dir(dataset, subfolder):
    """Ensure an output plot directory exists."""
    path = os.path.join("outputs", "plots", dataset, subfolder)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def base_plot(title, xlabel, ylabel):
    """Create a clean, consistent plot template."""
    plt.figure(figsize=(8, 5))
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
