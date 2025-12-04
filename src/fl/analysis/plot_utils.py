"""
Common plotting utilities for AEFL experiment analysis.
Defines consistent style, colors, and output folders.
"""

import os
import matplotlib.pyplot as plt
from src.fl.server.s3_io import upload_results_artifact


def ensure_plot_dir(dataset, plot_type):
    """Ensure an outputs/plots/... directory exists."""
    path = os.path.join("outputs", "plots", dataset, plot_type)
    os.makedirs(path, exist_ok=True)
    return path


def base_plot(title, xlabel, ylabel):
    """Apply consistent styling."""
    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)


def upload_plot(local_path, remote_prefix):
    """
    Uploads any plot to S3 under:
       s3://aefl/<remote_prefix>/<filename>
    """
    filename = os.path.basename(local_path)
    remote_path = f"{remote_prefix}/{filename}"

    upload_results_artifact(local_path, remote_path)
    print(f"[analysis] Uploaded plot → s3://aefl/{remote_path}")
