"""
Generate a thesis-ready accuracy comparison plot across all methods
(AEFL, FedAvg, FedProx) and datasets (SZ, PeMS08, LOS).

The script:
 - reads accuracy_all_datasets.csv
 - produces a compact 1×3 panel figure sized for A4 pages
 - uploads PNG + CSV to S3
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.s3_helpers import upload_to_s3

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
INPUT_CSV = "accuracy_all_datasets.csv"
BUCKET = os.environ.get("RESULTS_BUCKET", os.environ.get("S3_BUCKET", "aefl"))

METRICS = ["MAE", "RMSE", "MAPE"]

MODE_LABELS = {
    "aefl": "AEFL (Ours)",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
}

DATASET_LABELS = {
    "sz": "SZ",
    "pems08": "PEMS08",
    "los": "LOS",
}

BAR_COLORS = {
    "aefl": "#1f77b4",
    "fedavg": "#ff7f0e",
    "fedprox": "#2ca02c",
}


def plot_metric(ax, df, metric):
    """Draw a grouped bar chart for a single metric."""
    pivot = df.pivot(index="dataset", columns="mode", values=metric)
    pivot = pivot[["aefl", "fedavg", "fedprox"]]

    datasets = [DATASET_LABELS[d] for d in pivot.index]
    values = pivot.values

    x = np.arange(len(datasets))
    width = 0.26

    for i, mode in enumerate(["aefl", "fedavg", "fedprox"]):
        ax.bar(
            x + (i - 1) * width,
            values[:, i],
            width,
            label=MODE_LABELS[mode],
            color=BAR_COLORS[mode],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    if metric == "MAE":
        ax.set_ylabel("Error", fontsize=10)


def main():
    df = pd.read_csv(INPUT_CSV)
    df["dataset"] = df["dataset"].str.lower()
    df["mode"] = df["mode"].str.lower()

    # Thesis-optimised size: fits 1 column on A4
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.0))
    fig.suptitle("Accuracy Comparison Across Methods and Datasets", fontsize=12, fontweight="bold")

    for ax, metric in zip(axes, METRICS):
        plot_metric(ax, df, metric)

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out = Path("outputs/accuracy/accuracy_all_metrics.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")

    print(f"[OK] Saved → {out}")

    upload_to_s3(str(out), BUCKET, "outputs/accuracy/accuracy_all_metrics.png")
    upload_to_s3(INPUT_CSV, BUCKET, "outputs/accuracy/accuracy_all_datasets.csv")


if __name__ == "__main__":
    main()
