"""
Unified-style DP ablation visualisations:
  1) DP Noise vs MAE
  2) DP Noise vs Energy Consumption
"""

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Global Thesis Style
# -------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
})

from src.utils.s3_helpers import upload_to_s3

CSV_PATH = "privacy_dp_results.csv"
BUCKET = os.environ.get("RESULTS_BUCKET", os.environ.get("S3_BUCKET", "aefl"))

DATASET_LABELS = {"sz": "SZ", "pems08": "PEMS08", "los": "LOS"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def plot_accuracy(df):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))

    for dataset, color in zip(df["dataset"].unique(), COLORS):
        sub = df[df["dataset"] == dataset]
        ax.plot(sub["sigma"], sub["MAE"], marker="o", color=color, label=DATASET_LABELS[dataset])

    ax.set_xlabel("DP Noise σ")
    ax.set_ylabel("MAE")
    ax.set_title("Privacy–Accuracy Trade-off (DP Noise)")
    ax.legend()

    out = Path("outputs/privacy/dp_accuracy.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")

    print(f"[OK] Saved DP accuracy → {out}")
    upload_to_s3(str(out), BUCKET, "outputs/privacy/dp_accuracy.png")


def plot_energy(df):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))

    for dataset, color in zip(df["dataset"].unique(), COLORS):
        sub = df[df["dataset"] == dataset]
        ax.plot(sub["sigma"], sub["total_energy_j"], marker="o", color=color, label=DATASET_LABELS[dataset])

    ax.set_xlabel("DP Noise σ")
    ax.set_ylabel("Energy (J)")
    ax.set_title("DP Noise vs Energy Consumption")
    ax.legend()

    out = Path("outputs/privacy/dp_energy.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")

    print(f"[OK] Saved DP energy → {out}")
    upload_to_s3(str(out), BUCKET, "outputs/privacy/dp_energy.png")


def main():
    df = pd.read_csv(CSV_PATH)
    df["dataset"] = df["dataset"].str.lower()
    df["sigma"] = df["sigma"].astype(float)

    plot_accuracy(df)
    plot_energy(df)

    upload_to_s3(CSV_PATH, BUCKET, "outputs/privacy/privacy_dp_results.csv")


if __name__ == "__main__":
    main()
