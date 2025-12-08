"""
Plots for DP ablation experiments:
  1) σ vs MAE
  2) σ vs total energy

Both plots are generated in compact A4-friendly dimensions.
The PNGs + CSV are uploaded to S3.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.s3_helpers import upload_to_s3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CSV_PATH = "privacy_dp_results.csv"
BUCKET = os.environ.get("RESULTS_BUCKET", os.environ.get("S3_BUCKET", "aefl"))

DATASET_LABELS = {"sz": "SZ", "pems08": "PEMS08", "los": "LOS"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def plot_accuracy(df):
    """Plot σ vs MAE (compact thesis-optimised)."""
    fig, ax = plt.subplots(figsize=(6.2, 3.2))

    for dataset, color in zip(df["dataset"].unique(), COLORS):
        sub = df[df["dataset"] == dataset]
        ax.plot(sub["sigma"], sub["MAE"], marker="o", color=color, label=DATASET_LABELS[dataset])

    ax.set_xlabel("DP Noise σ", fontsize=10)
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_title("Privacy–Accuracy Trade-off (DP Noise)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)

    out = Path("outputs/privacy/dp_accuracy.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved DP accuracy → {out}")

    upload_to_s3(str(out), BUCKET, "outputs/privacy/dp_accuracy.png")


def plot_energy(df):
    """Plot σ vs total energy (compact thesis-optimised)."""
    fig, ax = plt.subplots(figsize=(6.2, 3.2))

    for dataset, color in zip(df["dataset"].unique(), COLORS):
        sub = df[df["dataset"] == dataset]
        ax.plot(sub["sigma"], sub["total_energy_j"], marker="o", color=color, label=DATASET_LABELS[dataset])

    ax.set_xlabel("DP Noise σ", fontsize=10)
    ax.set_ylabel("Energy (J)", fontsize=10)
    ax.set_title("DP Noise vs Energy Consumption", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)

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

    # Upload CSV too
    upload_to_s3(CSV_PATH, BUCKET, "outputs/privacy/privacy_dp_results.csv")


if __name__ == "__main__":
    main()
