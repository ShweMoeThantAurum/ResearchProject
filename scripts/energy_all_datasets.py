"""
Generate a thesis-ready energy comparison plot across AEFL, FedAvg, FedProx.

Unified visual style:
 - consistent colours and fonts
 - compact A4-friendly figsize
 - shared rcParams for grid, titles, labels and legends
"""

import sys
import os
from pathlib import Path
import numpy as np
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.s3_helpers import upload_to_s3

# -------------------------------------------------------------
INPUT_CSV = "energy_all_datasets.csv"
BUCKET = os.environ.get("RESULTS_BUCKET", os.environ.get("S3_BUCKET", "aefl"))

MODE_LABELS = {
    "aefl": "AEFL (Ours)",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
}

DATASET_LABELS = {"sz": "SZ", "pems08": "PEMS08", "los": "LOS"}

BAR_COLORS = {
    "aefl": "#1f77b4",
    "fedavg": "#ff7f0e",
    "fedprox": "#2ca02c",
}


def main():
    df = pd.read_csv(INPUT_CSV)
    df["dataset"] = df["dataset"].str.lower()
    df["mode"] = df["mode"].str.lower()

    pivot = df.pivot(index="dataset", columns="mode", values="total_energy_j")
    pivot = pivot[["aefl", "fedavg", "fedprox"]]

    datasets = [DATASET_LABELS[d] for d in pivot.index]
    values = pivot.values

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6.2, 3.2))

    ax.set_title("Energy Consumption Across Methods and Datasets")

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

    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    fig.tight_layout()

    out = Path("outputs/energy/energy_all_datasets.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")

    print(f"[OK] Saved → {out}")
    upload_to_s3(str(out), BUCKET, "outputs/energy/energy_all_datasets.png")
    upload_to_s3(INPUT_CSV, BUCKET, "outputs/energy/energy_all_datasets.csv")


if __name__ == "__main__":
    main()
