"""Plot communication latency per round for each dataset and FL mode."""

import os
import pandas as pd
import matplotlib.pyplot as plt

DATASETS = ["sz", "los", "pems08"]
MODES = ["aefl", "fedavg", "fedprox", "localonly"]

BASE_DIR = "outputs"
OUT_DIR = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_comm(dataset, mode):
    """Plot upload and download latency per round from summary_<mode>.csv."""
    csv_path = f"{BASE_DIR}/{dataset}/{mode}/summary_{mode}.csv"
    if not os.path.exists(csv_path):
        print(f"[SKIP] {dataset},{mode}: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8,5))
    plt.plot(df["round"], df["upload_latency_sec"], label="Upload Latency")
    plt.plot(df["round"], df["download_latency_sec"], label="Download Latency")
    plt.title(f"Communication Latency — {dataset.upper()} — {mode.upper()}")
    plt.xlabel("Round")
    plt.ylabel("Latency (s)")
    plt.legend()
    plt.grid(True)

    out = f"{OUT_DIR}/comm_{dataset}_{mode}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] Saved {out}")


def main():
    """Generate communication plots for all datasets and FL modes."""
    for ds in DATASETS:
        for mode in MODES:
            plot_comm(ds, mode)


if __name__ == "__main__":
    main()
