"""Plot total client energy per round for each dataset and FL mode."""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

DATASETS = ["sz", "los", "pems08"]
MODES = ["aefl", "fedavg", "fedprox", "localonly"]

RUN_LOGS = "run_logs"
OUT_DIR = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)


def load_energy(dataset, mode):
    """Load client energy log entries into a DataFrame."""
    path = f"{RUN_LOGS}/{dataset}/{mode}/client_energy.log"
    if not os.path.exists(path):
        print(f"[SKIP] {dataset},{mode}: {path} missing")
        return None

    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                pass

    if not rows:
        return None

    return pd.DataFrame(rows)


def plot_energy(dataset, mode):
    """Plot total energy consumption per round for one dataset and mode."""
    df = load_energy(dataset, mode)
    if df is None:
        return

    grouped = df.groupby("round")["total_j"].sum().reset_index()

    plt.figure(figsize=(8,5))
    plt.plot(grouped["round"], grouped["total_j"])
    plt.title(f"Total Client Energy per Round — {dataset.upper()} — {mode.upper()}")
    plt.xlabel("Round")
    plt.ylabel("Energy (J)")
    plt.grid(True)

    out = f"{OUT_DIR}/energy_{dataset}_{mode}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] Saved {out}")


def main():
    """Generate energy plots for all datasets and FL modes."""
    for ds in DATASETS:
        for mode in MODES:
            plot_energy(ds, mode)


if __name__ == "__main__":
    main()
