"""Plot final accuracy metrics (MAE, RMSE, MAPE) across datasets and FL modes."""

import os
import json
import matplotlib.pyplot as plt

DATASETS = ["sz", "los", "pems08"]
MODES = ["aefl", "fedavg", "fedprox", "localonly"]

BASE_DIR = "outputs"
OUT_DIR = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ["MAE", "RMSE", "MAPE"]


def load_metrics(dataset, mode):
    """Load final accuracy metrics JSON for a dataset and FL mode."""
    path = os.path.join(BASE_DIR, dataset, mode, f"final_metrics_{mode}.json")
    if not os.path.exists(path):
        print("[WARN] Missing metrics:", path)
        return None
    with open(path, "r") as f:
        return json.load(f)


def collect_metrics():
    """Collect MAE, RMSE, MAPE values for all datasets and FL modes."""
    results = {m: {ds: {} for ds in DATASETS} for m in METRICS}

    for ds in DATASETS:
        for mode in MODES:
            metrics = load_metrics(ds, mode)
            if not metrics:
                continue
            for m in METRICS:
                if m in metrics:
                    results[m][ds][mode] = metrics[m]
    return results


def plot_metric(metric_name, metric_data):
    """Create a 3-panel bar plot comparing a metric across datasets."""
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(14, 4), sharey=True)
    if len(DATASETS) == 1:
        axes = [axes]

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        modes_present = [m for m in MODES if m in metric_data[ds]]
        values = [metric_data[ds][m] for m in modes_present]

        ax.bar([m.upper() for m in modes_present], values)
        ax.set_title(f"{metric_name} – {ds.upper()}")
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    out_path = os.path.join(OUT_DIR, f"accuracy_{metric_name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[OK] Saved", out_path)


def main():
    """Generate accuracy comparison plots for all datasets."""
    results = collect_metrics()
    for metric in METRICS:
        plot_metric(metric, results[metric])


if __name__ == "__main__":
    main()
