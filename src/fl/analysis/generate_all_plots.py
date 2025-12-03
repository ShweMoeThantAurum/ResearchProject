"""
Convenience script to generate all core plots after experiments.

Usage (from project root):
    python -m src.fl.analysis.generate_all_plots
"""

from .plot_accuracy import plot_accuracy_metrics
from .plot_energy import plot_energy
from .plot_comparison import plot_cross_dataset_accuracy

# Datasets and modes used in the thesis
DATASETS = ["sz", "los", "pems08"]
MODES = ["localonly", "fedavg", "fedprox", "aefl"]


def main():
    # --------------------------------------------------
    # 1) Per-dataset accuracy plots (MAE / RMSE / MAPE)
    # --------------------------------------------------
    for d in DATASETS:
        try:
            print(f"[analysis] Plotting accuracy metrics for dataset={d}...")
            plot_accuracy_metrics(d)
        except FileNotFoundError as e:
            print(f"[analysis] Skipping accuracy plots for {d}: {e}")

    # --------------------------------------------------
    # 2) Per-dataset, per-mode energy plots
    # --------------------------------------------------
    for d in DATASETS:
        for m in MODES:
            try:
                print(f"[analysis] Plotting energy for dataset={d}, mode={m}...")
                plot_energy(d, m)
            except FileNotFoundError as e:
                print(f"[analysis] Skipping energy plot for {d}/{m}: {e}")

    # --------------------------------------------------
    # 3) Cross-dataset accuracy comparison (MAE only)
    # --------------------------------------------------
    for m in MODES:
        try:
            print(f"[analysis] Plotting cross-dataset MAE for mode={m}...")
            plot_cross_dataset_accuracy(m)
        except FileNotFoundError as e:
            print(f"[analysis] Skipping cross-dataset accuracy for {m}: {e}")

    print("[analysis] All requested plots generated (where data was available).")


if __name__ == "__main__":
    main()

