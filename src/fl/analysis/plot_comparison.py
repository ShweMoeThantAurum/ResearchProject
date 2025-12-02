"""
Composite plots comparing accuracy and energy across datasets and modes.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .load_results import load_all_final_metrics, load_client_energy_summary
from .plot_utils import set_plot_style, save_figure, nice_mode_label


def build_accuracy_energy_frame(datasets, modes, summaries_dir=None, logs_dir=None):
    """
    Build a joined DataFrame with final MAE and total energy per mode and dataset.
    """
    metrics_df = load_all_final_metrics(datasets, modes, summaries_dir=summaries_dir)

    energy_df = load_client_energy_summary(logs_dir=logs_dir)
    if not energy_df.empty:
        energy_grouped = (
            energy_df.groupby(["dataset", "mode"])["total_energy_j"]
            .sum()
            .reset_index()
        )
    else:
        energy_grouped = pd.DataFrame(columns=["dataset", "mode", "total_energy_j"])

    if metrics_df.empty:
        return pd.DataFrame()

    merged = metrics_df.merge(
        energy_grouped,
        on=["dataset", "mode"],
        how="left",
    )

    return merged


def plot_mae_vs_energy(datasets, modes, out_path=None, summaries_dir=None, logs_dir=None):
    """
    Plot MAE versus total energy for each dataset/mode combination.

    This makes the main trade-off between accuracy and energy visible.
    """
    df = build_accuracy_energy_frame(
        datasets=datasets,
        modes=modes,
        summaries_dir=summaries_dir,
        logs_dir=logs_dir,
    )

    if df.empty:
        raise RuntimeError("No combined accuracy/energy data found for plotting")

    df = df.copy()
    df["mode_label"] = df["mode"].apply(nice_mode_label)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.scatterplot(
        data=df,
        x="total_energy_j",
        y="MAE",
        hue="mode_label",
        style="dataset",
        s=60,
        ax=ax,
    )

    for _, row in df.iterrows():
        label = "%s-%s" % (row["dataset"], row["mode_label"])
        if not pd.isna(row["total_energy_j"]) and not pd.isna(row["MAE"]):
            ax.text(
                row["total_energy_j"],
                row["MAE"],
                " " + label,
                fontsize=8,
            )

    ax.set_xlabel("Total energy (J)")
    ax.set_ylabel("MAE")
    ax.set_title("MAE vs total energy across datasets and FL modes")
    ax.legend(title="Mode", frameon=True)

    if out_path is None:
        out_path = os.path.join("outputs", "plots", "mae_vs_energy.png")

    save_figure(fig, out_path)
    return out_path
