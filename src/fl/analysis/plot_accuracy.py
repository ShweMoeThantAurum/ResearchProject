"""
Plots focusing on accuracy metrics such as MAE, RMSE and MAPE.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

from .load_results import load_all_final_metrics
from .plot_utils import set_plot_style, save_figure, nice_mode_label


def plot_final_metric_bar(datasets, modes, metric, out_path=None, summaries_dir=None):
    """
    Plot a grouped bar chart of a final metric across datasets and modes.
    """
    set_plot_style()
    df = load_all_final_metrics(datasets, modes, summaries_dir=summaries_dir)

    if df.empty:
        raise RuntimeError("No final metrics found for plotting %s" % metric)

    if metric not in df.columns:
        raise KeyError("Metric %s not found in final metrics DataFrame" % metric)

    df = df.copy()
    df["mode_label"] = df["mode"].apply(nice_mode_label)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=df,
        x="dataset",
        y=metric,
        hue="mode_label",
        ax=ax,
    )

    ax.set_ylabel(metric)
    ax.set_xlabel("Dataset")
    ax.set_title("Final %s across datasets and FL modes" % metric)
    ax.legend(title="Mode", frameon=True)

    if out_path is None:
        out_path = os.path.join("outputs", "plots", "final_%s_bar.png" % metric.lower())

    save_figure(fig, out_path)
    return out_path


def plot_all_final_metrics(datasets, modes, out_dir=None, summaries_dir=None):
    """
    Convenience helper to plot MAE, RMSE and MAPE bar charts in one call.
    """
    if out_dir is None:
        out_dir = os.path.join("outputs", "plots")

    metrics = ["MAE", "RMSE", "MAPE"]
    paths = {}

    for m in metrics:
        out_path = os.path.join(out_dir, "final_%s_bar.png" % m.lower())
        paths[m] = plot_final_metric_bar(
            datasets=datasets,
            modes=modes,
            metric=m,
            out_path=out_path,
            summaries_dir=summaries_dir,
        )

    return paths
