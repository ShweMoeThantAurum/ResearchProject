"""
Plots for exploring privacy–utility trade-offs using DP noise settings.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .plot_utils import set_plot_style, save_figure, nice_mode_label


def plot_dp_sigma_vs_metric(df, metric, out_path=None):
    """
    Plot a line chart of a metric versus DP sigma for each mode.

    The input DataFrame is expected to contain columns:
    - dataset
    - mode
    - dp_sigma
    - metric (e.g. MAE, RMSE, MAPE)
    """
    if "dp_sigma" not in df.columns:
        raise KeyError("Input DataFrame must contain a 'dp_sigma' column")

    if metric not in df.columns:
        raise KeyError("Metric %s not found in DataFrame" % metric)

    data = df.copy()
    data["mode_label"] = data["mode"].apply(nice_mode_label)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.lineplot(
        data=data,
        x="dp_sigma",
        y=metric,
        hue="mode_label",
        style="dataset",
        marker="o",
        ax=ax,
    )

    ax.set_xlabel("DP noise sigma")
    ax.set_ylabel(metric)
    ax.set_title("%s vs DP sigma (privacy–utility trade-off)" % metric)
    ax.legend(title="Mode / Dataset", frameon=True)

    if out_path is None:
        out_path = os.path.join(
            "outputs", "plots", "dp_sigma_vs_%s.png" % metric.lower()
        )

    save_figure(fig, out_path)
    return out_path


def build_dp_experiment_frame(runs):
    """
    Build a DataFrame from a list of DP experiment records.

    Each record may contain dataset, mode, dp_sigma and metrics.
    This helper makes it easier to construct inputs for privacy plots.
    """
    # This helper is intentionally simple so you can assemble experiments by hand
    df = pd.DataFrame(runs)
    return df
