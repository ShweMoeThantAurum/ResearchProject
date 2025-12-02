"""
Plots focusing on energy use, such as total Joules per client and mode.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .load_results import load_client_energy_summary
from .plot_utils import set_plot_style, save_figure, nice_mode_label


def build_energy_summary(logs_dir=None):
    """
    Build a tidy DataFrame with client energy summaries.

    This aggregates the raw JSONL log client_energy_summary.log.
    """
    df = load_client_energy_summary(logs_dir=logs_dir)
    if df.empty:
        return df

    # Normalise column names that are useful for plotting
    if "role" not in df.columns and "client_role" in df.columns:
        df["role"] = df["client_role"]

    # Ensure numeric type for total energy
    df["total_energy_j"] = pd.to_numeric(df["total_energy_j"], errors="coerce")
    return df


def plot_total_energy_by_role(dataset_filter=None, out_path=None, logs_dir=None):
    """
    Plot total energy per client role and mode as a grouped bar chart.
    """
    set_plot_style()
    df = build_energy_summary(logs_dir=logs_dir)

    if df.empty:
        raise RuntimeError("No client energy summary data found for plotting")

    df = df.copy()

    if dataset_filter is not None:
        df = df[df["dataset"] == dataset_filter]

    if df.empty:
        raise RuntimeError(
            "No client energy data after filtering for dataset=%s" % dataset_filter
        )

    df["mode_label"] = df["mode"].apply(nice_mode_label)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=df,
        x="role",
        y="total_energy_j",
        hue="mode_label",
        ax=ax,
    )

    ax.set_xlabel("Client role")
    ax.set_ylabel("Total energy (J)")
    if dataset_filter:
        ax.set_title("Total energy per role and mode (%s)" % dataset_filter)
    else:
        ax.set_title("Total energy per role and mode")
    ax.legend(title="Mode", frameon=True)

    if out_path is None:
        if dataset_filter:
            filename = "energy_by_role_%s.png" % dataset_filter
        else:
            filename = "energy_by_role.png"
        out_path = os.path.join("outputs", "plots", filename)

    save_figure(fig, out_path)
    return out_path


def plot_total_energy_by_mode(dataset_filter=None, out_path=None, logs_dir=None):
    """
    Plot total energy summed across all roles for each mode.
    """
    df = build_energy_summary(logs_dir=logs_dir)
    if df.empty:
        raise RuntimeError("No client energy summary data found for plotting")

    df = df.copy()

    if dataset_filter is not None:
        df = df[df["dataset"] == dataset_filter]

    if df.empty:
        raise RuntimeError(
            "No client energy data after filtering for dataset=%s" % dataset_filter
        )

    grouped = (
        df.groupby(["dataset", "mode"])["total_energy_j"]
        .sum()
        .reset_index()
    )

    grouped["mode_label"] = grouped["mode"].apply(nice_mode_label)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    if dataset_filter is not None:
        plot_df = grouped[grouped["dataset"] == dataset_filter]
    else:
        plot_df = grouped

    sns.barplot(
        data=plot_df,
        x="dataset",
        y="total_energy_j",
        hue="mode_label",
        ax=ax,
    )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Total energy (J)")
    ax.set_title("Total energy across clients per mode")
    ax.legend(title="Mode", frameon=True)

    if out_path is None:
        if dataset_filter:
            filename = "energy_by_mode_%s.png" % dataset_filter
        else:
            filename = "energy_by_mode.png"
        out_path = os.path.join("outputs", "plots", filename)

    save_figure(fig, out_path)
    return out_path
