"""
Combined comparison plots for thesis publication.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fl.analysis.load_results import load_all_summaries
from src.fl.analysis.plot_utils import save_plot


def plot_overall_radar():
    """
    Create radar chart showing normalised MAE/RMSE/MAPE for all modes.
    """
    df = load_all_summaries()
    if df.empty:
        print("[PLOT] No summaries found.")
        return

    modes = sorted(df["mode"].unique())
    metrics = ["MAE", "RMSE", "MAPE"]

    plt.figure(figsize=(8, 8))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.append(angles, angles[0])

    for mode in modes:
        subset = df[df["mode"] == mode].mean()
        values = [subset[m] for m in metrics]
        values.append(values[0])

        plt.polar(angles, values, marker="o", label=mode)

    plt.title("Overall Comparison Radar Chart")
    plt.legend(loc="upper right")

    save_plot("comparison_radar.png")
