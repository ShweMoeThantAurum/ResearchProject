"""
Generate accuracy comparison plots across datasets.
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.fl.analysis.load_results import load_all_summaries
from src.fl.analysis.plot_utils import save_plot


def plot_accuracy_bars():
    """
    Create grouped bar charts for MAE, RMSE, MAPE across modes/datasets.
    """
    df = load_all_summaries()
    if df.empty:
        print("[PLOT] No summaries found.")
        return

    metrics = ["MAE", "RMSE", "MAPE"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        pivot = df.pivot(index="mode", columns="dataset", values=metric)
        pivot.plot(kind="bar")

        plt.title(metric + " comparison")
        plt.ylabel(metric)
        plt.xlabel("FL Mode")
        plt.xticks(rotation=0)

        save_plot("accuracy_{}.png".format(metric.lower()))
