"""
Plot DP privacy trade-offs (sigma vs. accuracy).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fl.analysis.load_results import load_all_summaries
from src.fl.analysis.plot_utils import save_plot


def plot_privacy_tradeoff(dp_sigmas, accuracy_values):
    """
    Plot DP sigma vs accuracy.
    """
    plt.figure(figsize=(7, 5))

    plt.plot(dp_sigmas, accuracy_values, marker="o")
    plt.title("Accuracy vs DP Noise Level (Sigma)")
    plt.xlabel("DP Sigma")
    plt.ylabel("MAE")

    save_plot("privacy_tradeoff.png")
