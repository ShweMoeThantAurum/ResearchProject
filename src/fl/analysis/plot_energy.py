"""
Generate plots showing energy usage across clients.
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.fl.analysis.load_results import load_energy_logs
from src.fl.analysis.plot_utils import save_plot


def plot_energy_distribution():
    """
    Plot mean energy usage per client role.
    """
    df = load_energy_logs()
    if df.empty:
        print("[PLOT] No energy logs found.")
        return

    grouped = df.groupby("role")["energy_total_j"].mean()

    plt.figure(figsize=(8, 5))
    grouped.plot(kind="bar")

    plt.title("Mean Energy Consumption per Client")
    plt.ylabel("Energy (J)")
    plt.xlabel("Client")

    save_plot("energy_per_role.png")
