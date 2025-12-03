"""
Plot total client energy consumption across modes.
Used for Experiment 2: Energy Comparison.
"""

import os
import matplotlib.pyplot as plt

from .load_results import load_energy_totals
from .plot_utils import base_plot, ensure_plot_dir


def plot_energy(dataset, mode):
    """
    Plot per-client total energy for a given dataset/mode.

    Reads:
        outputs/summaries/<dataset>/<mode>/energy/<role>.json
    """
    out_dir = ensure_plot_dir(dataset, "energy")

    energies = load_energy_totals(dataset, mode)
    if not energies:
        raise ValueError(f"No energy summaries found for {dataset} / {mode}")

    roles = list(sorted(energies.keys()))
    values = [energies[r] for r in roles]

    base_plot(
        f"{dataset.upper()} - {mode.upper()} Total Client Energy",
        "Client Role",
        "Energy (J)",
    )
    plt.bar(roles, values)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset}_{mode}_energy.png"))
    plt.close()

    print(f"[analysis] Saved energy plot to {out_dir}")
