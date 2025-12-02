"""
Plot total client energy consumption across modes.
Used for Experiment 2: Energy Comparison.
"""

import os
import matplotlib.pyplot as plt
from .plot_utils import base_plot, ensure_plot_dir

# Each client end-of-training prints:
#   "[role] Finished 20 rounds. Total estimated energy=36.40 J."
# We parse this from logs.


def parse_energy_from_logs(dataset, mode):
    """Extract total energy from client log files."""
    log_path = "outputs/logs/events.log"
    energies = {}

    with open(log_path, "r") as f:
        for line in f:
            if f"{dataset}" not in line or mode not in line:
                continue
            if "Total estimated energy" in line:
                parts = line.strip().split()
                role = parts[1].strip("[]")
                value = float(parts[-2])
                energies[role] = value

    return energies


def plot_energy(dataset, mode):
    """Plot per-client total energy."""
    out_dir = ensure_plot_dir(dataset, "energy")

    energies = parse_energy_from_logs(dataset, mode)
    roles = list(energies.keys())
    values = list(energies.values())

    base_plot(f"{dataset.upper()} - {mode.upper()} Energy", "Client Role", "Energy (J)")
    plt.bar(roles, values)
    plt.savefig(os.path.join(out_dir, f"{dataset}_{mode}_energy.png"))
    plt.close()

    print(f"[analysis] Saved energy plot to {out_dir}")
