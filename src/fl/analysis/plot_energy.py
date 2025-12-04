"""
Energy comparison plot for federated clients.
"""

import os
import json
import matplotlib.pyplot as plt
from .plot_utils import base_plot, ensure_plot_dir, upload_plot


def plot_energy(dataset, mode):
    out_dir = ensure_plot_dir(dataset, "energy")

    energy_path = f"outputs/summaries/{dataset}/{mode}/energy_{mode}.json"
    if not os.path.exists(energy_path):
        print(f"[analysis] No energy file found → skipping {dataset}/{mode}")
        return

    with open(energy_path, "r") as f:
        energies = json.load(f)

    roles = list(energies.keys())
    values = list(energies.values())

    base_plot(f"{dataset.upper()} - {mode.upper()} Energy", "Client Role", "Energy (J)")
    plt.bar(roles, values)

    filename = f"{dataset}_{mode}_energy.png"
    local_path = os.path.join(out_dir, filename)
    plt.savefig(local_path)
    plt.close()

    upload_plot(local_path, f"experiments/{dataset}/plots/energy")

    print(f"[analysis] Saved + uploaded energy plot for {dataset}/{mode}")
