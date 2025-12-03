"""
Utilities for loading experiment summaries and metrics from outputs/.
Provides clean access to MAE, RMSE, MAPE, and energy totals.
"""

import os
import csv
import json
import glob


def _summary_path(dataset, mode):
    return f"outputs/summaries/{dataset}/{mode}/summary_{mode}.csv"


def _metrics_path(dataset, mode):
    return f"outputs/summaries/{dataset}/{mode}/final_metrics_{mode}.json"


def _energy_dir(dataset, mode):
    return f"outputs/summaries/{dataset}/{mode}/energy"


def load_metrics(dataset, mode):
    """Load final MAE/RMSE/MAPE JSON metrics."""
    path = _metrics_path(dataset, mode)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_summary(dataset, mode):
    """Load summary CSV for a given dataset/mode."""
    path = _summary_path(dataset, mode)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows


def load_energy_totals(dataset, mode):
    """
    Load per-client total energy from JSON summaries.

    Returns:
        dict: {role: total_energy_j}
    """
    energy_dir = _energy_dir(dataset, mode)
    if not os.path.isdir(energy_dir):
        raise FileNotFoundError(f"Energy directory not found: {energy_dir}")

    energies = {}
    pattern = os.path.join(energy_dir, "*.json")

    for path in glob.glob(pattern):
        with open(path, "r") as f:
            data = json.load(f)
        role = data.get("role") or os.path.splitext(os.path.basename(path))[0]
        energies[role] = float(data.get("total_energy_j", 0.0))

    return energies
