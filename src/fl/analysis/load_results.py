"""
Load experiment summaries and prepare them for plotting.
"""

import os
import json
import pandas as pd


def load_all_summaries():
    """
    Load all summary JSON files for all datasets and FL modes.
    """
    base = "outputs/summaries"
    rows = []

    if not os.path.exists(base):
        return pd.DataFrame()

    for fname in os.listdir(base):
        if not fname.endswith(".json"):
            continue

        parts = fname.replace(".json", "").split("_")
        dataset = parts[1]
        mode = parts[2]

        with open(os.path.join(base, fname), "r") as f:
            metrics = json.load(f)

        rows.append({
            "dataset": dataset,
            "mode": mode,
            "MAE": metrics.get("MAE", None),
            "RMSE": metrics.get("RMSE", None),
            "MAPE": metrics.get("MAPE", None)
        })

    return pd.DataFrame(rows)


def load_energy_logs():
    """
    Load energy logs produced by clients (metadata).
    """
    base = "experiments/metadata"
    rows = []

    if not os.path.exists(base):
        return pd.DataFrame()

    for fname in os.listdir(base):
        if not fname.endswith(".json"):
            continue

        parts = fname.split("_r")
        role = parts[0]

        with open(os.path.join(base, fname), "r") as f:
            m = json.load(f)

        rows.append({
            "role": role,
            "round": m.get("round", None),
            "bandwidth_mbps": m.get("bandwidth_mbps", None),
            "energy_total_j": m.get("total_energy_j", None)
        })

    return pd.DataFrame(rows)
