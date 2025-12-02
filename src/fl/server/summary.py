"""
Persist experiment-level summaries for FL runs.
Stores JSON summaries and simple CSV-style round logs.
"""

import os
import csv

from ..utils.serialization import save_json
from .utils_server import get_dataset, get_fl_mode


def _summary_dir():
    """Return directory for summary outputs."""
    dataset = get_dataset()
    mode = get_fl_mode().lower()
    path = os.path.join("outputs", "summaries", dataset, mode)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_experiment_summary(final_metrics, round_records):
    """Save final metrics and round-level stats to disk."""
    out_dir = _summary_dir()

    # Final metrics JSON
    metrics_path = os.path.join(out_dir, "final_metrics.json")
    payload = {
        "dataset": get_dataset(),
        "mode": get_fl_mode(),
        "metrics": final_metrics,
    }
    save_json(metrics_path, payload)

    # Round-level CSV (simple overview)
    csv_path = os.path.join(out_dir, "rounds.csv")
    if round_records:
        fieldnames = sorted(round_records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in round_records:
                writer.writerow(row)

    print("[SERVER] Summary saved:")
    print(" ", metrics_path)
    if round_records:
        print(" ", csv_path)

    return metrics_path, csv_path

def generate_cloud_summary(final_metrics, round_records):
    """
    Backwards-compat wrapper expected by server_main.py.
    Calls save_experiment_summary() under the hood.
    """
    return save_experiment_summary(final_metrics, round_records)
