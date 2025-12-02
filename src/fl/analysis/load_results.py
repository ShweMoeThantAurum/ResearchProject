"""
Utilities for loading experiment summaries and metrics from outputs/.
Provides clean access to MAE, RMSE, MAPE, and energy totals.
"""

import os
import csv
import json


def _summary_path(dataset, mode):
    return f"outputs/summaries/{dataset}/{mode}/summary_{mode}.csv"


def _metrics_path(dataset, mode):
    return f"outputs/summaries/{dataset}/{mode}/final_metrics_{mode}.json"


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
