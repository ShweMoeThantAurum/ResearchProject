"""
Helpers for loading metrics and logs from experiment outputs.
"""

import os
import json
import pandas as pd


def _default_summaries_dir():
    """Return the default directory for summary artifacts."""
    return os.path.join("outputs", "summaries")


def _default_logs_dir():
    """Return the default directory for log files."""
    return os.path.join("outputs", "logs")


def load_final_metrics(dataset, mode, summaries_dir=None):
    """
    Load final metrics JSON for a dataset and FL mode.
    """
    if summaries_dir is None:
        summaries_dir = _default_summaries_dir()

    mode_lower = str(mode).lower()
    # Canonical file name: <dataset>_<mode>_final_metrics.json
    candidates = [
        os.path.join(summaries_dir, "%s_%s_final_metrics.json" % (dataset, mode_lower)),
        # Backwards compatible path from old layout, if still present
        os.path.join("outputs", dataset, mode_lower, "final_metrics_%s.json" % mode_lower),
    ]

    path = None
    for cand in candidates:
        if os.path.exists(cand):
            path = cand
            break

    if path is None:
        raise FileNotFoundError(
            "Could not find final metrics for dataset=%s mode=%s in %s"
            % (dataset, mode, summaries_dir)
        )

    with open(path, "r") as f:
        data = json.load(f)

    # Attach metadata for plotting convenience
    data["dataset"] = dataset
    data["mode"] = mode
    data["source_path"] = path
    return data


def load_all_final_metrics(datasets, modes, summaries_dir=None):
    """
    Load final metrics for many dataset/mode combinations into a DataFrame.
    """
    records = []
    for ds in datasets:
        for mode in modes:
            try:
                rec = load_final_metrics(ds, mode, summaries_dir=summaries_dir)
                records.append(rec)
            except FileNotFoundError:
                # Skip missing combinations so you can run partial experiments
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def _load_jsonl(path):
    """Load a JSONL file as a list of dictionaries."""
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # Ignore malformed lines
                continue
    return rows


def load_client_energy_summary(logs_dir=None):
    """
    Load per-client total energy summaries as a DataFrame.

    Expects a JSONL file named client_energy_summary.log written by clients.
    """
    if logs_dir is None:
        logs_dir = _default_logs_dir()

    path = os.path.join(logs_dir, "client_energy_summary.log")
    rows = _load_jsonl(path)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Ensure useful columns exist even if older runs logged slightly different keys
    if "total_energy_j" not in df.columns and "total_j" in df.columns:
        df["total_energy_j"] = df["total_j"]

    if "mode" not in df.columns:
        df["mode"] = "unknown"

    if "dataset" not in df.columns:
        df["dataset"] = "unknown"

    return df
