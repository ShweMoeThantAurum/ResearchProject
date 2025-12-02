"""
Summary generation for a completed training run.

This module:
  - builds a per-round CSV summary from server logs
  - saves final metrics as JSON
  - stores everything under outputs/summaries/<dataset>/<mode>/

S3 uploads of summaries can be added later if needed, but are not
required for your experiments to run.
"""

import os
import json
import pandas as pd

from src.fl.utils.logger import LOG_DIR


def _load_jsonl(path):
    """
    Load a JSONL file and return a list of dictionaries.

    If the file does not exist, an empty list is returned.
    """
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


def build_round_summary(num_rounds):
    """
    Build a per-round summary DataFrame using server logs.

    Columns include:
      - round
      - mean_update_download_latency_sec
      - mean_update_size_mb

    The function uses the "server_update_download.log" file.
    """
    update_log_path = os.path.join(LOG_DIR, "server_update_download.log")
    updates = _load_jsonl(update_log_path)

    rows = []

    for r in range(1, num_rounds + 1):
        entries = [e for e in updates if int(e.get("round", -1)) == r]

        if not entries:
            rows.append(
                {
                    "round": r,
                    "mean_update_download_latency_sec": 0.0,
                    "mean_update_size_mb": 0.0,
                }
            )
            continue

        mean_latency = sum(e.get("latency_sec", 0.0) for e in entries) / float(
            len(entries)
        )
        mean_size_mb = (
            sum(e.get("size_bytes", 0.0) for e in entries)
            / float(len(entries))
            / (1024.0 * 1024.0)
        )

        rows.append(
            {
                "round": r,
                "mean_update_download_latency_sec": mean_latency,
                "mean_update_size_mb": mean_size_mb,
            }
        )

    df = pd.DataFrame(rows)
    return df


def save_summaries(final_metrics, num_rounds, dataset_name, mode_name):
    """
    Save per-round CSV summary and final metrics JSON.

    Files are written to:
        outputs/summaries/<dataset>/<mode>/

    Returns the path to the CSV file.
    """
    dataset = dataset_name.lower()
    mode = mode_name.lower()

    out_dir = os.path.join("outputs", "summaries", dataset, mode)
    os.makedirs(out_dir, exist_ok=True)

    df = build_round_summary(num_rounds)
    csv_path = os.path.join(out_dir, "summary_{}.csv".format(mode))
    df.to_csv(csv_path, index=False)

    metrics_path = os.path.join(out_dir, "final_metrics_{}.json".format(mode))
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print("[SERVER] Summary written to:")
    print("  " + csv_path)
    print("  " + metrics_path)

    return csv_path
