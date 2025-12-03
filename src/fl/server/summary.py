"""
Experiment summary generation for federated learning runs.
Saves final metrics locally and uploads artifacts to S3.
"""

import os
import csv

from ..utils.serialization import save_json
from ..utils.logger import log_event
from .s3_io import upload_results_artifact
from .utils_server import get_dataset, get_fl_mode


def _summary_dir(dataset, mode):
    """Return local directory for summary outputs."""
    base = os.path.join("outputs", "summaries", dataset, mode)
    os.makedirs(base, exist_ok=True)
    return base


def _metrics_json_path(dataset, mode):
    """Return JSON path for final metrics."""
    base = _summary_dir(dataset, mode)
    return os.path.join(base, f"final_metrics_{mode}.json")


def _summary_csv_path(dataset, mode):
    """Return CSV path for high-level summary."""
    base = _summary_dir(dataset, mode)
    return os.path.join(base, f"summary_{mode}.csv")


def generate_cloud_summary(final_metrics, dataset, rounds, mode):
    """
    Save final metrics and a one-row summary CSV for an experiment.

    Layout:
      outputs/summaries/<dataset>/<mode>/
        - final_metrics_<mode>.json
        - summary_<mode>.csv
    """
    mode = mode.lower()
    dataset = dataset.lower()

    metrics_path = _metrics_json_path(dataset, mode)
    csv_path = _summary_csv_path(dataset, mode)

    # JSON metrics
    save_json(metrics_path, final_metrics)

    # One-row CSV summary
    fieldnames = ["dataset", "mode", "rounds", "MAE", "RMSE", "MAPE"]
    row = {
        "dataset": dataset,
        "mode": mode,
        "rounds": int(rounds),
        "MAE": final_metrics.get("MAE", 0.0),
        "RMSE": final_metrics.get("RMSE", 0.0),
        "MAPE": final_metrics.get("MAPE", 0.0),
    }

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    log_event(
        f"[SERVER] summary_saved dataset={dataset} mode={mode} "
        f"metrics_path={metrics_path} csv_path={csv_path}"
    )
    print(
        f"[SERVER] Summary saved | dataset={dataset} mode={mode}\n"
        f"  {csv_path}\n  {metrics_path}"
    )

    # Upload both to S3 results bucket
    remote_base = f"experiments/{dataset}/{mode}/"
    upload_results_artifact(csv_path, remote_base + os.path.basename(csv_path))
    upload_results_artifact(metrics_path, remote_base + os.path.basename(metrics_path))
