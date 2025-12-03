"""
Experiment summary writer for federated learning server.

Saves:
- final_metrics_<mode>.json
- summary_<mode>.csv  (one row with final metrics)
- rounds_<mode>.csv   (per-round summary: selection, timing, etc.)

and uploads final summary artifacts to S3 for later analysis.
"""

import os
import csv

from ..utils.serialization import save_json
from .utils_server import get_dataset, get_fl_mode
from .s3_io import upload_results_artifact


def _summary_dir(dataset: str, mode: str) -> str:
    """Return (and create) the base summary directory."""
    base_dir = os.path.join("outputs", "summaries", dataset, mode)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _rounds_csv_path(dataset: str, mode: str) -> str:
    """Path to the per-round summary CSV."""
    base_dir = _summary_dir(dataset, mode)
    return os.path.join(base_dir, f"rounds_{mode}.csv")


def log_round_summary(round_id, selected_clients, num_updates, aggregation_time_s, mode_label=None):
    """
    Append a single row to rounds_<mode>.csv with per-round summary.

    Columns:
        dataset,mode,round,num_selected,selected_clients,num_updates,aggregation_time_s
    """
    dataset = get_dataset()
    mode = (mode_label or get_fl_mode()).lower()

    path = _rounds_csv_path(dataset, mode)
    file_exists = os.path.exists(path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "dataset",
                    "mode",
                    "round",
                    "num_selected",
                    "selected_clients",
                    "num_updates",
                    "aggregation_time_s",
                ]
            )

        writer.writerow(
            [
                dataset,
                mode,
                int(round_id),
                int(len(selected_clients)),
                ";".join(selected_clients),
                int(num_updates),
                float(f"{aggregation_time_s:.6f}"),
            ]
        )

    print(f"[SERVER] Logged round {round_id} summary to {path}")


def generate_cloud_summary(final_metrics, rounds, mode_label=None):
    """Save final metrics locally and upload summaries to S3."""
    dataset = get_dataset()
    mode = (mode_label or get_fl_mode()).lower()

    base_dir = _summary_dir(dataset, mode)
    metrics_path = os.path.join(base_dir, f"final_metrics_{mode}.json")
    csv_path = os.path.join(base_dir, f"summary_{mode}.csv")

    # Save JSON metrics
    save_json(metrics_path, final_metrics)

    # Save a simple one-row CSV summary
    with open(csv_path, "w", newline="") as f:
        f.write("dataset,mode,rounds,MAE,RMSE,MAPE\n")
        f.write(
            f"{dataset},{mode},{rounds},"
            f"{final_metrics.get('MAE', 0.0):.6f},"
            f"{final_metrics.get('RMSE', 0.0):.6f},"
            f"{final_metrics.get('MAPE', 0.0):.6f}\n"
        )

    print(f"[SERVER] Summary saved | dataset={dataset} | mode={mode}")
    print(f"  {csv_path}")
    print(f"  {metrics_path}")

    # Upload to S3 in the same layout as before
    remote_prefix = f"experiments/{dataset}/{mode}"
    upload_results_artifact(csv_path, f"{remote_prefix}/summary_{mode}.csv")
    upload_results_artifact(metrics_path, f"{remote_prefix}/final_metrics_{mode}.json")
