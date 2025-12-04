"""
Experiment summary writer for federated learning server.

Saves:
- final_metrics_<mode>.json
- energy_<mode>.json
- summary_<mode>.csv         (final single-row summary incl. energy)
- rounds_<mode>.csv          (per-round summary)

Uploads all key results to S3.
"""

import os
import csv

from ..utils.serialization import save_json
from .utils_server import get_dataset, get_fl_mode
from .s3_io import upload_results_artifact


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

def _summary_dir(dataset: str, mode: str) -> str:
    base = os.path.join("outputs", "summaries", dataset, mode)
    os.makedirs(base, exist_ok=True)
    return base


def _rounds_csv_path(dataset: str, mode: str) -> str:
    return os.path.join(_summary_dir(dataset, mode), f"rounds_{mode}.csv")


# -------------------------------------------------------------------
# Per-round logging
# -------------------------------------------------------------------

def log_round_summary(round_id, selected_clients, num_updates, aggregation_time_s, mode_label=None):
    """Append one row to rounds_<mode>.csv"""
    dataset = get_dataset()
    mode = (mode_label or get_fl_mode()).lower()

    path = _rounds_csv_path(dataset, mode)
    is_new = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "dataset", "mode", "round",
                "num_selected", "selected_clients",
                "num_updates", "aggregation_time_s"
            ])
        w.writerow([
            dataset,
            mode,
            int(round_id),
            int(len(selected_clients)),
            ";".join(selected_clients),
            int(num_updates),
            float(f"{aggregation_time_s:.6f}")
        ])

    print(f"[SERVER] Logged round {round_id} summary → {path}")


# -------------------------------------------------------------------
# Final summary (metrics + energy)
# -------------------------------------------------------------------

def generate_cloud_summary(final_metrics, rounds, mode_label=None, energy_totals=None):
    dataset = get_dataset()
    mode = (mode_label or get_fl_mode()).lower()

    base_dir = _summary_dir(dataset, mode)

    metrics_path = os.path.join(base_dir, f"final_metrics_{mode}.json")
    energy_path  = os.path.join(base_dir, f"energy_{mode}.json")
    csv_path     = os.path.join(base_dir, f"summary_{mode}.csv")

    # Save metrics JSON
    save_json(metrics_path, final_metrics)

    # Save energy JSON (optional but recommended)
    if energy_totals is not None:
        save_json(energy_path, energy_totals)

    # Build summary CSV row
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "mode", "rounds",
            "MAE", "RMSE", "MAPE",
            "energy_roadside", "energy_vehicle",
            "energy_sensor", "energy_camera",
            "energy_bus"
        ])

        writer.writerow([
            dataset,
            mode,
            rounds,
            f"{final_metrics.get('MAE', 0.0):.6f}",
            f"{final_metrics.get('RMSE', 0.0):.6f}",
            f"{final_metrics.get('MAPE', 0.0):.6f}",
            f"{energy_totals.get('roadside', 0):.3f}",
            f"{energy_totals.get('vehicle', 0):.3f}",
            f"{energy_totals.get('sensor', 0):.3f}",
            f"{energy_totals.get('camera', 0):.3f}",
            f"{energy_totals.get('bus', 0):.3f}",
        ])

    print(f"[SERVER] Summary saved | dataset={dataset} mode={mode}")
    print(" ", csv_path)
    print(" ", metrics_path)
    if energy_totals:
        print(" ", energy_path)

    # Upload all artifacts to S3
    prefix = f"experiments/{dataset}/{mode}"
    upload_results_artifact(csv_path,     f"{prefix}/summary_{mode}.csv")
    upload_results_artifact(metrics_path, f"{prefix}/final_metrics_{mode}.json")
    if energy_totals:
        upload_results_artifact(energy_path, f"{prefix}/energy_{mode}.json")
