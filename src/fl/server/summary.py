"""
Experiment summary writer for federated learning server.

Saves:
- final_metrics_<mode>.json
- summary_<mode>.csv

and uploads them to S3 for later analysis.
"""

import os

from ..utils.serialization import save_json
from .utils_server import get_dataset, get_fl_mode
from .s3_io import upload_results_artifact


def generate_cloud_summary(final_metrics, rounds, mode_label=None):
    """Save final metrics locally and upload summaries to S3."""
    dataset = get_dataset()
    mode = (mode_label or get_fl_mode()).lower()

    base_dir = os.path.join("outputs", "summaries", dataset, mode)
    os.makedirs(base_dir, exist_ok=True)

    metrics_path = os.path.join(base_dir, f"final_metrics_{mode}.json")
    csv_path = os.path.join(base_dir, f"summary_{mode}.csv")

    # Save JSON metrics
    save_json(metrics_path, final_metrics)

    # Save a simple one-row CSV summary
    with open(csv_path, "w") as f:
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
