"""
Generate run summaries for the final evaluation outputs.

Writes:
- summary CSV (MAE, RMSE, MAPE)
- final_metrics JSON
"""

import os
import csv
from src.fl.utils.serialization import save_json


def generate_cloud_summary(metrics, rounds, mode, dataset="unknown"):
    """
    Save summary results into:
        outputs/summaries/<dataset>/<mode>/
    """
    out_dir = os.path.join("outputs", "summaries", dataset, mode.lower())
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "summary_%s.csv" % mode.lower())
    json_path = os.path.join(out_dir, "final_metrics_%s.json" % mode.lower())

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])

    # Write JSON
    save_json(json_path, metrics)

    print("[SERVER] Summary saved | dataset=%s | mode=%s" % (dataset, mode))
    print(" ", csv_path)
    print(" ", json_path)

    return csv_path, json_path
