"""
Save final evaluation summary.
"""

import os
import json

from src.fl.config import settings


def write_summary(metrics, mode):
    """
    Write CSV + JSON summaries.
    """
    out_dir = "outputs/summaries"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset = settings.get_dataset()

    csv_path = os.path.join(out_dir, "summary_{}_{}.csv".format(dataset, mode.lower()))
    json_path = os.path.join(out_dir, "summary_{}_{}.json".format(dataset, mode.lower()))

    # CSV
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write("{},{}\n".format(k, v))

    # JSON
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[SERVER] Summary saved:", csv_path, json_path)
