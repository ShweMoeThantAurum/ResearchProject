"""
Generate CSV and JSON summaries for the experiment.
"""

import os
import json
import pandas as pd


def generate_cloud_summary(metrics, dataset, mode, rounds):
    out_dir = os.path.join("outputs", "summaries", dataset, mode)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "summary_%s.csv" % mode)
    json_path = os.path.join(out_dir, "final_metrics_%s.json" % mode)

    df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump({
            "dataset": dataset,
            "mode": mode,
            "rounds": rounds,
            "metrics": metrics,
        }, f, indent=2)

    print("[SERVER] Summary saved to:")
    print(" ", csv_path)
    print(" ", json_path)

    return csv_path, json_path
