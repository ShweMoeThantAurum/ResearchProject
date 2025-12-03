"""
Writes experiment results into outputs/summaries/<dataset>/<mode>/
"""

import os
import json


def save_experiment_summary(dataset, mode, metrics):
    """Save final metrics JSON file."""
    out_dir = f"outputs/summaries/{dataset}/{mode}"
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"final_metrics_{mode}.json")

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[SERVER] Saved summary → {path}")
