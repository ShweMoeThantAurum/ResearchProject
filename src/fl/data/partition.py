"""
Partitions global dataset into 5 client splits (roadside, vehicle, sensor, camera, bus).
Implements simple round-robin assignment across nodes.
"""

import os
import sys
import torch

ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]

PROC_DIR = "datasets/processed"

def load_split(path):
    """Loads a (X, y) tuple saved by preprocess.py."""
    X, y = torch.load(path)
    return X, y

def save_role(out_dir, role, X, y):
    """Saves client-specific tensors."""
    path = os.path.join(out_dir, role + ".pt")
    torch.save((X, y), path)

def partition(dataset):
    """Creates client partitions for train/val/test combined."""
    base = os.path.join(PROC_DIR, dataset, "global")
    out_dir = os.path.join(PROC_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)

    X_train, y_train = load_split(os.path.join(base, "train.pt"))
    X_val, y_val = load_split(os.path.join(base, "val.pt"))
    X_test, y_test = load_split(os.path.join(base, "test.pt"))

    X_all = torch.cat([X_train, X_val, X_test], dim=0)
    y_all = torch.cat([y_train, y_val, y_test], dim=0)

    n = X_all.shape[0]
    k = len(ROLES)

    for idx, role in enumerate(ROLES):
        X_role = X_all[idx:n:k]
        y_role = y_all[idx:n:k]
        save_role(out_dir, role, X_role, y_role)

        print(role, "samples:", len(X_role))

    print("Partition complete:", dataset)
    print("Saved to:", out_dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.fl.data.partition <dataset>")
        sys.exit(1)
    partition(sys.argv[1])
