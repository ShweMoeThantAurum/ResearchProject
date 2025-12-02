"""
Preprocessing pipeline for the Los-Loop dataset.

Loads raw speed/adjacency CSVs, normalises traffic signals,
constructs sliding-window samples, and generates FL client partitions.
"""

import os
import boto3
import numpy as np
import pandas as pd
from data.dataloader import build_client_datasets

BUCKET = "aefl"
S3_PREFIX = "raw/los/"
s3 = boto3.client("s3")


def download_from_s3(raw_dir):
    """Download Los-Loop raw CSV files if missing."""
    os.makedirs(raw_dir, exist_ok=True)
    files = ["los_speed.csv", "los_adj.csv"]

    for fname in files:
        local = os.path.join(raw_dir, fname)
        if not os.path.exists(local):
            print(f"Downloading {fname}...")
            s3.download_file(BUCKET, S3_PREFIX + fname, local)
        else:
            print(f"{fname} already present.")

    print("Los-Loop raw data ready.")


def preprocess_and_split(raw_dir="data/raw/los",
                         out_dir="data/processed/los/prepared",
                         num_clients=5,
                         noniid=False,
                         imbalance=0.4,
                         seed=42):
    """Create sequences and federated partitions for Los-Loop data."""
    download_from_s3(raw_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Preprocessing Los-Loop data into {out_dir}...")

    speed = pd.read_csv(os.path.join(raw_dir, "los_speed.csv"), header=None).values
    adj = pd.read_csv(os.path.join(raw_dir, "los_adj.csv"), header=None).values

    # Normalise speed signals
    speed = (speed - speed.mean()) / speed.std()

    seq_len = 12
    X, y = [], []

    for i in range(len(speed) - seq_len - 1):
        X.append(speed[i:i+seq_len])
        y.append(speed[i+seq_len])

    X, y = np.array(X), np.array(y)

    n = len(X)
    n_train, n_val = int(0.7*n), int(0.85*n)

    np.save(os.path.join(out_dir, "X_train.npy"), X[:n_train])
    np.save(os.path.join(out_dir, "y_train.npy"), y[:n_train])
    np.save(os.path.join(out_dir, "X_valid.npy"), X[n_train:n_val])
    np.save(os.path.join(out_dir, "y_valid.npy"), y[n_train:n_val])
    np.save(os.path.join(out_dir, "X_test.npy"), X[n_val:])
    np.save(os.path.join(out_dir, "y_test.npy"), y[n_val:])
    np.save(os.path.join(out_dir, "adj.npy"), adj)

    print(f"Train={n_train}, Val={n_val-n_train}, Test={n-n_val}, Nodes={X.shape[-1]}")

    build_client_datasets(out_dir, num_clients, noniid, imbalance, seed)

    print("\nRenaming client partitions...")

    role_map = {0: "roadside", 1: "vehicle", 2: "sensor", 3: "camera", 4: "bus"}
    clients_dir = os.path.join(out_dir, "clients")

    for idx, role in role_map.items():
        old_x = os.path.join(clients_dir, f"client{idx}_X.npy")
        old_y = os.path.join(clients_dir, f"client{idx}_y.npy")
        new_x = os.path.join(clients_dir, f"client_{role}_X.npy")
        new_y = os.path.join(clients_dir, f"client_{role}_y.npy")

        if os.path.exists(old_x):
            os.rename(old_x, new_x)
        if os.path.exists(old_y):
            os.rename(old_y, new_y)

        print(f"Renamed client {idx} → {role}")

    print("\nCompleted preprocessing of Los-Loop dataset.")


if __name__ == "__main__":
    preprocess_and_split()
