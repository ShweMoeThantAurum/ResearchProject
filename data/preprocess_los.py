"""
Preprocesses Los-Loop CSVs into standardized arrays and builds client partitions.
Automatically downloads raw files from S3.
"""

import os
import boto3
import numpy as np
import pandas as pd
from data.dataloader import build_client_datasets

BUCKET = "aefl-results"
S3_PREFIX = "raw/los/"  # los_speed.csv, los_adj.csv

s3 = boto3.client("s3")

def download_from_s3(raw_dir):
    """Download Los-Loop raw files from S3 into raw_dir."""
    os.makedirs(raw_dir, exist_ok=True)
    files = ["los_speed.csv", "los_adj.csv"]

    for fname in files:
        local_path = os.path.join(raw_dir, fname)
        if not os.path.exists(local_path):
            print(f"Downloading {fname} from S3...")
            s3.download_file(BUCKET, S3_PREFIX + fname, local_path)
        else:
            print(f"{fname} already exists locally.")
    print("Los-Loop raw data ready.")

def preprocess_and_split(raw_dir="data/raw/los",
                         out_dir="data/processed/los/prepared",
                         num_clients=5,
                         noniid=False,
                         imbalance=0.4,
                         seed=42):

    download_from_s3(raw_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Preprocessing Los-Loop data into {out_dir}...")

    speed = pd.read_csv(os.path.join(raw_dir, "los_speed.csv"), header=None).values
    adj = pd.read_csv(os.path.join(raw_dir, "los_adj.csv"), header=None).values

    # Normalize
    speed = (speed - speed.mean()) / speed.std()

    # Sliding window (12 input → 1 output)
    seq_len = 12
    X, y = [], []
    for i in range(len(speed) - seq_len - 1):
        X.append(speed[i:i+seq_len])
        y.append(speed[i+seq_len])
    X, y = np.array(X), np.array(y)

    # Train/val/test split
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

    # Federated split
    build_client_datasets(out_dir, num_clients, noniid, imbalance, seed)

    print("Completed preprocessing of Los-Loop dataset.")


if __name__ == "__main__":
    preprocess_and_split()
