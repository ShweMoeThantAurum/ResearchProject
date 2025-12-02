"""
Preprocessing pipeline for the SZ-Taxi dataset.

Downloads raw CSVs from S3 if missing, prepares local arrays, and
creates federated client partitions mapped to IoT roles.
"""

import os
import boto3
import numpy as np
from data.dataloader import build_client_datasets

BUCKET = "aefl"
S3_PREFIX = "raw/sz/"
s3 = boto3.client("s3")


def download_from_s3(raw_dir):
    """Download SZ-Taxi raw CSV files if missing."""
    os.makedirs(raw_dir, exist_ok=True)
    files = ["sz_speed.csv", "sz_adj.csv"]

    for fname in files:
        local = os.path.join(raw_dir, fname)
        if not os.path.exists(local):
            print(f"Downloading {fname}...")
            s3.download_file(BUCKET, S3_PREFIX + fname, local)
        else:
            print(f"{fname} already exists locally.")

    print("SZ raw data ready.")


def preprocess_and_split(raw_dir="data/raw/sz",
                         out_dir="data/processed/sz/prepared",
                         num_clients=5,
                         noniid=False,
                         imbalance=0.4,
                         seed=42):
    """Create client splits for SZ-Taxi data (preprocessed offline)."""
    download_from_s3(raw_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Preprocessing SZ-Taxi data into {out_dir}...")

    # Dataset is already preprocessed externally
    build_client_datasets(out_dir, num_clients, noniid, imbalance, seed)

    # Rename numeric clients → IoT role names
    print("\nRenaming client partitions to IoT roles...")

    role_map = {
        0: "roadside",
        1: "vehicle",
        2: "sensor",
        3: "camera",
        4: "bus"
    }

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

    print(f"\nCompleted preprocessing. Final client directory: {clients_dir}")


if __name__ == "__main__":
    preprocess_and_split()
