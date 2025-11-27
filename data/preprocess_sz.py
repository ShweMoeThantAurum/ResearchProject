"""Preprocess SZ-Taxi dataset, generate train/val/test arrays, and create FL client splits."""

import os
import boto3
import numpy as np
from data.dataloader import build_client_datasets

BUCKET = "aefl"
S3_PREFIX = "raw/sz/"
s3 = boto3.client("s3")


def download_from_s3(raw_dir):
    """Ensure SZ raw CSVs exist locally by downloading from S3 if missing."""
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
    """Convert SZ raw CSVs into prepared arrays, then create and rename client partitions."""
    download_from_s3(raw_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Preprocessing SZ-Taxi data into {out_dir}...")

    # Client partitions (actual preprocessing omitted here — raw SZ preprocessed earlier)
    build_client_datasets(out_dir, num_clients, noniid, imbalance, seed)

    # Rename numeric clients → IoT role identifiers
    print("\nRenaming client partitions to semantic IoT roles...")

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

    print(f"\nCompleted preprocessing. Final partition dir: {clients_dir}")


if __name__ == "__main__":
    preprocess_and_split()
