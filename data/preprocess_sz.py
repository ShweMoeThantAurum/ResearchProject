"""
Preprocesses SZ-Taxi CSVs into normalized, windowed arrays and builds client partitions.
Automatically downloads raw files from S3 if not present locally.
"""

import os
import boto3
import numpy as np
from data.dataloader import build_client_datasets

BUCKET = "aefl-results"
S3_PREFIX = "raw/sz/"   # sz_speed.csv, sz_adj.csv

s3 = boto3.client("s3")

def download_from_s3(raw_dir):
    """Download SZ raw files from S3 into raw_dir."""
    os.makedirs(raw_dir, exist_ok=True)
    files = ["sz_speed.csv", "sz_adj.csv"]

    for fname in files:
        local_path = os.path.join(raw_dir, fname)
        if not os.path.exists(local_path):
            print(f"Downloading {fname} from S3...")
            s3.download_file(BUCKET, S3_PREFIX + fname, local_path)
        else:
            print(f"{fname} already exists locally.")
    print("SZ raw data ready.")

def preprocess_and_split(raw_dir="data/raw/sz",
                         out_dir="data/processed/sz/prepared",
                         num_clients=5,
                         noniid=False,
                         imbalance=0.4,
                         seed=42):
    """Converts raw SZ-Taxi data into prepared arrays and per-client splits."""
    download_from_s3(raw_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Preprocessing SZ-Taxi data into {out_dir}...")

    # NOTE: SZ preprocessing is minimal in your setup — if you add normalization,
    # do it here. For now, we just create client splits.
    build_client_datasets(
        proc_dir=out_dir,
        num_clients=num_clients,
        noniid=noniid,
        imbalance_factor=imbalance,
        seed=seed,
    )

    print(f"Completed preprocessing with {'Non-IID' if noniid else 'IID'} partitioning.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Preprocess SZ-Taxi dataset")
    p.add_argument("--raw_dir", type=str, default="data/raw/sz")
    p.add_argument("--out_dir", type=str, default="data/processed/sz/prepared")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--noniid", action="store_true")
    p.add_argument("--imbalance", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    preprocess_and_split(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        num_clients=args.clients,
        noniid=args.noniid,
        imbalance=args.imbalance,
        seed=args.seed
    )
