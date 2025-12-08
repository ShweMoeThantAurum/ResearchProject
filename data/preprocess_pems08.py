"""
Preprocessing pipeline for the PeMSD8 traffic dataset.

Loads NPZ traffic tensors, extracts sliding-window samples, normalises
signals, and creates federated client partitions mapped to IoT roles.
"""

import os
import boto3
import numpy as np

from data.dataloader import build_client_datasets

BUCKET = "aefl"
S3_PREFIX = "raw/pems08/"
s3 = boto3.client("s3")


def download_from_s3(raw_dir):
    """Download PeMSD8 NPZ file if it is missing."""
    os.makedirs(raw_dir, exist_ok=True)
    fname = "pems08.npz"
    local = os.path.join(raw_dir, fname)

    if not os.path.exists(local):
        print(f"Downloading {fname} from s3://{BUCKET}/{S3_PREFIX}{fname}...")
        s3.download_file(BUCKET, S3_PREFIX + fname, local)
    else:
        print(f"{fname} already exists locally.")

    print("PeMSD8 raw data ready.")


def preprocess_and_split(
    raw_dir="data/raw/pems08",
    out_dir="data/processed/pems08/prepared",
    num_clients=5,
    noniid=False,
    imbalance=0.4,
    seed=42,
):
    """Create sliding-window sequences and federated partitions for PeMSD8."""
    download_from_s3(raw_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Preprocessing PeMSD8 data into {out_dir}...")

    arr = np.load(os.path.join(raw_dir, "pems08.npz"))
    key = "data" if "data" in arr.files else arr.files[0]
    data = arr[key]

    # Expected shapes: [T, N, C] or [N, T, C] or [T, N]
    if data.ndim == 3:
        # Ensure time dimension is first
        if data.shape[0] < data.shape[1]:
            data = data.transpose(1, 0, 2)
        data = data[..., 0]  # keep first channel
    elif data.ndim != 2:
        raise ValueError(f"Unexpected PeMSD8 data shape: {data.shape}")

    # Global standardisation
    data = (data - data.mean()) / data.std()

    # Build sliding windows
    seq_len = 12
    X, y = [], []

    for i in range(len(data) - seq_len - 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])

    X = np.array(X)
    y = np.array(y)

    # Train/valid/test split
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.85 * n)

    np.save(os.path.join(out_dir, "X_train.npy"), X[:n_train])
    np.save(os.path.join(out_dir, "y_train.npy"), y[:n_train])
    np.save(os.path.join(out_dir, "X_valid.npy"), X[n_train:n_val])
    np.save(os.path.join(out_dir, "y_valid.npy"), y[n_train:n_val])
    np.save(os.path.join(out_dir, "X_test.npy"), X[n_val:])
    np.save(os.path.join(out_dir, "y_test.npy"), y[n_val:])

    # Build client partitions
    build_client_datasets(out_dir, num_clients, noniid, imbalance, seed)

    print("\nRenaming client partitions to IoT roles...")

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

    print("\nCompleted preprocessing of PeMSD8 dataset.")


if __name__ == "__main__":
    preprocess_and_split()
