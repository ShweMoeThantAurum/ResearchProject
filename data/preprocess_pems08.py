"""
Preprocesses PeMSD8 .npz file into standardized arrays and client splits.
Automatically downloads raw file from S3.
"""

import os
import boto3
import numpy as np
from data.dataloader import build_client_datasets

BUCKET = "aefl-results"
S3_PREFIX = "raw/pems08/"   # pems08.npz

s3 = boto3.client("s3")

def download_from_s3(raw_dir):
    """Download PeMSD8 NPZ file from S3 into raw_dir."""
    os.makedirs(raw_dir, exist_ok=True)
    fname = "pems08.npz"

    local_path = os.path.join(raw_dir, fname)

    if not os.path.exists(local_path):
        print(f"Downloading {fname} from S3...")
        s3.download_file(BUCKET, S3_PREFIX + fname, local_path)
    else:
        print(f"{fname} already exists locally.")

    print("PeMSD8 raw data ready.")

def preprocess_and_split(raw_dir="data/raw/pems08",
                         out_dir="data/processed/pems08/prepared",
                         num_clients=5,
                         noniid=False,
                         imbalance=0.4,
                         seed=42):

    download_from_s3(raw_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Preprocessing PeMSD8 data into {out_dir}...")

    arr = np.load(os.path.join(raw_dir, "pems08.npz"))
    key = "data" if "data" in arr.files else arr.files[0]
    arr = arr[key]

    # Handle shape
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[1]:
            arr = arr.transpose(1,0,2)
        arr = arr[..., 0]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected shape {arr.shape}")

    # Normalize
    arr = (arr - arr.mean()) / arr.std()

    # Sliding window 12->1
    seq_len=12
    X, y = [], []

    for i in range(len(arr) - seq_len - 1):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len])

    X, y = np.array(X), np.array(y)

    # Splits
    n=len(X)
    n_train, n_val = int(0.7*n), int(0.85*n)

    np.save(os.path.join(out_dir,"X_train.npy"),X[:n_train])
    np.save(os.path.join(out_dir,"y_train.npy"),y[:n_train])
    np.save(os.path.join(out_dir,"X_valid.npy"),X[n_train:n_val])
    np.save(os.path.join(out_dir,"y_valid.npy"),y[n_train:n_val])
    np.save(os.path.join(out_dir,"X_test.npy"),X[n_val:])
    np.save(os.path.join(out_dir,"y_test.npy"),y[n_val:])

    print(f"Train={n_train}, Val={n_val-n_train}, Test={n-n_val}, Nodes={X.shape[-1]}")

    # Federated split
    build_client_datasets(out_dir, num_clients, noniid, imbalance, seed)

    print("Completed preprocessing of PeMSD8 dataset.")


if __name__ == "__main__":
    preprocess_and_split()
