"""
Preprocessing pipelines for SZ-Taxi, Los-Loop, and PeMSD8.

Responsibilities:
    - Download raw CSV/NPZ from S3 if missing
    - Normalise speed/flow signals
    - Build sliding-window sequences
    - Split into train/valid/test
    - Partition nodes into IoT roles and save X_<role>.npy / y_<role>.npy
"""

import os
import boto3
import numpy as np
import pandas as pd

from .partition import save_role_partitions


BUCKET = os.environ.get("S3_BUCKET", "aefl")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=AWS_REGION)


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _download_if_missing(key_prefix, filenames, local_dir):
    """
    Download a set of files from S3 if they are missing locally.

    key_prefix: e.g. "raw/sz/"
    filenames:  list of names under that prefix
    local_dir:  local directory under datasets/raw/<dataset>/
    """
    _ensure_dir(local_dir)
    for fname in filenames:
        local_path = os.path.join(local_dir, fname)
        if os.path.exists(local_path):
            print("%s already present." % local_path)
            continue
        s3_key = os.path.join(key_prefix, fname)
        print("Downloading s3://%s/%s -> %s" % (BUCKET, s3_key, local_path))
        s3.download_file(BUCKET, s3_key, local_path)


def _split_and_save(X,
                    y,
                    processed_dir,
                    seq_len=12,
                    train_ratio=0.7,
                    val_ratio=0.15):
    """
    Split X, y into train/valid/test and save to processed_dir.

    X: [time, num_nodes] or [samples, seq_len, num_nodes]
    y: [time, num_nodes] or [samples, num_nodes]

    If X has shape [time, num_nodes], this function first builds
    sliding-window sequences of length seq_len.
    """
    _ensure_dir(processed_dir)

    if X.ndim == 2:
        # Build sequences
        seq_X = []
        seq_y = []
        for i in range(len(X) - seq_len - 1):
            seq_X.append(X[i:i + seq_len])
            seq_y.append(X[i + seq_len])
        X = np.array(seq_X)
        y = np.array(seq_y)

    n = len(X)
    n_train = int(train_ratio * n)
    n_val = int((train_ratio + val_ratio) * n)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_valid = X[n_train:n_val]
    y_valid = y[n_train:n_val]
    X_test = X[n_val:]
    y_test = y[n_val:]

    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "X_valid.npy"), X_valid)
    np.save(os.path.join(processed_dir, "y_valid.npy"), y_valid)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)

    print("Saved splits to %s" % processed_dir)
    print("  Train: %d" % len(X_train))
    print("  Valid: %d" % len(X_valid))
    print("  Test:  %d" % len(X_test))

    return X_train, y_train


def preprocess_sz(root_raw="datasets/raw",
                  root_processed="datasets/processed",
                  noniid=False,
                  imbalance_factor=0.4,
                  seed=42):
    """
    Preprocess SZ-Taxi dataset.

    Raw files expected (downloaded from S3 if missing):
        - sz_speed.csv
        - sz_adj.csv

    Output directory:
        datasets/processed/sz/
    """
    raw_dir = os.path.join(root_raw, "sz")
    processed_dir = os.path.join(root_processed, "sz")

    _download_if_missing("raw/sz", ["sz_speed.csv", "sz_adj.csv"], raw_dir)

    speed = pd.read_csv(os.path.join(raw_dir, "sz_speed.csv"), header=None).values
    adj = pd.read_csv(os.path.join(raw_dir, "sz_adj.csv"), header=None).values

    # Normalise
    speed = (speed - speed.mean()) / speed.std()

    X_train, y_train = _split_and_save(speed, speed, processed_dir)

    # Save adjacency for reference
    np.save(os.path.join(processed_dir, "adj.npy"), adj)

    # Partition nodes into roles and save per-role files
    save_role_partitions(
        X_train,
        y_train,
        processed_dir,
        roles=None,
        noniid=noniid,
        imbalance_factor=imbalance_factor,
        seed=seed,
    )


def preprocess_los(root_raw="datasets/raw",
                   root_processed="datasets/processed",
                   noniid=False,
                   imbalance_factor=0.4,
                   seed=42):
    """
    Preprocess Los-Loop dataset.

    Raw files:
        - los_speed.csv
        - los_adj.csv

    Output:
        datasets/processed/los/
    """
    raw_dir = os.path.join(root_raw, "los")
    processed_dir = os.path.join(root_processed, "los")

    _download_if_missing("raw/los", ["los_speed.csv", "los_adj.csv"], raw_dir)

    speed = pd.read_csv(os.path.join(raw_dir, "los_speed.csv"), header=None).values
    adj = pd.read_csv(os.path.join(raw_dir, "los_adj.csv"), header=None).values

    speed = (speed - speed.mean()) / speed.std()

    X_train, y_train = _split_and_save(speed, speed, processed_dir)

    np.save(os.path.join(processed_dir, "adj.npy"), adj)

    save_role_partitions(
        X_train,
        y_train,
        processed_dir,
        roles=None,
        noniid=noniid,
        imbalance_factor=imbalance_factor,
        seed=seed,
    )


def preprocess_pems08(root_raw="datasets/raw",
                      root_processed="datasets/processed",
                      noniid=False,
                      imbalance_factor=0.4,
                      seed=42):
    """
    Preprocess PeMSD8 dataset.

    Raw NPZ file expected:
        - pems08.npz

    Inside NPZ we expect a tensor of shape:
        [time, num_nodes, channels] or [num_nodes, time, channels]

    Output:
        datasets/processed/pems08/
    """
    raw_dir = os.path.join(root_raw, "pems08")
    processed_dir = os.path.join(root_processed, "pems08")

    _ensure_dir(raw_dir)
    _ensure_dir(processed_dir)

    fname = "pems08.npz"
    local_npz = os.path.join(raw_dir, fname)

    if not os.path.exists(local_npz):
        s3_key = os.path.join("raw/pems08", fname)
        print("Downloading s3://%s/%s -> %s" % (BUCKET, s3_key, local_npz))
        s3.download_file(BUCKET, s3_key, local_npz)
    else:
        print("%s already present." % local_npz)

    arr = np.load(local_npz)
    if "data" in arr.files:
        data = arr["data"]
    else:
        data = arr[arr.files[0]]

    # Collapse channel dimension to a single speed/flow channel
    if data.ndim == 3:
        if data.shape[0] < data.shape[1]:
            data = data.transpose(1, 0, 2)
        data = data[..., 0]
    elif data.ndim != 2:
        raise ValueError("Unexpected PeMSD8 shape %s" % (data.shape,))

    data = (data - data.mean()) / data.std()

    X_train, y_train = _split_and_save(data, data, processed_dir)

    save_role_partitions(
        X_train,
        y_train,
        processed_dir,
        roles=None,
        noniid=noniid,
        imbalance_factor=imbalance_factor,
        seed=seed,
    )
