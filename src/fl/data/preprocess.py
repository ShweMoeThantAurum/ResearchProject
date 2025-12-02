"""
Dataset preprocessing pipelines for traffic flow prediction.

This module provides helpers to:
    - Preprocess PeMSD8, Los-Loop, and SZ-Taxi datasets
    - Construct sliding-window sequences
    - Normalise signals
    - Build client partitions

Processed data is stored under:
    datasets/processed/<dataset_name>/
with files:
    X_train.npy, y_train.npy
    X_valid.npy, y_valid.npy
    X_test.npy,  y_test.npy
    clients/...
"""

import os
import numpy as np
import pandas as pd

from src.fl.data.partition import build_client_datasets


def _ensure_dir(path):
    """
    Create a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _sliding_window_sequences(data, seq_len):
    """
    Construct sliding-window sequences from a 2D array.

    Parameters:
        data    : np.ndarray [time, num_nodes]
        seq_len : number of time steps in input sequence

    Returns:
        X: [num_samples, seq_len, num_nodes]
        y: [num_samples, num_nodes]
    """
    X = []
    y = []
    for i in range(len(data) - seq_len - 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def preprocess_pems08(raw_dir, out_dir,
                      num_clients=5,
                      noniid=False,
                      imbalance_factor=0.4,
                      seed=42,
                      seq_len=12):
    """
    Preprocess the PeMSD8 dataset from an NPZ file.

    Expects:
        raw_dir/pems08.npz

    Steps:
        - Load traffic tensor
        - Normalise values
        - Build sliding-window sequences
        - Split into train/valid/test
        - Save npy arrays
        - Build client partitions
    """
    _ensure_dir(raw_dir)
    _ensure_dir(out_dir)

    npz_path = os.path.join(raw_dir, "pems08.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError("Missing pems08.npz in %s" % raw_dir)

    arr = np.load(npz_path)
    key = "data" if "data" in arr.files else arr.files[0]
    data = arr[key]

    if data.ndim == 3:
        if data.shape[0] < data.shape[1]:
            data = data.transpose(1, 0, 2)
        data = data[..., 0]
    elif data.ndim != 2:
        raise ValueError("Unexpected PeMSD8 shape: %s" % (data.shape,))

    data = (data - data.mean()) / data.std()

    X, y = _sliding_window_sequences(data, seq_len)
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.85 * n)

    np.save(os.path.join(out_dir, "X_train.npy"), X[:n_train])
    np.save(os.path.join(out_dir, "y_train.npy"), y[:n_train])
    np.save(os.path.join(out_dir, "X_valid.npy"), X[n_train:n_val])
    np.save(os.path.join(out_dir, "y_valid.npy"), y[n_train:n_val])
    np.save(os.path.join(out_dir, "X_test.npy"), X[n_val:])
    np.save(os.path.join(out_dir, "y_test.npy"), y[n_val:])

    print("PeMSD8: train=%d val=%d test=%d nodes=%d" %
          (n_train, n_val - n_train, n - n_val, X.shape[-1]))

    build_client_datasets(
        proc_dir=out_dir,
        num_clients=num_clients,
        noniid=noniid,
        imbalance_factor=imbalance_factor,
        seed=seed,
    )


def preprocess_los_loop(raw_dir, out_dir,
                        num_clients=5,
                        noniid=False,
                        imbalance_factor=0.4,
                        seed=42,
                        seq_len=12):
    """
    Preprocess the Los-Loop dataset from CSV files.

    Expects:
        raw_dir/los_speed.csv
        raw_dir/los_adj.csv

    Steps:
        - Load speed matrix and adjacency matrix
        - Normalise signals
        - Build sliding-window sequences
        - Split into train/valid/test
        - Save npy arrays
        - Build client partitions
    """
    _ensure_dir(raw_dir)
    _ensure_dir(out_dir)

    speed_path = os.path.join(raw_dir, "los_speed.csv")
    adj_path = os.path.join(raw_dir, "los_adj.csv")

    if not os.path.exists(speed_path) or not os.path.exists(adj_path):
        raise FileNotFoundError("Missing los_speed.csv or los_adj.csv in %s" %
                                raw_dir)

    speed = pd.read_csv(speed_path, header=None).values
    adj = pd.read_csv(adj_path, header=None).values

    speed = (speed - speed.mean()) / speed.std()

    X, y = _sliding_window_sequences(speed, seq_len)
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.85 * n)

    np.save(os.path.join(out_dir, "X_train.npy"), X[:n_train])
    np.save(os.path.join(out_dir, "y_train.npy"), y[:n_train])
    np.save(os.path.join(out_dir, "X_valid.npy"), X[n_train:n_val])
    np.save(os.path.join(out_dir, "y_valid.npy"), y[n_train:n_val])
    np.save(os.path.join(out_dir, "X_test.npy"), X[n_val:])
    np.save(os.path.join(out_dir, "y_test.npy"), y[n_val:])
    np.save(os.path.join(out_dir, "adj.npy"), adj)

    print("Los-Loop: train=%d val=%d test=%d nodes=%d" %
          (n_train, n_val - n_train, n - n_val, X.shape[-1]))

    build_client_datasets(
        proc_dir=out_dir,
        num_clients=num_clients,
        noniid=noniid,
        imbalance_factor=imbalance_factor,
        seed=seed,
    )


def preprocess_sz_taxi(proc_dir,
                       num_clients=5,
                       noniid=False,
                       imbalance_factor=0.4,
                       seed=42):
    """
    Prepare SZ-Taxi client partitions.

    Assumes that:
        proc_dir contains already prepared:
            X_train.npy, y_train.npy,
            X_valid.npy, y_valid.npy,
            X_test.npy,  y_test.npy

    This function does not recompute SZ-Taxi preprocessing
    (which is often done offline), but only builds client splits.
    """
    required = [
        "X_train.npy", "y_train.npy",
        "X_valid.npy", "y_valid.npy",
        "X_test.npy", "y_test.npy",
    ]

    for name in required:
        path = os.path.join(proc_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError("SZ-Taxi requires %s in %s" %
                                    (name, proc_dir))

    print("SZ-Taxi: found prepared arrays in %s" % proc_dir)

    build_client_datasets(
        proc_dir=proc_dir,
        num_clients=num_clients,
        noniid=noniid,
        imbalance_factor=imbalance_factor,
        seed=seed,
    )


def preprocess_dataset(dataset_name,
                       raw_root="datasets/raw",
                       proc_root="datasets/processed",
                       num_clients=5,
                       noniid=False,
                       imbalance_factor=0.4,
                       seed=42,
                       seq_len=12):
    """
    Dispatch function to preprocess a given dataset name.

    Supported names:
        - "pems08"
        - "los"
        - "sz"

    For SZ-Taxi, this expects that the prepared arrays
    already exist under:
        datasets/processed/sz
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "pems08":
        raw_dir = os.path.join(raw_root, "pems08")
        out_dir = os.path.join(proc_root, "pems08")
        preprocess_pems08(
            raw_dir=raw_dir,
            out_dir=out_dir,
            num_clients=num_clients,
            noniid=noniid,
            imbalance_factor=imbalance_factor,
            seed=seed,
            seq_len=seq_len,
        )
    elif dataset_name == "los":
        raw_dir = os.path.join(raw_root, "los")
        out_dir = os.path.join(proc_root, "los")
        preprocess_los_loop(
            raw_dir=raw_dir,
            out_dir=out_dir,
            num_clients=num_clients,
            noniid=noniid,
            imbalance_factor=imbalance_factor,
            seed=seed,
            seq_len=seq_len,
        )
    elif dataset_name == "sz":
        out_dir = os.path.join(proc_root, "sz")
        preprocess_sz_taxi(
            proc_dir=out_dir,
            num_clients=num_clients,
            noniid=noniid,
            imbalance_factor=imbalance_factor,
            seed=seed,
        )
    else:
        raise ValueError("Unsupported dataset name: %s" % dataset_name)


if __name__ == "__main__":
    # Simple CLI entrypoint, if you want to run:
    #   python -m src.fl.data.preprocess pems08
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess datasets for federated learning"
    )
    parser.add_argument("dataset", type=str,
                        help="Dataset name: pems08, los, sz")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--noniid", action="store_true")
    parser.add_argument("--imbalance", type=float, default=0.4)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    preprocess_dataset(
        dataset_name=args.dataset,
        num_clients=args.clients,
        noniid=args.noniid,
        imbalance_factor=args.imbalance,
        seed=args.seed,
        seq_len=args.seq_len,
    )
