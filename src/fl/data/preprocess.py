"""
Preprocess raw SZ / LOS / PEMS08 traffic datasets for GRU forecasting.
Creates normalized sliding-window sequences and saves train/val/test splits.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

RAW_DIR = "datasets/raw"
PROC_DIR = "datasets/processed"


# Utility functions
def create_sequences(data, seq_len, horizon):
    """Builds sliding window sequences for GRU training."""
    X, y = [], []
    total = len(data)
    end = total - seq_len - horizon

    for i in range(end):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len + horizon - 1])
    return np.array(X), np.array(y)


def normalize(data):
    """Min-max normalization per feature."""
    mn = data.min(axis=0, keepdims=True)
    mx = data.max(axis=0, keepdims=True)
    norm = (data - mn) / (mx - mn + 1e-6)
    return norm.astype(np.float32), mn, mx


def save_tensor(path, arr):
    """Saves numpy arrays or (X, y) tuples using torch.save."""
    # Tuple: (X, y)
    if isinstance(arr, tuple):
        X, y = arr
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        torch.save((X_t, y_t), path)
    else:
        torch.save(torch.tensor(arr, dtype=torch.float32), path)


# Dataset-specific raw loaders
def load_sz():
    """Loads SZ dataset from sz_speed.csv."""
    folder = os.path.join(RAW_DIR, "sz")
    path = os.path.join(folder, "sz_speed.csv")

    if not os.path.exists(path):
        raise FileNotFoundError("Missing SZ speed file: " + path)

    df = pd.read_csv(path)
    return df.values.astype(np.float32)


def load_los():
    """Loads Los Angeles dataset from los_speed.csv."""
    folder = os.path.join(RAW_DIR, "los")
    path = os.path.join(folder, "los_speed.csv")

    if not os.path.exists(path):
        raise FileNotFoundError("Missing LOS speed file: " + path)

    df = pd.read_csv(path)
    return df.values.astype(np.float32)


def load_pems08():
    """Loads PEMS08 dataset from pems08.npz."""
    folder = os.path.join(RAW_DIR, "pems08")
    path = os.path.join(folder, "pems08.npz")

    if not os.path.exists(path):
        raise FileNotFoundError("Missing PEMS08 npz file: " + path)

    data = np.load(path)["data"]  # (N, T)
    data = data.T                 # convert to (T, N)
    return data.astype(np.float32)


# Main preprocessing procedure
def preprocess(dataset):
    """Orchestrates preprocessing for dataset: sz, los, pems08."""
    dataset = dataset.lower()

    # Load raw dataset
    if dataset == "sz":
        values = load_sz()
    elif dataset == "los":
        values = load_los()
    elif dataset == "pems08":
        values = load_pems08()
    else:
        raise ValueError("Unknown dataset: " + dataset)

    print("Loaded raw dataset:", dataset, "| shape:", values.shape)

    # Normalize features
    values_norm, mn, mx = normalize(values)

    # Sequence configuration
    seq_len = 12
    horizon = 3

    # Create sliding window sequences
    X, y = create_sequences(values_norm, seq_len, horizon)
    print("Created sequences:", X.shape, y.shape)

    # Train / Val / Test split
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    # Output directory
    out_dir = os.path.join(PROC_DIR, dataset, "global")
    os.makedirs(out_dir, exist_ok=True)

    # Save splits
    save_tensor(os.path.join(out_dir, "train.pt"), (X_train, y_train))
    save_tensor(os.path.join(out_dir, "val.pt"), (X_val, y_val))
    save_tensor(os.path.join(out_dir, "test.pt"), (X_test, y_test))

    print("Preprocessing complete:", dataset)
    print("Saved to:", out_dir)
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


# CLI entry
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.fl.data.preprocess <sz|los|pems08>")
        sys.exit(1)

    preprocess(sys.argv[1])
