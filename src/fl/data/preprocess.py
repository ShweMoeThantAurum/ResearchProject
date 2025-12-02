"""
Preprocessing pipeline for supported datasets.
"""

import os
import numpy as np

from src.fl.config.settings import get_raw_dir, get_proc_dir, get_dataset


def _sliding_windows(data, seq_len=12):
    """Create sliding windows for GRU input."""
    x = []
    y = []
    for i in range(len(data) - seq_len):
        x.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(x), np.array(y)


def _normalise(data):
    """Min-max normalise each feature."""
    mn = data.min(axis=0)
    mx = data.max(axis=0)
    return (data - mn) / (mx - mn + 1e-6), mn, mx


def preprocess_dataset():
    """
    Preprocess one dataset from raw CSV into train/test numpy files.
    """

    dataset = get_dataset()
    raw_dir = get_raw_dir()
    proc_dir = get_proc_dir()

    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)

    # Dataset-specific raw filename
    fname = None
    if dataset == "sz":
        fname = "sz.csv"
    elif dataset == "los":
        fname = "los.csv"
    elif dataset == "pems08":
        fname = "pems08.csv"
    else:
        raise ValueError("Unsupported dataset {}".format(dataset))

    raw_path = os.path.join(raw_dir, fname)
    data = np.loadtxt(raw_path, delimiter=",")

    # Normalise
    data_norm, mn, mx = _normalise(data)

    # Sliding windows for GRU
    x, y = _sliding_windows(data_norm)

    # Train-test split (80/20)
    split = int(len(x) * 0.8)
    train_x = x[:split]
    train_y = y[:split]
    test_x = x[split:]
    test_y = y[split:]

    # Save processed arrays
    np.save(os.path.join(proc_dir, "train_x.npy"), train_x)
    np.save(os.path.join(proc_dir, "train_y.npy"), train_y)
    np.save(os.path.join(proc_dir, "test_x.npy"), test_x)
    np.save(os.path.join(proc_dir, "test_y.npy"), test_y)

    # Also save normalisation parameters
    np.save(os.path.join(proc_dir, "mn.npy"), mn)
    np.save(os.path.join(proc_dir, "mx.npy"), mx)
