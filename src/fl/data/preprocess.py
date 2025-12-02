"""
Dataset preprocessing utilities for SZ-taxi, LOS-loop, and PeMS08 datasets.
Transforms raw files into standardised tensors for FL experiments.
"""

import os
import numpy as np
import pandas as pd
import torch


def _save_tensor(path, tensor):
    """Save preprocessed tensor to disk."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(tensor, path)


def preprocess_sz(raw_dir, out_path, seq_len=12):
    """Preprocess SZ-taxi dataset into sequences."""
    raw = pd.read_csv(os.path.join(raw_dir, "sz.csv"))
    values = raw.values.astype(np.float32)

    data = []
    for i in range(len(values) - seq_len):
        window = values[i : i + seq_len]
        data.append(window)

    data = torch.tensor(data)  # [N, seq, nodes]
    _save_tensor(out_path, data)


def preprocess_los(raw_dir, out_path, seq_len=12):
    """Preprocess LOS-loop dataset into sequences."""
    raw = pd.read_csv(os.path.join(raw_dir, "los.csv"))
    values = raw.values.astype(np.float32)

    data = []
    for i in range(len(values) - seq_len):
        window = values[i : i + seq_len]
        data.append(window)

    data = torch.tensor(data)
    _save_tensor(out_path, data)


def preprocess_pems08(raw_dir, out_path, seq_len=12):
    """Preprocess PeMS08 dataset into sequences."""
    raw = pd.read_csv(os.path.join(raw_dir, "pems08.csv"))
    values = raw.values.astype(np.float32)

    data = []
    for i in range(len(values) - seq_len):
        window = values[i : i + seq_len]
        data.append(window)

    data = torch.tensor(data)
    _save_tensor(out_path, data)
