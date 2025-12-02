"""
Preprocessing utilities for raw traffic datasets.

Handles:
    - missing value cleanup
    - min–max normalization
    - sequence-to-supervised sliding window creation
    - saving processed datasets
"""

import os
import numpy as np
from src.fl.utils.serialization import save_numpy


def preprocess_dataset(raw_array, seq_len=12, horizon=1, save_dir=None):
    """
    Clean and convert a raw traffic matrix into supervised learning windows.

    Args:
        raw_array: numpy array of shape [time, num_nodes]
        seq_len: number of past timesteps for each input sequence
        horizon: number of timesteps ahead to predict
        save_dir: optional output directory to save processed files

    Returns:
        X: input sequences [num_samples, seq_len, num_nodes]
        y: targets [num_samples, num_nodes]
        norm_stats: dict with min/max for denormalization
    """
    # Fill missing values with column means
    clean = np.where(np.isnan(raw_array), np.nanmean(raw_array, axis=0), raw_array)

    # Min–max normalization
    data_min = clean.min(axis=0)
    data_max = clean.max(axis=0)
    norm = (clean - data_min) / (data_max - data_min + 1e-6)

    # Sliding windows
    X_list = []
    y_list = []

    T = len(norm)
    for t in range(T - seq_len - horizon + 1):
        X_list.append(norm[t : t + seq_len])
        y_list.append(norm[t + seq_len + horizon - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    norm_stats = {"min": data_min, "max": data_max}

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_numpy(os.path.join(save_dir, "X.npy"), X)
        save_numpy(os.path.join(save_dir, "y.npy"), y)
        save_numpy(os.path.join(save_dir, "norm_stats.npy"), np.array([data_min, data_max]))

    return X, y, norm_stats
