"""
Load processed datasets for federated learning.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.fl.config.settings import get_proc_dir


def _load_numpy(path):
    """Load a numpy file."""
    return np.load(path)


def load_prepared_data():
    """Load preprocessed train and test numpy arrays."""
    base = get_proc_dir()

    train_x = _load_numpy(os.path.join(base, "train_x.npy"))
    train_y = _load_numpy(os.path.join(base, "train_y.npy"))
    test_x = _load_numpy(os.path.join(base, "test_x.npy"))
    test_y = _load_numpy(os.path.join(base, "test_y.npy"))

    return train_x, train_y, test_x, test_y


def to_tensor_dataset(x, y):
    """Convert numpy arrays to a PyTorch TensorDataset."""
    tx = torch.tensor(x).float()
    ty = torch.tensor(y).float()
    return TensorDataset(tx, ty)


def load_datasets_as_torch():
    """Load processed data as PyTorch datasets."""
    train_x, train_y, test_x, test_y = load_prepared_data()
    return (
        to_tensor_dataset(train_x, train_y),
        to_tensor_dataset(test_x, test_y),
    )
