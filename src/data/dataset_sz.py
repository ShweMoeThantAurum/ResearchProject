"""
Loads windowed SZ arrays for PyTorch training and evaluation.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class WindowedArray(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)          # [S, L, N]
        self.y = np.load(y_path)          # [S, N]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        return x, y
