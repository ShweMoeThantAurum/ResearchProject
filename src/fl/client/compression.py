"""
Model update compression.

Two modes supported (controlled by COMPRESSION_MODE):
    - sparsify: zero out smallest values
    - topk: keep only largest K fraction
"""

import os
import torch
import numpy as np


def apply_compression(update):
    mode = os.environ.get("COMPRESSION_MODE", "sparsify").lower()

    if mode == "sparsify":
        sparsity = float(os.environ.get("COMPRESSION_SPARSITY", 0.5))
        return sparsify(update, sparsity)

    if mode == "topk":
        frac = float(os.environ.get("COMPRESSION_K_FRAC", 0.1))
        return topk(update, frac)

    return update


def sparsify(update, sparsity):
    """Zero out smallest values."""
    out = {}
    for k, v in update.items():
        arr = v.numpy()
        thresh = np.quantile(np.abs(arr), sparsity)
        mask = np.abs(arr) >= thresh
        out[k] = torch.tensor(arr * mask)
    return out


def topk(update, frac):
    """Keep only largest-K fraction of values."""
    out = {}
    for k, v in update.items():
        arr = v.numpy().flatten()
        k_top = max(1, int(len(arr) * frac))
        idx = np.argpartition(np.abs(arr), -k_top)[-k_top:]
        mask = np.zeros_like(arr)
        mask[idx] = 1
        out[k] = torch.tensor((v.numpy().flatten() * mask).reshape(v.numpy().shape))
    return out
