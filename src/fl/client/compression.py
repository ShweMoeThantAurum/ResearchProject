"""
Model update compression (sparsification or top-k).
"""

import torch
import numpy as np
from src.fl.config import settings


def apply_compression(update):
    """
    Apply sparsification or top-k depending on config.
    """
    mode = settings.get_compression_mode()

    if mode == "sparsify":
        return sparsify(update)
    elif mode == "topk":
        return topk(update)

    return update


def sparsify(update):
    """
    Zero out a fraction of smallest-magnitude weights.
    """
    frac = settings.get_compression_sparsity()
    new = {}

    for k, v in update.items():
        flat = v.view(-1)
        k_val = int(len(flat) * frac)
        thresh = flat.abs().kthvalue(k_val).values.item()

        mask = v.abs() >= thresh
        new[k] = v * mask

    return new


def topk(update):
    """
    Keep only top-K fraction of weights.
    """
    frac = settings.get_compression_k_frac()
    new = {}

    for k, v in update.items():
        flat = v.view(-1)
        k_val = int(len(flat) * frac)
        topk_vals, _ = torch.topk(flat.abs(), k_val)
        thresh = topk_vals.min()

        mask = v.abs() >= thresh
        new[k] = v * mask

    return new
