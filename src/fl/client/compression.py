"""
Model update compression utilities.

Currently supports:
    - sparsify       : zero-out a fraction of smallest values
    - topk           : keep only top-k fraction of values
"""

import torch


def apply_compression(state_dict, mode, sparsity=0.5, k_frac=0.1):
    """
    Apply compression method to each tensor in the state dict.

    Parameters:
        mode     : "sparsify" or "topk"
        sparsity : fraction to zero-out (sparsify mode)
        k_frac   : fraction to keep (top-k mode)
    """
    for name, tensor in state_dict.items():
        flat = tensor.view(-1)

        if mode == "sparsify":
            k = int(sparsity * flat.numel())
            if k > 0:
                thresh = torch.topk(flat.abs(), k, largest=False).values.max()
                flat[flat.abs() <= thresh] = 0

        elif mode == "topk":
            k = int(k_frac * flat.numel())
            if k > 0:
                thresh = torch.topk(flat.abs(), k).values.min()
                flat[flat.abs() < thresh] = 0

        tensor.copy_(flat.view_as(tensor))
