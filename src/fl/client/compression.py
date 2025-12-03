"""
Simple update compression utilities.
Implements optional top-k sparsification to reduce communication cost.
"""

import torch


def _topk_tensor(t: torch.Tensor, k_ratio: float):
    """
    Returns a sparsified version of the tensor where only the top-k%
    magnitude values are kept.
    """

    if k_ratio <= 0 or k_ratio >= 1:
        return t.clone()

    numel = t.numel()
    k = max(1, int(numel * k_ratio))

    # Flatten for easy processing
    flat = t.view(-1)

    # Top-k magnitude indices
    _, idx = torch.topk(flat.abs(), k)

    # Build sparse tensor
    out = torch.zeros_like(flat)
    out[idx] = flat[idx]

    return out.view_as(t)


def compress_update(update_dict: dict, settings):
    """
    Apply compression to each tensor in the update dict.
    Compression is optional and controlled by:
        settings.compression_enabled
        settings.compression_ratio   (0 < r < 1)
    """

    ratio = getattr(settings, "compression_ratio", 0.0)

    if ratio <= 0:
        # No compression
        return update_dict

    compressed = {}
    for key, tensor in update_dict.items():
        compressed[key] = _topk_tensor(tensor, ratio)

    return compressed
