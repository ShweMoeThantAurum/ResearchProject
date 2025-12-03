"""
Update compression utilities.
Implements simple top-k magnitude sparsification for gradients/weights.
"""

import torch


def _topk_tensor(t, k_ratio):
    """Keep only top-k magnitude values in a tensor."""
    if k_ratio <= 0 or k_ratio >= 1:
        return t.clone()

    flat = t.view(-1)
    numel = flat.numel()
    k = max(1, int(numel * k_ratio))

    _, idx = torch.topk(flat.abs(), k)
    out = torch.zeros_like(flat)
    out[idx] = flat[idx]

    return out.view_as(t)


def compress_update(update_dict, settings):
    """
    Apply compression to each tensor in an update dict.
    Uses settings.compression_mode and compression_sparsity/k_frac.
    """
    if not settings.compression_enabled:
        return update_dict

    # For now both "sparsify" and "topk" use the same ratio.
    if settings.compression_mode == "topk":
        ratio = settings.compression_k_frac
    else:
        ratio = settings.compression_sparsity

    if ratio <= 0:
        return update_dict

    compressed = {}
    for key, tensor in update_dict.items():
        compressed[key] = _topk_tensor(tensor, ratio)

    return compressed
