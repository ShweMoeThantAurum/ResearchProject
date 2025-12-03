"""
Simple update compression utilities.
Implements sparsify and top-k compression to reduce communication cost.
"""

import torch


def _sparsify_tensor(t, sparsity):
    """Zero out small-magnitude values to reach target sparsity."""
    if sparsity <= 0.0:
        return t.clone()
    if sparsity >= 1.0:
        return torch.zeros_like(t)

    flat = t.view(-1)
    numel = flat.numel()
    keep_frac = 1.0 - sparsity
    k = int(numel * keep_frac)

    if k <= 0:
        return torch.zeros_like(t)

    # Threshold so that only k largest magnitudes are kept
    values, _ = torch.topk(flat.abs(), k)
    threshold = values.min()

    mask = flat.abs() >= threshold
    out = torch.zeros_like(flat)
    out[mask] = flat[mask]

    return out.view_as(t)


def _topk_tensor(t, k_frac):
    """Keep only top-k% values by magnitude."""
    if k_frac <= 0.0 or k_frac >= 1.0:
        return t.clone()

    flat = t.view(-1)
    numel = flat.numel()
    k = max(1, int(numel * k_frac))

    _, idx = torch.topk(flat.abs(), k)
    out = torch.zeros_like(flat)
    out[idx] = flat[idx]

    return out.view_as(t)


def apply_compression(update_dict,
                      mode="sparsify",
                      sparsity=0.5,
                      k_frac=0.1):
    """
    Apply compression to a state dict or update dict.
    Returns a new dict with compressed tensors.
    """
    if not update_dict:
        return {}

    compressed = {}

    for name, tensor in update_dict.items():
        if mode == "sparsify":
            compressed[name] = _sparsify_tensor(tensor, sparsity)
        elif mode == "topk":
            compressed[name] = _topk_tensor(tensor, k_frac)
        else:
            # Unknown mode: return original tensor
            compressed[name] = tensor.clone()

    return compressed


def compress_update(update_dict, settings):
    """
    Backwards-compatible wrapper that reads compression config from settings.
    """
    enabled = getattr(settings, "compression_enabled", False)
    if not enabled:
        return update_dict

    mode = getattr(settings, "compression_mode", "sparsify")
    sparsity = getattr(settings, "compression_sparsity", 0.5)
    k_frac = getattr(settings, "compression_k_frac", 0.1)

    return apply_compression(
        update_dict,
        mode=mode,
        sparsity=sparsity,
        k_frac=k_frac,
    )
