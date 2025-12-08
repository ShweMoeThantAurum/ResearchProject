"""
Model compression utilities for communication-efficient FL.

Implements:
- magnitude-based sparsification
- top-k selection
- 8-bit per-tensor symmetric quantisation

Also provides helper functions for estimating payload size in bytes.
"""

import math
import torch


# -------------------------------
# Magnitude pruning
# -------------------------------


def sparsify_state(state_dict, sparsity=0.5):
    """
    Zero smallest-magnitude weights to reach the given sparsity level.

    Returns:
      compressed_state : same shapes, more zeros
      kept_ratio       : fraction of non-zero parameters
      payload_bytes    : estimated dense payload size (no index saving)
    """
    all_vals = torch.cat(
        [v.flatten().abs() for v in state_dict.values() if v.is_floating_point()]
    )
    k = int(len(all_vals) * sparsity)
    if k <= 0:
        return state_dict, 1.0, dense_state_size_bytes(state_dict)

    # Determine magnitude threshold for pruning
    threshold = torch.topk(all_vals, k, largest=False).values.max().item()
    compressed = {}
    kept_total = 0
    total = 0

    for name, v in state_dict.items():
        if not v.is_floating_point():
            compressed[name] = v
            continue

        mask = v.abs() >= threshold
        compressed[name] = v * mask
        kept_total += mask.sum().item()
        total += mask.numel()

    kept_ratio = kept_total / max(1, total)

    # Dense payload: tensor shapes unchanged, but many entries zeroed
    return compressed, kept_ratio, dense_state_size_bytes(compressed)


# -------------------------------
# Size utilities
# -------------------------------


def dense_state_size_bytes(state_dict):
    """Estimate dense float32 payload size in bytes for a state_dict."""
    total = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.element_size() * v.numel()
    return int(total)


def _int_index_bytes(numel, ndim):
    """
    Estimate bytes required to represent indices for sparse tensors.

    Uses int32 indices for each element times the effective dimensionality.
    """
    # For top-k we send 1D flat indices (int32), so ndim=1 effectively.
    return 4 * numel * max(1, ndim)


# -------------------------------
# Top-k sparsification
# -------------------------------


def topk_compress_state(state_dict, k_frac=0.1):
    """
    Apply per-tensor top-k sparsification by magnitude.

    Returns:
      decomp_state : dense tensors with zeros for dropped entries
      kept_ratio   : fraction of kept parameters
      payload_bytes: estimated compressed wire size (values + indices)
    """
    assert 0.0 < k_frac <= 1.0, "k_frac must be in (0, 1]"
    decomp = {}
    total = 0
    kept = 0
    payload_bytes = 0

    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            # Non-float tensors are sent densely
            decomp[name] = t
            payload_bytes += t.element_size() * t.numel()
            continue

        flat = t.flatten()
        numel = flat.numel()
        k = max(1, int(math.ceil(k_frac * numel)))

        # Top-k by magnitude
        vals, idxs = torch.topk(flat.abs(), k, largest=True, sorted=False)

        # Recover signed values
        signs = torch.sign(flat[idxs])
        top_vals = vals * signs

        # Reconstruct dense tensor for aggregation
        rec = torch.zeros_like(flat)
        rec[idxs] = top_vals
        decomp[name] = rec.view_as(t)

        kept += k
        total += numel

        # Payload = indices (int32) + values (float32)
        payload_bytes += _int_index_bytes(k, ndim=1) + 4 * k  # 4 bytes/float32

    kept_ratio = kept / max(1, total)
    return decomp, kept_ratio, int(payload_bytes)


# -------------------------------
# 8-bit per-tensor symmetric quantisation
# -------------------------------


def quantize8_state(state_dict):
    """
    Apply per-tensor symmetric int8 quantisation.

    Uses max-abs scaling with zero-point 0 and returns dequantised dense
    tensors for aggregation plus an estimate of compressed wire size.
    """
    dequant = {}
    payload_bytes = 0

    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            # Non-float tensors are sent densely
            dequant[name] = t
            payload_bytes += t.element_size() * t.numel()
            continue

        max_abs = t.abs().max()
        if max_abs == 0:
            # All zeros: keep zeros and a dummy scale
            q = torch.zeros_like(t, dtype=torch.int8)
            scale = torch.tensor(1.0, dtype=torch.float32, device=t.device)
        else:
            scale = (max_abs / 127.0).to(torch.float32)
            q = torch.clamp((t / scale).round(), -127, 127).to(torch.int8)

        # Dequantize for server-side aggregation
        dequant[name] = (q.to(torch.float32) * scale).to(t.dtype)

        # Payload size: int8 weights + 4-byte scale per tensor
        payload_bytes += q.numel() * 1 + 4

    kept_ratio = 1.0
    return dequant, kept_ratio, int(payload_bytes)
