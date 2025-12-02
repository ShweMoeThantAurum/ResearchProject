"""
Model compression utilities for client uploads.
Supports magnitude sparsification, Top-k and 8-bit quantisation.
"""

import os
import math
import torch


def _compression_enabled():
    """Return True if compression is enabled."""
    return os.environ.get("COMPRESSION_ENABLED", "false").strip().lower() == "true"


def _compression_mode():
    """Return compression mode name."""
    return os.environ.get("COMPRESSION_MODE", "sparsify").strip().lower()


def _compression_sparsity():
    """Return sparsity fraction for magnitude pruning."""
    return float(os.environ.get("COMPRESSION_SPARSITY", "0.5"))


def _compression_k_frac():
    """Return fraction of weights kept in Top-k."""
    return float(os.environ.get("COMPRESSION_K_FRAC", "0.1"))


def _dense_state_size_bytes(state_dict):
    """Estimate dense payload size in bytes."""
    total = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.element_size() * v.numel()
    return int(total)


def _sparsify_state(state_dict, sparsity):
    """Zero out smallest-magnitude weights to reach target sparsity."""
    floats = [v.flatten().abs() for v in state_dict.values()
              if isinstance(v, torch.Tensor) and v.is_floating_point()]
    if not floats or sparsity <= 0.0:
        return state_dict, 1.0, _dense_state_size_bytes(state_dict)

    all_vals = torch.cat(floats)
    k = int(len(all_vals) * sparsity)
    if k <= 0:
        return state_dict, 1.0, _dense_state_size_bytes(state_dict)

    threshold = torch.topk(all_vals, k, largest=False).values.max().item()

    compressed = {}
    kept = 0
    total = 0

    for name, v in state_dict.items():
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            compressed[name] = v
            continue

        mask = v.abs() >= threshold
        compressed[name] = v * mask
        kept += int(mask.sum().item())
        total += int(mask.numel())

    kept_ratio = kept / float(max(1, total))
    payload_bytes = _dense_state_size_bytes(compressed)
    return compressed, kept_ratio, payload_bytes


def _topk_state(state_dict, k_frac):
    """Apply per-tensor Top-k sparsification."""
    if k_frac <= 0.0 or k_frac > 1.0:
        return state_dict, 1.0, _dense_state_size_bytes(state_dict)

    decompressed = {}
    total = 0
    kept = 0
    payload_bytes = 0

    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            decompressed[name] = t
            payload_bytes += t.element_size() * t.numel()
            continue

        flat = t.flatten()
        numel = flat.numel()
        k = max(1, int(math.ceil(k_frac * numel)))

        vals, idxs = torch.topk(flat.abs(), k, largest=True, sorted=False)
        signs = torch.sign(flat[idxs])
        top_vals = vals * signs

        rec = torch.zeros_like(flat)
        rec[idxs] = top_vals
        decompressed[name] = rec.view_as(t)

        kept += k
        total += numel

        # Indices (int32) + values (float32)
        payload_bytes += 4 * k + 4 * k

    kept_ratio = kept / float(max(1, total))
    return decompressed, kept_ratio, payload_bytes


def _quantize8_state(state_dict):
    """Apply per-tensor symmetric int8 quantisation."""
    dequant = {}
    payload_bytes = 0

    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            dequant[name] = t
            payload_bytes += t.element_size() * t.numel()
            continue

        max_abs = t.abs().max()
        if max_abs == 0:
            q = torch.zeros_like(t, dtype=torch.int8)
            scale = torch.tensor(1.0, dtype=torch.float32, device=t.device)
        else:
            scale = (max_abs / 127.0).to(torch.float32)
            q = torch.clamp((t / scale).round(), -127, 127).to(torch.int8)

        dequant[name] = (q.to(torch.float32) * scale).to(t.dtype)
        payload_bytes += q.numel() * 1 + 4

    kept_ratio = 1.0
    return dequant, kept_ratio, payload_bytes


def apply_compression_if_enabled(state_dict, role, round_id):
    """
    Optionally compress the update before upload.
    Returns compressed state, kept ratio and modeled payload size.
    """
    if not _compression_enabled():
        dense_bytes = _dense_state_size_bytes(state_dict)
        return state_dict, 1.0, dense_bytes

    mode = _compression_mode()

    if mode in ("sparsify", "magnitude", "prune"):
        compressed, kept_ratio, payload_bytes = _sparsify_state(
            state_dict, _compression_sparsity()
        )
        print(
            f"[{role}] Compression r={round_id}: sparsify "
            f"kept={kept_ratio:.3f}, modeled={payload_bytes / 1e6:.3f} MB"
        )
        return compressed, kept_ratio, payload_bytes

    if mode == "topk":
        compressed, kept_ratio, payload_bytes = _topk_state(
            state_dict, _compression_k_frac()
        )
        print(
            f"[{role}] Compression r={round_id}: topk "
            f"kept={kept_ratio:.3f}, modeled={payload_bytes / 1e6:.3f} MB"
        )
        return compressed, kept_ratio, payload_bytes

    if mode in ("q8", "int8", "quant8"):
        compressed, kept_ratio, payload_bytes = _quantize8_state(state_dict)
        print(
            f"[{role}] Compression r={round_id}: int8 "
            f"kept={kept_ratio:.3f}, modeled={payload_bytes / 1e6:.3f} MB"
        )
        return compressed, kept_ratio, payload_bytes

    dense_bytes = _dense_state_size_bytes(state_dict)
    print(f"[{role}] Compression disabled or unknown mode='{mode}', using dense.")
    return state_dict, 1.0, dense_bytes
