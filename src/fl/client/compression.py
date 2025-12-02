"""
Local compression utilities applied before uploading updates.

Wraps sparsification, top-k selection, and 8-bit quantisation provided
by src.utils.compression.
"""

import os
import torch
from typing import Dict, Tuple
from src.utils.compression import (
    sparsify_state,
    topk_compress_state,
    quantize8_state,
    dense_state_size_bytes,
)


def maybe_compress(state_dict):
    """Optionally compress the state_dict according to env settings."""
    enabled = os.environ.get("COMPRESSION_ENABLED", "false").lower() == "true"
    if not enabled:
        return state_dict, 1.0, dense_state_size_bytes(state_dict)

    mode = os.environ.get("COMPRESSION_MODE", "sparsify").lower()
    sparsity = float(os.environ.get("COMPRESSION_SPARSITY", "0.5"))
    k_frac = float(os.environ.get("COMPRESSION_K_FRAC", "0.1"))

    if mode in ("sparsify", "magnitude", "prune"):
        return sparsify_state(state_dict, sparsity)

    if mode == "topk":
        return topk_compress_state(state_dict, k_frac)

    if mode in ("q8", "int8"):
        return quantize8_state(state_dict)

    return state_dict, 1.0, dense_state_size_bytes(state_dict)
