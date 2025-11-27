"""Optional lightweight local model compression for communication cost modelling."""

import os
import torch
from typing import Dict, Tuple
from src.utils.compression import (
    sparsify_state,
    topk_compress_state,
    quantize8_state,
    dense_state_size_bytes,
)


def maybe_compress(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict, float, int]:
    """
    Optionally apply compression to the model update before upload.

    Compression modes (controlled by env vars):
      - sparsify : zero out a fraction of weights
      - topk : keep top-k magnitude parameters
      - q8 : 8-bit quantisation
      - disabled : return dense weights unchanged

    Returns:
        compressed_state : dict
        kept_ratio : fraction of weights kept
        payload_bytes : estimated payload size for energy computation
    """
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
    