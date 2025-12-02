"""
Client-side model compression utilities.

This module wraps the compression primitives in src.fl.utils.compression
and applies them conditionally based on environment variables.

Supported modes:
    - sparsify  : magnitude pruning to a target sparsity
    - topk      : top-k sparsification
    - q8 / int8 : symmetric 8-bit quantisation
"""

import os

from src.fl.utils.compression import (
    sparsify_state,
    topk_compress_state,
    quantize8_state,
    dense_state_size_bytes,
)


def compression_is_enabled():
    """
    Return True if model compression is enabled.

    Reads COMPRESSION_ENABLED from environment variables.
    """
    flag = os.environ.get("COMPRESSION_ENABLED", "false").strip().lower()
    return flag in ["1", "true", "yes", "on"]


def get_compression_settings():
    """
    Return compression settings from environment variables.

    Returns:
        mode          : compression mode string
        sparsity      : target sparsity for magnitude pruning
        k_frac        : fraction of parameters kept for top-k
    """
    mode = os.environ.get("COMPRESSION_MODE", "sparsify").strip().lower()
    sparsity = float(os.environ.get("COMPRESSION_SPARSITY", "0.5"))
    k_frac = float(os.environ.get("COMPRESSION_K_FRAC", "0.1"))
    return mode, sparsity, k_frac


def maybe_compress_state(state_dict):
    """
    Optionally compress a model state_dict before upload.

    Returns:
        compressed_state : potentially modified state_dict
        kept_ratio       : fraction of kept parameters
        payload_bytes    : estimated number of bytes on the wire
    """
    if not compression_is_enabled():
        payload_bytes = dense_state_size_bytes(state_dict)
        return state_dict, 1.0, payload_bytes

    mode, sparsity, k_frac = get_compression_settings()

    if mode in ["sparsify", "magnitude", "prune"]:
        return sparsify_state(state_dict, sparsity)

    if mode == "topk":
        return topk_compress_state(state_dict, k_frac)

    if mode in ["q8", "int8"]:
        return quantize8_state(state_dict)

    payload_bytes = dense_state_size_bytes(state_dict)
    return state_dict, 1.0, payload_bytes
