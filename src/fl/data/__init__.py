"""
Data utilities for AEFL experiments.

Includes:
    - Preprocessing pipelines for SZ-Taxi, Los-Loop, PeMSD8
    - Node partitioning into IoT roles
    - Convenience loaders for train/valid/test splits
"""

from .preprocess import (
    preprocess_sz,
    preprocess_los,
    preprocess_pems08,
)

from .loader import (
    load_splits,
    get_processed_dir,
)

from .partition import (
    partition_nodes_to_roles,
    save_role_partitions,
)

__all__ = [
    "preprocess_sz",
    "preprocess_los",
    "preprocess_pems08",
    "load_splits",
    "get_processed_dir",
    "partition_nodes_to_roles",
    "save_role_partitions",
]
