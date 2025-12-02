"""
Data subpackage for federated learning.

Contains:
    - loader      : load global arrays and client partitions
    - partition   : create client splits from global data
    - preprocess  : dataset-specific preprocessing pipelines
"""

from .loader import (
    get_processed_dir,
    load_global_arrays,
    get_num_nodes,
    load_client_partition,
)

from .partition import (
    split_clients_by_nodes,
    build_client_datasets,
)

from .preprocess import (
    preprocess_dataset,
    preprocess_pems08,
    preprocess_los_loop,
    preprocess_sz_taxi,
)

__all__ = [
    "get_processed_dir",
    "load_global_arrays",
    "get_num_nodes",
    "load_client_partition",
    "split_clients_by_nodes",
    "build_client_datasets",
    "preprocess_dataset",
    "preprocess_pems08",
    "preprocess_los_loop",
    "preprocess_sz_taxi",
]
