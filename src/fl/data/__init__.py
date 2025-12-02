"""
Dataset utilities for preprocessing, loading, and partitioning traffic datasets.

Modules:
    - preprocess: cleaning, normalization, sliding windows
    - loader: dataset loaders for training/evaluation
    - partition: create client-specific shards for FL
"""

from .preprocess import preprocess_dataset
from .loader import TrafficDataset, load_dataset
from .partition import partition_non_iid

__all__ = [
    "preprocess_dataset",
    "TrafficDataset",
    "load_dataset",
    "partition_non_iid",
]
