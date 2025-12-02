"""
Client-side utility helpers.

This module handles:
    - Reading runtime settings from environment variables
    - Resolving dataset and processed directory paths
    - Constructing PyTorch DataLoaders from client partitions
    - Basic role and mode helpers
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.fl.utils.logger import log_event
from src.fl.data.loader import (
    get_processed_dir,
    get_num_nodes,
    load_client_partition,
)


def get_client_role():
    """
    Return the client role name from environment variables.

    Defaults to "roadside" if CLIENT_ROLE is not set.
    """
    return os.environ.get("CLIENT_ROLE", "roadside").strip()


def get_fl_mode():
    """
    Return the federated learning mode from environment variables.

    Supported modes:
        AEFL, FedAvg, FedProx, LocalOnly

    The returned string is always upper case.
    """
    mode = os.environ.get("FL_MODE", "AEFL").strip()
    if not mode:
        mode = "AEFL"
    return mode.upper()


def client_allows_training(mode):
    """
    Return True if this mode should perform local training.

    Currently all modes perform training, including LocalOnly,
    which simply does not aggregate globally on the server side.
    """
    return True


def get_dataset_name():
    """
    Return the dataset name from environment variables.

    Defaults to "sz" when DATASET is not set.
    """
    return os.environ.get("DATASET", "sz").strip().lower()


def get_fl_rounds():
    """
    Return the number of federated learning rounds.

    Reads FL_ROUNDS from environment variables, defaults to 20.
    """
    return int(os.environ.get("FL_ROUNDS", "20"))


def get_batch_size():
    """
    Return the local mini-batch size.

    Reads BATCH_SIZE from environment variables, defaults to 64.
    """
    return int(os.environ.get("BATCH_SIZE", "64"))


def get_local_epochs():
    """
    Return the number of local epochs per round.

    Reads LOCAL_EPOCHS from environment variables, defaults to 1.
    """
    return int(os.environ.get("LOCAL_EPOCHS", "1"))


def get_lr():
    """
    Return the client learning rate.

    Reads LR from environment variables, defaults to 0.001.
    """
    return float(os.environ.get("LR", "0.001"))


def get_hidden_size():
    """
    Return the GRU hidden size.

    Reads HIDDEN_SIZE from environment variables, defaults to 64.
    """
    return int(os.environ.get("HIDDEN_SIZE", "64"))


def get_energy_params():
    """
    Return client-side energy model parameters.

    Reads:
        DEVICE_POWER_WATTS   : nominal device power
        NET_J_PER_MB         : communication energy per megabyte
        FLOP_ENERGY_J        : energy per floating point operation

    FLOP_ENERGY_J defaults to a small illustrative constant.
    """
    device_power = float(os.environ.get("DEVICE_POWER_WATTS", "3.5"))
    net_j_per_mb = float(os.environ.get("NET_J_PER_MB", "0.6"))
    flop_energy = float(os.environ.get("FLOP_ENERGY_J", "1e-12"))
    return device_power, net_j_per_mb, flop_energy


def pad_or_trim_last_dim(arr, target_dim):
    """
    Ensure the last dimension of an array matches target_dim.

    If the array has more nodes than target_dim, extra nodes are trimmed.
    If the array has fewer nodes than target_dim, zeros are padded.
    """
    shape = list(arr.shape)
    current = shape[-1]

    if current == target_dim:
        return arr

    if current > target_dim:
        return arr[..., :target_dim]

    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, target_dim - current)
    return np.pad(arr, pad_width, mode="constant")


def build_client_dataloader(dataset_name,
                            role,
                            batch_size,
                            num_nodes):
    """
    Load a client partition and construct a PyTorch DataLoader.

    Parameters:
        dataset_name : name of the dataset (sz, los, pems08)
        role         : client role (roadside, vehicle, sensor, camera, bus)
        batch_size   : mini-batch size
        num_nodes    : global graph node count

    Returns:
        loader: PyTorch DataLoader over this client's local samples
    """
    proc_dir = get_processed_dir(dataset_name)
    X, y = load_client_partition(proc_dir, role)

    X = pad_or_trim_last_dim(X, num_nodes)
    y = pad_or_trim_last_dim(y, num_nodes)

    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    log_event("client_data.log", {
        "role": role,
        "dataset": dataset_name,
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "batch_size": batch_size,
    })

    print("[%s] Loaded local data: X=%s, y=%s, batch=%d" %
          (role, X.shape, y.shape, batch_size))

    return loader


def infer_num_nodes_for_dataset(dataset_name):
    """
    Infer the number of nodes for the given dataset using X_train.

    Uses the last dimension of X_train.npy stored under the processed
    directory for the dataset.
    """
    proc_dir = get_processed_dir(dataset_name)
    return get_num_nodes(proc_dir)
