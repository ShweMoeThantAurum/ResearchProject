"""
Client partitioning utilities for federated learning.

This module supports:
    - Splitting graph nodes into client subsets (IID or non-IID)
    - Building per-client datasets from global X_train/y_train
    - Mapping numeric client indices to semantic IoT roles
"""

import os
import json
import numpy as np


def split_clients_by_nodes(num_nodes,
                           num_clients,
                           noniid=False,
                           imbalance_factor=0.4,
                           seed=42):
    """
    Split node indices among clients.

    Parameters:
        num_nodes        : total number of graph nodes
        num_clients      : number of clients to create
        noniid           : if True, use Dirichlet-based unbalanced split
        imbalance_factor : not directly used here but kept for clarity
        seed             : random seed for reproducibility

    Returns:
        A list of numpy arrays, one per client, containing node indices.
    """
    rng = np.random.default_rng(seed)
    all_nodes = np.arange(num_nodes)

    if not noniid:
        return np.array_split(all_nodes, num_clients)

    # Non-IID using Dirichlet proportions
    shares = rng.dirichlet(np.ones(num_clients))
    sizes = np.maximum(1, (shares * num_nodes).astype(int))

    diff = num_nodes - sizes.sum()
    sizes[-1] += diff

    rng.shuffle(all_nodes)

    splits = []
    start = 0
    for s in sizes:
        splits.append(all_nodes[start:start + s])
        start += s

    return splits


def build_client_datasets(proc_dir,
                          num_clients=5,
                          noniid=False,
                          imbalance_factor=0.4,
                          seed=42,
                          role_map=None):
    """
    Create per-client training datasets from global X_train/y_train arrays.

    This function:
        - Loads X_train, y_train from proc_dir
        - Splits node dimension among clients
        - Saves per-client arrays into:
              <proc_dir>/clients/client_<role>_X.npy
              <proc_dir>/clients/client_<role>_y.npy
        - Writes a meta.json summary with partition details

    Parameters:
        proc_dir        : processed dataset directory
        num_clients     : number of clients to create
        noniid          : whether to simulate non-IID partitioning
        imbalance_factor: kept for documentation (future extension)
        seed            : random seed
        role_map        : optional mapping {client_index: "role_name"}
                          defaults to roadside, vehicle, sensor, camera, bus
    """
    print("Building client datasets | proc_dir=%s | noniid=%s | clients=%d" %
          (proc_dir, noniid, num_clients))

    x_path = os.path.join(proc_dir, "X_train.npy")
    y_path = os.path.join(proc_dir, "y_train.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Expected X_train.npy and y_train.npy in %s" %
                                proc_dir)

    X_train = np.load(x_path)
    y_train = np.load(y_path)

    num_nodes = X_train.shape[-1]

    splits = split_clients_by_nodes(
        num_nodes=num_nodes,
        num_clients=num_clients,
        noniid=noniid,
        imbalance_factor=imbalance_factor,
        seed=seed,
    )

    clients_dir = os.path.join(proc_dir, "clients")
    os.makedirs(clients_dir, exist_ok=True)

    if role_map is None:
        role_map = {
            0: "roadside",
            1: "vehicle",
            2: "sensor",
            3: "camera",
            4: "bus",
        }

    client_sizes = []

    for idx, node_idxs in enumerate(splits):
        role = role_map.get(idx, "client%d" % idx)

        X_local = X_train[:, :, node_idxs]
        y_local = y_train[:, node_idxs]

        x_out = os.path.join(clients_dir, "client_%s_X.npy" % role)
        y_out = os.path.join(clients_dir, "client_%s_y.npy" % role)

        np.save(x_out, X_local)
        np.save(y_out, y_local)

        client_sizes.append(len(node_idxs))

        print("  Client %d (%s): nodes=%d" %
              (idx, role, len(node_idxs)))

    meta = {
        "num_clients": num_clients,
        "num_nodes": num_nodes,
        "noniid": noniid,
        "imbalance_factor": imbalance_factor,
        "seed": seed,
        "clients": client_sizes,
        "role_map": role_map,
    }

    meta_path = os.path.join(proc_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved client datasets in %s" % clients_dir)
    print("Node allocation per client:", client_sizes)
