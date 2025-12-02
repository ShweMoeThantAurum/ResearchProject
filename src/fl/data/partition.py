"""
Node-level partitioning into IoT roles for federated learning.

This module:
    - Partitions graph nodes (sensors) into logical IoT roles
    - Supports both IID and Dirichlet Non-IID splits
    - Saves per-role X/y arrays for client training
"""

import os
import json
import numpy as np


DEFAULT_ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def partition_nodes_to_roles(num_nodes,
                             roles=None,
                             noniid=False,
                             imbalance_factor=0.4,
                             seed=42):
    """
    Partition node indices among roles.

    If noniid=False:
        - Split nodes as evenly as possible across roles.

    If noniid=True:
        - Sample node shares from a Dirichlet distribution and
          assign contiguous chunks of shuffled nodes.

    Returns:
        dict mapping role -> numpy array of node indices
    """
    if roles is None:
        roles = list(DEFAULT_ROLES)

    rng = np.random.default_rng(seed)
    all_nodes = np.arange(num_nodes)
    rng.shuffle(all_nodes)

    num_roles = len(roles)

    if not noniid:
        splits = np.array_split(all_nodes, num_roles)
        return {role: splits[i] for i, role in enumerate(roles)}

    # Non-IID: Dirichlet-based shares with a simple imbalance factor
    shares = rng.dirichlet(np.ones(num_roles) * imbalance_factor)
    sizes = np.maximum(1, (shares * num_nodes).astype(int))

    # Fix rounding errors
    diff = num_nodes - sizes.sum()
    sizes[-1] += diff

    role_to_nodes = {}
    start = 0
    for i, role in enumerate(roles):
        size = sizes[i]
        role_to_nodes[role] = all_nodes[start:start + size]
        start += size

    return role_to_nodes


def save_role_partitions(X_train,
                         y_train,
                         processed_dir,
                         roles=None,
                         noniid=False,
                         imbalance_factor=0.4,
                         seed=42):
    """
    Save per-role X/y arrays for each IoT role.

    Assumptions:
        - X_train: [samples, seq_len, num_nodes]
        - y_train: [samples, num_nodes]
        - Nodes are along the last dimension.

    Files written into processed_dir:
        - X_<role>.npy
        - y_<role>.npy
        - role_partitions.json (meta info)
    """
    if roles is None:
        roles = list(DEFAULT_ROLES)

    num_nodes = X_train.shape[-1]
    role_to_nodes = partition_nodes_to_roles(
        num_nodes,
        roles=roles,
        noniid=noniid,
        imbalance_factor=imbalance_factor,
        seed=seed,
    )

    os.makedirs(processed_dir, exist_ok=True)

    meta = {
        "num_nodes": int(num_nodes),
        "roles": roles,
        "partitions": {},
        "noniid": bool(noniid),
        "imbalance_factor": float(imbalance_factor),
        "seed": int(seed),
    }

    for role in roles:
        idxs = role_to_nodes[role]
        X_role = X_train[:, :, idxs]
        y_role = y_train[:, idxs]

        np.save(os.path.join(processed_dir, "X_%s.npy" % role), X_role)
        np.save(os.path.join(processed_dir, "y_%s.npy" % role), y_role)

        meta["partitions"][role] = {
            "num_nodes": int(len(idxs)),
            "indices": idxs.tolist(),
        }

    with open(os.path.join(processed_dir, "role_partitions.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved role partitions in %s" % processed_dir)
    for role in roles:
        print("  role=%s nodes=%d" %
              (role, meta["partitions"][role]["num_nodes"]))

    return role_to_nodes
