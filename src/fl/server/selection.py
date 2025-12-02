"""
Client selection strategies.

AEFL:
    - Score clients based on bandwidth + energy
    - Select top-K clients
"""

import os


def select_all_clients(roles):
    """Return all roles (FedAvg, FedProx, LocalOnly)."""
    return list(roles)


def select_clients_aefl(metadata, roles):
    """
    Compute AEFL scores from metadata and pick best K clients.

    Scores:
        0.6 * normalized_bandwidth
      + 0.4 * (1 - normalized_energy)
    """
    if not metadata:
        return list(roles)

    bw = {}
    en = {}

    for role in roles:
        if role in metadata:
            bw[role] = metadata[role].get("bandwidth_mbps", 0.0)
            en[role] = metadata[role].get("total_energy_j", 0.0)
        else:
            bw[role] = 0.0
            en[role] = 1.0

    bw_max = max(bw.values()) or 1
    en_max = max(en.values()) or 1

    scores = {}
    for r in roles:
        bw_norm = bw[r] / bw_max
        en_norm = 1 - (en[r] / en_max)
        scores[r] = 0.6 * bw_norm + 0.4 * en_norm

    sorted_roles = sorted(roles, key=lambda x: scores[x], reverse=True)

    # Hardcoded top-3 (your preferred option)
    max_clients = 3

    selected = sorted_roles[:max_clients]

    print("[SERVER] AEFL scores:", scores)
    print("[SERVER] AEFL selected:", selected)

    return selected
