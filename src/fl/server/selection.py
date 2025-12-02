"""
Client selection strategies used by the FL server.

Includes:
- select_all_clients: used for FedAvg, FedProx, LocalOnly, and AEFL rounds
- select_clients_aefl: AEFL's adaptive client selection using bandwidth and energy metadata
"""

import os


def select_all_clients(all_roles):
    """
    Return all clients. Used for:
    - FedAvg
    - FedProx
    - LocalOnly
    - AEFL (round 1 only)
    """
    return list(all_roles)


def select_clients_aefl(metadata, all_roles):
    """
    Select clients based on AEFL scoring:
        score = 0.6 * (bandwidth normalized)
               + 0.4 * (1 - normalized energy)

    metadata: dict from S3_IO.load_round_metadata
    all_roles: list of all possible clients
    """

    if not metadata:
        return list(all_roles)

    roles = list(metadata.keys())
    bw = {r: metadata[r].get("bandwidth_mbps", 0.0) for r in roles}
    en = {r: metadata[r].get("total_energy_j", 0.0) for r in roles}

    bw_max = max(bw.values()) or 1
    en_max = max(en.values()) or 1

    scores = {}
    for r in roles:
        bw_score = bw[r] / bw_max
        en_score = 1 - (en[r] / en_max)
        scores[r] = 0.6 * bw_score + 0.4 * en_score

    sorted_roles = sorted(roles, key=lambda r: scores[r], reverse=True)

    max_clients = int(os.environ.get("AEFL_MAX_CLIENTS", len(sorted_roles)))
    selected = sorted_roles[:max_clients]

    print("[SERVER] AEFL Scores:", scores)
    print("[SERVER] AEFL selected:", selected)

    return selected
