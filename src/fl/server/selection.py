"""Client selection logic for AEFL and baseline strategies."""

import os


def select_all_clients(all_roles: list):
    """
    Select all available clients.

    Used for FedAvg, FedProx, LocalOnly, and AEFL round 1.
    """
    return list(all_roles)


def select_clients_aefl(metadata: dict, all_roles: list):
    """
    Select clients adaptively using AEFL scoring.

    Score = 0.6 * (bandwidth / max_bw) + 0.4 * (1 - energy / max_energy)

    If metadata is empty (round 1), select all clients.
    A max client cap can be applied via AEFL_MAX_CLIENTS env var.
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
