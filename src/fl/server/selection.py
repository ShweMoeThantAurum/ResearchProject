"""
Client selection logic for AEFL and baseline modes.
"""

import os


def select_all_clients(roles):
    """
    Select all clients (baseline modes).
    """
    return list(roles)


def select_clients_aefl(metadata, roles):
    """
    Select clients using adaptive AEFL scoring.
    """
    if not metadata:
        return list(roles)

    scores = {}

    bw = {r: metadata[r].get("bandwidth_mbps", 0.0) for r in roles}
    en = {r: metadata[r].get("total_energy_j", 0.0) for r in roles}

    bw_max = max(bw.values()) or 1
    en_max = max(en.values()) or 1

    # AEFL score = weighted bandwidth + inverse energy
    for r in roles:
        bw_score = bw[r] / bw_max
        en_score = 1 - (en[r] / en_max)
        scores[r] = 0.6 * bw_score + 0.4 * en_score

    sorted_roles = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return sorted_roles[:3]
