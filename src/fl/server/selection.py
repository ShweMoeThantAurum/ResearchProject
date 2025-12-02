"""
Client selection strategies for the server.

This module makes AEFL a distinct algorithm via energy and bandwidth
aware client selection. Baselines select all clients every round.
"""

import os


def select_all_clients(all_roles):
    """
    Select all available roles.

    Used by FedAvg, FedProx, LocalOnly, and also by AEFL for the
    first training round when no metadata is yet available.
    """
    return list(all_roles)


def get_max_clients():
    """
    Return the maximum number of clients that AEFL is allowed to select.

    This is hard-coded to three clients by default. You can override
    it with AEFL_MAX_CLIENTS environment variable if needed.
    """
    default_max = 3
    value = os.environ.get("AEFL_MAX_CLIENTS")
    if value is None:
        return default_max

    try:
        parsed = int(value)
        if parsed <= 0:
            return default_max
        return parsed
    except ValueError:
        return default_max


def select_clients_aefl(metadata_by_role, all_roles):
    """
    Select a subset of clients using AEFL scoring.

    The score combines:
      - normalised bandwidth (higher is better)
      - normalised total energy (lower is better)

    The current scoring function is:
      score = 0.5 * bw_norm + 0.5 * (1 - energy_norm)

    Returns a list of selected role strings.
    """
    if not metadata_by_role:
        # No metadata from previous round, fall back to all clients
        return list(all_roles)

    roles = list(metadata_by_role.keys())

    # Extract raw values
    bw = {}
    en = {}

    for r in roles:
        meta = metadata_by_role.get(r, {})
        bw[r] = float(meta.get("bandwidth_mbps", 0.0))
        en[r] = float(meta.get("total_energy_j", 0.0))

    bw_max = max(bw.values()) if bw else 1.0
    en_max = max(en.values()) if en else 1.0

    if bw_max <= 0:
        bw_max = 1.0
    if en_max <= 0:
        en_max = 1.0

    scores = {}

    for r in roles:
        bw_norm = bw[r] / bw_max
        en_norm = en[r] / en_max
        score = 0.5 * bw_norm + 0.5 * (1.0 - en_norm)
        scores[r] = score

    sorted_roles = sorted(roles, key=lambda r: scores[r], reverse=True)
    max_clients = get_max_clients()
    selected = sorted_roles[:max_clients]

    print("[SERVER] AEFL Scores:", scores)
    print("[SERVER] AEFL selected:", selected)

    return selected
