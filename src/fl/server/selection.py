"""
Client selection logic for server-side FL.
Implements AEFL energy-aware selection based on client metadata.
"""

from .utils_server import ROLES, get_aefl_clients_per_round


def select_all_clients():
    """Select all available roles."""
    return list(ROLES)


def _normalise(values):
    """Normalise a dict of numeric values into [0, 1]."""
    if not values:
        return {k: 0.0 for k in values}
    vmax = max(values.values())
    if vmax <= 0:
        return {k: 0.0 for k in values}
    return {k: v / float(vmax) for k, v in values.items()}


def select_clients_aefl(metadata):
    """Select AEFL clients using bandwidth and energy scores."""
    if not metadata:
        return select_all_clients()

    roles = list(metadata.keys())

    bw = {r: metadata[r].get("bandwidth_mbps", 0.0) for r in roles}
    en = {r: metadata[r].get("total_energy_j", 0.0) for r in roles}

    bw_norm = _normalise(bw)
    en_norm = _normalise(en)

    # Higher is better for bandwidth; lower is better for energy
    scores = {}
    for r in roles:
        bw_score = bw_norm.get(r, 0.0)
        en_score = 1.0 - en_norm.get(r, 0.0)
        scores[r] = 0.6 * bw_score + 0.4 * en_score

    sorted_roles = sorted(roles, key=lambda k: scores[k], reverse=True)
    k = min(get_aefl_clients_per_round(), len(sorted_roles))
    selected = sorted_roles[:k]

    print("[SERVER] AEFL scores:", scores)
    print("[SERVER] AEFL selected clients:", selected)

    return selected
