"""
Client selection logic.
AEFL selects clients using bandwidth + energy metadata.
Other modes simply select all clients.
"""

from src.fl.server.utils_server import ROLES


def select_clients(metadata, mode):
    """Return list of roles to include this round."""
    if mode != "aefl":
        return list(ROLES)

    if not metadata:
        return list(ROLES)

    bw = {}
    en = {}

    for role in ROLES:
        if role in metadata:
            bw[role] = metadata[role].get("bandwidth_mbps", 0.0)
            en[role] = metadata[role].get("total_energy_j", 0.0)
        else:
            bw[role] = 0.0
            en[role] = 1e9

    # Normalise
    bw_max = max(bw.values())
    en_max = max(en.values())

    scores = {}
    for r in ROLES:
        bw_score = bw[r] / (bw_max + 1e-9)
        en_score = 1.0 - (en[r] / (en_max + 1e-9))
        scores[r] = 0.6 * bw_score + 0.4 * en_score

    sorted_roles = sorted(ROLES, key=lambda x: scores[x], reverse=True)
    return sorted_roles[:3]
