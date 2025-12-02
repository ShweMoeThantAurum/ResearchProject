"""
Client selection strategies for server-side FL rounds.

AEFL selects clients adaptively based on bandwidth and energy metadata.
"""

# Hardcoded AEFL max clients (Option 1)
AEFL_MAX_CLIENTS = 3


def select_all_clients(all_roles):
    """Return all clients for FedAvg, FedProx, LocalOnly and AEFL round 1."""
    return list(all_roles)


def select_clients_aefl(metadata, all_roles):
    """
    Select clients using AEFL scoring based on bandwidth and energy.

    Returns:
        selected_roles (list)
        scores (dict): role -> normalised AEFL score
    """
    if not metadata:
        # No metadata yet → select all
        roles = list(all_roles)
        scores = {r: 1.0 for r in roles}
        return roles, scores

    roles = list(metadata.keys())

    # Extract bandwidth & energy
    bw = {r: metadata[r].get("bandwidth_mbps", 0.0) for r in roles}
    en = {r: metadata[r].get("total_energy_j", 0.0) for r in roles}

    bw_max = max(bw.values()) or 1
    en_max = max(en.values()) or 1

    # Score = 0.6 * bw_norm + 0.4 * (1 - energy_norm)
    scores = {}
    for r in roles:
        bw_score = bw[r] / bw_max
        en_score = 1 - (en[r] / en_max)
        scores[r] = 0.6 * bw_score + 0.4 * en_score

    # Sort by score
    sorted_roles = sorted(roles, key=lambda r: scores[r], reverse=True)

    # Hardcoded AEFL max clients (option 1)
    max_clients = AEFL_MAX_CLIENTS

    selected = sorted_roles[:max_clients]

    print("[SERVER] AEFL Scores:", scores)
    print("[SERVER] AEFL selected:", selected)

    return selected, scores
