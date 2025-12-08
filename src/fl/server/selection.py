"""
Client selection strategies for server-side FL rounds.

 - FedAvg / FedProx:
      all clients participate in each round.
 - AEFL:
      clients are selected adaptively based on bandwidth and energy
      metadata from the previous round.
"""

from src.fl.logger import log_event

# Maximum number of clients AEFL will select per round
AEFL_MAX_CLIENTS = 3


def select_all_clients(all_roles):
    """
    Return all clients for FedAvg, FedProx and AEFL round 1.

    Args:
        all_roles (list[str]): list of all available client roles.
    """
    return list(all_roles)


def select_clients_aefl(metadata, all_roles, round_id=None):
    """
    Select clients using AEFL scoring based on bandwidth and energy.

    Args:
        metadata (dict):
            role -> metadata dict from the previous round.
            Expected keys per role include:
              - 'bandwidth_mbps'
              - 'total_energy_j'
        all_roles (list[str]):
            Complete list of client roles known to the server.
        round_id (int or None):
            Current round index (used only for logging).

    Returns:
        selected_roles (list[str]):
            List of roles chosen for this round.
        scores (dict):
            role -> normalised AEFL score used for weighting.
    """
    if not metadata:
        # No metadata yet → select all clients (typically round 1)
        roles = list(all_roles)
        scores = {r: 1.0 for r in roles}
        return roles, scores

    roles = list(metadata.keys())

    # Extract bandwidth and energy for each client
    bw = {r: float(metadata[r].get("bandwidth_mbps", 0.0)) for r in roles}
    en = {r: float(metadata[r].get("total_energy_j", 0.0)) for r in roles}

    bw_max = max(bw.values()) or 1.0
    en_max = max(en.values()) or 1.0

    # ------------------------------------------------------------------
    # Score = 0.6 * bw_norm + 0.4 * (1 - energy_norm)
    # Higher bandwidth and LOWER total energy are preferred.
    # ------------------------------------------------------------------
    scores = {}
    for r in roles:
        bw_score = bw[r] / bw_max
        en_score = 1.0 - (en[r] / en_max)
        scores[r] = 0.6 * bw_score + 0.4 * en_score

    # ------------------------------------------------------------------
    # Adaptive skipping:
    #   - Mark clients with energy > 80% of max as "high energy".
    #   - Prefer selecting from the remaining pool.
    #   - If all are high-energy, fall back to using everyone.
    # ------------------------------------------------------------------
    high_energy_threshold = 0.8 * en_max
    energy_filtered = [r for r in roles if en[r] <= high_energy_threshold]

    if energy_filtered:
        candidate_roles = energy_filtered
    else:
        # Edge case: all are high-energy → cannot skip everyone
        candidate_roles = roles

    # Sort candidates by AEFL score (descending)
    sorted_roles = sorted(candidate_roles, key=lambda r: scores[r], reverse=True)

    # Hardcoded AEFL max clients
    max_clients = AEFL_MAX_CLIENTS
    selected = sorted_roles[:max_clients]

    print("[SERVER] AEFL Scores:", scores)
    print("[SERVER] AEFL energy-aware candidates:", candidate_roles)
    print("[SERVER] AEFL selected:", selected)

    # Log selection for later analysis (adaptivity plots)
    log_event(
        "server_selection.log",
        {
            "round": round_id,
            "scores": scores,
            "selected": selected,
            "energy": en,
            "bandwidth": bw,
            "high_energy_threshold": high_energy_threshold,
        },
    )

    return selected, scores
