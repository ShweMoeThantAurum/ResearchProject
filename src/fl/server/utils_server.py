"""
Shared utilities for server-side FL logic.
Includes AEFL scoring, metric helpers and role definitions.
"""

ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def compute_energy_score(energy):
    """Normalizes energy so lower energy = higher score."""
    return 1.0 / (energy + 1e-9)


def compute_accuracy_score(loss):
    """Higher accuracy means lower loss, so invert loss."""
    return 1.0 / (loss + 1e-9)


def compute_privacy_score(noise_sigma):
    """Higher DP noise means higher privacy score."""
    return noise_sigma


def build_eval_metrics(loss, energy, dp_sigma):
    """Builds AEFL combined score per client."""
    e = compute_energy_score(energy)
    a = compute_accuracy_score(loss)
    p = compute_privacy_score(dp_sigma)
    return 0.4 * a + 0.4 * e + 0.2 * p


def get_aefl_clients_per_round(scores, k=3):
    """
    Selects top-k clients based on AEFL combined scores.
    scores: dict(client → float)
    """
    sorted_clients = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in sorted_clients[:k]]
