"""
Aggregation strategies for server-side model updates.

 - FedAvg  : uniform averaging across all participating clients
 - FedProx : same server-side aggregation as FedAvg
 - AEFL    : weighted aggregation using selection scores
"""

from typing import Dict
import torch


def aggregate_fedavg(states: Dict[str, Dict[str, torch.Tensor]]):
    """
    Average all client model parameters equally (FedAvg-style).

    Args:
        states: mapping role -> state_dict
    """
    if not states:
        return {}

    roles = list(states.keys())
    base = states[roles[0]]

    avg = {}
    for name in base.keys():
        tensors = [states[r][name].float() for r in roles]
        avg[name] = sum(tensors) / float(len(tensors))
    return avg


def aggregate_fedprox(states: Dict[str, Dict[str, torch.Tensor]]):
    """
    FedProx aggregation.

    Note:
        In this implementation, FedProx uses the same server-side
        aggregation as FedAvg. The proximal term is applied on the
        client side during training.
    """
    return aggregate_fedavg(states)


def aggregate_aefl(
    states: Dict[str, Dict[str, torch.Tensor]], scores: Dict[str, float]
):
    """
    AEFL weighted aggregation based on selection scores.

    Args:
        states: role -> model state_dict
        scores: role -> AEFL selection score

    Returns:
        Aggregated state_dict where each client is weighted
        according to its AEFL score.
    """
    if not states:
        return {}

    # Normalise AEFL scores to obtain aggregation weights
    roles = list(states.keys())
    raw_scores = {r: scores.get(r, 1.0) for r in roles}

    total = sum(raw_scores.values()) or 1.0
    weights = {r: raw_scores[r] / total for r in roles}

    base_role = roles[0]
    base_state = states[base_role]
    aggregated = {}

    for name in base_state.keys():
        tensors = []
        for r in roles:
            t = states[r][name].float() * weights[r]
            tensors.append(t)
        aggregated[name] = sum(tensors)

    return aggregated


# Deprecated aliases kept for backwards compatibility
def fedavg_aggregate(states):
    """Deprecated alias for aggregate_fedavg."""
    return aggregate_fedavg(states)


def aefl_aggregate(states, scores=None):
    """Deprecated alias for aggregate_aefl."""
    return aggregate_aefl(states, scores)
