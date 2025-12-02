"""
Aggregation strategies for server-side model updates.

FedAvg = uniform average
FedProx = same as FedAvg (server side)
AEFL = weighted aggregation using selection scores
"""

from typing import Dict
import torch


def aggregate_fedavg(states):
    """Average all client model parameters equally."""
    if not states:
        return {}

    roles = list(states.keys())
    base = states[roles[0]]

    avg = {}
    for name in base.keys():
        tensors = [states[r][name].float() for r in roles]
        avg[name] = sum(tensors) / float(len(tensors))
    return avg


def aggregate_fedprox(states):
    """FedProx uses FedAvg server-side."""
    return aggregate_fedavg(states)


def aggregate_aefl(states, scores):
    """
    AEFL weighted aggregation based on selection scores.

    Args:
        states (dict): role -> state_dict
        scores (dict): role -> AEFL selection score

    Returns:
        aggregated state_dict
    """
    if not states:
        return {}

    # Normalise AEFL scores
    roles = list(states.keys())
    raw_scores = {r: scores.get(r, 1.0) for r in roles}

    total = sum(raw_scores.values()) or 1
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


# Deprecated aliases
def fedavg_aggregate(states):
    return aggregate_fedavg(states)


def aefl_aggregate(states, scores=None):
    return aggregate_aefl(states, scores)
