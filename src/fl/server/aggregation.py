"""
Model aggregation utilities.
Implements FedAvg, FedProx, and AEFL.
"""

import torch


def _mean(states):
    """Average model parameters across roles."""
    if not states:
        return {}

    roles = list(states.keys())
    base = states[roles[0]]

    avg = {}
    for name in base:
        tensors = [states[r][name].float() for r in roles]
        stacked = torch.stack(tensors)
        avg[name] = torch.mean(stacked, dim=0)

    return avg


def aggregate_fedavg(states):
    """FedAvg aggregation."""
    return _mean(states)


def aggregate_fedprox(states):
    """FedProx uses same server aggregation as FedAvg."""
    return _mean(states)


def aggregate_aefl(states):
    """AEFL uses weighted or equal averaging (equal for now)."""
    return _mean(states)
