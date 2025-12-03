"""
Aggregation strategies for server-side federated learning.
Implements FedAvg-style parameter averaging.
"""

import torch


def _aggregate_mean(states):
    """Average all client model parameters equally."""
    if not states:
        return {}

    roles = list(states.keys())
    base = states[roles[0]]

    averaged = {}
    for name in base.keys():
        tensors = [states[r][name].float() for r in roles]
        stacked = torch.stack(tensors, dim=0)
        averaged[name] = torch.mean(stacked, dim=0)

    return averaged


def aggregate_fedavg(states):
    """Standard FedAvg parameter aggregation."""
    return _aggregate_mean(states)


def aggregate_fedprox(states):
    """FedProx uses same server-side aggregation as FedAvg."""
    return _aggregate_mean(states)


def aggregate_aefl(states):
    """AEFL currently uses equal-weight averaging on selected clients."""
    return _aggregate_mean(states)
