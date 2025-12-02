"""
Aggregation strategies for server-side FL updates.
Implements FedAvg-style averaging and an AEFL variant.
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
    """FedProx uses the same server-side aggregation as FedAvg."""
    return _aggregate_mean(states)


def aggregate_aefl(states):
    """AEFL aggregation currently uses equal-weight averaging."""
    return _aggregate_mean(states)
