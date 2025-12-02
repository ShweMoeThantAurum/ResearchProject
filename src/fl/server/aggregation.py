"""
Aggregation strategies for federated learning.

Implements:
- FedAvg (equal weighting)
- FedProx (same aggregation as FedAvg on server)
- AEFL (currently same as FedAvg)
"""

import torch


def aggregate_fedavg(states):
    """Average all client update tensors equally."""
    if not states:
        return {}

    roles = list(states.keys())
    base = states[roles[0]]

    avg = {}
    for name in base.keys():
        tensors = [states[r][name].float() for r in roles]
        avg[name] = sum(tensors) / float(len(tensors))
    return avg


def aggregate_aefl(states):
    """AEFL currently uses FedAvg aggregation."""
    return aggregate_fedavg(states)


def aggregate_fedprox(states):
    """FedProx uses FedAvg on the server side."""
    return aggregate_fedavg(states)
