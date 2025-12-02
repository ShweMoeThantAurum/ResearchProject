"""
Model aggregation strategies.

Includes:
    - FedAvg
    - FedProx (same aggregation here)
    - AEFL (same as FedAvg for now)
"""

import torch


def aggregate_fedavg(states):
    """Simple average over client updates."""
    if not states:
        return {}

    roles = list(states.keys())
    base = states[roles[0]]

    avg = {}
    for name in base:
        tensors = [states[r][name].float() for r in roles]
        avg[name] = sum(tensors) / float(len(tensors))
    return avg


def aggregate_fedprox(states):
    """FedProx server aggregation = FedAvg."""
    return aggregate_fedavg(states)


def aggregate_aefl(states):
    """AEFL currently uses FedAvg aggregation."""
    return aggregate_fedavg(states)
