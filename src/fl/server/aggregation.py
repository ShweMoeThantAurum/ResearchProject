"""
Aggregation methods for FL modes.
"""

import torch


def aggregate_fedavg(states):
    """
    Average model weights across clients.
    """
    roles = list(states.keys())
    base = states[roles[0]]

    new_state = {}
    for k in base.keys():
        tensors = [states[r][k].float() for r in roles]
        new_state[k] = sum(tensors) / float(len(tensors))

    return new_state


def aggregate_fedprox(states):
    """
    FedProx aggregation (same as FedAvg server-side).
    """
    return aggregate_fedavg(states)


def aggregate_aefl(states):
    """
    AEFL aggregation (currently same as FedAvg).
    """
    return aggregate_fedavg(states)
