"""Aggregation strategies for cloud-based federated learning."""

from typing import Dict
import torch


def aggregate_fedavg(states: Dict[str, dict]):
    """
    Perform FedAvg aggregation by averaging model parameters across clients.

    The states dict maps client roles to state_dicts. All tensors must
    have identical shapes. This is the standard unweighted mean.
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


def aggregate_aefl(states: Dict[str, dict]):
    """
    AEFL aggregation currently mirrors FedAvg.

    The adaptivity of AEFL is implemented in the client selection stage,
    not in weighted parameter averaging.
    """
    return aggregate_fedavg(states)


def aggregate_fedprox(states: Dict[str, dict]):
    """
    FedProx uses a proximal local objective but aggregates globally the same
    way as FedAvg in the canonical formulation.
    """
    return aggregate_fedavg(states)


# --- Compatibility aliases (older imports) ---

def fedavg_aggregate(states: Dict[str, dict]):
    return aggregate_fedavg(states)


def aefl_aggregate(states: Dict[str, dict], scores=None):
    return aggregate_aefl(states)
