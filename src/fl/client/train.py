"""
Local training loop for FL clients.
Supports FedAvg, AEFL, FedProx and LocalOnly modes.
"""

import torch
from torch.utils.data import DataLoader
from ..utils.logger import log_event

FEDPROX_MU = 0.01


def train_one_round(model,
                    loader,
                    role,
                    round_id,
                    device,
                    local_epochs,
                    lr,
                    mode,
                    global_state=None):
    """Train model for one FL round and return updated weights."""
    model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    use_prox = mode.lower() == "fedprox" and global_state is not None
    global_params = None

    if use_prox:
        # Clone global parameters as FedProx reference
        global_params = [global_state[k].to(device) for k in model.state_dict().keys()]

    total_loss = 0.0
    total_batches = 0
    total_samples = 0

    # Simple FLOPs estimate per batch (very rough)
    approx_flops = 0.0

    for _ in range(local_epochs):
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)

            if use_prox and global_params is not None:
                prox_term = 0.0
                for p, g0 in zip(model.parameters(), global_params):
                    prox_term += torch.sum((p - g0) ** 2)
                loss = loss + 0.5 * FEDPROX_MU * prox_term

            loss.backward()
            opt.step()

            batch_size = X.size(0)
            total_loss += loss.item()
            total_batches += 1
            total_samples += batch_size

            # Approximate GRU cost ~ O(seq_len * hidden * nodes)
            seq_len = X.size(1)
            num_nodes = X.size(2)
            hidden = next(model.parameters()).numel() // num_nodes
            approx_flops += float(batch_size * seq_len * num_nodes * hidden)

    avg_loss = total_loss / max(1, total_batches)

    log_event({
        "type": "client_train",
        "role": role,
        "round": round_id,
        "mode": mode,
        "local_epochs": local_epochs,
        "lr": lr,
        "avg_loss": avg_loss,
        "batches": total_batches,
        "samples": total_samples,
        "approx_flops": approx_flops,
    })

    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Time is measured outside this function by the caller
    return state_cpu, avg_loss, total_samples, approx_flops
