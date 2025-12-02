"""
Local training logic for federated learning clients.

This module runs one local training round on a client, supporting:
    - FedAvg (standard local SGD)
    - FedProx (proximal term around the previous global model)
    - AEFL and LocalOnly, which reuse the same local loop
"""

import torch
from src.fl.utils.logger import log_event, Timer

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
    """
    Train the local model for a single federated learning round.

    Parameters:
        model        : PyTorch model to train
        loader       : DataLoader with local samples
        role         : client role name
        round_id     : current federated round index (starting from 1)
        device       : "cpu" or "cuda"
        local_epochs : local epochs per round
        lr           : learning rate
        mode         : FL mode string, used to enable FedProx
        global_state : optional reference global state for FedProx

    Returns:
        updated_state : state_dict of the trained model on CPU
        elapsed       : local training time in seconds
        avg_loss      : mean loss across all batches and epochs
        total_samples : total number of training samples seen
    """
    timer = Timer()
    timer.start()

    model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    use_fedprox = (mode.lower() == "fedprox") and (global_state is not None)

    global_params = None
    if use_fedprox:
        global_params = [global_state[k].to(device)
                         for k in model.state_dict().keys()]

    total_loss = 0.0
    total_batches = 0
    total_samples = 0

    for _ in range(local_epochs):
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)

            if use_fedprox:
                prox_term = 0.0
                for p, g0 in zip(model.parameters(), global_params):
                    prox_term = prox_term + torch.sum((p - g0) ** 2)
                loss = loss + (FEDPROX_MU / 2.0) * prox_term

            loss.backward()
            opt.step()

            batch_size = X.size(0)
            total_loss = total_loss + loss.item()
            total_batches = total_batches + 1
            total_samples = total_samples + batch_size

    elapsed = timer.stop()
    avg_loss = total_loss / max(1, total_batches)

    log_event("client_train.log", {
        "role": role,
        "round": round_id,
        "training_time_sec": elapsed,
        "avg_loss": avg_loss,
        "batches": total_batches,
        "samples": total_samples,
        "local_epochs": local_epochs,
        "lr": lr,
        "mode": mode,
    })

    print("[%s] Round %d training | mode=%s, loss=%.6f, time=%.3fs, samples=%d"
          % (role, round_id, mode, avg_loss, elapsed, total_samples))

    updated_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return updated_state, elapsed, avg_loss, total_samples
