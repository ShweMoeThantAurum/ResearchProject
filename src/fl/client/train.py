"""
Local training routine for federated learning clients.

Supports FedAvg, AEFL, and FedProx (extra proximal term).
"""

import torch
from src.fl.logger import log_event, Timer

FEDPROX_MU = 0.01


def train_one_round(
    model,
    loader,
    role,
    round_id,
    device,
    local_epochs,
    lr,
    mode,
    global_state=None,
):
    """
    Train local model for one round. Returns:
        updated_state_dict, elapsed_time, avg_loss, num_samples
    """
    timer = Timer()
    timer.start()

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    use_prox = mode.lower() == "fedprox" and global_state is not None
    if use_prox:
        global_params = [global_state[k].to(device) for k in model.state_dict()]

    total_loss, batches, samples = 0.0, 0, 0

    for _ in range(local_epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)

            if use_prox:
                prox = 0.0
                for p, g0 in zip(model.parameters(), global_params):
                    prox += torch.sum((p - g0) ** 2)
                loss += (FEDPROX_MU / 2) * prox

            loss.backward()
            opt.step()

            total_loss += loss.item()
            batches += 1
            samples += X.size(0)

    elapsed = timer.stop()
    avg = total_loss / max(1, batches)

    log_event(
        "client_train.log",
        {
            "role": role,
            "round": round_id,
            "training_time_sec": elapsed,
            "avg_loss": avg,
            "samples": samples,
            "local_epochs": local_epochs,
            "lr": lr,
            "mode": mode,
        },
    )

    print(
        f"[{role}] Train r={round_id} | mode={mode}, "
        f"loss={avg:.6f}, time={elapsed:.3f}s, samples={samples}"
    )

    return {k: v.cpu() for k, v in model.state_dict().items()}, elapsed, avg, samples
