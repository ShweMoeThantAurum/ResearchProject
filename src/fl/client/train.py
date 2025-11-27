"""Client-side local model training, including FedProx support."""

import torch
from torch.utils.data import DataLoader
from src.fl.logger import log_event, Timer

FEDPROX_MU = 0.01  # constant proximal term coefficient


def train_one_round(model: torch.nn.Module,
                    loader: DataLoader,
                    role: str,
                    round_id: int,
                    device: str,
                    local_epochs: int,
                    lr: float,
                    mode: str,
                    global_state=None):
    """
    Train the model locally for a single federated learning round.

    Supports:
      - AEFL (same as FedAvg)
      - FedAvg
      - FedProx (adds proximal term)
      - LocalOnly (standard local training)

    Returns the updated state_dict, training duration, mean loss, and sample count.
    """
    timer = Timer()
    timer.start()

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    use_prox = (mode.lower() == "fedprox") and (global_state is not None)

    # Prepare global parameters for FedProx
    global_params = None
    if use_prox:
        global_params = [global_state[k].to(device) for k in model.state_dict().keys()]

    total_loss = 0.0
    total_batches = 0
    total_samples = 0

    for _ in range(local_epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)

            if use_prox:
                prox_term = 0.0
                for p, g0 in zip(model.parameters(), global_params):
                    prox_term += torch.sum((p - g0) ** 2)
                loss += (FEDPROX_MU / 2.0) * prox_term

            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_batches += 1
            total_samples += x.size(0)

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

    print(f"[{role}] Round {round_id} training | mode={mode}, "
          f"loss={avg_loss:.6f}, time={elapsed:.3f}s, samples={total_samples}")

    return {k: v.cpu() for k, v in model.state_dict().items()}, elapsed, avg_loss, total_samples
