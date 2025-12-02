"""
Local training for one FL round.

Loads global model weights, trains on the client's local partition,
and returns the parameter update (state_dict difference).
"""

import time
import torch
import torch.nn as nn

from src.fl.models import SimpleGRU


def run_local_training(global_state, X, y, hidden_size):
    """
    Load global model -> train locally -> return update dict.
    """
    device = torch.device("cpu")

    num_nodes = X.shape[-1]

    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden_size).to(device)
    model.load_state_dict(global_state)

    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)

    timer = time.time()
    optimizer.zero_grad()
    pred = model(X_t)
    loss = criterion(pred, y_t)
    loss.backward()
    optimizer.step()
    train_time = time.time() - timer

    # Compute update
    update = {}
    for name, param in model.state_dict().items():
        update[name] = param.cpu()

    return update, loss.item(), train_time
