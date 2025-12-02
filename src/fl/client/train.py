"""
Local training loop for a single FL round.
"""

import time
import torch
import torch.nn as nn


def train_one_round(model, x, y):
    """
    Train the model for one round on local data.
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    # Convert numpy -> tensors
    x_t = torch.tensor(x).float()
    y_t = torch.tensor(y).float()

    optimizer.zero_grad()
    preds = model(x_t)
    loss = criterion(preds.squeeze(), y_t)
    loss.backward()
    optimizer.step()

    train_time = time.time() - start

    # Fake FLOPS estimate for energy model
    total_params = sum(p.numel() for p in model.parameters())
    flops_j = total_params * 1e-9

    return float(loss.item()), train_time, flops_j
