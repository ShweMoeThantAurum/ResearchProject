"""
Local model training for a single FL round.

This module executes:
    - forward/backward passes
    - one or more epochs over the client's local data
    - computes FLOPs for energy tracking
"""

import time
import torch
import torch.nn as nn


def train_one_round(model,
                    X_local,
                    y_local,
                    lr,
                    batch_size,
                    epochs,
                    device):
    """
    Train the model for one FL round using the client's local dataset.

    Parameters:
        model      : PyTorch model
        X_local    : local inputs  [N, seq_len, nodes]
        y_local    : local targets [N, nodes]
        lr         : learning rate
        batch_size : training batch size
        epochs     : number of passes over dataset
        device     : compute device

    Returns:
        loss       : final epoch loss
        train_time : seconds spent training
        flops      : very rough estimate of floating-point operations
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.from_numpy(X_local).float().to(device)
    y = torch.from_numpy(y_local).float().to(device)

    n = X.shape[0]

    start = time.time()
    flops = 0.0

    for _ in range(epochs):
        for i in range(0, n, batch_size):
            xb = X[i:i + batch_size]
            yb = y[i:i + batch_size]

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            flops += xb.numel() * 2

    elapsed = time.time() - start

    return loss.item(), elapsed, flops
