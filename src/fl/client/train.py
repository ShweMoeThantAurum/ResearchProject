"""
Client-side local training utilities.
Runs one local epoch of GRU model training with optional DP and compression.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from .dp import apply_dp_noise
from .compression import apply_compression


def train_one_epoch(
    model,
    X,
    y,
    lr,
    batch_size,
    local_epochs,
    device,
    dp_enabled=False,
    dp_sigma=0.01,
    compression_enabled=False,
    compression_mode="sparsify",
    compression_sparsity=0.5,
    compression_k_frac=0.1,
):
    """
    Runs one round of local GRU training. Applies DP & compression to the final update.
    Returns: (loss_value, compute_time_seconds)
    """
    start = time.time()

    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    last_loss = 0.0

    for _ in range(local_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

    compute_time = time.time() - start

    # ----------------------------------------------------
    # DP: apply Gaussian noise to model parameters
    # ----------------------------------------------------
    if dp_enabled:
        apply_dp_noise(model.state_dict(), sigma=dp_sigma)

    # ----------------------------------------------------
    # Compression: sparsify or top-k
    # ----------------------------------------------------
    if compression_enabled:
        apply_compression(
            model.state_dict(),
            mode=compression_mode,
            sparsity=compression_sparsity,
            k_frac=compression_k_frac,
        )

    return last_loss, compute_time
