"""
Client-side local training utilities.
Runs local GRU training with optional DP and compression.
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
    Runs local GRU training and returns a DP/compressed update dict.

    Args:
        model: GRUModel (already moved to device)
        X, y: tensors or numpy arrays (time-series sequences)
        lr: learning rate
        batch_size: mini-batch size
        local_epochs: number of local epochs
        device: torch.device("cpu" | "cuda")
        dp_enabled: whether to add Gaussian noise to update
        dp_sigma: DP sigma value
        compression_enabled: whether to compress the update
        compression_mode: "sparsify" or "topk"
        compression_sparsity: fraction of weights to zero out (for sparsify)
        compression_k_frac: fraction of weights to keep (for topk)

    Returns:
        update_state: dict[name -> tensor] (CPU), after DP & compression
        last_loss: float
        compute_time_s: float
        batch_count: int
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
    batch_count = 0

    for _ in range(local_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
            batch_count += 1

    compute_time_s = time.time() - start

    # ----------------------------------------------------
    # Build update dict on CPU
    # ----------------------------------------------------
    update_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # ----------------------------------------------------
    # DP: apply Gaussian noise to the update (not model in-place)
    # ----------------------------------------------------
    if dp_enabled and dp_sigma > 0.0:
        update_state = apply_dp_noise(update_state, sigma=dp_sigma)

    # ----------------------------------------------------
    # Compression: sparsify or top-k the update
    # ----------------------------------------------------
    if compression_enabled:
        update_state = apply_compression(
            update_state,
            mode=compression_mode,
            sparsity=compression_sparsity,
            k_frac=compression_k_frac,
        )

    return update_state, last_loss, compute_time_s, batch_count
