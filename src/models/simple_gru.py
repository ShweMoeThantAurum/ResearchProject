"""
A small GRU model for multivariate time series forecasting.
"""
import torch
import torch.nn as nn

class SimpleGRU(nn.Module):
    def __init__(self, num_nodes, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size=num_nodes, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        out, _ = self.gru(x)       # x: [B, L, N]
        last = out[:, -1, :]       # [B, H]
        yhat = self.head(last)     # [B, N]
        return yhat
