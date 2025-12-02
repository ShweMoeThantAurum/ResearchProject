"""
Simple GRU model for traffic forecasting using sequence-to-one prediction.
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """A simple GRU regressor for traffic forecasting."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last time-step hidden
        out = self.fc(out)
        return out
