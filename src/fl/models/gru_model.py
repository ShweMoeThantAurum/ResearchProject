"""
GRU-based neural network for traffic flow prediction.
Takes sequences of shape [batch, seq_len, num_nodes] and outputs per-node predictions.
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """A simple GRU regressor for traffic forecasting."""
    def __init__(self, num_nodes, hidden_size):
        super().__init__()

        # GRU encoder
        self.gru = nn.GRU(
            input_size=num_nodes,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Linear decoder projecting hidden state to per-node output
        self.decoder = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        """Forward pass through GRU and linear decoder."""
        # x: [batch, seq, num_nodes]
        out, _ = self.gru(x)
        h_last = out[:, -1, :]           # last timestep hidden state
        pred = self.decoder(h_last)      # predict next traffic values
        return pred
