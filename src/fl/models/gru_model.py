"""
GRU-based predictor for traffic flow forecasting.
Takes a sequence of past traffic speeds and predicts the next step.
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    Simple GRU regressor used by all FL clients and server.
    Input:  (batch, seq_len, num_nodes)
    Output: (batch, num_nodes)
    """

    def __init__(self, num_nodes, hidden_size=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size

        # GRU processes each timestep; features = num_nodes
        self.gru = nn.GRU(
            input_size=num_nodes,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Final linear layer converts hidden state → node-wise forecast
        self.fc = nn.Linear(hidden_size, num_nodes)

        # Stable training: initialize weights
        self._init_weights()

    def _init_weights(self):
        """Applies Xavier initialization for stable convergence."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """
        Run GRU over full sequence.
        Take final hidden state → forecast next-step speed.
        """
        out, h = self.gru(x)       # h: (1, batch, hidden)
        h_last = h[-1]             # (batch, hidden)
        pred = self.fc(h_last)     # (batch, num_nodes)
        return pred
