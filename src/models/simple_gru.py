"""Defines a lightweight GRU-based model for spatio-temporal traffic prediction."""

import torch
import torch.nn as nn


class SimpleGRU(nn.Module):
    """
    A minimal GRU-based predictor used across all FL modes.
    Input:
        X: [batch, seq_len, num_nodes]
    Output:
        y_pred: [batch, num_nodes]
    """

    def __init__(self, num_nodes, hidden_size=64, seq_len=12):
        """
        Initialise GRU encoder and output projection layer.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # GRU encoder: processes sequences of node features
        self.gru = nn.GRU(
            input_size=num_nodes,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Final linear layer maps hidden state → node predictions
        self.fc = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        """
        Run GRU over temporal sequence and predict future traffic values.
        """
        # x shape: [batch, seq_len, num_nodes]
        out, _ = self.gru(x)           # out: [batch, seq_len, hidden_size]
        final = out[:, -1, :]          # use last timestep: [batch, hidden_size]
        pred = self.fc(final)          # [batch, num_nodes]
        return pred
