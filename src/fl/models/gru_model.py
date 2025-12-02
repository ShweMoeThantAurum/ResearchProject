"""
GRU-based traffic flow prediction model.
"""

import torch
import torch.nn as nn


class SimpleGRU(nn.Module):
    """
    A GRU model for sequence-to-one traffic prediction.
    """

    def __init__(self, input_dim=1, hidden_size=64, num_layers=1):
        """
        Create GRU + linear projection.
        """
        super(SimpleGRU, self).__init__()

        # GRU encoder
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True
        )

        # Final linear layer for 1-step prediction
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through GRU and final layer.
        """
        # x: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)

        # Take the last hidden state
        last = out[:, -1, :]

        # Predict next timestep
        return self.fc(last)
