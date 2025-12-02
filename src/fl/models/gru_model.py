"""
Lightweight GRU-based model for spatio-temporal traffic prediction.

The model:
    - Takes a sequence of node-level traffic readings
    - Produces the next-step prediction for all nodes

Input shape:
    [batch_size, seq_len, num_nodes]

Output shape:
    [batch_size, num_nodes]
"""

import torch
import torch.nn as nn


class SimpleGRU(nn.Module):
    """
    Minimal GRU predictor shared across all FL baselines and AEFL.

    Components:
        - GRU encoder over temporal dimension
        - Linear layer mapping hidden state to node-level outputs
    """

    def __init__(self, num_nodes, hidden_size=64, seq_len=12):
        """
        Initialise the GRU encoder and output projection.

        Parameters:
            num_nodes  : number of graph nodes (features per timestep)
            hidden_size: size of GRU hidden state
            seq_len    : number of timesteps in the input sequence
        """
        super(SimpleGRU, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.gru = nn.GRU(
            input_size=num_nodes,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        """
        Run GRU over the sequence and return next-step prediction.

        Parameters:
            x: tensor [batch_size, seq_len, num_nodes]

        Returns:
            pred: tensor [batch_size, num_nodes]
        """
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)
        return pred
