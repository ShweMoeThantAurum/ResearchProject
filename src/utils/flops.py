"""
FLOPs estimation utilities for the SimpleGRU model.

Approximate GRU FLOPs per timestep:
  FLOPs_t = 6 * H * (H + I)

where:
  H = hidden_size
  I = input_size (num_nodes)

For seq_len timesteps:
  FLOPs_total = seq_len * FLOPs_t
"""


def estimate_gru_flops(num_nodes, hidden_size, seq_len):
    """
    Estimate the number of floating-point operations for a GRU forward pass.

    This is a coarse analytical estimate used only for logging.
    """
    flops_per_timestep = 6 * hidden_size * (hidden_size + num_nodes)
    return flops_per_timestep * seq_len
