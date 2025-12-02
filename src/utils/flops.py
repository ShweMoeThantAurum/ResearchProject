"""
FLOPs estimation utilities for the SimpleGRU model.

GRU FLOPs per timestep approx:
FLOPs = 6 * H * (H + I)
where:
  H = hidden_size
  I = input_size (num_nodes)

For seq_len timesteps:
FLOPs_total = seq_len * FLOPs
"""

def estimate_gru_flops(num_nodes, hidden_size, seq_len):
    # GRU formula:
    flops_per_timestep = 6 * hidden_size * (hidden_size + num_nodes)

    # total FLOPs for the whole sequence
    return flops_per_timestep * seq_len
