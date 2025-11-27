"""Server-side placeholder for DP-aware aggregation (future Lambda support)."""

def maybe_account_dp(states):
    """
    Return states unchanged.

    Light DP is applied client-side only in this implementation.
    Server-side DP aggregation may be added later.
    """
    return states