"""
Convenience wrappers for loading metadata for a specific FL round.
"""

from src.fl.server.s3 import load_round_metadata


def load_metadata_for_round(round_id, prefix):
    """Load all client metadata for the given round."""
    return load_round_metadata(round_id, prefix)


def load_client_metadata(round_id, bucket, prefix):
    """Compatibility wrapper."""
    return load_metadata_for_round(round_id, prefix)
