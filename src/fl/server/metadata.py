"""
Lightweight wrappers for loading round-level client metadata.

These helpers delegate to the S3 utilities and keep the server
code readable when accessing per-round metadata.
"""

from src.fl.server.s3 import load_round_metadata


def load_metadata_for_round(round_id, prefix):
    """Load all client metadata for the given round."""
    return load_round_metadata(round_id, prefix)


def load_client_metadata(round_id, bucket, prefix):
    """
    Compatibility wrapper.

    Kept for backwards compatibility with earlier code that
    passed (round_id, bucket, prefix).
    """
    return load_metadata_for_round(round_id, prefix)
