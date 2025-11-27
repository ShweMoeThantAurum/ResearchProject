"""Wrapper functions for loading client metadata files from S3."""

from src.fl.server.s3 import load_round_metadata


def load_metadata_for_round(round_id: int, prefix: str):
    """
    Load metadata JSON for all clients for the specified round.
    """
    return load_round_metadata(round_id, prefix)


def load_client_metadata(round_id: int, bucket: str, prefix: str):
    """
    Backward-compatible wrapper matching an older signature.
    Bucket is ignored; prefix and round determine metadata.
    """
    return load_metadata_for_round(round_id, prefix)
