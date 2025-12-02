"""
Server-side configuration helpers for FL experiments.
Reads core settings from environment variables.
"""

import os


ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def get_dataset():
    """Return the active dataset name."""
    return os.environ.get("DATASET", "sz").strip().lower()


def get_fl_mode():
    """Return the federated learning mode."""
    return os.environ.get("FL_MODE", "AEFL").strip().upper()


def get_fl_rounds():
    """Return the number of federated learning rounds."""
    return int(os.environ.get("FL_ROUNDS", "20"))


def get_hidden_size():
    """Return the GRU hidden size."""
    return int(os.environ.get("HIDDEN_SIZE", "64"))


def get_batch_size():
    """Return batch size for server-side evaluation."""
    return int(os.environ.get("SERVER_BATCH_SIZE", "64"))


def get_s3_bucket():
    """Return S3 bucket used for FL model exchange."""
    return os.environ.get("S3_BUCKET", "aefl")


def get_results_bucket():
    """Return S3 bucket for storing experiment summaries."""
    return os.environ.get("RESULTS_BUCKET", get_s3_bucket())


def get_s3_prefix():
    """Return S3 prefix under which round artefacts are stored."""
    dataset = get_dataset()
    return f"fl/{dataset}"


def get_aefl_clients_per_round():
    """Return the fixed number of AEFL clients selected per round."""
    return int(os.environ.get("AEFL_CLIENTS_PER_ROUND", "3"))
