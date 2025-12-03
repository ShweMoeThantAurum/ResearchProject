"""
Shared utilities for server-side FL logic:
- roles and environment helpers
- AEFL configuration
- S3 path helpers
"""

import os


# Roles
ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


# Basic env helpers
def _get_env(name, default):
    value = os.environ.get(name, default)
    return value


def get_dataset():
    """Return dataset name such as sz, los, pems08."""
    return _get_env("DATASET", "sz").lower()


def get_fl_mode():
    """Return FL mode such as aefl, fedavg, fedprox, localonly."""
    return _get_env("FL_MODE", "AEFL").lower()


def get_fl_rounds():
    """Return total number of FL rounds."""
    return int(_get_env("FL_ROUNDS", "20"))


def get_hidden_size():
    """Return GRU hidden size."""
    return int(_get_env("HIDDEN_SIZE", "64"))


def get_batch_size():
    """Return evaluation batch size."""
    return int(_get_env("BATCH_SIZE", "64"))


def get_s3_bucket():
    """Return primary S3 bucket for FL models and updates."""
    return _get_env("S3_BUCKET", "aefl")


def get_results_bucket():
    """
    Return S3 bucket for experiment summaries.
    Falls back to S3_BUCKET if RESULTS_BUCKET is not set.
    """
    return _get_env("RESULTS_BUCKET", get_s3_bucket())


def get_s3_prefix():
    """Return S3 prefix for this experiment run."""
    dataset = get_dataset()
    mode = get_fl_mode().upper()
    return f"fl/{dataset}/{mode}"


# AEFL configuration
def get_aefl_max_clients():
    """Return the maximum number of clients per AEFL round."""
    return int(_get_env("AEFL_MAX_CLIENTS", "3"))
