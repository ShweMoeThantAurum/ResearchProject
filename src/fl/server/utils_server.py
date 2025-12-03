"""
Shared utilities for server-side federated learning:
- role definitions
- environment helpers
- S3 path helpers
- simple AEFL settings.
"""

import os

# ---------------------------------------------------------------------
# Roles used throughout the project
# ---------------------------------------------------------------------
ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def _get_env(name, default):
    """Return environment variable or default value."""
    v = os.environ.get(name, default)
    return v


# ---------------------------------------------------------------------
# S3 configuration
# ---------------------------------------------------------------------
def get_s3_bucket():
    """Return primary S3 bucket name for round data."""
    return _get_env("S3_BUCKET", "aefl")


def get_s3_prefix():
    """
    Return dataset/mode prefix for round data.

    Layout:
      fl/<dataset>/<MODE>/
    """
    dataset = _get_env("DATASET", "sz").lower()
    mode = _get_env("FL_MODE", "AEFL").upper()
    return f"fl/{dataset}/{mode}"


def get_results_bucket():
    """
    Return bucket name for experiment summaries and artifacts.
    Falls back to S3_BUCKET if RESULTS_BUCKET is not set.
    """
    return _get_env("RESULTS_BUCKET", _get_env("S3_BUCKET", "aefl"))


# ---------------------------------------------------------------------
# Dataset/mode for summaries
# ---------------------------------------------------------------------
def get_dataset():
    """Return dataset name as lowercase string."""
    return _get_env("DATASET", "sz").lower()


def get_fl_mode():
    """Return FL mode name as lowercase string."""
    return _get_env("FL_MODE", "AEFL").lower()


# ---------------------------------------------------------------------
# AEFL selection parameters
# ---------------------------------------------------------------------
def get_aefl_max_clients():
    """Return maximum number of clients to select per AEFL round."""
    return int(_get_env("AEFL_MAX_CLIENTS", "3"))
