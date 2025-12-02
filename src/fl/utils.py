"""
Shared helper functions for dataset paths, S3 prefixes, and
federated learning hyperparameters.

Centralises environment-variable parsing so clients and server remain
consistent across all experiment configurations.
"""

import os


# ------------------------------
# Dataset paths
# ------------------------------

def get_dataset():
    """Return the active dataset name (sz, los, pems08)."""
    return os.environ.get("DATASET", "sz").strip().lower()


def get_proc_dir():
    """Return the path to the preprocessed dataset directory."""
    dataset = get_dataset()
    return f"data/processed/{dataset}/prepared"


# ------------------------------
# S3 configuration
# ------------------------------

def get_bucket():
    """Return the S3 bucket name used for FL model exchange."""
    return os.environ.get("S3_BUCKET", "aefl")


def get_prefix():
    """Return the S3 prefix fl/<dataset> for storing model rounds."""
    dataset = get_dataset()
    return f"fl/{dataset}"


# ------------------------------
# FL hyperparameters
# ------------------------------

def get_fl_rounds():
    """Return number of federated learning rounds."""
    return int(os.environ.get("FL_ROUNDS", 20))


def get_batch_size():
    """Return mini-batch size for local training."""
    return int(os.environ.get("BATCH_SIZE", 64))


def get_local_epochs():
    """Return local epochs per FL round."""
    return int(os.environ.get("LOCAL_EPOCHS", 1))


def get_lr():
    """Return client learning rate."""
    return float(os.environ.get("LR", 0.001))


def get_hidden_size():
    """Return GRU hidden size for the model."""
    return int(os.environ.get("HIDDEN_SIZE", 64))
