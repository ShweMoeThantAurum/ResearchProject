"""Shared utility helpers for dataset paths, S3 configuration, and FL hyperparameters."""

import os


# ============================================================
# Dataset Paths
# ============================================================

def get_dataset():
    """
    Return the active dataset identifier.

    The dataset name is taken from the DATASET environment variable.
    Valid examples include: 'sz', 'los', 'pems08'.
    If unset, the default value 'sz' is used.
    """
    return os.environ.get("DATASET", "sz").strip().lower()


def get_proc_dir():
    """
    Return the path to the directory containing preprocessed data.

    The path is constructed as:
        data/processed/<dataset>/prepared
    where <dataset> is determined by get_dataset().
    """
    dataset = get_dataset()
    return f"data/processed/{dataset}/prepared"


# ============================================================
# S3 Bucket + Prefix
# ============================================================

def get_bucket():
    """
    Return the S3 bucket name used for cloud federated learning.

    The name is read from the S3_BUCKET environment variable.
    If unset, the default bucket name 'aefl' is used.
    """
    return os.environ.get("S3_BUCKET", "aefl")


def get_prefix():
    """
    Return the S3 key prefix used to store FL artefacts.

    The prefix is constructed as:
        fl/<dataset>
    where <dataset> is determined by get_dataset().
    """
    dataset = get_dataset()
    return f"fl/{dataset}"


# ============================================================
# FL Hyperparameters
# ============================================================

def get_fl_rounds():
    """
    Return the number of federated learning rounds.

    The value is read from the FL_ROUNDS environment variable.
    If unset, the default is 20.
    """
    return int(os.environ.get("FL_ROUNDS", 20))


def get_batch_size():
    """
    Return the mini-batch size used by FL clients.

    The value is read from the BATCH_SIZE environment variable.
    If unset, the default is 64.
    """
    return int(os.environ.get("BATCH_SIZE", 64))


def get_local_epochs():
    """
    Return the number of local epochs each client trains per round.

    The value is read from the LOCAL_EPOCHS environment variable.
    If unset, the default is 1.
    """
    return int(os.environ.get("LOCAL_EPOCHS", 1))


def get_lr():
    """
    Return the learning rate used by FL clients.

    The value is read from the LR environment variable.
    If unset, the default is 0.001.
    """
    return float(os.environ.get("LR", 0.001))


def get_hidden_size():
    """
    Return the hidden size for the SimpleGRU model.

    The value is read from the HIDDEN_SIZE environment variable.
    If unset, the default is 64.
    """
    return int(os.environ.get("HIDDEN_SIZE", 64))
