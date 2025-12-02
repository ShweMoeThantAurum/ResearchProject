"""
Helpers for reading configuration values from environment variables.
"""

import os


def get_dataset():
    """Return the active dataset name."""
    return os.environ.get("DATASET", "sz")


def get_proc_dir():
    """Return the directory containing processed data."""
    dataset = get_dataset()
    return "datasets/processed/{}".format(dataset)


def get_raw_dir():
    """Return the directory containing raw data."""
    dataset = get_dataset()
    return "datasets/raw/{}".format(dataset)


def get_fl_mode():
    """Return the active federated learning mode."""
    return os.environ.get("FL_MODE", "AEFL")


def get_fl_rounds():
    """Return the number of federated learning rounds."""
    value = os.environ.get("FL_ROUNDS", "20")
    return int(value)


def get_batch_size():
    """Return the local training batch size."""
    value = os.environ.get("BATCH_SIZE", "64")
    return int(value)


def get_learning_rate():
    """Return the learning rate for local optimisers."""
    value = os.environ.get("LR", "0.001")
    return float(value)


def get_hidden_size():
    """Return the GRU hidden size."""
    value = os.environ.get("HIDDEN_SIZE", "64")
    return int(value)


def get_device_power():
    """Return the assumed device power in watts."""
    value = os.environ.get("DEVICE_POWER_WATTS", "3.5")
    return float(value)


def get_comm_energy_per_mb():
    """Return the communication energy in joules per megabyte."""
    value = os.environ.get("NET_J_PER_MB", "0.6")
    return float(value)


def dp_enabled():
    """Return True if differential privacy is enabled."""
    return os.environ.get("DP_ENABLED", "false").lower() == "true"


def get_dp_sigma():
    """Return the standard deviation of DP noise."""
    value = os.environ.get("DP_SIGMA", "0.01")
    return float(value)


def compression_enabled():
    """Return True if model compression is enabled."""
    return os.environ.get("COMPRESSION_ENABLED", "false").lower() == "true"


def get_compression_mode():
    """Return the selected compression mode."""
    return os.environ.get("COMPRESSION_MODE", "sparsify")


def get_compression_sparsity():
    """Return the sparsity level for magnitude pruning."""
    value = os.environ.get("COMPRESSION_SPARSITY", "0.5")
    return float(value)


def get_compression_k_frac():
    """Return the retained fraction for top-k compression."""
    value = os.environ.get("COMPRESSION_K_FRAC", "0.1")
    return float(value)


def get_s3_bucket():
    """Return the name of the S3 bucket."""
    return os.environ.get("S3_BUCKET", "aefl")


def get_region():
    """Return the AWS region name."""
    return os.environ.get("AWS_REGION", "us-east-1")
