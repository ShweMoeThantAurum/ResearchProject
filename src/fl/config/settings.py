"""
Default configuration settings for FL clients and server.
Values may be overridden by environment variables or YAML configs.
"""

import os


def _get_env(key, default):
    """Return value from environment or default."""
    return os.environ.get(key, default)


class Settings:
    """Central settings object for server and clients."""

    def __init__(self):
        self.dataset = _get_env("DATASET", "sz").lower()
        self.fl_mode = _get_env("FL_MODE", "AEFL").lower()
        self.fl_rounds = int(_get_env("FL_ROUNDS", "20"))
        self.batch_size = int(_get_env("BATCH_SIZE", "64"))
        self.local_epochs = int(_get_env("LOCAL_EPOCHS", "1"))
        self.lr = float(_get_env("LR", "0.001"))
        self.hidden_size = int(_get_env("HIDDEN_SIZE", "64"))

        # Energy
        self.device_power_watts = float(_get_env("DEVICE_POWER_WATTS", "3.5"))
        self.net_j_per_mb = float(_get_env("NET_J_PER_MB", "0.6"))

        # DP
        self.dp_enabled = _get_env("DP_ENABLED", "false").lower() == "true"
        self.dp_sigma = float(_get_env("DP_SIGMA", "0.01"))

        # Compression
        self.compression_enabled = _get_env("COMPRESSION_ENABLED", "false").lower() == "true"
        self.compression_mode = _get_env("COMPRESSION_MODE", "sparsify")
        self.compression_sparsity = float(_get_env("COMPRESSION_SPARSITY", "0.5"))
        self.compression_k_frac = float(_get_env("COMPRESSION_K_FRAC", "0.1"))

        # AWS
        self.aws_region = _get_env("AWS_REGION", "us-east-1")
        self.s3_bucket = _get_env("S3_BUCKET", "aefl")
        self.s3_prefix = f"experiments/{self.dataset}/{self.fl_mode}"


settings = Settings()
