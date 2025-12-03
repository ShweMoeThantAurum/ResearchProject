"""
Shared utilities for server-side FL logic:
- AEFL scoring helpers
- environment access
- directory helpers
- S3 path helpers
"""

import os

# ---------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------
ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


# ---------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------
def _get_env(name, default):
    v = os.environ.get(name, default)
    return v


def get_mode():
    return _get_env("FL_MODE", "AEFL").lower()


def get_fl_rounds():
    return int(_get_env("FL_ROUNDS", "20"))


def get_hidden_size():
    return int(_get_env("HIDDEN_SIZE", "64"))


def get_batch_size():
    return int(_get_env("BATCH_SIZE", "64"))


def get_s3_bucket():
    return _get_env("S3_BUCKET", "aefl")


def get_s3_prefix():
    """
    Correct S3 prefix:
    fl/<dataset>/<FL_MODE>   # Mode must be uppercase and no bucket-name duplication
    """
    dataset = _get_env("DATASET", "sz").lower()
    mode = _get_env("FL_MODE", "AEFL").upper()  # KEEP UPPERCASE
    return f"fl/{dataset}/{mode}"


# ---------------------------------------------------------------------
# Mode checks
# ---------------------------------------------------------------------
def is_aefl(mode):
    return mode.lower() == "aefl"


def is_fedavg(mode):
    return mode.lower() == "fedavg"


def is_fedprox(mode):
    return mode.lower() == "fedprox"


def is_localonly(mode):
    return mode.lower() == "localonly"


# ---------------------------------------------------------------------
# AEFL scoring utilities
# ---------------------------------------------------------------------
def compute_energy_score(total_energy):
    return 1.0 / (total_energy + 1e-9)


def compute_accuracy_score(loss):
    return 1.0 / (loss + 1e-9)


def compute_privacy_score(sigma):
    return sigma


def build_eval_metrics(loss, energy, sigma):
    """
    AEFL combined score = 0.4*accuracy + 0.4*energy + 0.2*privacy
    """
    a = compute_accuracy_score(loss)
    e = compute_energy_score(energy)
    p = compute_privacy_score(sigma)
    return 0.4 * a + 0.4 * e + 0.2 * p


def get_aefl_clients_per_round(scores, k=3):
    """
    Selects top-k clients based on AEFL scores.
    scores: dict(client → float)
    """
    sorted_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_pairs[:k]]


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
def get_proc_dir():
    return "datasets/processed"

# ---------------------------------------------------------------------
# Required by summary.py and server_main.py
# ---------------------------------------------------------------------
def get_dataset():
    """Returns dataset name from environment."""
    return _get_env("DATASET", "sz").lower()


def get_fl_mode():
    """Returns FL mode from environment (AEFL, FedAvg, etc.)."""
    return _get_env("FL_MODE", "AEFL").lower()

def get_results_bucket():
    """
    Returns the S3 bucket name used for storing client updates and results.
    Falls back to S3_BUCKET if no RESULTS_BUCKET is explicitly set.
    """
    return _get_env("RESULTS_BUCKET", _get_env("S3_BUCKET", "aefl"))
