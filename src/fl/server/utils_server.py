"""
Server-side helper utilities including directory management,
naming helpers, and round-specific file paths.
"""

import os
from src.fl.config.settings import settings


def ensure_dir(path):
    """Create directory if missing."""
    if not os.path.exists(path):
        os.makedirs(path)


def server_output_dir():
    """Base directory for server outputs for this experiment."""
    path = os.path.join("outputs", "models", settings.dataset, settings.fl_mode)
    ensure_dir(path)
    return path


def summary_dir():
    """Directory for experiment summaries."""
    path = os.path.join("outputs", "summaries", settings.dataset, settings.fl_mode)
    ensure_dir(path)
    return path


def round_model_path(round_id):
    """Path for storing a round-specific global model."""
    base = server_output_dir()
    return os.path.join(base, f"global_round_{round_id}.pt")


def summary_csv_path():
    """CSV summary path."""
    return os.path.join(summary_dir(), f"summary_{settings.fl_mode}.csv")


def final_metrics_path():
    """Final JSON metrics path."""
    return os.path.join(summary_dir(), f"final_metrics_{settings.fl_mode}.json")
