"""
Global configuration paths and constant utilities.
"""

import os
from pathlib import Path

# Root directory (detected dynamically)
ROOT = Path(__file__).resolve().parents[3]

CONFIG_DIR = ROOT / "configs"
DATASET_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "outputs"
EXPERIMENT_DIR = ROOT / "experiments"

DEFAULT_CONFIG = CONFIG_DIR / "aefl_default.yaml"

# Resolves dataset from environment variable
def get_dataset_name():
    return os.environ.get("DATASET", "sz")
