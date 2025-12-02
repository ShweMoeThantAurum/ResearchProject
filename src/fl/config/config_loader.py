"""
Simple YAML configuration loader.
"""

import yaml


def load_config(path):
    """Load a YAML configuration file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
