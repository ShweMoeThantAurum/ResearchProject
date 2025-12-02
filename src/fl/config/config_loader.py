"""
YAML configuration loader.
Loads override settings for modes like FedAvg, FedProx, AEFL, DP-on, compression-on.
"""

import os
import yaml
from .settings import settings


def apply_yaml_config(path):
    """Load and apply configuration overrides from YAML."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            print(f"[config_loader] Warning: Unknown config key '{key}'")

    print(f"[config_loader] Loaded config overrides from {path}")
