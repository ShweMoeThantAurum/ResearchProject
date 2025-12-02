"""
YAML configuration loader and merged config manager.
Handles:
- base config (AEFL, FedAvg, ...)
- overrides (DP, compression sweeps)
- runtime environment overrides
"""

import yaml
from pathlib import Path
from .settings import CONFIG_DIR, DEFAULT_CONFIG


def load_yaml(path: Path):
    """Load YAML file and return dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_dict(base, override):
    """Recursively merge dictionaries (override takes priority)."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            base[k] = merge_dict(base[k], v)
        else:
            base[k] = v
    return base


def load_config(mode: str, override_dp=False, override_compression=False):
    """
    Load the base configuration (fedavg.yaml, aefl_default.yaml, ...)
    and optionally apply DP/compression overrides.
    """

    base_path = CONFIG_DIR / f"{mode.lower()}.yaml"

    if not base_path.exists():
        base_path = DEFAULT_CONFIG

    config = load_yaml(base_path)

    # DP override
    if override_dp:
        dp_cfg = load_yaml(CONFIG_DIR / "dp_on.yaml")
        config = merge_dict(config, dp_cfg)

    # Compression override
    if override_compression:
        comp_cfg = load_yaml(CONFIG_DIR / "compression_on.yaml")
        config = merge_dict(config, comp_cfg)

    return config
