"""
YAML configuration loader.

Goal:
- Make experiments reproducible and easy to describe.
- Allow mode-specific configs (AEFL, FedAvg, FedProx, LocalOnly).
- Allow optional overlays (e.g. dp_on.yaml, compression_on.yaml).

Precedence:
  1. Python defaults in Settings (settings.py)
  2. YAML configs (baseline + overlays)
  3. Environment variables (if you set them explicitly)

At runtime:
- Map YAML keys into both `settings` attributes AND environment variables,
  so legacy code that reads os.environ continues to work.
"""

import os
import yaml
from .settings import settings

CONFIG_DIR = "configs"


def _set_attr_and_env(attr_name, env_name, value, cast=None):
    """
    Helper to keep settings.<attr> and os.environ[ENV] consistent.
    Optionally casts value.
    """
    if cast is not None:
        value = cast(value)
    setattr(settings, attr_name, value)
    if env_name is not None:
        os.environ[env_name] = str(value)


def _apply_root_level(cfg: dict):
    """Apply root-level keys like 'mode' and 'dataset'."""
    if "mode" in cfg:
        mode = str(cfg["mode"]).lower()
        _set_attr_and_env("fl_mode", "FL_MODE", mode)

    if "dataset" in cfg:
        ds = str(cfg["dataset"]).lower()
        _set_attr_and_env("dataset", "DATASET", ds)


def _apply_training(cfg: dict):
    """Map training.* into settings + env."""
    tr = cfg.get("training", {})
    if not tr:
        return

    if "batch_size" in tr:
        _set_attr_and_env("batch_size", "BATCH_SIZE", tr["batch_size"], int)
    if "local_epochs" in tr:
        _set_attr_and_env("local_epochs", "LOCAL_EPOCHS", tr["local_epochs"], int)
    if "learning_rate" in tr:
        _set_attr_and_env("lr", "LR", tr["learning_rate"], float)
    if "hidden_size" in tr:
        _set_attr_and_env("hidden_size", "HIDDEN_SIZE", tr["hidden_size"], int)


def _apply_fl(cfg: dict):
    """Map fl.* into settings + env (rounds, max_clients_per_round)."""
    fl = cfg.get("fl", {})
    if not fl:
        return

    if "rounds" in fl:
        _set_attr_and_env("fl_rounds", "FL_ROUNDS", fl["rounds"], int)

    # Specific to AEFL
    if "max_clients_per_round" in fl:
        # Used by utils_server.get_aefl_max_clients via AEFL_MAX_CLIENTS
        os.environ["AEFL_MAX_CLIENTS"] = str(int(fl["max_clients_per_round"]))


def _apply_energy(cfg: dict):
    """Map energy.* into settings + env."""
    en = cfg.get("energy", {})
    if not en:
        return

    if "device_power_watts" in en:
        _set_attr_and_env(
            "device_power_watts", "DEVICE_POWER_WATTS", en["device_power_watts"], float
        )
    if "net_j_per_mb" in en:
        _set_attr_and_env("net_j_per_mb", "NET_J_PER_MB", en["net_j_per_mb"], float)


def _apply_privacy(cfg: dict):
    """Map privacy.* into settings + env."""
    pr = cfg.get("privacy", {})
    if not pr:
        return

    if "dp_enabled" in pr:
        enabled = bool(pr["dp_enabled"])
        _set_attr_and_env("dp_enabled", "DP_ENABLED", str(enabled).lower())
        # settings.dp_enabled is a bool, env is "true"/"false"
        settings.dp_enabled = enabled

    if "dp_sigma" in pr:
        _set_attr_and_env("dp_sigma", "DP_SIGMA", pr["dp_sigma"], float)


def _apply_compression(cfg: dict):
    """Map compression.* into settings + env."""
    comp = cfg.get("compression", {})
    if not comp:
        return

    if "enabled" in comp:
        enabled = bool(comp["enabled"])
        _set_attr_and_env(
            "compression_enabled", "COMPRESSION_ENABLED", str(enabled).lower()
        )
        settings.compression_enabled = enabled

    if "mode" in comp:
        _set_attr_and_env(
            "compression_mode", "COMPRESSION_MODE", comp["mode"], str
        )
    if "sparsity" in comp:
        _set_attr_and_env(
            "compression_sparsity",
            "COMPRESSION_SPARSITY",
            comp["sparsity"],
            float,
        )
    if "k_frac" in comp:
        _set_attr_and_env(
            "compression_k_frac", "COMPRESSION_K_FRAC", comp["k_frac"], float
        )


def apply_yaml_config(path: str):
    """
    Load and apply configuration overrides from YAML.

    Supported structure (any subset is fine):

        mode: "AEFL"
        dataset: "sz"

        training:
          batch_size: 64
          local_epochs: 1
          learning_rate: 0.001
          hidden_size: 64

        fl:
          rounds: 20
          max_clients_per_round: 3

        energy:
          device_power_watts: 3.5
          net_j_per_mb: 0.6

        privacy:
          dp_enabled: true
          dp_sigma: 0.05

        compression:
          enabled: true
          mode: "sparsify"
          sparsity: 0.5
          k_frac: 0.1
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping object.")

    _apply_root_level(cfg)
    _apply_training(cfg)
    _apply_fl(cfg)
    _apply_energy(cfg)
    _apply_privacy(cfg)
    _apply_compression(cfg)

    print(f"[config_loader] Loaded config overrides from {path}")


def _resolve_config_paths_from_env():
    """
    Resolve list of YAML config paths from FL_CONFIG env var.

    Examples:
        FL_CONFIG=aefl_default
        FL_CONFIG=aefl_default,dp_on,compression_on
        FL_CONFIG=configs/aefl_default.yaml

    Returns a list of paths (may be empty).
    """
    cfg_str = os.environ.get("FL_CONFIG", "").strip()
    if not cfg_str:
        return []

    names = [c.strip() for c in cfg_str.split(",") if c.strip()]
    paths = []

    for name in names:
        # Allow full path
        if name.endswith(".yaml") and os.path.exists(name):
            paths.append(name)
            continue

        # Otherwise assume it's under CONFIG_DIR
        fname = name if name.endswith(".yaml") else name + ".yaml"
        candidate = os.path.join(CONFIG_DIR, fname)
        paths.append(candidate)

    return paths


def _resolve_default_mode_config():
    """
    If FL_CONFIG is not set, infer a baseline config based on current mode.

    AEFL   -> configs/aefl_default.yaml
    FedAvg -> configs/fedavg.yaml
    FedProx-> configs/fedprox.yaml
    LocalOnly -> configs/localonly.yaml

    Returns a list with 0 or 1 paths.
    """
    mode = settings.fl_mode.lower()
    mapping = {
        "aefl": "aefl_default.yaml",
        "fedavg": "fedavg.yaml",
        "fedprox": "fedprox.yaml",
        "localonly": "localonly.yaml",
    }
    fname = mapping.get(mode)
    if not fname:
        return []

    return [os.path.join(CONFIG_DIR, fname)]


def load_experiment_config():
    """
    Main entry used by server/client.

    Strategy:
      - If FL_CONFIG is set  -> use those YAML(s) in order.
      - Else                 -> load mode-specific baseline YAML if it exists.
      - If a file is missing -> print a warning, but do not crash.

    This should be called once at the start of:
      - src.fl.server.server_main.main()
      - src.fl.client.client_main.main()
    """
    # 1) Explicit FL_CONFIG overrides everything
    paths = _resolve_config_paths_from_env()

    # 2) If nothing specified, try mode-based default
    if not paths:
        paths = _resolve_default_mode_config()

    if not paths:
        print("[config_loader] No YAML configs applied (using defaults/env only).")
        return

    for path in paths:
        try:
            apply_yaml_config(path)
        except FileNotFoundError as e:
            print(f"[config_loader] WARNING: {e}")
        except Exception as e:
            print(f"[config_loader] WARNING: Failed to apply {path}: {e}")
