"""
YAML configuration loader for FL experiments.

Supports:
- Mode-specific default configs (aefl_default, fedavg, fedprox, localonly)
- Optional extra overlay configs (e.g., dp_on.yaml, compression_on.yaml)
- Applying nested YAML structure onto the central `settings` object.

Typical usage (server / client):
    from src.fl.config.config_loader import load_experiment_config
    from src.fl.config.settings import settings

    load_experiment_config(settings)

Environment variables:
    DATASET          - sz | los | pems08 (handled by Settings)
    FL_MODE          - aefl | fedavg | fedprox | localonly (handled by Settings)
    EXPERIMENT_CONFIG - optional explicit config path or name, e.g. "configs/aefl_default.yaml"
    EXTRA_CONFIG     - optional comma-separated overlays, e.g. "dp_on.yaml,compression_on.yaml"
"""

import os
import yaml

from .settings import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> str:
    """
    Return project root directory assuming this file lives in src/fl/config/.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _configs_dir() -> str:
    """Return absolute path to configs/ directory."""
    root = _project_root()
    return os.path.join(root, "configs")


def _resolve_config_path(name_or_path: str) -> str:
    """
    Resolve a config file path.

    - If `name_or_path` is absolute or contains a '/', treat as path.
    - Otherwise, assume it lives under configs/ directory.
    """
    if os.path.isabs(name_or_path) or "/" in name_or_path:
        path = name_or_path
    else:
        path = os.path.join(_configs_dir(), name_or_path)

    if not path.lower().endswith(".yaml"):
        path = path + ".yaml"

    return path


def _load_yaml(path: str) -> dict:
    """Load YAML file into a dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[config_loader] Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    print(f"[config_loader] Loaded YAML config: {path}")
    return data


# ---------------------------------------------------------------------------
# Mapping YAML → Settings
# ---------------------------------------------------------------------------

def _apply_config_dict(cfg: dict, s=settings) -> None:
    """
    Apply a parsed YAML config dict onto the Settings object `s`.

    Expected YAML structure (examples):

        mode: "AEFL"

        training:
          batch_size: 64
          local_epochs: 1
          learning_rate: 0.001
          hidden_size: 64

        fl:
          rounds: 20

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
    # Top-level mode override (maps to settings.fl_mode)
    mode = cfg.get("mode")
    if mode is not None:
        s.fl_mode = str(mode).lower()

    # training.*
    training = cfg.get("training", {})
    if "batch_size" in training:
        s.batch_size = int(training["batch_size"])
    if "local_epochs" in training:
        s.local_epochs = int(training["local_epochs"])
    if "learning_rate" in training:
        s.lr = float(training["learning_rate"])
    if "hidden_size" in training:
        s.hidden_size = int(training["hidden_size"])

    # fl.*
    fl_cfg = cfg.get("fl", {})
    if "rounds" in fl_cfg:
        s.fl_rounds = int(fl_cfg["rounds"])
    # NOTE: max_clients_per_round is handled on the server side via AEFL_MAX_CLIENTS,
    # so we intentionally do not wire it into Settings here.

    # energy.*
    energy = cfg.get("energy", {})
    if "device_power_watts" in energy:
        s.device_power_watts = float(energy["device_power_watts"])
    if "net_j_per_mb" in energy:
        s.net_j_per_mb = float(energy["net_j_per_mb"])

    # privacy.*
    privacy = cfg.get("privacy", {})
    if "dp_enabled" in privacy:
        s.dp_enabled = bool(privacy["dp_enabled"])
    if "dp_sigma" in privacy:
        s.dp_sigma = float(privacy["dp_sigma"])

    # compression.*
    comp = cfg.get("compression", {})
    if "enabled" in comp:
        s.compression_enabled = bool(comp["enabled"])
    if "mode" in comp:
        s.compression_mode = str(comp["mode"])
    if "sparsity" in comp:
        s.compression_sparsity = float(comp["sparsity"])
    if "k_frac" in comp:
        s.compression_k_frac = float(comp["k_frac"])


def apply_yaml_config(path: str, s=settings) -> None:
    """
    Backwards-compatible helper: load YAML and apply it to `s`.
    """
    cfg = _load_yaml(path)
    _apply_config_dict(cfg, s)


# ---------------------------------------------------------------------------
# Main entry: load_experiment_config
# ---------------------------------------------------------------------------

def _default_mode_config_name(mode: str) -> str:
    """
    Return the default config filename (without path) for a given FL mode.
    """
    mode = mode.lower()
    if mode == "aefl":
        return "aefl_default.yaml"
    if mode == "fedavg":
        return "fedavg.yaml"
    if mode == "fedprox":
        return "fedprox.yaml"
    if mode == "localonly":
        return "localonly.yaml"
    # Fallback: no default
    return ""


def load_experiment_config(s=settings) -> None:
    """
    Load experiment configuration based on:
      - settings.fl_mode (for default per-mode YAML)
      - EXPERIMENT_CONFIG (explicit config, overrides default selection)
      - EXTRA_CONFIG (comma-separated overlay configs)

    Order of application:
        1. Base config:
           - If EXPERIMENT_CONFIG set → use that
           - Else use per-mode default (aefl_default / fedavg / fedprox / localonly)
        2. Overlay configs from EXTRA_CONFIG (if any), in listed order.

    Each later config overrides earlier ones.
    """
    mode = s.fl_mode.lower()
    config_dir = _configs_dir()

    paths = []

    # 1) Explicit experiment config (takes precedence as base)
    explicit = os.environ.get("EXPERIMENT_CONFIG") or os.environ.get("CONFIG_FILE")
    if explicit:
        base_path = _resolve_config_path(explicit)
        paths.append(base_path)
    else:
        # 2) Mode-specific default
        default_name = _default_mode_config_name(mode)
        if default_name:
            base_path = os.path.join(config_dir, default_name)
            paths.append(base_path)

    # 3) Optional overlays (e.g., dp_on.yaml, compression_on.yaml)
    extra = os.environ.get("EXTRA_CONFIG", "").strip()
    if extra:
        for token in extra.split(","):
            token = token.strip()
            if not token:
                continue
            overlay_path = _resolve_config_path(token)
            paths.append(overlay_path)

    # If nothing to load, just return gracefully
    if not paths:
        print("[config_loader] No YAML configs selected (using pure env/default Settings).")
        return

    # Apply all configs in order
    for p in paths:
        try:
            apply_yaml_config(p, s)
        except FileNotFoundError as e:
            print(f"[config_loader] WARNING: {e}")

    # Final debug print
    print(
        "[config_loader] Final settings after YAML load: "
        f"mode={s.fl_mode}, dataset={s.dataset}, "
        f"rounds={s.fl_rounds}, batch_size={s.batch_size}, "
        f"dp_enabled={s.dp_enabled}, compression_enabled={s.compression_enabled}"
    )

