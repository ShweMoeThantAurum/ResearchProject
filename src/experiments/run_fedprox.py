"""
Runs the FedProx baseline (for non-IID federated learning) using
the same unified federation engine. The behavior is controlled entirely
by the YAML config (strategy: fedprox, mu, etc.).
"""

from src.core.federation import run_federated


def run(config_path: str = "configs/fedprox_sz.yaml") -> None:
    """Execute FedProx baseline for the selected dataset."""
    run_federated(config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FedProx baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fedprox_sz.yaml",
        help=(
            "Path to FedProx config file. Examples:\n"
            "  configs/fedprox_sz.yaml\n"
            "  configs/fedprox_los.yaml\n"
            "  configs/fedprox_pems08.yaml"
        ),
    )
    args = parser.parse_args()
    run(args.config)
