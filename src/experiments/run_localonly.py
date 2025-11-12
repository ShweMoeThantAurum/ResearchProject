"""
Runs the Local-Only baseline (no federation, clients train independently
and only ensemble at evaluation).
"""

from src.core.federation import run_federated


def run(config_path: str = "configs/localonly_sz.yaml") -> None:
    """Execute Local-Only experiment for the selected dataset."""
    run_federated(config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Local-Only baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/localonly_sz.yaml",
        help=(
            "Path to Local-Only config file. Examples:\n"
            "  configs/localonly_sz.yaml\n"
            "  configs/localonly_los.yaml\n"
            "  configs/localonly_pems08.yaml"
        ),
    )
    args = parser.parse_args()
    run(args.config)
