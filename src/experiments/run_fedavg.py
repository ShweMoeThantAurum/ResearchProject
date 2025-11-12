"""
Runs the standard FedAvg baseline using the unified federation engine.
"""

from src.core.federation import run_federated


def run(config_path: str = "configs/fedavg_sz.yaml") -> None:
    """Execute FedAvg baseline for the selected dataset."""
    run_federated(config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FedAvg baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fedavg_sz.yaml",
        help=(
            "Path to FedAvg config file. Examples:\n"
            "  configs/fedavg_sz.yaml\n"
            "  configs/fedavg_los.yaml\n"
            "  configs/fedavg_pems08.yaml"
        ),
    )
    args = parser.parse_args()
    run(args.config)
