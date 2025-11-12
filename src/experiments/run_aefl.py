"""
Runs AEFL (Adaptive Energy-Aware Federated Learning) experiments
across different traffic datasets using the unified federation engine.
"""

from src.core.federation import run_federated


def run(config_path: str = "configs/aefl_sz.yaml") -> None:
    """Execute AEFL experiment for the selected dataset."""
    run_federated(config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AEFL experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/aefl_sz.yaml",
        help=(
            "Path to AEFL config file. Examples:\n"
            "  configs/aefl_sz.yaml\n"
            "  configs/aefl_los.yaml\n"
            "  configs/aefl_pems08.yaml"
        ),
    )
    args = parser.parse_args()
    run(args.config)
