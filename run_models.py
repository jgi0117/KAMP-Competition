from __future__ import annotations

import argparse
from pathlib import Path

from kamp_models.runner import SUPPORTED_MODELS, load_config, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare KAMP electricity forecasting candidates."
    )
    parser.add_argument("--data", required=True, help="Preprocessed NPZ file")
    parser.add_argument(
        "--config", default="model_config.toml", help="Experiment TOML configuration"
    )
    parser.add_argument("--output", default="outputs/latest", help="Output directory")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        help="Subset of models to run; defaults to all configured models",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: one tree combination and one-step F0 foundation fine-tuning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_experiment(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        config=load_config(args.config),
        models=args.models,
        quick=args.quick,
    )
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison to {Path(args.output) / 'comparison.csv'}")


if __name__ == "__main__":
    main()
