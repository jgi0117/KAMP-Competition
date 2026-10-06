from __future__ import annotations

import argparse
from pathlib import Path

from kamp_models.preparation import prepare_csv_to_npz
from kamp_models.runner import SUPPORTED_MODELS, load_config, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare KAMP electricity forecasting candidates."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Prepared NPZ, kjh raw CSV, or cleaned CSV file",
    )
    parser.add_argument(
        "--config", default="model_config.toml", help="Experiment TOML configuration"
    )
    parser.add_argument("--output", default="outputs/latest", help="Output directory")
    parser.add_argument(
        "--cleaned-reference",
        help="Optional cleaned CSV used to verify the kjh preprocessing output",
    )
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
    config = load_config(args.config)
    data_path = Path(args.data)
    output_dir = Path(args.output)
    if data_path.suffix.lower() == ".csv":
        data_config = config.get("data", {})
        experiment = config.get("experiment", {})
        reference = Path(args.cleaned_reference) if args.cleaned_reference else None
        if reference is None:
            candidate = data_path.with_name("okm_cleaned_2021.csv")
            if candidate != data_path and candidate.is_file():
                reference = candidate
        prepared_path = output_dir / "prepared" / "model_input_7_2_1.npz"
        report = prepare_csv_to_npz(
            data_path,
            prepared_path,
            context_length=int(experiment.get("context_length", 168)),
            horizon=int(experiment.get("horizon", 1)),
            train_ratio=float(data_config.get("train_ratio", 0.7)),
            val_ratio=float(data_config.get("val_ratio", 0.2)),
            test_ratio=float(data_config.get("test_ratio", 0.1)),
            cleaned_reference=reference,
        )
        data_path = prepared_path
        print(
            "Prepared chronological splits: "
            f"train={report.train_samples}, val={report.val_samples}, "
            f"test={report.test_samples}"
        )
        if report.cleaned_reference_match:
            print("KJH preprocessing output exactly matches the cleaned CSV reference.")
    comparison = run_experiment(
        data_path=data_path,
        output_dir=output_dir,
        config=config,
        models=args.models,
        quick=args.quick,
    )
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison to {output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
