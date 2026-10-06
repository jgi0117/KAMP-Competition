from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from kamp_models.preparation import prepare_csv_to_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the KJH preprocessing pipeline and create 7:2:1 model splits."
    )
    parser.add_argument("--input", required=True, help="Raw or cleaned OKM CSV")
    parser.add_argument(
        "--output",
        default="outputs/prepared/model_input_7_2_1.npz",
        help="Destination NPZ",
    )
    parser.add_argument(
        "--cleaned-reference",
        help="Optional okm_cleaned_2021.csv for exact verification",
    )
    parser.add_argument("--context-length", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_csv_to_npz(
        Path(args.input),
        Path(args.output),
        context_length=args.context_length,
        horizon=args.horizon,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        cleaned_reference=(
            Path(args.cleaned_reference) if args.cleaned_reference else None
        ),
    )
    for key, value in asdict(report).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
