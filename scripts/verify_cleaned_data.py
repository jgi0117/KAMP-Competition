"""Recreate the cleaned CSV from the original data and compare all cells."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import run_preprocessing_pipeline  # noqa: E402


def main():
    recreated = run_preprocessing_pipeline()
    # The supplied CSV serializes this datetime column as YYYY-MM-DD text.
    recreated["날짜"] = pd.to_datetime(recreated["날짜"]).dt.strftime("%Y-%m-%d")
    supplied = pd.read_csv(ROOT / "data/okm_cleaned_2021.csv", encoding="utf-8-sig")
    pd.testing.assert_frame_equal(
        recreated, supplied, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10
    )
    print(f"정제 데이터 일치: {len(supplied)}행 × {len(supplied.columns)}열")


if __name__ == "__main__":
    main()
