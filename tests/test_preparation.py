from __future__ import annotations

import numpy as np
import pandas as pd

from kamp_models.data import load_preprocessed_npz
from kamp_models.preparation import (
    TARGET_COLUMN,
    make_supervised_windows,
    prepare_csv_to_npz,
    write_cleaned_csv,
)


def _cleaned_frame(row_count: int = 20) -> pd.DataFrame:
    timestamps = pd.date_range("2021-01-01", periods=row_count, freq="h")
    return pd.DataFrame(
        {
            "날짜": timestamps.normalize(),
            "시간": timestamps.hour,
            TARGET_COLUMN: np.arange(row_count, dtype=np.float64),
            "생산량": np.arange(row_count, dtype=np.int64) * 10,
            "공장_셧다운_여부": np.zeros(row_count, dtype=bool),
        }
    )


def test_windows_predict_the_next_hour_average():
    X, y, history, feature_names, timestamps = make_supervised_windows(
        _cleaned_frame(), context_length=4, horizon=1
    )

    assert X.shape == (16, 4, 4)
    assert y.shape == (16, 1)
    assert feature_names[0] == TARGET_COLUMN
    np.testing.assert_array_equal(history[0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(y[0], [4.0])
    assert timestamps[0] == "2021-01-01T04:00:00"


def test_prepare_csv_creates_chronological_7_2_1_npz(tmp_path):
    source = write_cleaned_csv(_cleaned_frame(), tmp_path / "cleaned.csv")
    output = tmp_path / "prepared.npz"

    report = prepare_csv_to_npz(
        source,
        output,
        context_length=4,
        horizon=1,
        cleaned_reference=source,
    )
    bundle = load_preprocessed_npz(output, context_length=4, horizon=1)

    assert report.cleaned_reference_match is True
    assert (report.train_samples, report.val_samples, report.test_samples) == (11, 3, 2)
    assert (bundle.train.n_samples, bundle.val.n_samples, bundle.test.n_samples) == (
        11,
        3,
        2,
    )
    assert bundle.train.y[-1, 0] < bundle.val.y[0, 0] < bundle.test.y[0, 0]
