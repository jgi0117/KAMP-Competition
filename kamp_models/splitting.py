from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.data_pipeline import EARLY_STOP_DAYS as LHS_EARLY_STOP_DAYS
from src.data_pipeline import FOLD_VAL_RANGES as _LHS_FOLD_VAL_RANGES
from src.data_pipeline import TEST_START as _LHS_TEST_START

from .data import DatasetBundle


LHS_TEST_START = np.datetime64(_LHS_TEST_START.to_datetime64())
LHS_FOLD_VAL_RANGES = tuple(
    (
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(1, "D"),
    )
    for start, end in _LHS_FOLD_VAL_RANGES
)


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train: np.ndarray
    early_stop: np.ndarray
    evaluate: np.ndarray


def combined_target_timestamps(data: DatasetBundle) -> np.ndarray | None:
    if data.train.target_timestamps is None or data.val.target_timestamps is None:
        return None
    return np.concatenate(
        [data.train.target_timestamps, data.val.target_timestamps]
    ).astype("datetime64[ns]")


def _lhs_folds(timestamps: np.ndarray) -> list[TemporalFold]:
    folds = []
    early_delta = np.timedelta64(LHS_EARLY_STOP_DAYS, "D")
    for number, (start, end) in enumerate(LHS_FOLD_VAL_RANGES, start=1):
        early_start = start - early_delta
        folds.append(
            TemporalFold(
                name=f"fold{number}",
                train=np.flatnonzero(timestamps < early_start),
                early_stop=np.flatnonzero(
                    (timestamps >= early_start) & (timestamps < start)
                ),
                evaluate=np.flatnonzero(
                    (timestamps >= start) & (timestamps < end)
                ),
            )
        )
    if any(
        min(len(fold.train), len(fold.early_stop), len(fold.evaluate)) == 0
        for fold in folds
    ):
        raise ValueError("The prepared timestamps do not cover all lhs date folds")
    return folds


def _fallback_folds(sample_count: int, n_splits: int) -> list[TemporalFold]:
    folds = []
    for number, (available_train, evaluate) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(np.arange(sample_count)), start=1
    ):
        early_size = max(1, min(len(available_train) // 10, len(evaluate)))
        folds.append(
            TemporalFold(
                name=f"fold{number}",
                train=available_train[:-early_size],
                early_stop=available_train[-early_size:],
                evaluate=evaluate,
            )
        )
    return folds


def get_temporal_folds(data: DatasetBundle, n_splits: int) -> list[TemporalFold]:
    sample_count = data.train.n_samples + data.val.n_samples
    timestamps = combined_target_timestamps(data)
    if timestamps is not None and n_splits == len(LHS_FOLD_VAL_RANGES):
        return _lhs_folds(timestamps)
    return _fallback_folds(sample_count, n_splits)
