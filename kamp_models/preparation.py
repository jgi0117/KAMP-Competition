from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from .schema import DATE_COLUMN, FEATURE_COLUMNS, HOUR_COLUMN, TARGET_COLUMN
from .splitting import LHS_EARLY_STOP_DAYS, LHS_TEST_START


@dataclass(frozen=True)
class PreparationReport:
    source_csv: str
    output_npz: str
    source_kind: str
    source_rows: int
    context_length: int
    horizon: int
    feature_count: int
    split_strategy: str
    window_count: int
    sample_count: int
    train_samples: int
    val_samples: int
    test_samples: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    cleaned_reference: str | None
    cleaned_reference_match: bool | None
    generated_cleaned_sha256: str | None
    reference_cleaned_sha256: str | None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _normalise_cleaned_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if DATE_COLUMN not in result or HOUR_COLUMN not in result:
        raise KeyError(f"Cleaned data requires {DATE_COLUMN!r} and {HOUR_COLUMN!r}")
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN])
    result[HOUR_COLUMN] = result[HOUR_COLUMN].astype(np.int64)
    timestamp = result[DATE_COLUMN] + pd.to_timedelta(result[HOUR_COLUMN], unit="h")
    if timestamp.duplicated().any():
        raise ValueError("Cleaned data contains duplicate hourly timestamps")
    if not timestamp.is_monotonic_increasing:
        order = np.argsort(timestamp.to_numpy(), kind="stable")
        result = result.iloc[order].reset_index(drop=True)
        timestamp = timestamp.iloc[order].reset_index(drop=True)
    gaps = timestamp.diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Cleaned data must be a continuous hourly series")
    if TARGET_COLUMN not in result:
        raise KeyError(f"Cleaned data requires target column {TARGET_COLUMN!r}")
    if result.isna().any().any():
        raise ValueError("Cleaned data still contains missing values")
    return result


def load_or_clean_csv(source_csv: str | Path) -> tuple[pd.DataFrame, str]:
    source = Path(source_csv)
    frame = _read_csv(source)
    if TARGET_COLUMN in frame.columns:
        return _normalise_cleaned_frame(frame), "cleaned"

    from src.preprocessing import run_preprocessing_pipeline

    cleaned = run_preprocessing_pipeline(source, save_summary=False)
    return _normalise_cleaned_frame(cleaned), "raw_kjh_pipeline"


def verify_cleaned_reference(
    cleaned: pd.DataFrame,
    reference_csv: str | Path,
) -> None:
    reference = _normalise_cleaned_frame(_read_csv(Path(reference_csv)))
    assert_frame_equal(
        cleaned.reset_index(drop=True),
        reference.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )


def write_cleaned_csv(cleaned: pd.DataFrame, output_csv: str | Path) -> Path:
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\r\n",
    )
    return destination


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(value <= 0 for value in ratios):
        raise ValueError("train/val/test ratios must be positive")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")


def make_supervised_windows(
    cleaned: pd.DataFrame,
    *,
    context_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if context_length <= 0 or horizon <= 0:
        raise ValueError("context_length and horizon must be positive")
    if len(cleaned) < context_length + horizon:
        raise ValueError("Not enough rows to create one forecasting window")

    feature_columns = list(FEATURE_COLUMNS)
    missing_features = [column for column in feature_columns if column not in cleaned]
    if missing_features:
        raise KeyError(f"Cleaned data is missing canonical features: {missing_features}")
    feature_values = cleaned[feature_columns].astype(np.float32).to_numpy()
    target_values = cleaned[TARGET_COLUMN].astype(np.float32).to_numpy()

    contexts = np.lib.stride_tricks.sliding_window_view(
        feature_values, window_shape=context_length, axis=0
    )
    contexts = np.moveaxis(contexts, -1, 1)
    sample_count = len(cleaned) - context_length - horizon + 1
    X = np.ascontiguousarray(contexts[:sample_count], dtype=np.float32)
    target_history = np.ascontiguousarray(X[:, :, 0], dtype=np.float32)
    y = np.ascontiguousarray(
        np.lib.stride_tricks.sliding_window_view(
            target_values[context_length:], window_shape=horizon
        )[:sample_count],
        dtype=np.float32,
    )
    timestamps = (
        cleaned[DATE_COLUMN] + pd.to_timedelta(cleaned[HOUR_COLUMN], unit="h")
    )
    target_timestamps = timestamps.iloc[
        context_length : context_length + sample_count
    ].dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(dtype="U19")
    return X, y, target_history, np.asarray(feature_columns, dtype="U64"), target_timestamps


def prepare_csv_to_npz(
    source_csv: str | Path,
    output_npz: str | Path,
    *,
    context_length: int = 168,
    horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    split_strategy: str = "lhs",
    exclude_shutdown_targets: bool = True,
    cleaned_reference: str | Path | None = None,
    report_json: str | Path | None = None,
) -> PreparationReport:
    if split_strategy not in {"lhs", "ratio"}:
        raise ValueError("split_strategy must be 'lhs' or 'ratio'")
    if split_strategy == "ratio":
        _validate_ratios(train_ratio, val_ratio, test_ratio)
    source = Path(source_csv)
    destination = Path(output_npz)
    cleaned, source_kind = load_or_clean_csv(source)

    reference_path = Path(cleaned_reference) if cleaned_reference is not None else None
    reference_match: bool | None = None
    generated_hash: str | None = None
    reference_hash: str | None = None
    if reference_path is not None:
        verify_cleaned_reference(cleaned, reference_path)
        reference_match = True
        reference_hash = _sha256(reference_path)
        generated_csv = destination.with_suffix(".cleaned.csv")
        write_cleaned_csv(cleaned, generated_csv)
        generated_hash = _sha256(generated_csv)
        if generated_hash != reference_hash:
            raise AssertionError(
                "Cleaned values match, but serialized CSV bytes differ from the reference"
            )

    X, y, target_history, feature_names, target_timestamps = make_supervised_windows(
        cleaned,
        context_length=context_length,
        horizon=horizon,
    )
    window_count = len(y)
    if split_strategy == "ratio":
        train_end = int(window_count * train_ratio)
        val_end = train_end + int(window_count * val_ratio)
        boundaries: dict[str, slice | np.ndarray] = {
            "train": slice(0, train_end),
            "val": slice(train_end, val_end),
            "test": slice(val_end, window_count),
        }
    else:
        timestamps = target_timestamps.astype("datetime64[ns]")
        val_start = LHS_TEST_START - np.timedelta64(LHS_EARLY_STOP_DAYS, "D")
        keep = np.ones(window_count, dtype=bool)
        if exclude_shutdown_targets:
            shutdown = cleaned.get(
                "공장_셧다운_여부", cleaned[TARGET_COLUMN].eq(0)
            ).astype(bool).to_numpy()
            shutdown_targets = np.lib.stride_tricks.sliding_window_view(
                shutdown[context_length:], window_shape=horizon
            )[:window_count]
            keep &= ~shutdown_targets.any(axis=1)
        boundaries = {
            "train": np.flatnonzero(keep & (timestamps < val_start)),
            "val": np.flatnonzero(
                keep & (timestamps >= val_start) & (timestamps < LHS_TEST_START)
            ),
            "test": np.flatnonzero(keep & (timestamps >= LHS_TEST_START)),
        }
    split_counts = {
        name: len(np.arange(window_count)[boundary])
        for name, boundary in boundaries.items()
    }
    if any(count == 0 for count in split_counts.values()):
        raise ValueError("Split strategy produces an empty train, validation, or test split")
    sample_count = sum(split_counts.values())
    payload: dict[str, np.ndarray] = {
        "feature_names": feature_names,
        "split_strategy": np.asarray(split_strategy),
        "split_ratios": np.asarray(
            [split_counts[name] / sample_count for name in ("train", "val", "test")],
            dtype=np.float32,
        ),
    }
    for split_name, boundary in boundaries.items():
        payload[f"X_{split_name}"] = X[boundary]
        payload[f"y_{split_name}"] = y[boundary]
        payload[f"target_history_{split_name}"] = target_history[boundary]
        payload[f"target_timestamp_{split_name}"] = target_timestamps[boundary]

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    report = PreparationReport(
        source_csv=str(source.resolve()),
        output_npz=str(destination.resolve()),
        source_kind=source_kind,
        source_rows=len(cleaned),
        context_length=context_length,
        horizon=horizon,
        feature_count=X.shape[2],
        split_strategy=split_strategy,
        window_count=window_count,
        sample_count=sample_count,
        train_samples=split_counts["train"],
        val_samples=split_counts["val"],
        test_samples=split_counts["test"],
        train_ratio=split_counts["train"] / sample_count,
        val_ratio=split_counts["val"] / sample_count,
        test_ratio=split_counts["test"] / sample_count,
        cleaned_reference=(str(reference_path.resolve()) if reference_path else None),
        cleaned_reference_match=reference_match,
        generated_cleaned_sha256=generated_hash,
        reference_cleaned_sha256=reference_hash,
    )
    report_path = (
        Path(report_json)
        if report_json is not None
        else destination.with_suffix(".report.json")
    )
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
