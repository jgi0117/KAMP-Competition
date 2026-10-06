from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    X: np.ndarray
    y: np.ndarray
    target_history: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        return int(self.y.shape[0])


@dataclass(frozen=True)
class DatasetBundle:
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    context_length: int
    horizon: int


def _as_float_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _normalise_target(value: np.ndarray, name: str, horizon: int) -> np.ndarray:
    array = _as_float_array(value, name)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, horizon]")
    if array.shape[1] != horizon:
        raise ValueError(
            f"{name} horizon is {array.shape[1]}, but configuration expects {horizon}"
        )
    return array


def _infer_target_history(
    X: np.ndarray,
    context_length: int,
    target_feature_index: int,
) -> np.ndarray | None:
    if X.ndim == 2 and X.shape[1] == context_length:
        return X
    if X.ndim == 3 and X.shape[1] == context_length:
        if not 0 <= target_feature_index < X.shape[2]:
            raise ValueError(
                f"target_feature_index={target_feature_index} is outside X feature dimension "
                f"({X.shape[2]})"
            )
        return X[:, :, target_feature_index]
    return None


def _load_split(
    archive: np.lib.npyio.NpzFile,
    split: str,
    context_length: int,
    horizon: int,
    target_feature_index: int,
) -> DatasetSplit:
    x_key = f"X_{split}"
    y_key = f"y_{split}"
    missing = [key for key in (x_key, y_key) if key not in archive.files]
    if missing:
        raise KeyError(f"Missing keys in NPZ: {', '.join(missing)}")

    X = _as_float_array(archive[x_key], x_key)
    y = _normalise_target(archive[y_key], y_key, horizon)
    if X.ndim < 2:
        raise ValueError(f"{x_key} must have at least two dimensions")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{x_key} and {y_key} have different sample counts")

    history_key = f"target_history_{split}"
    if history_key in archive.files:
        history = _as_float_array(archive[history_key], history_key)
        if history.shape != (X.shape[0], context_length):
            raise ValueError(
                f"{history_key} must have shape [{X.shape[0]}, {context_length}]"
            )
    else:
        history = _infer_target_history(X, context_length, target_feature_index)

    return DatasetSplit(X=X, y=y, target_history=history)


def load_preprocessed_npz(
    path: str | Path,
    *,
    context_length: int = 168,
    horizon: int = 24,
    target_feature_index: int = 0,
) -> DatasetBundle:
    """Load chronological train/validation/test windows prepared by the data team.

    Required keys are X_train, y_train, X_val, y_val, X_test and y_test.
    Foundation models additionally need target_history_<split>, unless it can be
    inferred from X shaped [N, context] or [N, context, features].
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    with np.load(source, allow_pickle=False) as archive:
        splits = {
            name: _load_split(
                archive,
                name,
                context_length,
                horizon,
                target_feature_index,
            )
            for name in ("train", "val", "test")
        }

    return DatasetBundle(
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        context_length=context_length,
        horizon=horizon,
    )


def flatten_features(X: np.ndarray) -> np.ndarray:
    """Convert window features to the 2-D matrix expected by tree models."""

    return np.asarray(X, dtype=np.float32).reshape(X.shape[0], -1)


def require_target_history(split: DatasetSplit, split_name: str) -> np.ndarray:
    if split.target_history is None:
        raise ValueError(
            f"Foundation models need target_history_{split_name} in the NPZ when "
            f"X_{split_name} is not [N, context] or [N, context, features]."
        )
    return split.target_history

