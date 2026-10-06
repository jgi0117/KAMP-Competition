from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

from .data import DatasetBundle, flatten_features
from .metrics import regression_metrics


@dataclass
class TreeRunResult:
    model_name: str
    predictions: np.ndarray
    best_params: dict[str, Any]
    search_results: pd.DataFrame
    train_seconds: float
    inference_seconds: float
    model_path: Path


def run_tree_search(
    model_name: str,
    data: DatasetBundle,
    output_dir: Path,
    *,
    seed: int,
    cv_splits: int,
    n_jobs: int,
    search_profile: str = "full",
    param_grid: dict[str, list[Any]] | None = None,
    full_grid: dict[str, list[Any]],
    quick_grid: dict[str, list[Any]],
    estimator_factory: Callable[[dict[str, Any], int, int], Any],
) -> TreeRunResult:
    X_search = flatten_features(np.concatenate([data.train.X, data.val.X], axis=0))
    y_search = np.concatenate([data.train.y, data.val.y], axis=0)
    if len(X_search) <= cv_splits:
        raise ValueError(
            f"Need more than {cv_splits} train+validation samples for TimeSeriesSplit"
        )

    if param_grid is None:
        param_grid = full_grid if search_profile == "full" else quick_grid

    splitter = TimeSeriesSplit(n_splits=cv_splits)
    rows: list[dict[str, Any]] = []
    search_started = time.perf_counter()

    for candidate_index, params in enumerate(ParameterGrid(param_grid), start=1):
        fold_metrics: list[dict[str, float]] = []
        fold_started = time.perf_counter()
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X_search), start=1):
            estimator = estimator_factory(params, seed, n_jobs)
            estimator.fit(X_search[train_idx], y_search[train_idx])
            prediction = estimator.predict(X_search[val_idx])
            metrics = regression_metrics(y_search[val_idx], prediction)
            fold_metrics.append(metrics)
            rows.append(
                {
                    "candidate": candidate_index,
                    "fold": fold,
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    **metrics,
                }
            )

        rows.append(
            {
                "candidate": candidate_index,
                "fold": "mean",
                "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "rmse": float(np.mean([item["rmse"] for item in fold_metrics])),
                "mae": float(np.mean([item["mae"] for item in fold_metrics])),
                "r2": float(np.mean([item["r2"] for item in fold_metrics])),
                "elapsed_seconds": time.perf_counter() - fold_started,
            }
        )

    results = pd.DataFrame(rows)
    mean_rows = results[results["fold"] == "mean"].copy()
    winner = mean_rows.sort_values(
        ["rmse", "mae", "r2"], ascending=[True, True, False]
    ).iloc[0]
    best_params = json.loads(winner["params"])

    final_model = estimator_factory(best_params, seed, n_jobs)
    final_model.fit(X_search, y_search)
    train_seconds = time.perf_counter() - search_started

    inference_started = time.perf_counter()
    predictions = np.asarray(
        final_model.predict(flatten_features(data.test.X)), dtype=np.float32
    )
    inference_seconds = time.perf_counter() - inference_started

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(final_model, model_path)

    return TreeRunResult(
        model_name=model_name,
        predictions=predictions,
        best_params=best_params,
        search_results=results,
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
        model_path=model_path,
    )


def run_tree_model(
    model_name: str,
    data: DatasetBundle,
    output_dir: Path,
    **settings: Any,
) -> TreeRunResult:
    """Backward-compatible dispatcher; the implementations live per model."""

    if model_name == "xgboost":
        from .models.xgboost import run_xgboost

        return run_xgboost(data, output_dir, **settings)
    if model_name == "lightgbm":
        from .models.lightgbm import run_lightgbm

        return run_lightgbm(data, output_dir, **settings)
    raise ValueError(f"Unknown tree model: {model_name}")
