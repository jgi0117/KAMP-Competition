from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from .data import DatasetBundle, flatten_features
from .metrics import regression_metrics
from .splitting import get_temporal_folds


@dataclass
class TreeRunResult:
    model_name: str
    device: str
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
    device: str = "cpu",
    full_grid: dict[str, list[Any]],
    quick_grid: dict[str, list[Any]],
    estimator_factory: Callable[[dict[str, Any], int, int, str], Any],
) -> TreeRunResult:
    if device not in {"cpu", "cuda"}:
        raise ValueError("tree device must be 'cpu' or 'cuda'")

    X_search = flatten_features(np.concatenate([data.train.X, data.val.X], axis=0))
    y_search = np.concatenate([data.train.y, data.val.y], axis=0)
    if len(X_search) <= cv_splits:
        raise ValueError(
            f"Need more than {cv_splits} train+validation samples for TimeSeriesSplit"
        )

    if param_grid is None:
        param_grid = full_grid if search_profile == "full" else quick_grid

    folds = get_temporal_folds(data, cv_splits)
    rows: list[dict[str, Any]] = []
    search_started = time.perf_counter()

    candidates = list(ParameterGrid(param_grid))
    with tqdm(
        total=len(candidates) * len(folds),
        desc=f"{model_name} grid search [{device}]",
        unit="fold",
        dynamic_ncols=True,
    ) as progress:
        for candidate_index, params in enumerate(candidates, start=1):
            fold_metrics: list[dict[str, float]] = []
            fold_started = time.perf_counter()
            for fold_number, split in enumerate(folds, start=1):
                estimator = estimator_factory(params, seed, n_jobs, device)
                estimator.fit(X_search[split.train], y_search[split.train])
                prediction = estimator.predict(X_search[split.evaluate])
                metrics = regression_metrics(y_search[split.evaluate], prediction)
                fold_metrics.append(metrics)
                rows.append(
                    {
                        "candidate": candidate_index,
                        "fold": split.name or fold_number,
                        "n_train": len(split.train),
                        "n_early_stop": len(split.early_stop),
                        "n_eval": len(split.evaluate),
                        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                        **metrics,
                    }
                )
                progress.set_postfix(
                    candidate=f"{candidate_index}/{len(candidates)}",
                    fold=split.name or fold_number,
                    rmse=f"{metrics['rmse']:.3f}",
                )
                progress.update(1)

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

    final_model = estimator_factory(best_params, seed, n_jobs, device)
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
        device=device,
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
