from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.multioutput import MultiOutputRegressor

from ..data import DatasetBundle
from ..tree_models import TreeRunResult, run_tree_search


FULL_GRID = {
    "num_leaves": [7, 15, 31],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 500, 1000],
    "min_child_samples": [20, 50],
    "colsample_bytree": [0.8, 1.0],
}

QUICK_GRID = {
    "num_leaves": [15],
    "learning_rate": [0.05],
    "n_estimators": [20],
    "min_child_samples": [20],
    "colsample_bytree": [1.0],
}


def _build_estimator(
    params: dict[str, Any], seed: int, n_jobs: int
) -> MultiOutputRegressor:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("Install lightgbm to run the LightGBM candidate") from exc

    base = LGBMRegressor(
        objective="regression",
        max_depth=-1,
        subsample=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
        **params,
    )
    return MultiOutputRegressor(base, n_jobs=n_jobs)


def run_lightgbm(
    data: DatasetBundle,
    output_dir: Path,
    **settings: Any,
) -> TreeRunResult:
    return run_tree_search(
        "lightgbm",
        data,
        output_dir,
        full_grid=FULL_GRID,
        quick_grid=QUICK_GRID,
        estimator_factory=_build_estimator,
        **settings,
    )
